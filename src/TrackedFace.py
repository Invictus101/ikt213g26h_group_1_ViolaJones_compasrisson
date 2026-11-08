import numpy as np


# --- confidence = similarity_base * certainty_factor ---
# similarity is the main measure (how well the face is recognized), certainty can only pull it DOWN
# when someone else is close on second place, never up.

# similarity_base = logistic curve, 0.5 at sface_threshold. higher = steeper, more gets pushed towards 0/1
SIMILARITY_STEEPNESS = 8
# certainty_factor = 1 - (1 - MIN_CERTAINTY_FACTOR) * e^(-certainty / CERTAINTY_FALLOFF)
# MIN_CERTAINTY_FACTOR: factor when certainty is 0 (two persons equally similar).
#   constraints: >= 0.5 -> never red above sface_threshold, < 0.75 -> never green when certainty is 0
MIN_CERTAINTY_FACTOR = 0.6
# CERTAINTY_FALLOFF: how fast the penalty disappears as certainty grows. ~63% of it is gone at this certainty
CERTAINTY_FALLOFF = 0.05

# time penalty when the face isnt detected: time_factor = e^(-TIME_PENALTY * seconds_missing²)
# x² growth, so a few missed frames barely matter, but long gaps fall off fast
# the trackedFace is removed when time_factor < LOST_THRESHOLD, which happens after LOST_AFTER_SECONDS
LOST_THRESHOLD = 0.05
LOST_AFTER_SECONDS = 1.0
TIME_PENALTY = np.log(1 / LOST_THRESHOLD) / LOST_AFTER_SECONDS ** 2

# velocity
VEL_SMOOTHING = 0.3 # moving avg (exp) weight for the newest measurement, since the detector values are jittery. lower = smoother, but slower to react.
VEL_KEEP_PER_SECOND = 0.05 # when missing, how much of the speed is left after 1 s. so it glides to a stop instead of flying off


class TrackedFace:
    def __init__(self, pos: np.ndarray, size: np.ndarray, scores, sface_threshold: float, candidate_threshold: float, now: float):
        self.sface_threshold = sface_threshold # similarity score needed to count as recognized
        self.candidate_threshold = candidate_threshold # min score for a person to be included in scores
        self.vel = np.zeros(2) # px/s
        self.last_pos = pos # vec2, last seen
        self.pos = pos # vec2, current position. same as last_pos when detected, predicted from vel when missing
        self.size = size # vec2
        self.scores = scores # dict med person -> score
        self.last_seen = now # timestamp (s) of the last frame detected
        self.last_update = now # timestamp (s) of the last frame
        self.seconds_missing = 0.0 # time since last_seen
        self.spoofing = {"label":"fakereal", "real_score":0.67}

    def process_new_data(self, detected_pos, detected_size, scores, now: float, spoofing):
        """called per frame, new data is inputted and processed"""
        Δt = now - self.last_seen
        # vel from 2 newest detected positions
        measured_vel = (detected_pos - self.last_pos) / Δt
        self.vel = VEL_SMOOTHING * measured_vel + (1 - VEL_SMOOTHING) * self.vel # smooth so its not choppy and too fast
        self.last_pos = detected_pos
        self.pos = detected_pos
        self.size = detected_size
        self.scores = scores # will interpolate based on data, instead of just replacing. this is to prevent flickering and losing track from just 1/a few failed frame
        self.last_seen = now
        self.last_update = now
        self.seconds_missing = 0.0
        self.spoofing = spoofing

    def mark_missing(self, now: float):
        """called per frame when the face isnt detected. moves pos by vel, then slows vel down"""
        Δt = now - self.last_update
        self.pos = self.pos + self.vel * Δt
        self.vel = self.vel * VEL_KEEP_PER_SECOND ** Δt # speed decreases, and faster for longer delays
        self.last_update = now
        self.seconds_missing = now - self.last_seen

    @property
    def is_missing(self) -> bool:
        """not detected this frame"""
        return self.seconds_missing > 0

    @property
    def time_factor(self) -> float:
        """0-1, penalty for not being detected. 1 = detected this frame"""
        return float(np.exp(-TIME_PENALTY * self.seconds_missing ** 2))

    @property
    def is_lost(self) -> bool:
        """missing for so long that this trackedPerson should be removed"""
        return self.time_factor < LOST_THRESHOLD

    @property
    def diagonal(self) -> float:
        """distance between opposite corners"""
        return np.linalg.norm(self.size) # pytagoras

    @property
    def box(self) -> np.ndarray:
        """(x, y, w, h) with top left corner"""
        top_left = self.pos - self.size / 2
        return np.array([top_left[0], top_left[1], self.size[0], self.size[1]])

    @property
    def best_match(self) -> tuple[str, float]:
        """person with the highest score. (name, score) ?: ("Unknown", 0.0)"""
        return max(self.scores.items(), key=lambda item: item[1], default=("Unknown", 0.0))

    @property
    def certainty(self) -> float:
        """ diff between first and second score, if exists. if there isnt any scores then 0.
        if just 1 score, then pretend second is right at candidate_threshold (the edge of being included)"""
        top = sorted(self.scores.values(), reverse=True)
        if len(top) == 0:
            return 0.0
        if len(top) == 1:
            return top[0] - self.candidate_threshold
        return top[0] - top[1]

    @property
    def similarity_base(self) -> float:
        """0-1, how well the face is recognized. logistic curve, 0.5 at sface_threshold"""
        _, similarity = self.best_match
        return float(1 / (1 + np.exp(-SIMILARITY_STEEPNESS * (similarity - self.sface_threshold))))

    @property
    def certainty_factor(self) -> float:
        """MIN_CERTAINTY_FACTOR-1, penalty for someone close on second place. 1 = no penalty"""
        return float(1 - (1 - MIN_CERTAINTY_FACTOR) * np.exp(-self.certainty / CERTAINTY_FALLOFF))

    @property
    def certainty_strength(self) -> float:
        """0-1, certainty_factor stretched to 0-1 (0 = max penalty, 1 = no penalty). used for coloring"""
        return (self.certainty_factor - MIN_CERTAINTY_FACTOR) / (1 - MIN_CERTAINTY_FACTOR)

    @property
    def confidence(self) -> float:
        """0-1, similarity_base pulled down by certainty_factor and time_factor"""
        return self.similarity_base * self.certainty_factor * self.time_factor
