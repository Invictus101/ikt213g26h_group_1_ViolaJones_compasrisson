import time

import numpy as np

from src.TrackedFace import TrackedFace
from src.utils.Detection import Detection


class FaceTracker:
    # how far a detection can be from a tracked face and still counting as the same face. measured relative to face edge diagonal
    MAX_MATCH_DIST_FACTOR = 0.5

    def __init__(self, sface_threshold, candidate_threshold):
        self.sface_threshold = sface_threshold
        self.candidate_threshold = candidate_threshold
        self.trackedFaces: list[TrackedFace] = []

    def process_raw_detections(self, detections: list[Detection], now: float | None = None):
        """takes in raw detections, this happens every single frame. now: timestamp in seconds, defaults to current time"""
        if now is None:
            now = time.monotonic()
        matches, new_detections = self.match(detections)

        matched_tracks = set()
        for tracked_i, detected_i in matches:
            detection = detections[detected_i]
            self.trackedFaces[tracked_i].process_new_data(detection.pos, detection.size, detection.scores, now, detection.spoofing)
            matched_tracks.add(tracked_i)

        # trackedFaces without a detection this frame are kept, but get weaker over time (see TrackedFace.time_factor)
        for tracked_i, tracked_face in enumerate(self.trackedFaces):
            if tracked_i not in matched_tracks:
                tracked_face.mark_missing(now)

        for detected_i in new_detections:
            detection = detections[detected_i]
            self.trackedFaces.append(TrackedFace(detection.pos, detection.size, detection.scores, self.sface_threshold, self.candidate_threshold, now))

        # remove trackedFaces that have been missing for too long
        self.trackedFaces = [face for face in self.trackedFaces if not face.is_lost]

    def match(self, detections: list[Detection]) -> tuple[list[tuple[int, int]], list[int]]:
        """finds out which detection is conected with each trackedFace, returns a list of pair<trackedFace, detection>"""
        # make pair<track, detection> for each track x direction, but only if close enough
        pairs = []
        for track_i, track in enumerate(self.trackedFaces):
            for detection_i, detection in enumerate(detections):
                dist = np.linalg.norm(detection.pos - track.pos)
                if dist <= track.diagonal * self.MAX_MATCH_DIST_FACTOR:
                    pairs.append((dist, track_i, detection_i))

        # shorter distance at the top
        pairs.sort()

        # taking the shortest first bc theyre more certain
        matches: list[tuple[int, int]] = []
        used_tracks = set()
        used_detections = set()
        for dist, track_i, detection_i in pairs:
            if track_i in used_tracks or detection_i in used_detections:
                continue
            matches.append((track_i, detection_i))
            used_tracks.add(track_i)
            used_detections.add(detection_i)

        # new detections not conected to a TrackedFace
        newDetections = [detected_i for detected_i in range(len(detections)) if detected_i not in used_detections] # = detections.indices.filter { it !in usedDetections }
        return matches, newDetections # would have used the objs directly if this was kt
