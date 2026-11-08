from dataclasses import dataclass
import numpy as np

@dataclass
class Detection:
    pos: np.ndarray
    size: np.ndarray
    scores: dict[str, float] # map<>
    spoofing: dict[str, str | float]