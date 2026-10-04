from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Tuple

from .io_utils import read_json, write_json
from .schemas import FaultChain


def wilson_lower_bound(correct: int, total: int, z: float = 1.96) -> float:
    if total <= 0:
        return 0.0
    p = correct / total
    denominator = 1 + z * z / total
    centre = p + z * z / (2 * total)
    margin = z * math.sqrt((p * (1 - p) + z * z / (4 * total)) / total)
    return max(0.0, (centre - margin) / denominator)


def select_threshold(
    scored_correctness: Iterable[Tuple[float, bool]],
    target_precision_lower_bound: float,
    minimum_accepted: int = 30,
) -> Dict[str, float]:
    values = sorted(scored_correctness, key=lambda pair: pair[0], reverse=True)
    best = {"threshold": 1.01, "accepted": 0, "precision": 0.0, "precision_lower_bound": 0.0}
    correct = 0
    for index, (score, is_correct) in enumerate(values, start=1):
        correct += int(is_correct)
        if index < minimum_accepted:
            continue
        lower = wilson_lower_bound(correct, index)
        if lower >= target_precision_lower_bound:
            best = {
                "threshold": score,
                "accepted": index,
                "precision": correct / index,
                "precision_lower_bound": lower,
            }
    return best


@dataclass
class ThresholdPolicy:
    thresholds: Dict[str, float] = field(default_factory=lambda: {
        "subsystem": 0.98,
        "component": 0.98,
        "failure_mode": 0.95,
        "cause": 0.92,
        "chain": 0.90,
    })

    def decide(self, chain: FaultChain, has_warnings: bool = False) -> str:
        if has_warnings:
            return "REVIEW"
        required = ("subsystem", "component", "failure_mode")
        if all(chain.confidence.get(field, 0.0) >= self.thresholds.get(field, 1.0) for field in required):
            if chain.cause_id in (None, "UNKNOWN") or chain.confidence.get("cause", 0.0) >= self.thresholds.get("cause", 1.0):
                return "ACCEPT"
        return "REVIEW"

    def save(self, path: str) -> None:
        write_json(path, {"thresholds": self.thresholds})

    @classmethod
    def load(cls, path: str) -> "ThresholdPolicy":
        return cls(read_json(path).get("thresholds", {}))

