from __future__ import annotations

from typing import Dict, Iterable, List, Set

from .schemas import WorkOrder
from .text import character_ngrams


def _uncertainty(prediction: Dict[str, object]) -> float:
    chains = prediction.get("fault_chains", [])
    if not chains:
        return 1.0
    lowest = 1.0
    for chain in chains:
        confidence = chain.get("confidence", {})
        fields = [confidence.get(field, 0.0) for field in ("subsystem", "component", "failure_mode", "cause")]
        lowest = min(lowest, min(fields))
        if chain.get("decision") == "REVIEW":
            lowest = min(lowest, 0.49)
    warning_bonus = min(0.25, 0.05 * len(prediction.get("warnings", [])))
    return min(1.0, 1.0 - lowest + warning_bonus)


def _signature(order: WorkOrder) -> Set[str]:
    return set(character_ngrams(order.model_text(), (2, 3)))


def _jaccard(left: Set[str], right: Set[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 0.0


def select_for_annotation(
    orders: Iterable[WorkOrder], predictions: Iterable[Dict[str, object]], limit: int = 100
) -> List[WorkOrder]:
    order_map = {order.work_order_id: order for order in orders}
    ranked = sorted(
        ((str(row["work_order_id"]), _uncertainty(row)) for row in predictions if str(row["work_order_id"]) in order_map),
        key=lambda pair: pair[1],
        reverse=True,
    )
    selected: List[WorkOrder] = []
    signatures: List[Set[str]] = []
    pool = ranked[: max(limit * 8, limit)]
    while pool and len(selected) < limit:
        best_index = 0
        best_score = -1.0
        for index, (order_id, uncertainty) in enumerate(pool):
            signature = _signature(order_map[order_id])
            diversity = 1.0 if not signatures else 1.0 - max(_jaccard(signature, existing) for existing in signatures)
            score = 0.75 * uncertainty + 0.25 * diversity
            if score > best_score:
                best_score, best_index = score, index
        order_id, _ = pool.pop(best_index)
        order = order_map[order_id]
        selected.append(order)
        signatures.append(_signature(order))
    return selected

