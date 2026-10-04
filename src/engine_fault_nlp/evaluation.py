from __future__ import annotations

from collections import Counter
from typing import Dict, Iterable, List, Set, Tuple


FIELDS = ("subsystem_id", "component_id", "failure_mode_id", "cause_id")


def _pairs(chains: List[Dict[str, object]], field: str) -> Set[str]:
    return {str(chain[field]) for chain in chains if chain.get(field)}


def precision_recall_f1(tp: int, fp: int, fn: int) -> Dict[str, float]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "support": tp + fn}


def evaluate_records(predictions: Iterable[Dict[str, object]], gold: Iterable[Dict[str, object]]) -> Dict[str, object]:
    pred_map = {str(row["work_order_id"]): row for row in predictions}
    gold_map = {str(row.get("work_order_id", row.get("document_id"))): row for row in gold}
    counts = {field: Counter() for field in FIELDS}
    exact = 0
    total = 0
    for order_id, gold_row in gold_map.items():
        gold_chains = gold_row.get("fault_chains", gold_row.get("chains", []))
        pred_chains = pred_map.get(order_id, {}).get("fault_chains", [])
        gold_tuples = {
            tuple(chain.get(field) for field in FIELDS) for chain in gold_chains
        }
        pred_tuples = {
            tuple(chain.get(field) for field in FIELDS) for chain in pred_chains
        }
        exact += int(gold_tuples == pred_tuples)
        total += 1
        for field in FIELDS:
            expected = _pairs(gold_chains, field)
            actual = _pairs(pred_chains, field)
            counts[field]["tp"] += len(expected & actual)
            counts[field]["fp"] += len(actual - expected)
            counts[field]["fn"] += len(expected - actual)
    metrics = {
        field: precision_recall_f1(count["tp"], count["fp"], count["fn"])
        for field, count in counts.items()
    }
    metrics["chain_exact_match"] = exact / total if total else 0.0
    metrics["documents"] = total
    return metrics

