from __future__ import annotations

import hashlib
import random
import re
from collections import defaultdict
from typing import Dict, Iterable, List, Tuple

from .schemas import WorkOrder
from .text import normalize_text


def _template_signature(order: WorkOrder) -> str:
    text = normalize_text(order.fault_name + " " + order.fault_description + " " + order.diagnosis)
    text = re.sub(r"\d+", "#", text)
    text = re.sub(r"[^\w\u4e00-\u9fff]+", "", text)
    return hashlib.sha1(text[:200].encode("utf-8")).hexdigest()[:12]


def group_key(order: WorkOrder) -> str:
    identity = order.chassis_id or order.engine_id
    return "{}|{}".format(identity, _template_signature(order))


def _connected_groups(orders: List[WorkOrder]) -> List[List[WorkOrder]]:
    """Join records connected by chassis, engine, work-order ID, or template.

    A concatenated key is insufficient: the same chassis with a different text
    template (or the same template on a different chassis) could then leak into
    another split.  Connected components enforce every constraint separately.
    """

    parent = list(range(len(orders)))
    size = [1] * len(orders)

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root == right_root:
            return
        if size[left_root] < size[right_root]:
            left_root, right_root = right_root, left_root
        parent[right_root] = left_root
        size[left_root] += size[right_root]

    first_seen: Dict[Tuple[str, str], int] = {}
    for index, order in enumerate(orders):
        tokens = [
            ("work_order", normalize_text(order.work_order_id)),
            ("chassis", normalize_text(order.chassis_id)),
            ("engine", normalize_text(order.engine_id)),
            ("template", _template_signature(order)),
        ]
        for kind, value in tokens:
            if not value:
                continue
            token = kind, value.lower()
            if token in first_seen:
                union(index, first_seen[token])
            else:
                first_seen[token] = index

    groups: Dict[int, List[WorkOrder]] = defaultdict(list)
    for index, order in enumerate(orders):
        groups[find(index)].append(order)
    return list(groups.values())


def grouping_audit(orders: Iterable[WorkOrder]) -> Dict[str, object]:
    values = list(orders)
    groups = _connected_groups(values)
    sizes = sorted((len(group) for group in groups), reverse=True)
    total = len(values)
    return {
        "grouping_constraints": ["work_order_id", "chassis_id", "engine_id", "normalized_text_template"],
        "connected_groups": len(groups),
        "largest_connected_group": sizes[0] if sizes else 0,
        "largest_connected_group_ratio": sizes[0] / total if sizes and total else 0.0,
        "top_connected_group_sizes": sizes[:10],
        "orders_without_chassis_id": sum(not normalize_text(order.chassis_id) for order in values),
        "orders_without_engine_id": sum(not normalize_text(order.engine_id) for order in values),
    }


def grouped_split(
    orders: Iterable[WorkOrder],
    train_ratio: float = 0.70,
    validation_ratio: float = 0.15,
    seed: int = 42,
) -> Dict[str, List[WorkOrder]]:
    if not 0.0 < train_ratio < 1.0:
        raise ValueError("train_ratio must be between 0 and 1")
    if not 0.0 <= validation_ratio < 1.0 or train_ratio + validation_ratio >= 1.0:
        raise ValueError("train_ratio + validation_ratio must be smaller than 1")
    values = list(orders)
    groups = _connected_groups(values)
    random.Random(seed).shuffle(groups)
    groups.sort(key=len, reverse=True)
    total = len(values)
    targets = {
        "train": total * train_ratio,
        "validation": total * validation_ratio,
        "test": total * (1.0 - train_ratio - validation_ratio),
    }
    result = {"train": [], "validation": [], "test": []}
    for group in groups:
        target = max(
            result,
            key=lambda name: (
                (targets[name] - len(result[name])) / max(1.0, targets[name]),
                targets[name],
            ),
        )
        result[target].extend(group)
    return result
