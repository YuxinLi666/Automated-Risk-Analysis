from __future__ import annotations

from typing import Dict, Iterable, List

from .schemas import AnnotatedDocument, WorkOrder


ANNOTATION_GUIDE = {
    "entity_labels": ["COMPONENT", "FAILURE_MODE", "CAUSE", "SYMPTOM", "DTC"],
    "relation_labels": ["HAS_FAILURE_MODE", "CAUSED_BY", "INDICATES"],
    "rules": [
        "只标原文证据，不根据常识补写原因",
        "故障症状、故障模式、失效原因必须分开",
        "否定或排除项不标为正向实体",
        "同一工单可以有多条故障链",
        "原因未说明时在chain中使用UNKNOWN，不伪造CAUSE跨度",
    ],
}


def export_annotation_documents(orders: Iterable[WorkOrder], split_map: Dict[str, str] = None) -> List[Dict[str, object]]:
    split_map = split_map or {}
    documents = []
    for order in orders:
        document = AnnotatedDocument(
            document_id=order.work_order_id,
            text=order.model_text(),
            split=split_map.get(order.work_order_id, ""),
        )
        payload = document.to_dict()
        payload["metadata"] = {
            "engine_id": order.engine_id,
            "chassis_id": order.chassis_id,
            "fault_date": order.fault_date,
            "source_rows": order.source_rows,
        }
        documents.append(payload)
    return documents

