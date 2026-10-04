from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, Iterator, List, Tuple

from .schemas import PartMention, WorkOrder
from .text import deduplicate_fragments, normalize_text, parse_dtc_codes
from .xlsx import XlsxReader


COLUMNS = {
    "id": "工单：序号",
    "settlement_date": "工单结算日期",
    "fault_date": "故障日期",
    "engine_id": "发动机",
    "engine_series": "发动机系列",
    "province": "省份",
    "city": "城市",
    "chassis_id": "底盘号",
    "mileage": "故障: 行驶里程",
    "total_cost": "工单费用-报表专用",
    "fault_name": "故障: 故障名称",
    "description": "故障: 故障描述",
    "diagnosis": "故障: 故障原因",
    "solution": "故障: 解决方案",
    "part_no": "旧零件",
    "part_name": "旧零件名称",
    "culprit": "元凶件",
    "dtc": "故障: 故障诊断码",
}

# The full ``Report`` sheet contains hundreds of derived/formula columns.  The
# method is allowed to read only these raw source columns.  Positions are
# zero-based and are validated against their headers before any record is
# yielded, so a shifted workbook fails closed instead of silently ingesting a
# derived label column.
REPORT_SOURCE_COLUMN_INDICES = {
    "id": 0,                 # A
    "settlement_date": 2,    # C
    "engine_id": 7,          # H
    "engine_series": 8,      # I
    "province": 9,           # J
    "city": 10,              # K
    "chassis_id": 11,        # L (the historical pipeline's primary chassis field)
    "mileage": 16,           # Q
    "fault_name": 49,        # AX
    "description": 160,      # FE
    "diagnosis": 270,        # JK
    "solution": 274,         # JO
    "part_no": 275,          # JP
    "part_name": 276,        # JQ
    "culprit": 277,          # JR
    "dtc": 278,              # JS
    "total_cost": 289,       # KD
}


def _get(row: Dict[str, str], key: str) -> str:
    return normalize_text(row.get(COLUMNS[key], ""))


def _get_float(row: Dict[str, str], key: str) -> float:
    raw = _get(row, key).replace(",", "").replace("¥", "").replace("￥", "")
    try:
        return max(0.0, float(raw))
    except ValueError:
        return 0.0


def load_report_rows(path: str, sheet_name: str = "Report") -> Iterator[Tuple[int, Dict[str, str]]]:
    with XlsxReader(path) as workbook:
        if sheet_name != "Report":
            yield from workbook.rows_as_dicts(sheet_name)
            return
        rows = workbook.iter_rows(sheet_name)
        try:
            _, headers = next(rows)
        except StopIteration:
            return
        for key, index in REPORT_SOURCE_COLUMN_INDICES.items():
            actual = headers[index] if index < len(headers) else ""
            expected = COLUMNS[key]
            if actual != expected:
                raise ValueError(
                    "Report source column mismatch at index {}: expected {!r}, got {!r}".format(
                        index + 1, expected, actual
                    )
                )
        for row_number, values in rows:
            record = {
                COLUMNS[key]: values[index] if index < len(values) else ""
                for key, index in REPORT_SOURCE_COLUMN_INDICES.items()
            }
            yield row_number, record


def aggregate_work_orders(rows: Iterable[Tuple[int, Dict[str, str]]]) -> List[WorkOrder]:
    groups: Dict[str, List[Tuple[int, Dict[str, str]]]] = defaultdict(list)
    for row_number, row in rows:
        order_id = _get(row, "id")
        if order_id:
            groups[order_id].append((row_number, row))

    orders = []
    for order_id, group in groups.items():
        first = group[0][1]
        parts = []
        part_keys = set()
        for _, row in group:
            part = PartMention(
                part_no=_get(row, "part_no"),
                part_name=_get(row, "part_name"),
                culprit=_get(row, "culprit") in ("是", "true", "1", "Y", "yes"),
            )
            key = (part.part_no, part.part_name, part.culprit)
            if (part.part_no or part.part_name) and key not in part_keys:
                part_keys.add(key)
                parts.append(part)
        dtc_raw = deduplicate_fragments(_get(row, "dtc") for _, row in group)
        total_cost = max((_get_float(row, "total_cost") for _, row in group), default=0.0)
        orders.append(
            WorkOrder(
                work_order_id=order_id,
                settlement_date=_get(first, "settlement_date"),
                fault_date=_get(first, "fault_date"),
                engine_id=_get(first, "engine_id"),
                engine_series=_get(first, "engine_series"),
                chassis_id=_get(first, "chassis_id"),
                province=_get(first, "province"),
                city=_get(first, "city"),
                mileage=_get(first, "mileage"),
                total_cost=total_cost,
                fault_name=deduplicate_fragments(_get(row, "fault_name") for _, row in group),
                fault_description=deduplicate_fragments(_get(row, "description") for _, row in group),
                diagnosis=deduplicate_fragments(_get(row, "diagnosis") for _, row in group),
                solution=deduplicate_fragments(_get(row, "solution") for _, row in group),
                dtc_raw=dtc_raw,
                dtc_codes=parse_dtc_codes(dtc_raw),
                parts=parts,
                source_rows=[row_number for row_number, _ in group],
            )
        )
    return sorted(orders, key=lambda item: item.work_order_id)


def load_work_orders(path: str, sheet_name: str = "Report") -> List[WorkOrder]:
    return aggregate_work_orders(load_report_rows(path, sheet_name))


def audit_workbook(path: str, sheet_name: str = "Report") -> Dict[str, object]:
    rows = list(load_report_rows(path, sheet_name))
    orders = aggregate_work_orders(rows)
    duplicate_groups = sum(1 for order in orders if len(order.source_rows) > 1)
    return {
        "sheet": sheet_name,
        "rows": len(rows),
        "rows_with_work_order_id": sum(len(order.source_rows) for order in orders),
        "rows_without_work_order_id": len(rows) - sum(len(order.source_rows) for order in orders),
        "unique_work_orders": len(orders),
        "duplicate_work_order_groups": duplicate_groups,
        "max_rows_per_work_order": max((len(order.source_rows) for order in orders), default=0),
        "orders_with_numeric_dtc": sum(bool(order.dtc_codes) for order in orders),
        "unique_part_numbers": len({p.part_no for o in orders for p in o.parts if p.part_no}),
        "unique_part_names": len({p.part_name for o in orders for p in o.parts if p.part_name}),
    }
