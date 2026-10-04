from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean, median
from typing import Dict, Iterable, List, Sequence, Set, Tuple
from xml.etree import ElementTree as ET

from .io_utils import write_csv, write_json
from .schemas import WorkOrder
from .text import normalize_text


LAYER_NAMES = {
    0: "发动机",
    1: "子系统",
    2: "零部件",
    3: "故障模式",
    4: "失效原因",
}

CAUSE_RULES = [
    ("ELECTRICAL_CONNECTION", "电气连接/线路故障", ("线束磨破", "线路磨损", "接触不良", "插头", "搭铁", "短路", "开路", "断路", "接线")),
    ("QUALITY_DEFECT", "制造/质量缺陷", ("质量问题", "质量原因", "材料缺陷", "制造缺陷")),
    ("SEAL_ASSEMBLY", "密封或装配不良", ("密封不严", "紧固不严", "密封失效", "卡箍松", "装配不当", "安装不到位", "松动漏")),
    ("CONTAMINATION_BLOCKAGE", "污染/异物/堵塞", ("异物堵塞", "堵塞", "积碳", "污染", "杂质", "内部有水", "进水")),
    ("SOFTWARE_CALIBRATION", "软件/标定/参数设置", ("参数设置有误", "标定问题", "程序问题", "软件问题", "数据问题")),
    ("WEAR_FATIGUE", "磨损/疲劳/老化", ("异常磨损", "疲劳", "老化", "长期磨损")),
    ("INSUFFICIENT_LUBRICATION", "润滑不足", ("润滑不足", "缺机油", "机油不足")),
    ("THERMAL_OVERLOAD", "过热/热损伤", ("高温导致", "过热导致", "烧蚀", "热损伤")),
    ("EXTERNAL_DAMAGE", "外部损伤", ("外力", "碰撞", "异物打坏", "外部损伤")),
]

NON_FAILURE_MATERIAL_TERMS = (
    "天然气专用油", "发动机专用油", "润滑油", "防冻液", "冷却液", "尿素液",
)

COMPONENT_CANONICAL_MAP = {
    "ECM标定": ("ECM_CONTROL_MODULE", "发动机控制模块(ECM)"),
    "电子故障检修": ("ECM_SYSTEM_UNLOCATED", "发动机电控系统（未定位部件）"),
    "气门间隙": ("VALVETRAIN_ASSEMBLY", "配气机构"),
    "15N整机": ("ENGINE_ASSEMBLY", "发动机本体"),
    "其它零件": ("UNLOCATED_COMPONENT", "未定位零部件"),
    "其他零件": ("UNLOCATED_COMPONENT", "未定位零部件"),
}


@dataclass
class GraphNode:
    node_id: str
    label: str
    layer: int
    node_type: str
    order_ids: Set[str] = field(default_factory=set)
    allocated_cost: float = 0.0


@dataclass
class GraphEdge:
    source: str
    target: str
    source_layer: int
    target_layer: int
    order_ids: Set[str] = field(default_factory=set)
    allocated_cost: float = 0.0


@dataclass
class PathAggregate:
    path_id: str
    node_ids: Tuple[str, str, str, str, str]
    labels: Tuple[str, str, str, str, str]
    order_ids: Set[str] = field(default_factory=set)
    allocated_costs: List[float] = field(default_factory=list)


@dataclass
class FaultGraphResult:
    nodes: Dict[str, GraphNode]
    edges: Dict[Tuple[str, str], GraphEdge]
    ranking: List[Dict[str, object]]
    summary: Dict[str, object]


def _safe_id(prefix: str, value: str) -> str:
    digest = hashlib.sha1(value.encode("utf-8")).hexdigest()[:16].upper()
    return "{}_{}".format(prefix, digest)


def _canonical_cause(chain: Dict[str, object]) -> Tuple[str, str]:
    cause_id = str(chain.get("cause_id") or "")
    cause_name = normalize_text(chain.get("cause_name") or "")
    if cause_id and cause_id != "UNKNOWN":
        return cause_id, cause_name or cause_id
    text = normalize_text(chain.get("cause_text") or "")
    for canonical_id, label, keywords in CAUSE_RULES:
        if any(keyword in text for keyword in keywords):
            return canonical_id, label
    return "UNDETERMINED", "失效原因未说明/未定位"


def _is_non_failure_material(chain: Dict[str, object]) -> bool:
    name = normalize_text(chain.get("component_name") or "")
    return any(term in name for term in NON_FAILURE_MATERIAL_TERMS)


def _canonical_component(chain: Dict[str, object]) -> Tuple[str, str]:
    original_id = str(chain.get("component_id") or "")
    name = normalize_text(chain.get("component_name") or "未识别部件")
    if name in COMPONENT_CANONICAL_MAP:
        return COMPONENT_CANONICAL_MAP[name]
    if name.startswith("其他类GENERAL"):
        return "UNLOCATED_COMPONENT", "未定位零部件"
    return original_id or _safe_id("COMP", name), name


def _path_from_chain(engine_name: str, chain: Dict[str, object]):
    subsystem_id = str(chain.get("subsystem_id") or "ENGINE_GENERAL")
    subsystem_name = normalize_text(chain.get("subsystem_name") or "发动机综合/未定位")
    component_id, component_name = _canonical_component(chain)
    mode_id = str(chain.get("failure_mode_id") or "FAILURE_MODE_UNDETERMINED")
    mode_name = normalize_text(chain.get("failure_mode_name") or "故障模式未定位")
    cause_id, cause_name = _canonical_cause(chain)
    node_ids = (
        "ENGINE_ROOT",
        "SUBSYSTEM:" + subsystem_id,
        "COMPONENT:" + component_id,
        "FAILURE_MODE:" + mode_id,
        "CAUSE:" + cause_id,
    )
    labels = (engine_name, subsystem_name, component_name, mode_name, cause_name)
    return node_ids, labels


def percentile(values: Sequence[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    left = int(math.floor(position))
    right = int(math.ceil(position))
    if left == right:
        return ordered[left]
    return ordered[left] + (ordered[right] - ordered[left]) * (position - left)


def winsorized_mean(values: Sequence[float], lower: float = 0.05, upper: float = 0.95) -> float:
    if not values:
        return 0.0
    low, high = percentile(values, lower), percentile(values, upper)
    return mean(min(high, max(low, value)) for value in values)


def wilson_interval(successes: int, total: int, z: float = 1.96) -> Tuple[float, float]:
    if total <= 0:
        return 0.0, 0.0
    p = successes / total
    denominator = 1 + z * z / total
    centre = p + z * z / (2 * total)
    margin = z * math.sqrt((p * (1 - p) + z * z / (4 * total)) / total)
    return max(0.0, (centre - margin) / denominator), min(1.0, (centre + margin) / denominator)


def entropy_weights(matrix: Sequence[Tuple[float, float]]) -> Tuple[float, float]:
    if not matrix:
        return 0.5, 0.5
    rows = len(matrix)
    if rows == 1:
        return 0.5, 0.5
    divergences = []
    scale = 1.0 / math.log(rows)
    for column in range(2):
        total = sum(max(0.0, row[column]) for row in matrix)
        if total <= 0:
            divergences.append(0.0)
            continue
        probabilities = [max(0.0, row[column]) / total for row in matrix]
        entropy = -scale * sum(p * math.log(p) for p in probabilities if p > 0)
        divergences.append(max(0.0, 1.0 - entropy))
    total_divergence = sum(divergences)
    if total_divergence <= 1e-12:
        return 0.5, 0.5
    return divergences[0] / total_divergence, divergences[1] / total_divergence


def topsis_scores(matrix: Sequence[Tuple[float, float]], weights: Tuple[float, float]) -> List[float]:
    if not matrix:
        return []
    denominators = [math.sqrt(sum(row[column] ** 2 for row in matrix)) or 1.0 for column in range(2)]
    weighted = [
        tuple((row[column] / denominators[column]) * weights[column] for column in range(2))
        for row in matrix
    ]
    ideal_best = [max(row[column] for row in weighted) for column in range(2)]
    ideal_worst = [min(row[column] for row in weighted) for column in range(2)]
    scores = []
    for row in weighted:
        distance_best = math.sqrt(sum((row[column] - ideal_best[column]) ** 2 for column in range(2)))
        distance_worst = math.sqrt(sum((row[column] - ideal_worst[column]) ** 2 for column in range(2)))
        denominator = distance_best + distance_worst
        scores.append(distance_worst / denominator if denominator else 1.0)
    return scores


def build_fault_graph(
    orders: Iterable[WorkOrder],
    predictions: Iterable[Dict[str, object]],
    engine_name: str = "15N发动机",
    shrinkage_strength: float = 5.0,
    entropy_strength: float = 0.5,
) -> FaultGraphResult:
    order_map = {order.work_order_id: order for order in orders}
    prediction_map = {str(row["work_order_id"]): row for row in predictions}
    nodes: Dict[str, GraphNode] = {}
    edges: Dict[Tuple[str, str], GraphEdge] = {}
    paths: Dict[Tuple[str, ...], PathAggregate] = {}
    fault_order_ids: Set[str] = set()
    candidate_fault_order_ids: Set[str] = set()
    known_cost_orders = 0
    input_fault_cost = 0.0
    allocated_fault_cost = 0.0
    excluded_material_chains = 0

    for order_id, order in order_map.items():
        prediction = prediction_map.get(order_id)
        if not prediction or prediction.get("record_type", "FAULT") != "FAULT":
            continue
        candidate_fault_order_ids.add(order_id)
        raw_paths = []
        for chain in prediction.get("fault_chains", []):
            if _is_non_failure_material(chain):
                excluded_material_chains += 1
                continue
            raw_paths.append(_path_from_chain(engine_name, chain))
        distinct = {(node_ids, labels) for node_ids, labels in raw_paths}
        if not distinct:
            continue
        fault_order_ids.add(order_id)
        if order.total_cost > 0:
            known_cost_orders += 1
            input_fault_cost += order.total_cost
        allocated_cost = order.total_cost / len(distinct) if order.total_cost > 0 else 0.0
        allocated_fault_cost += allocated_cost * len(distinct)
        for node_ids, labels in distinct:
            path_key = tuple(node_ids)
            if path_key not in paths:
                path_id = _safe_id("PATH", "|".join(path_key))
                paths[path_key] = PathAggregate(path_id, node_ids, labels)
            aggregate = paths[path_key]
            aggregate.order_ids.add(order_id)
            if allocated_cost > 0:
                aggregate.allocated_costs.append(allocated_cost)

            for layer, (node_id, label) in enumerate(zip(node_ids, labels)):
                if node_id not in nodes:
                    nodes[node_id] = GraphNode(node_id, label, layer, LAYER_NAMES[layer])
                nodes[node_id].order_ids.add(order_id)
                nodes[node_id].allocated_cost += allocated_cost
            for layer in range(4):
                edge_key = (node_ids[layer], node_ids[layer + 1])
                if edge_key not in edges:
                    edges[edge_key] = GraphEdge(edge_key[0], edge_key[1], layer, layer + 1)
                edges[edge_key].order_ids.add(order_id)
                edges[edge_key].allocated_cost += allocated_cost

    all_allocations = [cost for path in paths.values() for cost in path.allocated_costs]
    global_cost_median = median(all_allocations) if all_allocations else 0.0
    path_rows = []
    total_fault_orders = len(fault_order_ids)
    for path in paths.values():
        count = len(path.order_ids)
        costs = path.allocated_costs
        robust_mean = winsorized_mean(costs)
        shrunk_cost = (
            (len(costs) * robust_mean + shrinkage_strength * global_cost_median)
            / (len(costs) + shrinkage_strength)
            if costs
            else 0.0
        )
        low, high = wilson_interval(count, total_fault_orders)
        path_rows.append({
            "path_id": path.path_id,
            "engine_node_id": path.node_ids[0],
            "subsystem_node_id": path.node_ids[1],
            "component_node_id": path.node_ids[2],
            "failure_mode_node_id": path.node_ids[3],
            "cause_node_id": path.node_ids[4],
            "engine": path.labels[0],
            "subsystem": path.labels[1],
            "component": path.labels[2],
            "failure_mode": path.labels[3],
            "failure_cause": path.labels[4],
            "cause_identified": not path.node_ids[4].endswith("UNDETERMINED"),
            "path_evidence_grade": (
                "LOW" if path.node_ids[2].endswith("UNLOCATED_COMPONENT")
                else "HIGH" if not path.node_ids[4].endswith("UNDETERMINED")
                else "MEDIUM"
            ),
            "unique_work_orders": count,
            "frequency_rate": count / total_fault_orders if total_fault_orders else 0.0,
            "frequency_ci_low": low,
            "frequency_ci_high": high,
            "cost_observations": len(costs),
            "cost_coverage": len(costs) / count if count else 0.0,
            "allocated_total_cost": sum(costs),
            "allocated_mean_cost": mean(costs) if costs else 0.0,
            "allocated_median_cost": median(costs) if costs else 0.0,
            "allocated_p95_cost": percentile(costs, 0.95),
            "winsorized_mean_cost": robust_mean,
            "shrunk_cost_severity": shrunk_cost,
            "sample_work_orders": "|".join(sorted(path.order_ids)[:8]),
        })

    decision_matrix = [
        (math.log1p(row["unique_work_orders"]), math.log1p(row["shrunk_cost_severity"]))
        for row in path_rows
    ]
    raw_entropy_weights = entropy_weights(decision_matrix)
    entropy_strength = min(1.0, max(0.0, entropy_strength))
    weights = (
        entropy_strength * raw_entropy_weights[0] + (1.0 - entropy_strength) * 0.5,
        entropy_strength * raw_entropy_weights[1] + (1.0 - entropy_strength) * 0.5,
    )
    scores = topsis_scores(decision_matrix, weights)
    for row, score in zip(path_rows, scores):
        row["frequency_weight"] = weights[0]
        row["cost_weight"] = weights[1]
        row["topsis_score"] = score
    path_rows.sort(
        key=lambda row: (
            -row["topsis_score"], -row["unique_work_orders"], -row["allocated_total_cost"], row["path_id"]
        )
    )
    count_paths = len(path_rows)
    for rank, row in enumerate(path_rows, start=1):
        percentile_rank = rank / max(1, count_paths)
        row["rank"] = rank
        row["priority"] = "P1" if percentile_rank <= 0.10 else "P2" if percentile_rank <= 0.30 else "P3" if percentile_rank <= 0.60 else "P4"

    summary = {
        "engine_root": engine_name,
        "fault_work_orders": total_fault_orders,
        "candidate_fault_work_orders": len(candidate_fault_order_ids),
        "material_only_excluded_orders": len(candidate_fault_order_ids - fault_order_ids),
        "known_cost_orders": known_cost_orders,
        "cost_coverage": known_cost_orders / total_fault_orders if total_fault_orders else 0.0,
        "input_fault_cost": input_fault_cost,
        "allocated_fault_cost": allocated_fault_cost,
        "cost_conservation_error": allocated_fault_cost - input_fault_cost,
        "nodes": len(nodes),
        "edges": len(edges),
        "paths": len(path_rows),
        "paths_with_identified_cause": sum(bool(row["cause_identified"]) for row in path_rows),
        "paths_with_unidentified_cause": sum(not bool(row["cause_identified"]) for row in path_rows),
        "frequency_weight": weights[0],
        "cost_weight": weights[1],
        "raw_entropy_frequency_weight": raw_entropy_weights[0],
        "raw_entropy_cost_weight": raw_entropy_weights[1],
        "entropy_strength": entropy_strength,
        "ranking_method": "log transform + empirical-Bayes cost shrinkage + regularized entropy-weight TOPSIS",
        "shrinkage_strength": shrinkage_strength,
        "excluded_non_failure_material_chains": excluded_material_chains,
    }
    return FaultGraphResult(nodes, edges, path_rows, summary)


def export_fault_graph(result: FaultGraphResult, output_dir: str) -> None:
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    node_rows = [
        {
            "node_id": node.node_id,
            "label": node.label,
            "layer": node.layer,
            "node_type": node.node_type,
            "unique_work_orders": len(node.order_ids),
            "allocated_cost": round(node.allocated_cost, 2),
        }
        for node in sorted(result.nodes.values(), key=lambda item: (item.layer, item.label, item.node_id))
    ]
    edge_rows = [
        {
            "source": edge.source,
            "target": edge.target,
            "source_layer": edge.source_layer,
            "target_layer": edge.target_layer,
            "unique_work_orders": len(edge.order_ids),
            "allocated_cost": round(edge.allocated_cost, 2),
        }
        for edge in sorted(result.edges.values(), key=lambda item: (item.source_layer, item.source, item.target))
    ]
    ranking_rows = []
    for row in result.ranking:
        ranking_rows.append({
            **row,
            "frequency_rate": round(row["frequency_rate"], 8),
            "frequency_ci_low": round(row["frequency_ci_low"], 8),
            "frequency_ci_high": round(row["frequency_ci_high"], 8),
            "cost_coverage": round(row["cost_coverage"], 6),
            "allocated_total_cost": round(row["allocated_total_cost"], 2),
            "allocated_mean_cost": round(row["allocated_mean_cost"], 2),
            "allocated_median_cost": round(row["allocated_median_cost"], 2),
            "allocated_p95_cost": round(row["allocated_p95_cost"], 2),
            "winsorized_mean_cost": round(row["winsorized_mean_cost"], 2),
            "shrunk_cost_severity": round(row["shrunk_cost_severity"], 2),
            "frequency_weight": round(row["frequency_weight"], 6),
            "cost_weight": round(row["cost_weight"], 6),
            "topsis_score": round(row["topsis_score"], 8),
        })
    write_csv(str(target / "fault_graph_nodes.csv"), list(node_rows[0]) if node_rows else [], node_rows)
    write_csv(str(target / "fault_graph_edges.csv"), list(edge_rows[0]) if edge_rows else [], edge_rows)
    write_csv(str(target / "fault_path_ranking.csv"), list(ranking_rows[0]) if ranking_rows else [], ranking_rows)
    write_json(str(target / "fault_graph_summary.json"), result.summary)
    _write_graphml(result, str(target / "fault_graph.graphml"))


def _write_graphml(result: FaultGraphResult, path: str) -> None:
    namespace = "http://graphml.graphdrawing.org/xmlns"
    ET.register_namespace("", namespace)
    root = ET.Element("{%s}graphml" % namespace)
    for key_id, target, name, attr_type in [
        ("label", "node", "label", "string"),
        ("layer", "node", "layer", "int"),
        ("node_type", "node", "node_type", "string"),
        ("orders", "all", "unique_work_orders", "int"),
        ("cost", "all", "allocated_cost", "double"),
    ]:
        ET.SubElement(root, "{%s}key" % namespace, id=key_id, **{"for": target, "attr.name": name, "attr.type": attr_type})
    graph = ET.SubElement(root, "{%s}graph" % namespace, id="fault_graph", edgedefault="directed")
    for node in result.nodes.values():
        element = ET.SubElement(graph, "{%s}node" % namespace, id=node.node_id)
        for key, value in [
            ("label", node.label), ("layer", node.layer), ("node_type", node.node_type),
            ("orders", len(node.order_ids)), ("cost", round(node.allocated_cost, 2)),
        ]:
            ET.SubElement(element, "{%s}data" % namespace, key=key).text = str(value)
    for index, edge in enumerate(result.edges.values(), start=1):
        element = ET.SubElement(
            graph, "{%s}edge" % namespace, id="e{}".format(index), source=edge.source, target=edge.target
        )
        ET.SubElement(element, "{%s}data" % namespace, key="orders").text = str(len(edge.order_ids))
        ET.SubElement(element, "{%s}data" % namespace, key="cost").text = str(round(edge.allocated_cost, 2))
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)
