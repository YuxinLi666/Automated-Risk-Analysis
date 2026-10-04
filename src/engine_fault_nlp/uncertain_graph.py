from __future__ import annotations

"""Uncertain work-order association graph and confidence-aware risk ranking.

The source workbook has no fault-event sequence.  Consequently this module
models reported/observed associations and maintenance burden, not causal
propagation probabilities.
"""

import hashlib
import math
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean, median
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple
from xml.etree import ElementTree as ET

from .graph import percentile, wilson_interval, winsorized_mean
from .io_utils import write_csv, write_json
from .ontology import KnowledgeBase
from .record_type import RECORD_TYPE_RULE_VERSION, classify_record_type
from .schemas import WorkOrder
from .text import normalize_text, stable_id


NON_FAILURE_MATERIAL_TERMS = (
    "天然气专用油", "发动机专用油", "润滑油", "防冻液", "冷却液", "尿素液",
)


@dataclass
class _NodeAggregate:
    node_id: str
    label: str
    entity_type: str
    order_ids: Set[str] = field(default_factory=set)
    allocated_cost: float = 0.0


@dataclass
class _EdgeAggregate:
    edge_id: str
    source: str
    relation: str
    target: str
    context_component_id: str = ""
    semantic_by_order: Dict[str, float] = field(default_factory=dict)
    confidence_by_order: Dict[str, float] = field(default_factory=dict)
    structure_by_order: Dict[str, float] = field(default_factory=dict)
    evidence: Set[str] = field(default_factory=set)
    allocated_cost: float = 0.0


@dataclass
class _PathAggregate:
    path_id: str
    node_ids: Tuple[str, str, str, str]
    labels: Tuple[str, str, str, str]
    subsystem_id: str
    component_id: str
    failure_mode_id: str
    semantic_by_order: Dict[str, float] = field(default_factory=dict)
    confidence_by_order: Dict[str, float] = field(default_factory=dict)
    structure_by_order: Dict[str, float] = field(default_factory=dict)
    allocated_costs: List[float] = field(default_factory=list)
    order_ids: Set[str] = field(default_factory=set)
    associated_edge_keys: Set[Tuple[str, str, str, str]] = field(default_factory=set)
    associated_node_ids: Set[str] = field(default_factory=set)


@dataclass
class UncertainGraphResult:
    nodes: List[Dict[str, Any]]
    edges: List[Dict[str, Any]]
    path_ranking: List[Dict[str, Any]]
    node_ranking: List[Dict[str, Any]]
    summary: Dict[str, Any]


def _safe_id(prefix: str, value: str) -> str:
    digest = hashlib.sha1(value.encode("utf-8")).hexdigest()[:16].upper()
    return "{}_{}".format(prefix, digest)


def _node_id(entity_type: str, canonical_id: str) -> str:
    return "{}:{}".format(entity_type, canonical_id)


def _edge_key(
    source: str,
    relation: str,
    target: str,
    context_component_id: str = "",
) -> Tuple[str, str, str, str]:
    context = context_component_id if relation in ("CAUSES_REPORTED", "MANIFESTS_AS") else ""
    return source, relation, target, context


def _is_non_failure_material(path: Dict[str, Any]) -> bool:
    name = normalize_text(path.get("component_name") or "")
    return any(term in name for term in NON_FAILURE_MATERIAL_TERMS)


def _traceable_branch_evidence(order: WorkOrder, evidence: Any) -> List[str]:
    if isinstance(evidence, str):
        evidence = [evidence]
    if not isinstance(evidence, list):
        return []
    source = normalize_text(
        " ".join(
            [order.fault_name, order.fault_description, order.diagnosis, order.dtc_raw]
            + order.dtc_codes
            + [
                "{} {}".format(part.part_no, part.part_name)
                for part in order.parts
                if part.culprit
            ]
        )
    ).lower()
    return sorted(
        {
            normalize_text(value)
            for value in evidence
            if isinstance(value, str)
            and normalize_text(value)
            and normalize_text(value).lower() in source
        }
    )
def _canonical_branch_label(
    branch: Dict[str, Any],
    relation: str,
    evidence: Any,
    kb: Optional[KnowledgeBase],
) -> Optional[str]:
    """Validate that traceable evidence also supports the branch's ontology ID."""

    if kb is None:
        return None
    if isinstance(evidence, str):
        evidence = [evidence]
    if not isinstance(evidence, list):
        return None
    quotes = [
        normalize_text(value).lower()
        for value in evidence
        if isinstance(value, str) and normalize_text(value)
    ]
    if not quotes:
        return None

    if relation == "CAUSES_REPORTED":
        cause_id = str(branch.get("cause_id") or "")
        entry = kb.causes.get(cause_id)
        if not entry or cause_id == "UNKNOWN":
            return None
        canonical_name = normalize_text(entry.get("name") or cause_id)
        supplied_name = normalize_text(branch.get("cause_name") or "")
        if supplied_name and supplied_name.lower() != canonical_name.lower():
            return None
        terms = [canonical_name] + [
            normalize_text(value) for value in entry.get("keywords", [])
        ]
        if any(
            term and term.lower() in quote
            for term in terms
            for quote in quotes
        ):
            return canonical_name
        return None

    if relation == "MANIFESTS_AS":
        supplied_name = normalize_text(branch.get("effect_name") or "")
        symptom_by_normalized_name = {
            normalize_text(value).lower(): normalize_text(value)
            for value in kb.symptoms
            if normalize_text(value)
        }
        canonical_name = symptom_by_normalized_name.get(supplied_name.lower())
        if not canonical_name:
            return None
        if str(branch.get("effect_id") or "") != stable_id("EFFECT", canonical_name):
            return None
        if any(canonical_name.lower() in quote for quote in quotes):
            return canonical_name
    return None


def _mean(values: Iterable[float], default: float = 0.0) -> float:
    items = list(values)
    return mean(items) if items else default


def _clamp(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(number):
        return default
    return min(1.0, max(0.0, number))


def _empirical_percentile(value: float, values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return sum(item <= value for item in values) / len(values)


def _positive_empirical_percentile(value: float, values: Sequence[float]) -> float:
    """Rank non-negative burden metrics without rewarding a zero observation."""

    if value <= 0 or not values:
        return 0.0
    return sum(0 < item <= value for item in values) / len(values)


def _fuse_evidence(
    semantic: float,
    statistical: float,
    structure: float,
    alpha: float,
    method: str,
    epsilon: float = 1e-6,
) -> float:
    semantic = _clamp(semantic)
    statistical = _clamp(statistical)
    structure = _clamp(structure)
    alpha = _clamp(alpha, 0.5)
    if structure <= 0:
        return 0.0
    if method == "linear":
        score = alpha * semantic + (1.0 - alpha) * statistical
    elif method == "geometric":
        score = (semantic + epsilon) ** alpha * (statistical + epsilon) ** (1.0 - alpha)
    else:
        raise ValueError("fusion_method must be 'geometric' or 'linear'")
    return min(1.0, structure * score)


def _add_node(
    nodes: Dict[str, _NodeAggregate],
    node_id: str,
    label: str,
    entity_type: str,
    order_id: str,
    allocated_cost: float,
) -> None:
    node = nodes.setdefault(node_id, _NodeAggregate(node_id, label, entity_type))
    node.order_ids.add(order_id)
    node.allocated_cost += allocated_cost


def _add_edge(
    edges: Dict[Tuple[str, str, str, str], _EdgeAggregate],
    source: str,
    relation: str,
    target: str,
    order_id: str,
    semantic_support: float,
    model_confidence: float,
    structure_score: float,
    allocated_cost: float,
    context_component_id: str = "",
    evidence: Optional[Iterable[str]] = None,
) -> Tuple[str, str, str, str]:
    key = _edge_key(source, relation, target, context_component_id)
    edge = edges.get(key)
    if edge is None:
        edge = _EdgeAggregate(
            _safe_id("EDGE", "|".join(key)), source, relation, target, key[3]
        )
        edges[key] = edge
    edge.semantic_by_order[order_id] = max(
        edge.semantic_by_order.get(order_id, 0.0), _clamp(semantic_support)
    )
    edge.confidence_by_order[order_id] = max(
        edge.confidence_by_order.get(order_id, 0.0), _clamp(model_confidence, 0.5)
    )
    edge.structure_by_order[order_id] = max(
        edge.structure_by_order.get(order_id, 0.0), _clamp(structure_score)
    )
    edge.allocated_cost += allocated_cost
    if evidence:
        edge.evidence.update(normalize_text(value) for value in evidence if normalize_text(value))
    return key


def build_uncertain_graph(
    orders: Iterable[WorkOrder],
    candidate_records: Iterable[Dict[str, Any]],
    engine_name: str = "15N发动机",
    minimum_path_support: float = 0.60,
    include_review: bool = False,
    alpha: float = 0.50,
    fusion_method: str = "geometric",
    shrinkage_strength: float = 5.0,
    risk_weights: Tuple[float, float, float] = (0.45, 0.35, 0.20),
    kb: Optional[KnowledgeBase] = None,
) -> UncertainGraphResult:
    """Build a confidence-weighted association graph from canonical paths."""

    if (
        not math.isfinite(minimum_path_support)
        or not 0.0 <= minimum_path_support <= 1.0
    ):
        raise ValueError("minimum_path_support must be between 0 and 1")
    if not math.isfinite(alpha) or not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be between 0 and 1")
    if fusion_method not in ("geometric", "linear"):
        raise ValueError("fusion_method must be 'geometric' or 'linear'")
    if not math.isfinite(shrinkage_strength) or shrinkage_strength < 0:
        raise ValueError("shrinkage_strength cannot be negative")
    if len(risk_weights) != 3:
        raise ValueError("risk_weights must contain occurrence, severity, and burden")
    if any(not math.isfinite(value) or value < 0 for value in risk_weights):
        raise ValueError("risk_weights cannot be negative")

    order_map = {order.work_order_id: order for order in orders}
    candidate_map = {str(row.get("work_order_id") or ""): row for row in candidate_records}
    nodes: Dict[str, _NodeAggregate] = {}
    edges: Dict[Tuple[str, str, str, str], _EdgeAggregate] = {}
    paths: Dict[Tuple[str, str, str], _PathAggregate] = {}
    context_head_orders: Dict[Tuple[str, str, str], Set[str]] = defaultdict(set)

    classified_fault_ids: Set[str] = set()
    represented_fault_ids: Set[str] = set()
    candidate_fault_ids: Set[str] = set()
    source_types: Set[str] = set()
    record_type_mismatches = 0
    excluded_record_types: Dict[str, int] = defaultdict(int)
    excluded_low_support_paths = 0
    excluded_review_paths = 0
    excluded_material_paths = 0
    rejected_structure_paths = 0
    excluded_incomplete_consensus_orders = 0
    excluded_unverified_branches = 0
    excluded_untraceable_branches = 0
    excluded_semantically_invalid_branches = 0
    empty_canonical_path_orders = 0
    candidate_warning_counts: Dict[str, int] = defaultdict(int)
    classified_known_cost_orders = 0
    classified_fault_total_cost = 0.0
    represented_known_cost_orders = 0
    represented_input_cost = 0.0
    allocated_fault_cost = 0.0

    for order_id, order in order_map.items():
        actual_type = classify_record_type(order)
        row = candidate_map.get(order_id)
        reported_type = (
            str(row.get("reported_record_type") or row.get("record_type") or "")
            if row else ""
        )
        if reported_type and reported_type != actual_type:
            record_type_mismatches += 1
        if actual_type != "FAULT":
            excluded_record_types[actual_type] += 1
            continue
        classified_fault_ids.add(order_id)
        if order.total_cost > 0:
            classified_known_cost_orders += 1
            classified_fault_total_cost += order.total_cost
        if not row:
            continue
        candidate_fault_ids.add(order_id)
        support_source = str(row.get("support_source") or "unknown")
        source_types.add(support_source)
        if (
            support_source == "planned_independent_run_frequency"
            and row.get("formal_eligible") is not True
        ):
            excluded_incomplete_consensus_orders += 1
            continue
        warning_values = row.get("warnings", [])
        if isinstance(warning_values, str):
            warning_values = [warning_values]
        for warning in warning_values:
            candidate_warning_counts[str(warning)] += 1
        if not row.get("canonical_paths"):
            empty_canonical_path_orders += 1

        accepted: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
        for item in row.get("canonical_paths", []):
            if not isinstance(item, dict):
                continue
            if _is_non_failure_material(item):
                excluded_material_paths += 1
                continue
            structure_score = _clamp(item.get("structural_gate"))
            if structure_score <= 0:
                rejected_structure_paths += 1
                continue
            support = _clamp(item.get("semantic_support"))
            if support < minimum_path_support:
                excluded_low_support_paths += 1
                continue
            if not include_review and str(item.get("decision") or "REVIEW") != "ACCEPT":
                excluded_review_paths += 1
                continue
            subsystem_id = str(item.get("subsystem_id") or "")
            component_id = str(item.get("component_id") or "")
            failure_mode_id = str(item.get("failure_mode_id") or "")
            if not subsystem_id or not component_id or not failure_mode_id:
                rejected_structure_paths += 1
                continue
            key = subsystem_id, component_id, failure_mode_id
            if key not in accepted or support > _clamp(accepted[key].get("semantic_support")):
                accepted[key] = item

        if not accepted:
            continue
        represented_fault_ids.add(order_id)
        if order.total_cost > 0:
            represented_known_cost_orders += 1
            represented_input_cost += order.total_cost
        allocated_cost = order.total_cost / len(accepted) if order.total_cost > 0 else 0.0
        allocated_fault_cost += allocated_cost * len(accepted)

        for (subsystem_id, component_id, failure_mode_id), item in accepted.items():
            subsystem_name = normalize_text(item.get("subsystem_name") or subsystem_id)
            component_name = normalize_text(item.get("component_name") or component_id)
            mode_name = normalize_text(item.get("failure_mode_name") or failure_mode_id)
            root_node = "SYSTEM:ENGINE_ROOT"
            subsystem_node = _node_id("SUBSYSTEM", subsystem_id)
            component_node = _node_id("COMPONENT", component_id)
            mode_node = _node_id("FAILURE_MODE", failure_mode_id)
            node_ids = root_node, subsystem_node, component_node, mode_node
            labels = engine_name, subsystem_name, component_name, mode_name

            for node_id, label, entity_type in zip(
                node_ids, labels, ("SYSTEM", "SUBSYSTEM", "COMPONENT", "FAILURE_MODE")
            ):
                _add_node(nodes, node_id, label, entity_type, order_id, allocated_cost)

            relation_supports = item.get("relation_supports", {})
            if not isinstance(relation_supports, dict):
                relation_supports = {}
            path_support = _clamp(item.get("semantic_support"))
            membership_support = _clamp(relation_supports.get("HAS_COMPONENT"), path_support)
            mode_support = _clamp(relation_supports.get("HAS_FAILURE_MODE"), path_support)
            model_confidence = _clamp(item.get("mean_model_confidence"), 0.5)
            structure_score = _clamp(item.get("structural_gate"))
            evidence = item.get("evidence", [])

            _add_edge(
                edges, root_node, "CONTAINS", subsystem_node, order_id,
                path_support, model_confidence, 1.0, allocated_cost, evidence=evidence,
            )
            membership_key = _add_edge(
                edges, subsystem_node, "HAS_COMPONENT", component_node, order_id,
                membership_support, model_confidence, structure_score, allocated_cost,
                evidence=evidence,
            )
            mode_key = _add_edge(
                edges, component_node, "HAS_FAILURE_MODE", mode_node, order_id,
                mode_support, model_confidence, structure_score, allocated_cost,
                evidence=evidence,
            )

            path_key = subsystem_id, component_id, failure_mode_id
            path = paths.get(path_key)
            if path is None:
                path = _PathAggregate(
                    _safe_id("PATH", "|".join(path_key)), node_ids, labels,
                    subsystem_id, component_id, failure_mode_id,
                )
                paths[path_key] = path
            path.order_ids.add(order_id)
            path.semantic_by_order[order_id] = max(
                path.semantic_by_order.get(order_id, 0.0), path_support
            )
            path.confidence_by_order[order_id] = max(
                path.confidence_by_order.get(order_id, 0.0), model_confidence
            )
            path.structure_by_order[order_id] = max(
                path.structure_by_order.get(order_id, 0.0), structure_score
            )
            if allocated_cost > 0:
                path.allocated_costs.append(allocated_cost)
            path.associated_edge_keys.update((membership_key, mode_key))
            context_head_orders[(mode_node, "MANIFESTS_AS", component_id)].add(order_id)

            for cause in item.get("causes", []):
                if not isinstance(cause, dict):
                    continue
                cause_id = str(cause.get("cause_id") or "")
                cause_support = _clamp(cause.get("semantic_support"))
                cause_evidence = cause.get("evidence", [])
                cause_gate = _clamp(cause.get("structural_gate"))
                if (
                    not cause_id
                    or cause_support < minimum_path_support
                    or str(cause.get("decision") or "REVIEW") != "ACCEPT"
                    or cause_gate <= 0
                ):
                    excluded_unverified_branches += 1
                    continue
                traceable_cause_evidence = _traceable_branch_evidence(
                    order, cause_evidence
                )
                if not traceable_cause_evidence:
                    excluded_unverified_branches += 1
                    excluded_untraceable_branches += 1
                    continue
                canonical_cause_name = _canonical_branch_label(
                    cause, "CAUSES_REPORTED", traceable_cause_evidence, kb
                )
                if not canonical_cause_name:
                    excluded_unverified_branches += 1
                    excluded_semantically_invalid_branches += 1
                    continue
                cause_node = _node_id("CAUSE", cause_id)
                context_head_orders[(cause_node, "CAUSES_REPORTED", component_id)].add(order_id)
                _add_node(
                    nodes, cause_node, canonical_cause_name,
                    "CAUSE", order_id, allocated_cost,
                )
                edge_key = _add_edge(
                    edges, cause_node, "CAUSES_REPORTED", mode_node, order_id,
                    cause_support, model_confidence, cause_gate, allocated_cost,
                    context_component_id=component_id,
                    evidence=traceable_cause_evidence,
                )
                path.associated_edge_keys.add(edge_key)
                path.associated_node_ids.add(cause_node)

            for effect in item.get("effects", []):
                if not isinstance(effect, dict):
                    continue
                effect_id = str(effect.get("effect_id") or "")
                effect_support = _clamp(effect.get("semantic_support"))
                effect_evidence = effect.get("evidence", [])
                effect_gate = _clamp(effect.get("structural_gate"))
                if (
                    not effect_id
                    or effect_support < minimum_path_support
                    or str(effect.get("decision") or "REVIEW") != "ACCEPT"
                    or effect_gate <= 0
                ):
                    excluded_unverified_branches += 1
                    continue
                traceable_effect_evidence = _traceable_branch_evidence(
                    order, effect_evidence
                )
                if not traceable_effect_evidence:
                    excluded_unverified_branches += 1
                    excluded_untraceable_branches += 1
                    continue
                canonical_effect_name = _canonical_branch_label(
                    effect, "MANIFESTS_AS", traceable_effect_evidence, kb
                )
                if not canonical_effect_name:
                    excluded_unverified_branches += 1
                    excluded_semantically_invalid_branches += 1
                    continue
                effect_node = _node_id("EFFECT", effect_id)
                _add_node(
                    nodes, effect_node, canonical_effect_name,
                    "EFFECT", order_id, allocated_cost,
                )
                edge_key = _add_edge(
                    edges, mode_node, "MANIFESTS_AS", effect_node, order_id,
                    effect_support, model_confidence, effect_gate, allocated_cost,
                    context_component_id=component_id,
                    evidence=traceable_effect_evidence,
                )
                path.associated_edge_keys.add(edge_key)
                path.associated_node_ids.add(effect_node)

    # Edges use a conditional Wilson lower bound n_edge / n_head.  This is an
    # evidence score, never interpreted as a calibrated probability.
    edge_rows: List[Dict[str, Any]] = []
    edge_score_lookup: Dict[Tuple[str, str, str, str], float] = {}
    for key, edge in edges.items():
        if edge.relation in ("CAUSES_REPORTED", "MANIFESTS_AS"):
            head_count = len(
                context_head_orders[(edge.source, edge.relation, edge.context_component_id)]
            )
        else:
            head_count = len(nodes[edge.source].order_ids)
        count = len(edge.semantic_by_order)
        low, high = wilson_interval(count, head_count)
        semantic = _mean(edge.semantic_by_order.values())
        structure = _mean(edge.structure_by_order.values())
        model_confidence = _mean(edge.confidence_by_order.values(), 0.5)
        fused = _fuse_evidence(semantic, low, structure, alpha, fusion_method)
        edge_score_lookup[key] = fused
        epistemic_status = (
            "ontology" if edge.relation in ("CONTAINS", "HAS_COMPONENT")
            else "reported" if edge.relation in ("CAUSES_REPORTED", "MANIFESTS_AS")
            else "observed"
        )
        edge_rows.append(
            {
                "edge_id": edge.edge_id,
                "source": edge.source,
                "relation": edge.relation,
                "target": edge.target,
                "context_component_id": edge.context_component_id,
                "supporting_work_orders": count,
                "head_work_orders": head_count,
                "conditional_frequency": count / head_count if head_count else 0.0,
                "frequency_ci_low": low,
                "frequency_ci_high": high,
                "semantic_support": semantic,
                "mean_model_confidence": model_confidence,
                "structure_score": structure,
                "evidence_confidence": fused,
                "allocated_cost": edge.allocated_cost,
                "epistemic_status": epistemic_status,
                "decision": "ACCEPT" if structure > 0 else "REJECT",
                "sample_work_orders": "|".join(sorted(edge.semantic_by_order)[:8]),
                "sample_evidence": "|".join(sorted(edge.evidence)[:5]),
            }
        )
    edge_rows.sort(key=lambda value: (-value["evidence_confidence"], value["edge_id"]))

    all_allocations = [cost for path in paths.values() for cost in path.allocated_costs]
    global_cost_median = median(all_allocations) if all_allocations else 0.0
    total_fault_orders = len(classified_fault_ids)
    path_rows: List[Dict[str, Any]] = []
    for path in paths.values():
        count = len(path.order_ids)
        low, high = wilson_interval(count, total_fault_orders)
        costs = path.allocated_costs
        robust_mean = winsorized_mean(costs)
        shrunk_cost = (
            (len(costs) * robust_mean + shrinkage_strength * global_cost_median)
            / (len(costs) + shrinkage_strength)
            if costs
            else 0.0
        )
        membership_key = _edge_key(path.node_ids[1], "HAS_COMPONENT", path.node_ids[2])
        mode_key = _edge_key(path.node_ids[2], "HAS_FAILURE_MODE", path.node_ids[3])
        core_scores = [edge_score_lookup.get(membership_key, 0.0), edge_score_lookup.get(mode_key, 0.0)]
        path_confidence = math.sqrt(core_scores[0] * core_scores[1])
        path_rows.append(
            {
                "path_id": path.path_id,
                "engine_node_id": path.node_ids[0],
                "subsystem_node_id": path.node_ids[1],
                "component_node_id": path.node_ids[2],
                "failure_mode_node_id": path.node_ids[3],
                "engine": path.labels[0],
                "subsystem": path.labels[1],
                "component": path.labels[2],
                "failure_mode": path.labels[3],
                "unique_work_orders": count,
                "frequency_rate": count / total_fault_orders if total_fault_orders else 0.0,
                "frequency_ci_low": low,
                "frequency_ci_high": high,
                "mean_semantic_support": _mean(path.semantic_by_order.values()),
                "mean_model_confidence": _mean(path.confidence_by_order.values(), 0.5),
                "mean_structure_score": _mean(path.structure_by_order.values()),
                "path_evidence_confidence": path_confidence,
                "cost_observations": len(costs),
                "cost_coverage": len(costs) / count if count else 0.0,
                "allocated_total_cost": sum(costs),
                "allocated_mean_cost": mean(costs) if costs else 0.0,
                "allocated_median_cost": median(costs) if costs else 0.0,
                "allocated_p95_cost": percentile(costs, 0.95),
                "winsorized_mean_cost": robust_mean,
                "shrunk_cost_severity": shrunk_cost,
                "sample_work_orders": "|".join(sorted(path.order_ids)[:8]),
                "_associated_nodes": set(path.associated_node_ids),
                "_associated_edges": set(path.associated_edge_keys),
            }
        )

    occurrence_values = [float(row["frequency_ci_low"]) for row in path_rows]
    severity_values = [float(row["shrunk_cost_severity"]) for row in path_rows]
    burden_values = [float(row["allocated_total_cost"]) for row in path_rows]
    weight_sum = sum(risk_weights)
    if weight_sum <= 0:
        raise ValueError("risk_weights must have a positive sum")
    occurrence_weight, severity_weight, burden_weight = [value / weight_sum for value in risk_weights]
    for row in path_rows:
        occurrence_score = _empirical_percentile(float(row["frequency_ci_low"]), occurrence_values)
        severity_score = _positive_empirical_percentile(
            float(row["shrunk_cost_severity"]), severity_values
        )
        burden_score = _positive_empirical_percentile(
            float(row["allocated_total_cost"]), burden_values
        )
        base_risk = (
            occurrence_weight * math.sqrt(occurrence_score * severity_score)
            + severity_weight * severity_score
            + burden_weight * burden_score
        )
        row["occurrence_percentile"] = occurrence_score
        row["severity_percentile"] = severity_score
        row["burden_percentile"] = burden_score
        row["base_risk_score"] = base_risk
        row["association_risk_score"] = base_risk * float(row["path_evidence_confidence"])
    path_rows.sort(
        key=lambda value: (
            -value["association_risk_score"], -value["unique_work_orders"],
            -value["allocated_total_cost"], value["path_id"],
        )
    )
    for rank, row in enumerate(path_rows, start=1):
        row["rank"] = rank
        percentile_rank = rank / max(1, len(path_rows))
        row["priority"] = (
            "P1" if percentile_rank <= 0.10 else "P2" if percentile_rank <= 0.30
            else "P3" if percentile_rank <= 0.60 else "P4"
        )

    node_raw: Dict[str, float] = defaultdict(float)
    node_paths: Dict[str, Set[str]] = defaultdict(set)
    for row in path_rows:
        contribution = float(row["association_risk_score"])
        for node_id in (
            row["subsystem_node_id"], row["component_node_id"], row["failure_mode_node_id"]
        ):
            node_raw[str(node_id)] += contribution
            node_paths[str(node_id)].add(str(row["path_id"]))
        for node_id in row["_associated_nodes"]:
            incident_scores = [
                edge_score_lookup.get(key, 0.0)
                for key in row["_associated_edges"]
                if key[0] == node_id or key[2] == node_id
            ]
            node_raw[str(node_id)] += contribution * max(incident_scores or [0.0])
            node_paths[str(node_id)].add(str(row["path_id"]))

    maxima_by_type: Dict[str, float] = defaultdict(float)
    for node_id, raw_score in node_raw.items():
        maxima_by_type[nodes[node_id].entity_type] = max(
            maxima_by_type[nodes[node_id].entity_type], raw_score
        )
    node_rows: List[Dict[str, Any]] = []
    for node in nodes.values():
        raw_score = node_raw.get(node.node_id, 0.0)
        maximum = maxima_by_type.get(node.entity_type, 0.0)
        node_rows.append(
            {
                "node_id": node.node_id,
                "label": node.label,
                "entity_type": node.entity_type,
                "unique_work_orders": len(node.order_ids),
                "allocated_cost": node.allocated_cost,
                "associated_paths": len(node_paths.get(node.node_id, set())),
                "association_importance_raw": raw_score,
                "association_importance": raw_score / maximum if maximum else 0.0,
            }
        )
    node_rows.sort(
        key=lambda value: (
            value["entity_type"], -value["association_importance"],
            -value["unique_work_orders"], value["node_id"],
        )
    )
    ranks: Dict[str, int] = defaultdict(int)
    for row in node_rows:
        ranks[row["entity_type"]] += 1
        row["rank_within_type"] = ranks[row["entity_type"]]

    # Remove internal-only association sets before returning/exporting.
    for row in path_rows:
        row.pop("_associated_nodes", None)
        row.pop("_associated_edges", None)

    summary = {
        "schema_version": "uncertain-association-graph-v1",
        "record_type_rule_version": RECORD_TYPE_RULE_VERSION,
        "engine_root": engine_name,
        "input_work_orders": len(order_map),
        "classified_fault_work_orders": len(classified_fault_ids),
        "candidate_fault_work_orders": len(candidate_fault_ids),
        "represented_fault_work_orders": len(represented_fault_ids),
        "excluded_record_types": dict(sorted(excluded_record_types.items())),
        "record_type_mismatches": record_type_mismatches,
        "excluded_low_support_paths": excluded_low_support_paths,
        "excluded_review_paths": excluded_review_paths,
        "excluded_non_failure_material_paths": excluded_material_paths,
        "rejected_structure_paths": rejected_structure_paths,
        "excluded_incomplete_consensus_orders": excluded_incomplete_consensus_orders,
        "excluded_unverified_branches": excluded_unverified_branches,
        "excluded_untraceable_branches": excluded_untraceable_branches,
        "excluded_semantically_invalid_branches": excluded_semantically_invalid_branches,
        "empty_canonical_path_orders": empty_canonical_path_orders,
        "fault_orders_without_represented_path": len(classified_fault_ids - represented_fault_ids),
        "candidate_warning_counts": dict(
            sorted(candidate_warning_counts.items(), key=lambda item: (-item[1], item[0]))
        ),
        "classified_known_cost_orders": classified_known_cost_orders,
        "classified_cost_coverage": (
            classified_known_cost_orders / len(classified_fault_ids) if classified_fault_ids else 0.0
        ),
        "classified_fault_total_cost": round(classified_fault_total_cost, 2),
        "represented_known_cost_orders": represented_known_cost_orders,
        "represented_cost_coverage": (
            represented_known_cost_orders / len(represented_fault_ids) if represented_fault_ids else 0.0
        ),
        "represented_input_cost": round(represented_input_cost, 2),
        "excluded_unrepresented_fault_cost": round(
            classified_fault_total_cost - represented_input_cost, 2
        ),
        "allocated_fault_cost": round(allocated_fault_cost, 2),
        "represented_cost_conservation_error": round(
            allocated_fault_cost - represented_input_cost, 8
        ),
        "nodes": len(node_rows),
        "edges": len(edge_rows),
        "paths": len(path_rows),
        "support_sources": sorted(source_types),
        "minimum_path_support": minimum_path_support,
        "include_review": include_review,
        "fusion_method": fusion_method,
        "fusion_epsilon": 1e-6,
        "semantic_weight_alpha": alpha,
        "statistical_support": "conditional Wilson 95% lower bound using unique work orders",
        "edge_score_semantics": "evidence confidence, not occurrence or propagation probability",
        "branch_validation": (
            "the same traceable source quote must satisfy the knowledge-base semantic ID"
        ),
        "ranking_method": "confidence-weighted association risk",
        "risk_weights": {
            "occurrence": occurrence_weight,
            "severity": severity_weight,
            "burden": burden_weight,
        },
        "risk_scope": "work-order association and maintenance burden; no temporal/causal claim",
        "shrinkage_strength": shrinkage_strength,
    }
    node_ranking = [
        dict(row) for row in node_rows
        if row["entity_type"] != "SYSTEM" and row["associated_paths"] > 0
    ]
    return UncertainGraphResult(node_rows, edge_rows, path_rows, node_ranking, summary)


def _round_rows(rows: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    result: List[Dict[str, Any]] = []
    for row in rows:
        result.append(
            {
                key: round(value, 8) if isinstance(value, float) else value
                for key, value in row.items()
            }
        )
    return result


def _write_graphml(result: UncertainGraphResult, path: str) -> None:
    namespace = "http://graphml.graphdrawing.org/xmlns"
    ET.register_namespace("", namespace)
    root = ET.Element("{%s}graphml" % namespace)
    key_specs = (
        ("n_label", "node", "label", "string"),
        ("n_type", "node", "entity_type", "string"),
        ("n_importance", "node", "association_importance", "double"),
        ("e_relation", "edge", "relation", "string"),
        ("e_confidence", "edge", "evidence_confidence", "double"),
        ("e_context", "edge", "context_component_id", "string"),
        ("e_status", "edge", "epistemic_status", "string"),
    )
    for key_id, target, name, value_type in key_specs:
        ET.SubElement(
            root,
            "{%s}key" % namespace,
            id=key_id,
            **{"for": target, "attr.name": name, "attr.type": value_type},
        )
    graph = ET.SubElement(root, "{%s}graph" % namespace, edgedefault="directed", id="G")
    for node in result.nodes:
        item = ET.SubElement(graph, "{%s}node" % namespace, id=str(node["node_id"]))
        for key_id, value in (
            ("n_label", node["label"]),
            ("n_type", node["entity_type"]),
            ("n_importance", node["association_importance"]),
        ):
            ET.SubElement(item, "{%s}data" % namespace, key=key_id).text = str(value)
    for edge in result.edges:
        item = ET.SubElement(
            graph, "{%s}edge" % namespace, id=str(edge["edge_id"]),
            source=str(edge["source"]), target=str(edge["target"]),
        )
        for key_id, value in (
            ("e_relation", edge["relation"]),
            ("e_confidence", edge["evidence_confidence"]),
            ("e_context", edge["context_component_id"]),
            ("e_status", edge["epistemic_status"]),
        ):
            ET.SubElement(item, "{%s}data" % namespace, key=key_id).text = str(value)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)


def export_uncertain_graph(result: UncertainGraphResult, output_dir: str) -> None:
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    nodes = _round_rows(result.nodes)
    edges = _round_rows(result.edges)
    paths = _round_rows(result.path_ranking)
    rankings = _round_rows(result.node_ranking)
    write_csv(str(target / "uncertain_graph_nodes.csv"), list(nodes[0]) if nodes else [], nodes)
    write_csv(str(target / "uncertain_graph_edges.csv"), list(edges[0]) if edges else [], edges)
    write_csv(str(target / "fault_path_risk_ranking.csv"), list(paths[0]) if paths else [], paths)
    write_csv(str(target / "node_association_importance.csv"), list(rankings[0]) if rankings else [], rankings)
    write_json(str(target / "uncertain_graph_summary.json"), result.summary)
    write_json(
        str(target / "uncertain_graph.json"),
        {"summary": result.summary, "nodes": nodes, "edges": edges},
    )
    _write_graphml(result, str(target / "uncertain_graph.graphml"))
