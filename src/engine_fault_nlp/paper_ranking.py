"""Four-layer graph aggregation and Balanced Risk computation, without plotting.

The statistical defaults follow the historical unseen-ranking implementation.
Labels come only from supplied predictions. Optional graph alias merging does
not alter neural inference, and must be enabled explicitly.
"""
from __future__ import annotations

from collections import defaultdict
import hashlib
import math
from pathlib import Path
from statistics import median

from .graph import (FaultGraphResult, GraphEdge, GraphNode, _canonical_component,
                    _is_non_failure_material, _write_graphml, wilson_interval, winsorized_mean)
from .io_utils import write_csv, write_json

FIELDS = ("subsystem_id", "component_id", "failure_mode_id")
TYPES = ("SUBSYSTEM", "COMPONENT", "FAILURE_MODE")


def document_id(row):
    value = row.get("document_id", row.get("work_order_id"))
    if value is None or not str(value).strip():
        raise ValueError("Every input row must have a document_id or work_order_id")
    return str(value)


def accepted_chains(row, threshold=0.5):
    """Support current three-head, legacy classification, and fault-chain JSONL."""
    if "predicted" in row:
        chain = dict(row["predicted"])
        confidence = row.get("confidence", {})
        if any(field in confidence and float(confidence[field]) < threshold for field in FIELDS):
            return []
        candidates = [chain]
    elif "classifications" in row:
        chain = {}
        for field in FIELDS:
            values = row["classifications"].get(field) or []
            if not values:
                return []
            top = max(values, key=lambda item: float(item.get("confidence", 0.0)))
            if float(top.get("confidence", 0.0)) < threshold:
                return []
            chain[field] = top.get("label")
        candidates = [chain]
    else:
        candidates = row.get("fault_chains", row.get("chains", []))
    return [dict(chain) for chain in candidates if all(chain.get(field) for field in FIELDS)]


def empirical_percentiles(values):
    """Average tied ranks, retaining the historical monotonic log1p transform."""
    if len(values) <= 1:
        return [1.0] * len(values)
    ordered = sorted((math.log1p(max(0.0, float(value))), index) for index, value in enumerate(values))
    result = [0.0] * len(values)
    start = 0
    while start < len(ordered):
        end = start + 1
        while end < len(ordered) and ordered[end][0] == ordered[start][0]:
            end += 1
        percentile = ((start + end - 1) / 2.0) / (len(values) - 1.0)
        for _, index in ordered[start:end]:
            result[index] = percentile
        start = end
    return result


def build_ranking(orders, predictions, *, cohort_ids=None, knowledge_base=None,
                  root_name="Equipment", threshold=0.5, canonicalize=False,
                  screen_materials=False, fault_only=False, shrinkage=5.0):
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("threshold must be finite and within [0, 1]")
    if not math.isfinite(shrinkage) or shrinkage < 0:
        raise ValueError("shrinkage must be finite and nonnegative")
    order_map = {}
    for row in orders:
        key = document_id(row)
        if key in order_map and row != order_map[key]:
            raise ValueError("Conflicting duplicate work-order records")
        order_map[key] = row
    ids = set(map(str, cohort_ids)) if cohort_ids is not None else set(order_map)
    if ids - set(order_map):
        raise ValueError("Cohort contains IDs missing from work-order inputs")
    prediction_map = defaultdict(list)
    record_types = {}
    for row in predictions:
        key = document_id(row)
        if key not in ids:
            continue
        record_types[key] = row.get("record_type")
        prediction_map[key].extend(accepted_chains(row, threshold))
    kb = knowledge_base or {}
    groups = {}
    excluded = 0
    all_cost = represented_cost = 0.0
    complete = known_cost = 0
    for key in sorted(ids):
        if fault_only and record_types.get(key) != "FAULT":
            excluded += 1
            continue
        paths = {}
        candidates = prediction_map[key]
        material_count = 0
        for chain in candidates:
            for field, collection in zip(FIELDS, ("subsystems", "components", "failure_modes")):
                entry = kb.get(collection, {}).get(str(chain[field]), {})
                name = entry.get("name", str(chain[field])) if isinstance(entry, dict) else str(entry)
                chain.setdefault(field.replace("_id", "_name"), name)
            if screen_materials and _is_non_failure_material(chain):
                material_count += 1
                continue
            if canonicalize:
                chain["component_id"], chain["component_name"] = _canonical_component(chain)
            path = tuple(str(chain[field]) for field in FIELDS)
            paths[path] = tuple(str(chain[field.replace("_id", "_name")]) for field in FIELDS)
        if candidates and material_count == len(candidates):
            excluded += 1
            continue
        raw_cost = order_map[key].get("total_cost")
        cost = None if raw_cost is None or raw_cost == "" else float(raw_cost)
        if cost is not None and (not math.isfinite(cost) or cost < 0):
            raise ValueError("Costs must be finite and nonnegative; audit refunds separately")
        if cost is not None:
            known_cost += 1
            all_cost += cost
        if not paths:
            continue
        complete += 1
        if cost is not None:
            represented_cost += cost
        allocated = None if cost is None else cost / len(paths)
        for path, names in paths.items():
            group = groups.setdefault(path, {"names": names, "orders": set(), "costs": [], "positive": []})
            group["orders"].add(key)
            if allocated is not None:
                group["costs"].append(allocated)
                if allocated > 0:
                    group["positive"].append(allocated)
    total = len(ids) - excluded
    positives = [cost for group in groups.values() for cost in group["positive"]]
    prior = median(positives) if positives else 0.0
    ranking, nodes, edges = [], {}, {}
    for path, group in sorted(groups.items()):
        count = len(group["orders"])
        costs = group["positive"]
        robust = winsorized_mean(costs)
        severity = (len(costs) * robust + shrinkage * prior) / (len(costs) + shrinkage) if costs else 0.0
        burden = sum(group["costs"])
        low, high = wilson_interval(count, total)
        node_ids = ("E0", *(kind + ":" + value for kind, value in zip(TYPES, path)))
        for layer, (node_id, name) in enumerate(zip(node_ids, (root_name, *group["names"]))):
            node = nodes.setdefault(node_id, GraphNode(node_id, name, layer, ("ROOT", *TYPES)[layer]))
            node.order_ids.update(group["orders"])
            node.allocated_cost += burden
        for layer, (source, target) in enumerate(zip(node_ids, node_ids[1:])):
            edge = edges.setdefault((source, target), GraphEdge(source, target, layer, layer + 1))
            edge.order_ids.update(group["orders"])
            edge.allocated_cost += burden
        ranking.append({"path_id": "PATH4_" + hashlib.sha1("|".join(path).encode()).hexdigest()[:16].upper(),
                        **dict(zip(FIELDS, path)), **dict(zip(("subsystem", "component", "failure_mode"), group["names"])),
                        "unique_work_orders": count, "frequency_rate": count / total,
                        "frequency_ci_low": low, "frequency_ci_high": high,
                        "known_cost_observations": len(group["costs"]), "positive_cost_observations": len(costs),
                        "winsorized_mean_cost_cny": robust, "shrunk_cost_severity_cny": severity,
                        "economic_burden_cny": burden,
                        "work_order_ids": "|".join(sorted(group["orders"]))})
    for raw, target in (("frequency_ci_low", "occurrence_percentile"),
                        ("shrunk_cost_severity_cny", "severity_percentile"),
                        ("economic_burden_cny", "burden_percentile")):
        for row, value in zip(ranking, empirical_percentiles([row[raw] for row in ranking])):
            row[target] = value
    for row in ranking:
        row["interaction_contribution"] = 0.45 * math.sqrt(max(row["occurrence_percentile"], 1e-12) * max(row["severity_percentile"], 1e-12))
        row["severity_contribution"] = 0.35 * row["severity_percentile"]
        row["burden_contribution"] = 0.20 * row["burden_percentile"]
        row["balanced_risk_score"] = sum(row[key] for key in ("interaction_contribution", "severity_contribution", "burden_contribution"))
    ranking.sort(key=lambda row: tuple(-row[key] for key in ("balanced_risk_score", "severity_percentile", "burden_percentile", "occurrence_percentile", "economic_burden_cny")) + (row["path_id"],))
    for rank, row in enumerate(ranking, 1):
        row["rank"] = rank
    summary = {"cohort_orders": len(ids), "excluded_orders": excluded, "frequency_denominator": total,
               "complete_path_orders": complete, "incomplete_path_orders": total - complete,
               "known_cost_orders": known_cost, "total_cost_cny": all_cost,
               "represented_cost_cny": represented_cost,
               "allocated_cost_cny": sum(row["economic_burden_cny"] for row in ranking),
               "cost_conservation_error": sum(row["economic_burden_cny"] for row in ranking) - represented_cost,
               "nodes": len(nodes), "edges": len(edges), "paths": len(ranking),
               "global_positive_cost_median": prior, "shrinkage_strength": shrinkage,
               "wilson_z": 1.96, "winsor_quantiles": [0.05, 0.95], "risk_weights": [0.45, 0.35, 0.20],
               "cost_severity_policy": "Historical positive-only observations; zero-cost paths have severity 0",
               "canonicalize_graph_labels": canonicalize, "screen_materials": screen_materials,
               "fault_only": fault_only, "acceptance_threshold": threshold,
               "interpretation": "Relative maintenance priority, not failure probability"}
    return FaultGraphResult(nodes, edges, ranking, summary)


def export_ranking(result, output_dir):
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    tables = {"fault_path_ranking.csv": result.ranking,
              "fault_graph_nodes.csv": [{"node_id": node.node_id, "label": node.label, "layer": node.layer,
                                          "unique_work_orders": len(node.order_ids), "allocated_cost_cny": node.allocated_cost}
                                         for node in result.nodes.values()],
              "fault_graph_edges.csv": [{"source": edge.source, "target": edge.target,
                                          "source_layer": edge.source_layer, "target_layer": edge.target_layer,
                                          "unique_work_orders": len(edge.order_ids), "allocated_cost_cny": edge.allocated_cost}
                                         for edge in result.edges.values()]}
    for name, rows in tables.items():
        write_csv(str(target / name), list(rows[0]) if rows else [], rows)
    write_json(str(target / "summary.json"), result.summary)
    _write_graphml(result, str(target / "four_layer_fault_graph.graphml"))
