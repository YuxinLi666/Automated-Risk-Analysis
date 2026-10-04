from __future__ import annotations

"""Repeated extraction, canonicalisation, and semantic consensus.

The Word draft describes a candidate triple as being supported when it is
independently recovered in multiple LLM runs.  This module makes that idea
executable without tying it to a particular model provider.  Raw run records
remain auditable; only canonical triples are counted, and a duplicate inside a
single run never increases support.
"""

import math
from collections import defaultdict
from statistics import mean
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from .normalizer import ComponentNormalizer
from .ontology import AMBIGUOUS_COMPONENT_NAMES, KnowledgeBase, infer_subsystem
from .record_type import RECORD_TYPE_RULE_VERSION, classify_record_type
from .schemas import WorkOrder
from .text import normalize_text, stable_id


RELATION_TYPES = {
    "HAS_COMPONENT": ("SUBSYSTEM", "COMPONENT"),
    "HAS_FAILURE_MODE": ("COMPONENT", "FAILURE_MODE"),
    "CAUSES_REPORTED": ("CAUSE", "FAILURE_MODE"),
    "MANIFESTS_AS": ("FAILURE_MODE", "EFFECT"),
}

ACCEPT_STRUCTURE_THRESHOLD = 0.75


def _clamp(value: Any, default: float = 0.5) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(number):
        return default
    return min(1.0, max(0.0, number))


def _confidence(chain: Dict[str, Any], field: str, default: float = 0.5) -> float:
    values = chain.get("confidence", {})
    if isinstance(values, dict):
        if field in values:
            return _clamp(values[field], default)
        if "chain" in values:
            return _clamp(values["chain"], default)
    return _clamp(values, default) if not isinstance(values, dict) else default


def _valid_quotes(chain: Dict[str, Any], order: WorkOrder) -> List[str]:
    """Return evidence strings that can be traced back to the work order."""

    values: List[str] = []
    evidence = chain.get("evidence", {})
    if isinstance(evidence, dict):
        for item in evidence.values():
            if isinstance(item, dict):
                values.append(str(item.get("text") or ""))
            elif isinstance(item, str):
                values.append(item)
    for field in ("cause_text", "cause_raw", "evidence_text"):
        values.append(str(chain.get(field) or ""))

    # Solution text and non-culprit replacement parts are deliberately omitted:
    # a repair action cannot by itself prove a fault or its cause.
    evidence_source = " ".join(
        [order.fault_name, order.fault_description, order.diagnosis, order.dtc_raw]
        + order.dtc_codes
        + [
            "{} {}".format(part.part_no, part.part_name)
            for part in order.parts
            if part.culprit
        ]
    )
    normalized_order = normalize_text(evidence_source).lower()
    accepted: Set[str] = set()
    for value in values:
        quote = normalize_text(value)
        if quote and quote.lower() in normalized_order:
            accepted.add(quote)
    return sorted(accepted)


def _match_enumeration(
    value_id: Any,
    value_name: Any,
    mapping: Dict[str, Dict[str, Any]],
    context: str,
) -> Tuple[Optional[str], Optional[str]]:
    candidate_id = str(value_id or "").strip()
    if candidate_id in mapping:
        return candidate_id, str(mapping[candidate_id].get("name") or candidate_id)

    target = normalize_text(value_name).lower()
    for item_id, entry in mapping.items():
        name = normalize_text(entry.get("name") or "")
        if target and target == name.lower():
            return item_id, name

    search_text = normalize_text("{} {}".format(value_name or "", context)).lower()
    matches: List[Tuple[int, str, str]] = []
    for item_id, entry in mapping.items():
        if item_id == "UNKNOWN":
            continue
        name = normalize_text(entry.get("name") or item_id)
        terms = [name] + [normalize_text(term) for term in entry.get("keywords", [])]
        for term in terms:
            if term and term.lower() in search_text:
                matches.append((len(term), item_id, name))
    if not matches:
        return None, None
    _, item_id, name = max(matches)
    return item_id, name


def _component_from_chain(
    order: WorkOrder,
    chain: Dict[str, Any],
    kb: KnowledgeBase,
    normalizer: ComponentNormalizer,
    minimum_score: float,
) -> Tuple[Optional[str], Optional[str], Optional[str], float, List[str]]:
    def supported(component_id: str, names: Sequence[str]) -> bool:
        diagnostic_text = normalize_text(
            " ".join((order.fault_name, order.fault_description, order.diagnosis))
        ).lower()
        if any(normalize_text(value).lower() in diagnostic_text for value in names if normalize_text(value)):
            return True
        for part in order.parts:
            if not part.culprit:
                continue
            mapped = kb.component_by_part_no(part.part_no) if part.part_no else None
            mapped = mapped or kb.component_by_name(part.part_name)
            if mapped and mapped.id == component_id:
                return True
        return any(
            code in kb.dtc and kb.dtc[code].component_id == component_id
            for code in order.dtc_codes
        )

    warnings: List[str] = []
    mention = normalize_text(chain.get("component_name") or chain.get("component") or "")
    component_id = str(chain.get("component_id") or "").strip()
    if component_id in kb.components:
        component = kb.components[component_id]
        mention_candidate = (
            normalizer.best_unique(mention, minimum_score=minimum_score) if mention else None
        )
        if mention_candidate and mention_candidate.component.id != component_id:
            warnings.append("component_id_name_conflict")
            return None, None, None, 0.0, warnings
        names = list(component.aliases) + [component.name]
        if not supported(component.id, names):
            warnings.append("component_without_diagnostic_or_structured_evidence")
            return None, None, None, 0.0, warnings
        gate = 1.0
        if mention and mention_candidate is None:
            warnings.append("component_name_unmapped_but_id_supported")
            gate = 0.75
        return component.id, component.name, component.subsystem_id, gate, warnings

    candidate = normalizer.best_unique(mention, minimum_score=minimum_score) if mention else None
    if candidate:
        names = list(candidate.component.aliases) + [candidate.component.name, mention]
        if not supported(candidate.component.id, names):
            warnings.append("component_without_diagnostic_or_structured_evidence")
            return None, None, None, 0.0, warnings
        gate = 1.0 if candidate.source == "exact" else max(0.60, candidate.score)
        if candidate.source != "exact":
            warnings.append("component_fuzzy_normalized")
        component = candidate.component
        return component.id, component.name, component.subsystem_id, gate, warnings

    # A missing model mention may be recovered only from an unambiguous, single
    # culprit part recorded on the same work order.
    culprit_parts = [part for part in order.parts if part.culprit and (part.part_no or part.part_name)]
    if not mention and len(culprit_parts) == 1:
        part = culprit_parts[0]
        component = kb.component_by_part_no(part.part_no) if part.part_no else None
        component = component or kb.component_by_name(part.part_name)
        if component:
            warnings.append("component_from_single_culprit_part")
            return component.id, component.name, component.subsystem_id, 0.80, warnings

    warnings.append("component_unresolved")
    return None, None, None, 0.0, warnings


def _subsystem_from_chain(
    chain: Dict[str, Any],
    kb: KnowledgeBase,
    component_id: Optional[str],
    component_name: Optional[str],
    component_subsystem: Optional[str],
) -> Tuple[Optional[str], Optional[str], float, List[str]]:
    warnings: List[str] = []
    proposed = str(chain.get("subsystem_id") or "").strip()
    proposed_name = normalize_text(chain.get("subsystem_name") or "")

    if component_subsystem and component_subsystem != "UNKNOWN":
        gate = 1.0
        if proposed and proposed in kb.subsystems and proposed != component_subsystem:
            source = (
                kb.components[component_id].subsystem_source
                if component_id in kb.components else "derived_keyword"
            )
            if source in ("expert", "curated", "bom", "functional_structure"):
                warnings.append("subsystem_component_conflict_rejected")
                gate = 0.0
            else:
                warnings.append("subsystem_component_conflict_requires_review")
                return proposed, kb.subsystems[proposed], 0.50, warnings
        return component_subsystem, kb.subsystems.get(component_subsystem, component_subsystem), gate, warnings

    if proposed in kb.subsystems and proposed != "UNKNOWN":
        return proposed, kb.subsystems[proposed], 1.0, warnings

    by_name = {normalize_text(name).lower(): item_id for item_id, name in kb.subsystems.items()}
    if proposed_name.lower() in by_name and by_name[proposed_name.lower()] != "UNKNOWN":
        item_id = by_name[proposed_name.lower()]
        return item_id, kb.subsystems[item_id], 1.0, warnings

    inferred = infer_subsystem("{} {}".format(component_name or "", proposed_name))
    if inferred != "UNKNOWN":
        warnings.append("subsystem_inferred_from_component")
        return inferred, kb.subsystems.get(inferred, inferred), 0.75, warnings
    warnings.append("subsystem_unresolved")
    return None, None, 0.0, warnings


def _explicit_cause(
    chain: Dict[str, Any],
    kb: KnowledgeBase,
    quotes: Sequence[str],
) -> Tuple[Optional[str], Optional[str], List[str], List[str]]:
    warnings: List[str] = []
    if not quotes:
        if str(chain.get("cause_id") or "") not in ("", "UNKNOWN"):
            warnings.append("cause_without_source_quote_removed")
        return None, None, [], warnings

    context = " ".join(quotes)
    cause_id, cause_name = _match_enumeration(
        chain.get("cause_id"), chain.get("cause_name"), kb.causes, context
    )
    if not cause_id or cause_id == "UNKNOWN":
        return None, None, [], warnings

    entry = kb.causes[cause_id]
    terms = [normalize_text(entry.get("name") or "")] + [
        normalize_text(value) for value in entry.get("keywords", [])
    ]
    matching_quotes = [
        quote for quote in quotes
        if any(term and term.lower() in quote.lower() for term in terms)
    ]
    if not matching_quotes:
        warnings.append("cause_category_not_supported_by_quote_removed")
        return None, None, [], warnings
    return cause_id, cause_name, sorted(set(matching_quotes)), warnings


def _effects(chain: Dict[str, Any], order: WorkOrder, kb: KnowledgeBase) -> List[Dict[str, Any]]:
    raw = chain.get("symptoms", [])
    if isinstance(raw, str):
        raw = [raw]
    raw_text = normalize_text(" ".join(str(value) for value in raw))
    source_text = normalize_text(" ".join((order.fault_name, order.fault_description, order.diagnosis)))
    result: Dict[str, Dict[str, Any]] = {}
    for symptom in kb.symptoms:
        normalized = normalize_text(symptom)
        if normalized and normalized in raw_text and normalized in source_text:
            effect_id = stable_id("EFFECT", normalized)
            result[effect_id] = {
                "effect_id": effect_id,
                "effect_name": normalized,
                "evidence": [normalized],
                "structural_gate": 1.0,
            }
    return [result[key] for key in sorted(result)]


def _entity(entity_id: str, name: str, entity_type: str) -> Dict[str, str]:
    return {"id": entity_id, "name": name, "type": entity_type}


def _triple(
    head: Dict[str, str],
    relation: str,
    tail: Dict[str, str],
    confidence: float,
    structural_gate: float,
    evidence: Sequence[str],
    context_component_id: Optional[str] = None,
) -> Dict[str, Any]:
    expected = RELATION_TYPES[relation]
    valid = head["type"] == expected[0] and tail["type"] == expected[1]
    return {
        "head": head,
        "relation": relation,
        "tail": tail,
        "model_confidence": _clamp(confidence),
        "structural_gate": _clamp(structural_gate, 0.0) if valid else 0.0,
        "evidence": list(evidence),
        "structural_valid": valid and structural_gate > 0.0,
        "context_component_id": context_component_id,
    }


def normalize_chain(
    order: WorkOrder,
    chain: Dict[str, Any],
    kb: KnowledgeBase,
    normalizer: Optional[ComponentNormalizer] = None,
    minimum_component_score: float = 0.25,
) -> Dict[str, Any]:
    """Canonicalise one extracted chain and enforce type/direction constraints."""

    normalizer = normalizer or ComponentNormalizer(kb)
    quotes = _valid_quotes(chain, order)
    component_id, component_name, component_subsystem, component_gate, warnings = _component_from_chain(
        order, chain, kb, normalizer, minimum_component_score
    )
    subsystem_id, subsystem_name, subsystem_gate, subsystem_warnings = _subsystem_from_chain(
        chain, kb, component_id, component_name, component_subsystem
    )
    warnings.extend(subsystem_warnings)

    mode_context = " ".join(
        [
            str(chain.get("failure_mode_name") or ""),
            str(chain.get("failure_mode_text") or ""),
            " ".join(quotes),
            order.fault_name,
            order.diagnosis,
        ]
    )
    mode_id, mode_name = _match_enumeration(
        chain.get("failure_mode_id"), chain.get("failure_mode_name"), kb.failure_modes, mode_context
    )
    if mode_id:
        mode_entry = kb.failure_modes[mode_id]
        mode_terms = [normalize_text(mode_entry.get("name") or "")] + [
            normalize_text(value) for value in mode_entry.get("keywords", [])
        ]
        diagnostic_text = normalize_text(
            " ".join((" ".join(quotes), order.fault_name, order.fault_description, order.diagnosis))
        ).lower()
        dtc_support = any(
            code in kb.dtc and kb.dtc[code].failure_mode_id == mode_id
            for code in order.dtc_codes
        )
        if not dtc_support and not any(
            term and term.lower() in diagnostic_text for term in mode_terms
        ):
            warnings.append("failure_mode_without_diagnostic_or_dtc_evidence")
            mode_id, mode_name = None, None
    if not mode_id:
        warnings.append("failure_mode_unresolved")

    cause_id, cause_name, cause_evidence, cause_warnings = _explicit_cause(chain, kb, quotes)
    warnings.extend(cause_warnings)
    effects = _effects(chain, order, kb)

    core_gate = min(component_gate, subsystem_gate) if component_id and subsystem_id and mode_id else 0.0
    triples: List[Dict[str, Any]] = []
    if subsystem_id and component_id:
        triples.append(
            _triple(
                _entity(subsystem_id, subsystem_name or subsystem_id, "SUBSYSTEM"),
                "HAS_COMPONENT",
                _entity(component_id, component_name or component_id, "COMPONENT"),
                min(_confidence(chain, "subsystem"), _confidence(chain, "component")),
                min(component_gate, subsystem_gate),
                quotes,
                component_id,
            )
        )
    if component_id and mode_id:
        triples.append(
            _triple(
                _entity(component_id, component_name or component_id, "COMPONENT"),
                "HAS_FAILURE_MODE",
                _entity(mode_id, mode_name or mode_id, "FAILURE_MODE"),
                min(_confidence(chain, "component"), _confidence(chain, "failure_mode")),
                component_gate,
                quotes,
                component_id,
            )
        )
    if cause_id and mode_id:
        triples.append(
            _triple(
                _entity(cause_id, cause_name or cause_id, "CAUSE"),
                "CAUSES_REPORTED",
                _entity(mode_id, mode_name or mode_id, "FAILURE_MODE"),
                min(_confidence(chain, "cause"), _confidence(chain, "failure_mode")),
                1.0,
                cause_evidence,
                component_id,
            )
        )
    for effect in effects:
        if mode_id:
            triples.append(
                _triple(
                    _entity(mode_id, mode_name or mode_id, "FAILURE_MODE"),
                    "MANIFESTS_AS",
                    _entity(effect["effect_id"], effect["effect_name"], "EFFECT"),
                    _confidence(chain, "chain"),
                    effect["structural_gate"],
                    effect["evidence"],
                    component_id,
                )
            )

    return {
        "subsystem_id": subsystem_id,
        "subsystem_name": subsystem_name,
        "component_id": component_id,
        "component_name": component_name,
        "failure_mode_id": mode_id,
        "failure_mode_name": mode_name,
        "cause_id": cause_id,
        "cause_name": cause_name,
        "cause_evidence": cause_evidence,
        "effects": effects,
        "model_confidence": min(
            _confidence(chain, "subsystem"),
            _confidence(chain, "component"),
            _confidence(chain, "failure_mode"),
        ),
        "structural_gate": core_gate,
        "evidence": quotes,
        "warnings": sorted(set(warnings)),
        "triples": triples,
    }


def _triple_key(triple: Dict[str, Any]) -> Tuple[str, str, str, str]:
    return (
        str(triple["head"]["id"]),
        str(triple["relation"]),
        str(triple["tail"]["id"]),
        str(triple.get("context_component_id") or ""),
    )


def _run_payload(row: Dict[str, Any]) -> Dict[str, Any]:
    payload = row.get("extraction", row)
    return payload if isinstance(payload, dict) else {}


def build_order_consensus(
    order: WorkOrder,
    run_rows: Sequence[Dict[str, Any]],
    kb: KnowledgeBase,
    planned_runs: Optional[int] = None,
    minimum_support: float = 0.60,
    minimum_component_score: float = 0.25,
) -> Dict[str, Any]:
    """Fuse repeated runs for one order using n_e / K semantic support."""

    if planned_runs is not None and planned_runs < 1:
        raise ValueError("planned_runs must be at least 1")
    if not math.isfinite(minimum_support) or not 0.0 <= minimum_support <= 1.0:
        raise ValueError("minimum_support must be between 0 and 1")
    if (
        not math.isfinite(minimum_component_score)
        or not 0.0 <= minimum_component_score <= 1.0
    ):
        raise ValueError("minimum_component_score must be between 0 and 1")
    normalizer = ComponentNormalizer(kb)
    record_warnings: Set[str] = set()
    indexed_map: Dict[int, Dict[str, Any]] = {}
    invalid_run_indices: List[str] = []
    for position, row in enumerate(run_rows):
        raw_run_index = row.get("run_index")
        if isinstance(raw_run_index, bool):
            valid_run_index = False
        elif isinstance(raw_run_index, int):
            valid_run_index = True
            run_index = raw_run_index
        elif isinstance(raw_run_index, str) and raw_run_index.strip().lstrip("-").isdigit():
            valid_run_index = True
            run_index = int(raw_run_index.strip())
        else:
            valid_run_index = False
        if not valid_run_index:
            record_warnings.add("invalid_run_index")
            if planned_runs is not None:
                invalid_run_indices.append(normalize_text(raw_run_index) or "<missing>")
                continue
            record_warnings.add("invalid_run_index_position_fallback")
            run_index = position
        if run_index in indexed_map:
            record_warnings.add("duplicate_run_index_deduplicated")
            existing_failed = indexed_map[run_index].get("ok") is False or indexed_map[run_index].get("error")
            replacement_ok = row.get("ok") is not False and not row.get("error")
            if existing_failed and replacement_ok:
                indexed_map[run_index] = row
        else:
            indexed_map[run_index] = row
    indexed_rows = sorted(indexed_map.items())

    denominator = int(planned_runs or len(indexed_rows) or 1)
    if denominator < len(indexed_rows):
        raise ValueError("planned_runs cannot be smaller than the number of supplied runs")
    supplied_indices = {run_index for run_index, _ in indexed_rows}
    expected_indices = (
        set(range(denominator)) if planned_runs is not None else set(supplied_indices)
    )
    missing_run_indices = sorted(expected_indices - supplied_indices)
    unexpected_run_indices = sorted(supplied_indices - expected_indices)
    if missing_run_indices:
        record_warnings.add("planned_run_slots_missing")
    if unexpected_run_indices:
        record_warnings.add("unexpected_run_indices")

    triple_observations: Dict[Tuple[str, str, str, str], Dict[str, Any]] = {}
    path_observations: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    successful_runs = 0
    failed_runs: List[Dict[str, Any]] = [
        {"run_index": value, "error": "invalid_run_index"}
        for value in invalid_run_indices
    ]

    for run_index, row in indexed_rows:
        if row.get("ok") is False or row.get("error"):
            failed_runs.append({"run_index": run_index, "error": str(row.get("error") or "failed")})
            continue
        payload = _run_payload(row)
        chains = payload.get("fault_chains", [])
        if not isinstance(chains, list):
            failed_runs.append({"run_index": run_index, "error": "fault_chains_not_a_list"})
            continue
        successful_runs += 1
        run_triples: Dict[Tuple[str, str, str, str], Dict[str, Any]] = {}
        run_paths: Dict[Tuple[str, str, str], Dict[str, Any]] = {}

        for raw_chain in chains:
            if not isinstance(raw_chain, dict):
                record_warnings.add("non_object_chain_skipped")
                continue
            normalized = normalize_chain(
                order, raw_chain, kb, normalizer, minimum_component_score
            )
            record_warnings.update(normalized["warnings"])
            for triple in normalized["triples"]:
                key = _triple_key(triple)
                previous = run_triples.get(key)
                if previous is None or triple["model_confidence"] > previous["model_confidence"]:
                    run_triples[key] = triple

            core = (
                normalized.get("subsystem_id"),
                normalized.get("component_id"),
                normalized.get("failure_mode_id"),
            )
            if all(core) and normalized["structural_gate"] > 0:
                path_key = (str(core[0]), str(core[1]), str(core[2]))
                path_item = run_paths.setdefault(
                    path_key,
                    {
                        **normalized,
                        "causes": {},
                        "effect_map": {},
                    },
                )
                if normalized["model_confidence"] > path_item["model_confidence"]:
                    for field in (
                        "subsystem_name", "component_name", "failure_mode_name",
                        "model_confidence", "structural_gate", "evidence",
                    ):
                        path_item[field] = normalized[field]
                if normalized.get("cause_id"):
                    path_item["causes"][normalized["cause_id"]] = {
                        "cause_id": normalized["cause_id"],
                        "cause_name": normalized.get("cause_name") or normalized["cause_id"],
                        "evidence": normalized.get("cause_evidence", []),
                        "structural_gate": 1.0,
                    }
                for effect in normalized.get("effects", []):
                    path_item["effect_map"][effect["effect_id"]] = effect

        for key, triple in run_triples.items():
            aggregate = triple_observations.setdefault(
                key,
                {
                    "head": triple["head"],
                    "relation": triple["relation"],
                    "tail": triple["tail"],
                    "context_component_id": triple.get("context_component_id"),
                    "run_indices": set(),
                    "confidences": [],
                    "gates": [],
                    "evidence": set(),
                },
            )
            aggregate["run_indices"].add(run_index)
            aggregate["confidences"].append(triple["model_confidence"])
            aggregate["gates"].append(triple["structural_gate"])
            aggregate["evidence"].update(triple["evidence"])

        for key, path in run_paths.items():
            aggregate = path_observations.setdefault(
                key,
                {
                    "subsystem_id": key[0],
                    "subsystem_name": path["subsystem_name"],
                    "component_id": key[1],
                    "component_name": path["component_name"],
                    "failure_mode_id": key[2],
                    "failure_mode_name": path["failure_mode_name"],
                    "run_indices": set(),
                    "confidences": [],
                    "gates": [],
                    "evidence": set(),
                    "causes": defaultdict(set),
                    "cause_names": {},
                    "cause_evidence": defaultdict(set),
                    "cause_gates": defaultdict(list),
                    "effects": defaultdict(set),
                    "effect_names": {},
                    "effect_evidence": defaultdict(set),
                    "effect_gates": defaultdict(list),
                },
            )
            aggregate["run_indices"].add(run_index)
            aggregate["confidences"].append(path["model_confidence"])
            aggregate["gates"].append(path["structural_gate"])
            aggregate["evidence"].update(path["evidence"])
            for cause_id, cause in path["causes"].items():
                aggregate["causes"][cause_id].add(run_index)
                aggregate["cause_names"][cause_id] = cause["cause_name"]
                aggregate["cause_evidence"][cause_id].update(cause.get("evidence", []))
                aggregate["cause_gates"][cause_id].append(cause.get("structural_gate", 0.0))
            for effect_id, effect in path["effect_map"].items():
                aggregate["effects"][effect_id].add(run_index)
                aggregate["effect_names"][effect_id] = effect["effect_name"]
                aggregate["effect_evidence"][effect_id].update(effect.get("evidence", []))
                aggregate["effect_gates"][effect_id].append(effect.get("structural_gate", 0.0))

    candidate_triples: List[Dict[str, Any]] = []
    support_lookup: Dict[Tuple[str, str, str, str], float] = {}
    for key, aggregate in triple_observations.items():
        count = len(aggregate["run_indices"])
        support = count / denominator
        support_lookup[key] = support
        gate = mean(aggregate["gates"]) if aggregate["gates"] else 0.0
        candidate_triples.append(
            {
                "head": aggregate["head"],
                "relation": aggregate["relation"],
                "tail": aggregate["tail"],
                "context_component_id": aggregate["context_component_id"],
                "support_count": count,
                "planned_runs": denominator,
                "semantic_support": support,
                "mean_model_confidence": mean(aggregate["confidences"]),
                "structural_gate": gate,
                "decision": (
                    "ACCEPT"
                    if support >= minimum_support and gate >= ACCEPT_STRUCTURE_THRESHOLD
                    else "REVIEW"
                ),
                "evidence": sorted(aggregate["evidence"]),
                "run_indices": sorted(aggregate["run_indices"]),
            }
        )
    candidate_triples.sort(
        key=lambda item: (
            -item["semantic_support"], item["relation"], item["head"]["id"], item["tail"]["id"]
        )
    )

    canonical_paths: List[Dict[str, Any]] = []
    for key, aggregate in path_observations.items():
        path_count = len(aggregate["run_indices"])
        membership_support = support_lookup.get((key[0], "HAS_COMPONENT", key[1], key[1]), 0.0)
        mode_support = support_lookup.get((key[1], "HAS_FAILURE_MODE", key[2], key[1]), 0.0)
        semantic_support = min(path_count / denominator, membership_support, mode_support)
        causes = []
        for cause_id, indices in aggregate["causes"].items():
            support = len(indices) / denominator
            evidence = sorted(aggregate["cause_evidence"][cause_id])
            structure = mean(aggregate["cause_gates"][cause_id]) if aggregate["cause_gates"][cause_id] else 0.0
            causes.append(
                {
                    "cause_id": cause_id,
                    "cause_name": aggregate["cause_names"][cause_id],
                    "support_count": len(indices),
                    "semantic_support": support,
                    "structural_gate": structure,
                    "decision": (
                        "ACCEPT"
                        if support >= minimum_support
                        and structure >= ACCEPT_STRUCTURE_THRESHOLD
                        and bool(evidence)
                        else "REVIEW"
                    ),
                    "evidence": evidence,
                    "run_indices": sorted(indices),
                }
            )
        effects = []
        for effect_id, indices in aggregate["effects"].items():
            support = len(indices) / denominator
            evidence = sorted(aggregate["effect_evidence"][effect_id])
            structure = mean(aggregate["effect_gates"][effect_id]) if aggregate["effect_gates"][effect_id] else 0.0
            effects.append(
                {
                    "effect_id": effect_id,
                    "effect_name": aggregate["effect_names"][effect_id],
                    "support_count": len(indices),
                    "semantic_support": support,
                    "structural_gate": structure,
                    "decision": (
                        "ACCEPT"
                        if support >= minimum_support
                        and structure >= ACCEPT_STRUCTURE_THRESHOLD
                        and bool(evidence)
                        else "REVIEW"
                    ),
                    "evidence": evidence,
                    "run_indices": sorted(indices),
                }
            )
        gate = mean(aggregate["gates"]) if aggregate["gates"] else 0.0
        canonical_paths.append(
            {
                "subsystem_id": key[0],
                "subsystem_name": aggregate["subsystem_name"],
                "component_id": key[1],
                "component_name": aggregate["component_name"],
                "failure_mode_id": key[2],
                "failure_mode_name": aggregate["failure_mode_name"],
                "path_support_count": path_count,
                "planned_runs": denominator,
                "semantic_support": semantic_support,
                "relation_supports": {
                    "HAS_COMPONENT": membership_support,
                    "HAS_FAILURE_MODE": mode_support,
                },
                "mean_model_confidence": mean(aggregate["confidences"]),
                "structural_gate": gate,
                "decision": (
                    "ACCEPT"
                    if semantic_support >= minimum_support and gate >= ACCEPT_STRUCTURE_THRESHOLD
                    else "REVIEW"
                ),
                "causes": sorted(causes, key=lambda value: (-value["semantic_support"], value["cause_id"])),
                "effects": sorted(effects, key=lambda value: (-value["semantic_support"], value["effect_id"])),
                "evidence": sorted(aggregate["evidence"]),
                "run_indices": sorted(aggregate["run_indices"]),
            }
        )
    canonical_paths.sort(
        key=lambda item: (
            -item["semantic_support"], item["subsystem_id"], item["component_id"], item["failure_mode_id"]
        )
    )

    complete_extraction = (
        successful_runs == denominator
        and not failed_runs
        and not missing_run_indices
        and not unexpected_run_indices
    )
    formal_eligible = planned_runs is not None and complete_extraction

    return {
        "work_order_id": order.work_order_id,
        "record_type": classify_record_type(order),
        "record_type_rule_version": RECORD_TYPE_RULE_VERSION,
        "support_source": (
            "planned_independent_run_frequency"
            if planned_runs is not None else "observed_independent_run_frequency"
        ),
        "support_denominator_kind": "planned_runs" if planned_runs is not None else "observed_runs",
        "planned_runs": denominator,
        "successful_runs": successful_runs,
        "complete_extraction": complete_extraction,
        "formal_eligible": formal_eligible,
        "missing_run_indices": missing_run_indices,
        "unexpected_run_indices": unexpected_run_indices,
        "invalid_run_indices": invalid_run_indices,
        "minimum_support": minimum_support,
        "candidate_triples": candidate_triples,
        "canonical_paths": canonical_paths,
        "failed_runs": failed_runs,
        "warnings": sorted(record_warnings),
    }


def build_consensus_records(
    orders: Iterable[WorkOrder],
    run_rows: Iterable[Dict[str, Any]],
    kb: KnowledgeBase,
    planned_runs: Optional[int] = None,
    minimum_support: float = 0.60,
    minimum_component_score: float = 0.25,
    include_orders_without_runs: bool = False,
) -> List[Dict[str, Any]]:
    order_map = {order.work_order_id: order for order in orders}
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in run_rows:
        order_id = str(row.get("work_order_id") or _run_payload(row).get("work_order_id") or "")
        if order_id in order_map:
            grouped[order_id].append(row)
    selected_orders = [
        order for order in order_map.values()
        if include_orders_without_runs or order.work_order_id in grouped
    ]
    return [
        build_order_consensus(
            order,
            grouped.get(order.work_order_id, []),
            kb,
            planned_runs,
            minimum_support,
            minimum_component_score,
        )
        for order in selected_orders
    ]


def prediction_to_candidate_record(
    order: WorkOrder,
    prediction: Dict[str, Any],
    kb: KnowledgeBase,
    minimum_component_score: float = 0.25,
    normalizer: Optional[ComponentNormalizer] = None,
) -> Dict[str, Any]:
    """Adapt a single-model prediction for graph smoke tests.

    This adapter deliberately labels its support source as model confidence; it
    must not be reported as repeated-run semantic consistency.
    """

    normalizer = normalizer or ComponentNormalizer(kb)
    triples: Dict[Tuple[str, str, str, str], Dict[str, Any]] = {}
    paths: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    warnings: Set[str] = set()
    for raw_chain in prediction.get("fault_chains", []):
        if not isinstance(raw_chain, dict):
            continue
        normalized = normalize_chain(order, raw_chain, kb, normalizer, minimum_component_score)
        warnings.update(normalized["warnings"])
        for triple in normalized["triples"]:
            key = _triple_key(triple)
            # Keep model evidence and structure separate; the graph fusion
            # applies the structural gate exactly once.
            support = triple["model_confidence"]
            item = {
                "head": triple["head"],
                "relation": triple["relation"],
                "tail": triple["tail"],
                "context_component_id": triple.get("context_component_id"),
                "support_count": 1,
                "planned_runs": 1,
                "semantic_support": support,
                "mean_model_confidence": triple["model_confidence"],
                "structural_gate": triple["structural_gate"],
                "decision": (
                    "ACCEPT"
                    if support >= 0.50 and triple["structural_gate"] >= ACCEPT_STRUCTURE_THRESHOLD
                    else "REVIEW"
                ),
                "evidence": triple["evidence"],
                "run_indices": [0],
            }
            if key not in triples or support > triples[key]["semantic_support"]:
                triples[key] = item

        core = (
            normalized.get("subsystem_id"),
            normalized.get("component_id"),
            normalized.get("failure_mode_id"),
        )
        if all(core) and normalized["structural_gate"] > 0:
            key = (str(core[0]), str(core[1]), str(core[2]))
            relation_supports = {
                "HAS_COMPONENT": triples.get((key[0], "HAS_COMPONENT", key[1], key[1]), {}).get("semantic_support", 0.0),
                "HAS_FAILURE_MODE": triples.get((key[1], "HAS_FAILURE_MODE", key[2], key[1]), {}).get("semantic_support", 0.0),
            }
            support = min(relation_supports.values())
            causes = []
            if normalized.get("cause_id"):
                cause_support = triples.get(
                    (normalized["cause_id"], "CAUSES_REPORTED", key[2], key[1]), {}
                ).get("semantic_support", 0.0)
                causes.append(
                    {
                        "cause_id": normalized["cause_id"],
                        "cause_name": normalized.get("cause_name") or normalized["cause_id"],
                        "support_count": 1,
                        "semantic_support": cause_support,
                        "structural_gate": 1.0,
                        "decision": (
                            "ACCEPT"
                            if cause_support >= 0.50 and bool(normalized.get("cause_evidence"))
                            else "REVIEW"
                        ),
                        "evidence": normalized.get("cause_evidence", []),
                        "run_indices": [0],
                    }
                )
            effects = []
            for effect in normalized.get("effects", []):
                effect_support = triples.get(
                    (key[2], "MANIFESTS_AS", effect["effect_id"], key[1]), {}
                ).get("semantic_support", 0.0)
                effects.append(
                    {
                        **effect,
                        "support_count": 1,
                        "semantic_support": effect_support,
                        "decision": (
                            "ACCEPT"
                            if effect_support >= 0.50
                            and effect.get("structural_gate", 0.0) >= ACCEPT_STRUCTURE_THRESHOLD
                            and bool(effect.get("evidence"))
                            else "REVIEW"
                        ),
                        "run_indices": [0],
                    }
                )
            item = {
                "subsystem_id": key[0],
                "subsystem_name": normalized["subsystem_name"],
                "component_id": key[1],
                "component_name": normalized["component_name"],
                "failure_mode_id": key[2],
                "failure_mode_name": normalized["failure_mode_name"],
                "path_support_count": 1,
                "planned_runs": 1,
                "semantic_support": support,
                "relation_supports": relation_supports,
                "mean_model_confidence": normalized["model_confidence"],
                "structural_gate": normalized["structural_gate"],
                "decision": (
                    "ACCEPT"
                    if support >= 0.50 and normalized["structural_gate"] >= ACCEPT_STRUCTURE_THRESHOLD
                    else "REVIEW"
                ),
                "causes": causes,
                "effects": effects,
                "evidence": normalized["evidence"],
                "run_indices": [0],
            }
            if key not in paths or support > paths[key]["semantic_support"]:
                paths[key] = item

    return {
        "work_order_id": order.work_order_id,
        "record_type": classify_record_type(order),
        "record_type_rule_version": RECORD_TYPE_RULE_VERSION,
        "reported_record_type": prediction.get("record_type"),
        "support_source": "single_model_confidence",
        "planned_runs": 1,
        "successful_runs": 1,
        "complete_extraction": True,
        "formal_eligible": False,
        "experiment_complete": False,
        "minimum_support": 0.50,
        "candidate_triples": sorted(
            triples.values(),
            key=lambda item: (-item["semantic_support"], item["relation"], item["head"]["id"]),
        ),
        "canonical_paths": sorted(
            paths.values(),
            key=lambda item: (-item["semantic_support"], item["component_id"], item["failure_mode_id"]),
        ),
        "failed_runs": [],
        "warnings": sorted(warnings),
    }


def predictions_to_candidate_records(
    orders: Iterable[WorkOrder],
    predictions: Iterable[Dict[str, Any]],
    kb: KnowledgeBase,
    minimum_component_score: float = 0.25,
) -> List[Dict[str, Any]]:
    if (
        not math.isfinite(minimum_component_score)
        or not 0.0 <= minimum_component_score <= 1.0
    ):
        raise ValueError("minimum_component_score must be between 0 and 1")
    prediction_map = {str(row.get("work_order_id") or ""): row for row in predictions}
    normalizer = ComponentNormalizer(kb)
    return [
        prediction_to_candidate_record(
            order,
            prediction_map.get(order.work_order_id, {"fault_chains": []}),
            kb,
            minimum_component_score,
            normalizer,
        )
        for order in orders
    ]
