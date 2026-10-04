from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Tuple

from .cause_extraction import extract_diagnostic_cause
from .normalizer import ComponentNormalizer
from .ontology import AMBIGUOUS_COMPONENT_NAMES, KnowledgeBase, infer_subsystem
from .record_type import classify_record_type
from .schemas import Evidence, ExtractionResult, FaultChain, PartMention, WorkOrder
from .text import is_negated, normalize_text


def _find_entry(text: str, entries: Dict[str, Dict[str, object]]) -> Tuple[Optional[str], Optional[str]]:
    for entry_id, entry in entries.items():
        for keyword in entry.get("keywords", []):
            start = text.find(str(keyword))
            if start >= 0 and not is_negated(text, start):
                return entry_id, str(keyword)
    return None, None


class WeakSupervisor:
    def __init__(self, kb: KnowledgeBase):
        self.kb = kb
        self.normalizer = ComponentNormalizer(kb)

    def label(self, order: WorkOrder) -> ExtractionResult:
        record_type = classify_record_type(order)
        text = normalize_text("\n".join([order.fault_name, order.fault_description, order.diagnosis, order.solution]))
        selected_parts = [part for part in order.parts if part.culprit] or order.parts
        chains: List[FaultChain] = []
        for part in selected_parts:
            chain = self._chain_from_part(order, part, text)
            if chain:
                chains.append(chain)

        if not chains:
            chains.extend(self._chains_from_dtc(order, text))
        if not chains:
            candidate = self._component_from_text(text)
            if candidate:
                chains.append(self._new_chain(order, candidate, text, 0.82, "alias_text"))

        if not chains:
            chain = FaultChain(dtc=order.dtc_codes, sources=["no_component"], decision="REVIEW")
            self._add_semantics(chain, order, text)
            chains.append(chain)
        chains = self._deduplicate(chains)
        if record_type != "FAULT":
            for chain in chains:
                chain.cause_text = ""
                chain.cause_method = "non_fault_record"
                chain.confidence["cause_text"] = 0.0
                chain.evidence.pop("cause", None)
        return ExtractionResult(
            order.work_order_id,
            chains,
            record_type=record_type,
            model_version="weak-supervision-v1",
        )

    def _chain_from_part(self, order: WorkOrder, part: PartMention, text: str) -> Optional[FaultChain]:
        component = self.kb.component_by_part_no(part.part_no) if part.part_no else None
        score, source = (0.995, "part_number_exact") if component else (0.0, "")
        if component is None and part.part_name:
            component = self.kb.component_by_name(part.part_name)
            score, source = (0.97, "part_name_exact") if component else (0.0, "")
        if component is None and part.part_name:
            candidate = self.normalizer.best(part.part_name)
            if candidate:
                component, score, source = candidate.component, candidate.score, candidate.source
        if component is None:
            return None
        return self._new_chain(order, component, text, score, source)

    def _chains_from_dtc(self, order: WorkOrder, text: str) -> List[FaultChain]:
        chains = []
        for code in order.dtc_codes:
            node = self.kb.dtc.get(code)
            if not node or not node.component_id or node.component_id not in self.kb.components:
                continue
            chain = self._new_chain(order, self.kb.components[node.component_id], text, 0.94, "dtc_dictionary")
            if node.failure_mode_id and node.failure_mode_id in self.kb.failure_modes:
                chain.failure_mode_id = node.failure_mode_id
                chain.failure_mode_name = self.kb.failure_modes[node.failure_mode_id]["name"]
                chain.confidence["failure_mode"] = 0.93
                chain.sources.append("dtc_failure_mode")
            chains.append(chain)
        return chains

    def _component_from_text(self, text: str):
        matches = []
        for component in self.kb.components.values():
            for alias in set(component.aliases + [component.name]):
                start = text.find(alias)
                if start >= 0 and not is_negated(text, start):
                    matches.append((len(alias), component))
        return max(matches, key=lambda pair: pair[0])[1] if matches else None

    def _new_chain(self, order, component, text, score, source):
        subsystem_id = component.subsystem_id
        subsystem_source = "component_ontology"
        subsystem_score = max(0.8, score - 0.01)
        if subsystem_id == "UNKNOWN" or component.name in AMBIGUOUS_COMPONENT_NAMES:
            contextual = infer_subsystem(text)
            if contextual != "UNKNOWN" and contextual != subsystem_id:
                subsystem_id = contextual
                subsystem_source = "work_order_context"
                subsystem_score = 0.88 if component.subsystem_id != "UNKNOWN" else 0.84
            elif subsystem_id == "UNKNOWN":
                subsystem_id = "ENGINE_GENERAL"
                subsystem_source = "general_unlocated_fallback"
                subsystem_score = 0.50
        chain = FaultChain(
            subsystem_id=subsystem_id,
            subsystem_name=self.kb.subsystems.get(subsystem_id, subsystem_id),
            component_id=component.id,
            component_name=component.name,
            dtc=list(order.dtc_codes),
            confidence={"component": score, "subsystem": subsystem_score},
            sources=[source, subsystem_source],
        )
        mention = next((a for a in component.aliases + [component.name] if a and a in text), component.name)
        chain.evidence["component"] = Evidence("document", mention)
        self._add_semantics(chain, order, text)
        return chain

    def _add_semantics(self, chain: FaultChain, order: WorkOrder, text: str) -> None:
        mode_id, mode_evidence = _find_entry(text, self.kb.failure_modes)
        if mode_id:
            chain.failure_mode_id = mode_id
            chain.failure_mode_name = self.kb.failure_modes[mode_id]["name"]
            chain.evidence["failure_mode"] = Evidence("document", mode_evidence or "")
            chain.confidence["failure_mode"] = 0.86
            chain.sources.append("failure_mode_pattern")
        direct_cause = extract_diagnostic_cause(order.diagnosis, chain.component_name or "")
        chain.cause_text = direct_cause.text
        chain.cause_raw = direct_cause.raw
        chain.cause_method = direct_cause.method
        chain.confidence["cause_text"] = direct_cause.confidence
        if direct_cause.text:
            chain.evidence["cause"] = Evidence("diagnosis", direct_cause.text)
            chain.sources.append("diagnostic_cause_extractor")

        cause_id, cause_evidence = _find_entry(order.diagnosis, self.kb.causes)
        if cause_id and cause_id != "UNKNOWN":
            chain.cause_id = cause_id
            chain.cause_name = self.kb.causes[cause_id]["name"]
            chain.evidence["root_cause"] = Evidence("diagnosis", cause_evidence or "")
            chain.confidence["cause"] = 0.78
            chain.sources.append("explicit_cause_pattern")
        else:
            chain.cause_id = "UNKNOWN"
            chain.cause_name = self.kb.causes["UNKNOWN"]["name"]
            chain.confidence["cause"] = 0.95
            chain.sources.append("cause_not_stated")
        chain.symptoms = [symptom for symptom in self.kb.symptoms if symptom in text]
        required = (chain.confidence.get("subsystem", 0), chain.confidence.get("component", 0), chain.confidence.get("failure_mode", 0))
        chain.confidence["chain"] = min(required) if all(required) else 0.0

    @staticmethod
    def _deduplicate(chains: Iterable[FaultChain]) -> List[FaultChain]:
        best = {}
        for chain in chains:
            key = (chain.component_id, chain.failure_mode_id, chain.cause_id)
            if key not in best or chain.confidence.get("chain", 0) > best[key].confidence.get("chain", 0):
                best[key] = chain
        return list(best.values())
