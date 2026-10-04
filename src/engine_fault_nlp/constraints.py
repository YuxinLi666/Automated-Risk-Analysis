from __future__ import annotations

from typing import List

from .ontology import AMBIGUOUS_COMPONENT_NAMES, KnowledgeBase
from .schemas import FaultChain


class ConstraintDecoder:
    def __init__(self, kb: KnowledgeBase):
        self.kb = kb

    def apply(self, chain: FaultChain) -> List[str]:
        warnings = []
        if chain.component_id and chain.component_id in self.kb.components:
            component = self.kb.components[chain.component_id]
            if (
                component.subsystem_id != "UNKNOWN"
                and component.name not in AMBIGUOUS_COMPONENT_NAMES
                and chain.subsystem_id
                and chain.subsystem_id != component.subsystem_id
            ):
                warnings.append("subsystem_component_conflict")
                chain.subsystem_id = component.subsystem_id
                chain.subsystem_name = self.kb.subsystems.get(component.subsystem_id, component.subsystem_id)
                chain.confidence["subsystem"] = min(chain.confidence.get("subsystem", 1.0), 0.8)
            elif not chain.subsystem_id:
                chain.subsystem_id = component.subsystem_id
                chain.subsystem_name = self.kb.subsystems.get(component.subsystem_id, component.subsystem_id)
        if chain.cause_id and chain.cause_id != "UNKNOWN" and "root_cause" not in chain.evidence:
            warnings.append("cause_without_evidence")
            chain.cause_id = "UNKNOWN"
            chain.cause_name = self.kb.causes.get("UNKNOWN", {}).get("name", "未说明")
            chain.confidence["cause"] = 0.99
        known_codes = [code for code in chain.dtc if code in self.kb.dtc]
        dtc_components = {self.kb.dtc[code].component_id for code in known_codes if self.kb.dtc[code].component_id}
        if chain.component_id and dtc_components and chain.component_id not in dtc_components:
            warnings.append("dtc_component_conflict")
            chain.confidence["chain"] = min(chain.confidence.get("chain", 1.0), 0.49)
        return warnings
