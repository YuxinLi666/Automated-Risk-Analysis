from __future__ import annotations

from typing import List, Optional

from .baseline import BaselineModel
from .calibration import ThresholdPolicy
from .constraints import ConstraintDecoder
from .ontology import KnowledgeBase
from .schemas import ExtractionResult, FaultChain, WorkOrder
from .weak_supervision import WeakSupervisor


class HybridExtractor:
    def __init__(
        self,
        kb: KnowledgeBase,
        baseline: Optional[BaselineModel] = None,
        policy: Optional[ThresholdPolicy] = None,
    ):
        self.kb = kb
        self.weak = WeakSupervisor(kb)
        self.baseline = baseline
        self.policy = policy or ThresholdPolicy()
        self.decoder = ConstraintDecoder(kb)

    def predict(self, order: WorkOrder) -> ExtractionResult:
        result = self.weak.label(order)
        if self.baseline:
            scores = self.baseline.predict(order.model_text())
            self._fuse_baseline(result.fault_chains, scores)
        all_warnings = []
        for chain in result.fault_chains:
            warnings = self.decoder.apply(chain)
            chain.decision = "NON_FAULT" if result.record_type != "FAULT" else self.policy.decide(chain, bool(warnings))
            all_warnings.extend(warnings)
        result.model_version = "hybrid-v1" if self.baseline else "rules-v1"
        result.warnings = sorted(set(all_warnings))
        return result

    def _fuse_baseline(self, chains: List[FaultChain], scores) -> None:
        if not chains:
            chains.append(FaultChain())
        for chain in chains:
            self._fill_field(chain, "subsystem", scores.get("subsystem_id", []))
            self._fill_field(chain, "component", scores.get("component_id", []))
            self._fill_field(chain, "failure_mode", scores.get("failure_mode_id", []))
            self._fill_field(chain, "cause", scores.get("cause_id", []))
            required = [chain.confidence.get(name, 0.0) for name in ("subsystem", "component", "failure_mode")]
            chain.confidence["chain"] = min(required) if all(required) else 0.0

    def _fill_field(self, chain: FaultChain, field: str, ranked) -> None:
        if not ranked:
            return
        if field == "subsystem":
            known_ranked = [item for item in ranked if item[0] != "UNKNOWN"]
            if known_ranked:
                ranked = known_ranked
        label, score = ranked[0]
        current_id = getattr(chain, field + "_id")
        current_score = chain.confidence.get(field, 0.0)
        if field == "subsystem" and current_id == "UNKNOWN":
            current_score = 0.0
        if current_id and current_score >= score:
            return
        setattr(chain, field + "_id", label)
        if field == "subsystem":
            setattr(chain, field + "_name", self.kb.subsystems.get(label, label))
        elif field == "component" and label in self.kb.components:
            setattr(chain, field + "_name", self.kb.components[label].name)
        elif field == "failure_mode" and label in self.kb.failure_modes:
            setattr(chain, field + "_name", self.kb.failure_modes[label]["name"])
        elif field == "cause" and label in self.kb.causes:
            setattr(chain, field + "_name", self.kb.causes[label]["name"])
        chain.confidence[field] = score
        chain.sources.append("baseline_{}".format(field))
