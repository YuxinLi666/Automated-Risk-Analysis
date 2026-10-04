from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class PartMention:
    part_no: str = ""
    part_name: str = ""
    culprit: bool = False


@dataclass
class WorkOrder:
    work_order_id: str
    settlement_date: str = ""
    fault_date: str = ""
    engine_id: str = ""
    engine_series: str = ""
    chassis_id: str = ""
    province: str = ""
    city: str = ""
    mileage: str = ""
    total_cost: float = 0.0
    fault_name: str = ""
    fault_description: str = ""
    diagnosis: str = ""
    solution: str = ""
    dtc_raw: str = ""
    dtc_codes: List[str] = field(default_factory=list)
    parts: List[PartMention] = field(default_factory=list)
    source_rows: List[int] = field(default_factory=list)

    def model_text(self) -> str:
        parts = [
            ("NAME", self.fault_name),
            ("DESC", self.fault_description),
            ("DIAGNOSIS", self.diagnosis),
            ("SOLUTION", self.solution),
            ("DTC", " ".join(self.dtc_codes) or self.dtc_raw),
            ("PART_NO", " ".join(p.part_no for p in self.parts if p.part_no)),
            ("PART_NAME", " ".join(p.part_name for p in self.parts if p.part_name)),
        ]
        return "\n".join("[{}] {}".format(k, v) for k, v in parts if v)

    def candidate_text(self) -> str:
        """Field-labelled text for auditable candidate extraction.

        Cost, location, and asset identifiers are intentionally excluded: they
        are graph/ranking metadata rather than semantic extraction evidence.
        """

        part_values = [
            "{}|{}|{}".format(part.part_no, part.part_name, "CULPRIT" if part.culprit else "OTHER")
            for part in self.parts
            if part.part_no or part.part_name
        ]
        values = [
            ("FAULT_NAME", self.fault_name),
            ("FAULT_DESCRIPTION", self.fault_description),
            ("DIAGNOSIS", self.diagnosis),
            ("SOLUTION", self.solution),
            ("DTC", " ".join(self.dtc_codes) or self.dtc_raw),
            ("PART", " ; ".join(part_values)),
        ]
        return "\n".join("[{}] {}".format(key, value) for key, value in values if value)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Dict[str, Any]) -> "WorkOrder":
        value = dict(value)
        value["parts"] = [PartMention(**p) for p in value.get("parts", [])]
        return cls(**value)


@dataclass
class Evidence:
    field: str
    text: str
    start: int = -1
    end: int = -1


@dataclass
class FaultChain:
    subsystem_id: Optional[str] = None
    subsystem_name: Optional[str] = None
    component_id: Optional[str] = None
    component_name: Optional[str] = None
    failure_mode_id: Optional[str] = None
    failure_mode_name: Optional[str] = None
    cause_id: Optional[str] = None
    cause_name: Optional[str] = None
    cause_text: str = ""
    cause_raw: str = ""
    cause_method: str = ""
    symptoms: List[str] = field(default_factory=list)
    dtc: List[str] = field(default_factory=list)
    evidence: Dict[str, Evidence] = field(default_factory=dict)
    confidence: Dict[str, float] = field(default_factory=dict)
    sources: List[str] = field(default_factory=list)
    decision: str = "REVIEW"

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        return result


@dataclass
class ExtractionResult:
    work_order_id: str
    fault_chains: List[FaultChain]
    record_type: str = "FAULT"
    model_version: str = ""
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EntityAnnotation:
    entity_id: str
    label: str
    start: int
    end: int
    text: str
    canonical_id: Optional[str] = None


@dataclass
class RelationAnnotation:
    relation: str
    head_id: str
    tail_id: str


@dataclass
class AnnotatedDocument:
    document_id: str
    text: str
    entities: List[EntityAnnotation] = field(default_factory=list)
    relations: List[RelationAnnotation] = field(default_factory=list)
    chains: List[Dict[str, Any]] = field(default_factory=list)
    split: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
