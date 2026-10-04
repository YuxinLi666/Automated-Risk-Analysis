import unittest

from engine_fault_nlp.constraints import ConstraintDecoder
from engine_fault_nlp.ontology import ComponentNode, DtcNode, KnowledgeBase, merge_knowledge_bases
from engine_fault_nlp.schemas import FaultChain, PartMention, WorkOrder
from engine_fault_nlp.weak_supervision import WeakSupervisor


class WeakConstraintTest(unittest.TestCase):
    def setUp(self):
        self.component = ComponentNode(
            id="COMP_SENSOR",
            name="排气温度传感器",
            subsystem_id="AFTERTREATMENT",
            aliases=["排温传感器"],
            part_numbers=["P100"],
        )
        self.kb = KnowledgeBase(components={self.component.id: self.component})
        self.kb.dtc["1877"] = DtcNode("1877", "排气温度信号异常", component_id=self.component.id)

    def test_part_dtc_and_explicit_cause(self):
        order = WorkOrder(
            work_order_id="G1",
            fault_name="故障灯亮",
            diagnosis="排气温度传感器线束磨破导致信号异常",
            dtc_codes=["1877"],
            parts=[PartMention("P100", "排气温度传感器", True)],
        )
        result = WeakSupervisor(self.kb).label(order)
        chain = result.fault_chains[0]
        self.assertEqual(chain.component_id, "COMP_SENSOR")
        self.assertEqual(chain.subsystem_id, "AFTERTREATMENT")
        self.assertEqual(chain.cause_id, "HARNESS_INTERFERENCE")
        self.assertIn("cause", chain.evidence)

    def test_unstated_cause_is_unknown(self):
        order = WorkOrder(
            work_order_id="G2",
            diagnosis="排气温度传感器损坏",
            parts=[PartMention("P100", "排气温度传感器", True)],
        )
        chain = WeakSupervisor(self.kb).label(order).fault_chains[0]
        self.assertEqual(chain.cause_id, "UNKNOWN")

    def test_constraint_removes_unsupported_cause(self):
        chain = FaultChain(
            subsystem_id="IGNITION",
            component_id="COMP_SENSOR",
            cause_id="MATERIAL_FATIGUE",
            cause_name="材料疲劳",
        )
        warnings = ConstraintDecoder(self.kb).apply(chain)
        self.assertIn("subsystem_component_conflict", warnings)
        self.assertIn("cause_without_evidence", warnings)
        self.assertEqual(chain.subsystem_id, "AFTERTREATMENT")
        self.assertEqual(chain.cause_id, "UNKNOWN")

    def test_merge_keeps_base_dtc_and_adds_parts(self):
        update_component = ComponentNode("COMP_NEW", "新传感器", "ECM_ELECTRICAL", ["新传感器"], ["N1"])
        update = KnowledgeBase(components={update_component.id: update_component})
        merged = merge_knowledge_bases(self.kb, update)
        self.assertIn("1877", merged.dtc)
        self.assertIn("COMP_NEW", merged.components)

    def test_unknown_component_uses_context_subsystem(self):
        unknown = ComponentNode("COMP_OTHER", "其它零件", "UNKNOWN", ["其它零件"], ["O1"])
        kb = KnowledgeBase(components={unknown.id: unknown})
        order = WorkOrder(
            "G4",
            fault_name="发动机一缸失火",
            diagnosis="一缸失火",
            parts=[PartMention("O1", "其它零件", True)],
        )
        chain = WeakSupervisor(kb).label(order).fault_chains[0]
        self.assertEqual(chain.subsystem_id, "IGNITION")
        ConstraintDecoder(kb).apply(chain)
        self.assertEqual(chain.subsystem_id, "IGNITION")

    def test_unknown_component_without_context_uses_general_bucket(self):
        unknown = ComponentNode("COMP_OTHER", "其它零件", "UNKNOWN", ["其它零件"], ["O1"])
        kb = KnowledgeBase(components={unknown.id: unknown})
        order = WorkOrder("G5", fault_name="问题待查", parts=[PartMention("O1", "其它零件", True)])
        chain = WeakSupervisor(kb).label(order).fault_chains[0]
        self.assertEqual(chain.subsystem_id, "ENGINE_GENERAL")

    def test_generic_sensor_is_disambiguated_by_context(self):
        sensor = ComponentNode("COMP_TEMP", "温度传感器", "ECM_ELECTRICAL", ["温度传感器"], ["T1"])
        kb = KnowledgeBase(components={sensor.id: sensor})
        order = WorkOrder(
            "G6",
            diagnosis="后处理排温传感器损坏",
            parts=[PartMention("T1", "温度传感器", True)],
        )
        chain = WeakSupervisor(kb).label(order).fault_chains[0]
        self.assertEqual(chain.subsystem_id, "AFTERTREATMENT")
        ConstraintDecoder(kb).apply(chain)
        self.assertEqual(chain.subsystem_id, "AFTERTREATMENT")


if __name__ == "__main__":
    unittest.main()
