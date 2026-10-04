import unittest

from engine_fault_nlp.consensus import build_order_consensus
from engine_fault_nlp.ontology import ComponentNode, KnowledgeBase
from engine_fault_nlp.schemas import WorkOrder


def make_kb():
    kb = KnowledgeBase()
    kb.components = {
        "STARTER": ComponentNode(
            "STARTER", "起动机", "START_CHARGE", aliases=["起动机", "启动机"],
            part_numbers=["S-1"], subsystem_source="expert"
        )
    }
    return kb


def extraction_chain(component_name="起动机", subsystem_id="START_CHARGE", cause=False):
    value = {
        "subsystem_id": subsystem_id,
        "subsystem_name": "起动与充电系统",
        "component_name": component_name,
        "failure_mode_id": "FUNCTION_FAILED",
        "failure_mode_name": "功能失效/损坏",
        "cause_id": "MATERIAL_FATIGUE" if cause else "UNKNOWN",
        "cause_name": "材料疲劳" if cause else "未说明",
        "evidence": {
            "component": {"field": "DIAGNOSIS", "text": "起动机损坏"},
            "failure_mode": {"field": "DIAGNOSIS", "text": "起动机损坏"},
        },
        "confidence": {
            "subsystem": 0.9,
            "component": 0.9,
            "failure_mode": 0.8,
            "cause": 0.8,
        },
    }
    return value


class ConsensusTest(unittest.TestCase):
    def test_duplicate_in_one_run_counts_once_and_three_of_five_is_point_six(self):
        order = WorkOrder("W1", fault_name="无法启动", diagnosis="检查起动机损坏")
        chain_a = extraction_chain("启动机")
        chain_b = extraction_chain("起动机")
        runs = [
            {"work_order_id": "W1", "run_index": 0, "fault_chains": [chain_a, chain_a]},
            {"work_order_id": "W1", "run_index": 1, "fault_chains": [chain_b]},
            {"work_order_id": "W1", "run_index": 2, "fault_chains": [chain_a]},
            {"work_order_id": "W1", "run_index": 3, "fault_chains": []},
            {"work_order_id": "W1", "run_index": 4, "fault_chains": []},
        ]
        result = build_order_consensus(order, runs, make_kb(), planned_runs=5)
        mode_edge = next(
            item for item in result["candidate_triples"]
            if item["relation"] == "HAS_FAILURE_MODE"
        )
        self.assertEqual(mode_edge["support_count"], 3)
        self.assertAlmostEqual(mode_edge["semantic_support"], 0.6)
        self.assertEqual(mode_edge["context_component_id"], "STARTER")
        self.assertEqual(result["canonical_paths"][0]["decision"], "ACCEPT")

    def test_specific_cause_without_matching_source_quote_is_removed(self):
        order = WorkOrder("W1", diagnosis="检查起动机损坏")
        runs = [{"work_order_id": "W1", "run_index": 0, "fault_chains": [extraction_chain(cause=True)]}]
        result = build_order_consensus(order, runs, make_kb(), planned_runs=1)
        self.assertFalse(any(item["relation"] == "CAUSES_REPORTED" for item in result["candidate_triples"]))
        self.assertIn("cause_category_not_supported_by_quote_removed", result["warnings"])

    def test_component_subsystem_conflict_is_rejected(self):
        order = WorkOrder("W1", diagnosis="检查起动机损坏")
        chain = extraction_chain(subsystem_id="LUBRICATION")
        runs = [{"work_order_id": "W1", "run_index": 0, "fault_chains": [chain]}]
        result = build_order_consensus(order, runs, make_kb(), planned_runs=1)
        self.assertEqual(result["canonical_paths"], [])
        membership = next(
            item for item in result["candidate_triples"] if item["relation"] == "HAS_COMPONENT"
        )
        self.assertEqual(membership["structural_gate"], 0.0)
        self.assertEqual(membership["decision"], "REVIEW")

    def test_duplicate_run_identifier_is_not_a_second_vote(self):
        order = WorkOrder("W1", diagnosis="检查起动机损坏")
        chain = extraction_chain()
        runs = [
            {"work_order_id": "W1", "run_index": 7, "fault_chains": [chain]},
            {"work_order_id": "W1", "run_index": 7, "fault_chains": [chain]},
        ]
        result = build_order_consensus(order, runs, make_kb(), planned_runs=2)
        mode_edge = next(
            item for item in result["candidate_triples"]
            if item["relation"] == "HAS_FAILURE_MODE"
        )
        self.assertEqual(mode_edge["support_count"], 1)
        self.assertAlmostEqual(mode_edge["semantic_support"], 0.5)
        self.assertIn("duplicate_run_index_deduplicated", result["warnings"])

    def test_solution_only_replacement_does_not_prove_component_failure(self):
        order = WorkOrder("W1", fault_name="无法启动", solution="更换起动机后恢复正常")
        chain = extraction_chain()
        chain["evidence"] = {
            "component": {"field": "SOLUTION", "text": "起动机"},
            "failure_mode": {"field": "SOLUTION", "text": "更换起动机"},
        }
        result = build_order_consensus(
            order,
            [{"work_order_id": "W1", "run_index": 0, "fault_chains": [chain]}],
            make_kb(),
            planned_runs=1,
        )
        self.assertEqual(result["canonical_paths"], [])
        self.assertIn("component_without_diagnostic_or_structured_evidence", result["warnings"])

    def test_component_id_and_name_conflict_is_not_accepted(self):
        kb = make_kb()
        kb.components["WATER_PUMP"] = ComponentNode(
            "WATER_PUMP", "水泵", "COOLING", aliases=["水泵"], subsystem_source="expert"
        )
        order = WorkOrder("W1", diagnosis="检查水泵泄漏")
        chain = extraction_chain(component_name="水泵")
        chain["component_id"] = "STARTER"
        chain["failure_mode_id"] = "LEAK"
        chain["failure_mode_name"] = "泄漏"
        chain["evidence"] = {
            "component": {"field": "DIAGNOSIS", "text": "水泵"},
            "failure_mode": {"field": "DIAGNOSIS", "text": "水泵泄漏"},
        }
        result = build_order_consensus(
            order,
            [{"work_order_id": "W1", "run_index": 0, "fault_chains": [chain]}],
            kb,
            planned_runs=1,
        )
        self.assertEqual(result["canonical_paths"], [])
        self.assertIn("component_id_name_conflict", result["warnings"])

    def test_reported_cause_keeps_its_own_traceable_evidence(self):
        order = WorkOrder("W1", diagnosis="材料疲劳导致起动机损坏")
        chain = extraction_chain(cause=True)
        chain["evidence"]["root_cause"] = {
            "field": "DIAGNOSIS", "text": "材料疲劳"
        }
        result = build_order_consensus(
            order,
            [{"work_order_id": "W1", "run_index": 0, "fault_chains": [chain]}],
            make_kb(),
            planned_runs=1,
        )
        branch = result["canonical_paths"][0]["causes"][0]
        self.assertEqual(branch["decision"], "ACCEPT")
        self.assertEqual(branch["evidence"], ["材料疲劳"])

    def test_missing_planned_slot_is_not_formally_eligible(self):
        order = WorkOrder("W1", diagnosis="检查起动机损坏")
        chain = extraction_chain()
        result = build_order_consensus(
            order,
            [{"work_order_id": "W1", "run_index": 0, "fault_chains": [chain]}],
            make_kb(),
            planned_runs=2,
        )
        self.assertFalse(result["complete_extraction"])
        self.assertFalse(result["formal_eligible"])
        self.assertEqual(result["missing_run_indices"], [1])
        self.assertIn("planned_run_slots_missing", result["warnings"])

    def test_all_planned_slots_are_formally_eligible(self):
        order = WorkOrder("W1", diagnosis="检查起动机损坏")
        chain = extraction_chain()
        result = build_order_consensus(
            order,
            [
                {"work_order_id": "W1", "run_index": 0, "fault_chains": [chain]},
                {"work_order_id": "W1", "run_index": 1, "fault_chains": []},
            ],
            make_kb(),
            planned_runs=2,
        )
        self.assertTrue(result["complete_extraction"])
        self.assertTrue(result["formal_eligible"])

    def test_invalid_planned_run_index_is_not_position_filled(self):
        order = WorkOrder("W1", diagnosis="检查起动机损坏")
        result = build_order_consensus(
            order,
            [{
                "work_order_id": "W1",
                "run_index": "not-an-index",
                "fault_chains": [extraction_chain()],
            }],
            make_kb(),
            planned_runs=1,
        )
        self.assertFalse(result["complete_extraction"])
        self.assertFalse(result["formal_eligible"])
        self.assertEqual(result["missing_run_indices"], [0])
        self.assertEqual(result["invalid_run_indices"], ["not-an-index"])
        self.assertEqual(result["failed_runs"][0]["error"], "invalid_run_index")


if __name__ == "__main__":
    unittest.main()
