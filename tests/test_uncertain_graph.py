import tempfile
import unittest
from pathlib import Path
from xml.etree import ElementTree as ET

from engine_fault_nlp.ontology import KnowledgeBase
from engine_fault_nlp.schemas import WorkOrder
from engine_fault_nlp.text import stable_id
from engine_fault_nlp.uncertain_graph import build_uncertain_graph, export_uncertain_graph


def path(component_id, component_name, mode_id, mode_name, support=0.8, cause=False):
    return {
        "subsystem_id": "START_CHARGE",
        "subsystem_name": "起动与充电系统",
        "component_id": component_id,
        "component_name": component_name,
        "failure_mode_id": mode_id,
        "failure_mode_name": mode_name,
        "semantic_support": support,
        "relation_supports": {"HAS_COMPONENT": support, "HAS_FAILURE_MODE": support},
        "mean_model_confidence": 0.85,
        "structural_gate": 1.0,
        "decision": "ACCEPT",
        "causes": ([{
            "cause_id": "MATERIAL_FATIGUE",
            "cause_name": "材料疲劳",
            "semantic_support": support,
            "structural_gate": 1.0,
            "decision": "ACCEPT",
            "evidence": ["材料疲劳"],
            "run_indices": [0, 1, 2],
        }] if cause else []),
        "effects": [],
        "evidence": ["起动机损坏"],
    }


class UncertainGraphTest(unittest.TestCase):
    def test_record_type_guard_cost_conservation_and_reported_cause(self):
        orders = [
            WorkOrder("F1", fault_name="起动机损坏", diagnosis="材料疲劳导致", total_cost=100.0),
            WorkOrder("F2", fault_name="起动机损坏", diagnosis="材料疲劳导致", total_cost=300.0),
            WorkOrder("M1", fault_name="主动升级", solution="数据刷写", total_cost=500.0),
        ]
        starter = path("STARTER", "起动机", "FUNCTION_FAILED", "功能失效/损坏", cause=True)
        candidates = [
            {"work_order_id": "F1", "record_type": "FAULT", "support_source": "independent_run_frequency", "canonical_paths": [starter]},
            {"work_order_id": "F2", "record_type": "FAULT", "support_source": "independent_run_frequency", "canonical_paths": [starter]},
            # Deliberately forged upstream type: the graph must recompute it.
            {"work_order_id": "M1", "record_type": "FAULT", "support_source": "independent_run_frequency", "canonical_paths": [starter]},
        ]
        result = build_uncertain_graph(orders, candidates, kb=KnowledgeBase())
        self.assertEqual(result.summary["classified_fault_work_orders"], 2)
        self.assertEqual(result.summary["record_type_mismatches"], 1)
        self.assertAlmostEqual(result.summary["classified_fault_total_cost"], 400.0)
        self.assertAlmostEqual(result.summary["represented_input_cost"], 400.0)
        self.assertAlmostEqual(result.summary["allocated_fault_cost"], 400.0)
        self.assertAlmostEqual(result.summary["represented_cost_conservation_error"], 0.0)
        cause_edge = next(item for item in result.edges if item["relation"] == "CAUSES_REPORTED")
        self.assertEqual(cause_edge["epistemic_status"], "reported")
        self.assertEqual(cause_edge["context_component_id"], "STARTER")

    def test_two_paths_split_one_order_cost(self):
        orders = [WorkOrder("F1", fault_name="部件损坏", total_cost=200.0)]
        candidates = [{
            "work_order_id": "F1",
            "record_type": "FAULT",
            "support_source": "independent_run_frequency",
            "canonical_paths": [
                path("STARTER", "起动机", "FUNCTION_FAILED", "功能失效/损坏"),
                path("GENERATOR", "发电机", "BROKEN", "断裂"),
            ],
        }]
        result = build_uncertain_graph(orders, candidates)
        self.assertEqual(len(result.path_ranking), 2)
        self.assertAlmostEqual(sum(item["allocated_total_cost"] for item in result.path_ranking), 200.0)

    def test_higher_semantic_support_increases_evidence_score(self):
        orders = [WorkOrder("F1", fault_name="起动机损坏", total_cost=100.0)]
        def build(support):
            return build_uncertain_graph(orders, [{
                "work_order_id": "F1",
                "record_type": "FAULT",
                "support_source": "independent_run_frequency",
                "canonical_paths": [path("STARTER", "起动机", "FUNCTION_FAILED", "功能失效/损坏", support)],
            }], minimum_path_support=0.0)
        low, high = build(0.4), build(0.9)
        low_score = next(item for item in low.edges if item["relation"] == "HAS_FAILURE_MODE")["evidence_confidence"]
        high_score = next(item for item in high.edges if item["relation"] == "HAS_FAILURE_MODE")["evidence_confidence"]
        self.assertGreater(high_score, low_score)

    def test_exports(self):
        orders = [WorkOrder("F1", fault_name="起动机损坏", total_cost=100.0)]
        candidates = [{
            "work_order_id": "F1", "record_type": "FAULT",
            "support_source": "independent_run_frequency",
            "canonical_paths": [path("STARTER", "起动机", "FUNCTION_FAILED", "功能失效/损坏")],
        }]
        result = build_uncertain_graph(orders, candidates)
        with tempfile.TemporaryDirectory() as directory:
            export_uncertain_graph(result, directory)
            for name in (
                "uncertain_graph_nodes.csv", "uncertain_graph_edges.csv",
                "fault_path_risk_ranking.csv", "node_association_importance.csv",
                "uncertain_graph_summary.json", "uncertain_graph.json", "uncertain_graph.graphml",
            ):
                self.assertTrue((Path(directory) / name).exists())
            root = ET.parse(Path(directory) / "uncertain_graph.graphml").getroot()
            namespace = {"g": "http://graphml.graphdrawing.org/xmlns"}
            self.assertGreaterEqual(len(root.findall("g:key", namespace)), 7)
            self.assertTrue(root.findall(".//g:data", namespace))

    def test_zero_cost_does_not_receive_severity_or_burden_credit(self):
        orders = [WorkOrder("F1", fault_name="起动机损坏", total_cost=0.0)]
        candidates = [{
            "work_order_id": "F1", "record_type": "FAULT",
            "support_source": "independent_run_frequency",
            "canonical_paths": [path("STARTER", "起动机", "FUNCTION_FAILED", "功能失效/损坏")],
        }]
        result = build_uncertain_graph(orders, candidates)
        ranked = result.path_ranking[0]
        self.assertEqual(ranked["severity_percentile"], 0.0)
        self.assertEqual(ranked["burden_percentile"], 0.0)
        self.assertEqual(ranked["association_risk_score"], 0.0)

    def test_branch_requires_own_traceable_evidence(self):
        orders = [WorkOrder("F1", fault_name="起动机损坏", total_cost=100.0)]
        candidate_path = path(
            "STARTER", "起动机", "FUNCTION_FAILED", "功能失效/损坏", cause=True
        )
        candidates = [{
            "work_order_id": "F1", "record_type": "FAULT",
            "support_source": "planned_independent_run_frequency",
            "formal_eligible": True,
            "canonical_paths": [candidate_path],
        }]
        result = build_uncertain_graph(orders, candidates, kb=KnowledgeBase())
        self.assertFalse(any(item["relation"] == "CAUSES_REPORTED" for item in result.edges))
        self.assertEqual(result.summary["excluded_unverified_branches"], 1)

    def test_effect_wilson_denominator_is_component_context_specific(self):
        orders = [
            WorkOrder("F1", fault_name="部件损坏", fault_description="动力不足"),
            WorkOrder("F2", fault_name="部件损坏"),
        ]
        path_a = path("A", "部件A", "FUNCTION_FAILED", "功能失效/损坏")
        path_b = path("B", "部件B", "FUNCTION_FAILED", "功能失效/损坏")
        path_a["effects"] = [{
            "effect_id": stable_id("EFFECT", "动力不足"),
            "effect_name": "动力不足",
            "semantic_support": 0.8,
            "structural_gate": 1.0,
            "decision": "ACCEPT",
            "evidence": ["动力不足"],
            "run_indices": [0, 1, 2, 3],
        }]
        candidates = [
            {"work_order_id": "F1", "record_type": "FAULT", "canonical_paths": [path_a]},
            {"work_order_id": "F2", "record_type": "FAULT", "canonical_paths": [path_b]},
        ]
        result = build_uncertain_graph(orders, candidates, kb=KnowledgeBase())
        edge = next(item for item in result.edges if item["relation"] == "MANIFESTS_AS")
        self.assertEqual(edge["supporting_work_orders"], 1)
        self.assertEqual(edge["head_work_orders"], 1)
        self.assertEqual(edge["conditional_frequency"], 1.0)

    def test_cost_summary_separates_unrepresented_fault_orders(self):
        orders = [
            WorkOrder("F1", fault_name="起动机损坏", total_cost=100.0),
            WorkOrder("F2", fault_name="水泵损坏", total_cost=50.0),
        ]
        candidates = [{
            "work_order_id": "F1", "record_type": "FAULT",
            "canonical_paths": [path("STARTER", "起动机", "FUNCTION_FAILED", "功能失效/损坏")],
        }]
        result = build_uncertain_graph(orders, candidates)
        self.assertEqual(result.summary["classified_fault_total_cost"], 150.0)
        self.assertEqual(result.summary["represented_input_cost"], 100.0)
        self.assertEqual(result.summary["excluded_unrepresented_fault_cost"], 50.0)
        self.assertEqual(result.summary["represented_cost_conservation_error"], 0.0)

    def test_traceable_quote_must_semantically_support_cause_id(self):
        orders = [WorkOrder("F1", fault_name="起动机损坏", diagnosis="起动机损坏")]
        candidate_path = path(
            "STARTER", "起动机", "FUNCTION_FAILED", "功能失效/损坏", cause=True
        )
        # One quote matches the cause semantics and a different quote is
        # traceable.  The branch must require the same quote to satisfy both.
        candidate_path["causes"][0]["evidence"] = ["材料疲劳", "起动机损坏"]
        candidates = [{
            "work_order_id": "F1",
            "record_type": "FAULT",
            "support_source": "single_model_confidence",
            "canonical_paths": [candidate_path],
        }]
        result = build_uncertain_graph(orders, candidates, kb=KnowledgeBase())
        self.assertFalse(any(item["relation"] == "CAUSES_REPORTED" for item in result.edges))
        self.assertEqual(result.summary["excluded_semantically_invalid_branches"], 1)

    def test_traceable_quote_must_semantically_support_effect_id(self):
        orders = [WorkOrder("F1", fault_name="部件损坏")]
        candidate_path = path("A", "部件A", "FUNCTION_FAILED", "功能失效/损坏")
        candidate_path["effects"] = [{
            "effect_id": stable_id("EFFECT", "动力不足"),
            "effect_name": "动力不足",
            "semantic_support": 0.8,
            "structural_gate": 1.0,
            "decision": "ACCEPT",
            "evidence": ["动力不足", "部件损坏"],
            "run_indices": [0, 1, 2, 3],
        }]
        candidates = [{
            "work_order_id": "F1",
            "record_type": "FAULT",
            "support_source": "single_model_confidence",
            "canonical_paths": [candidate_path],
        }]
        result = build_uncertain_graph(orders, candidates, kb=KnowledgeBase())
        self.assertFalse(any(item["relation"] == "MANIFESTS_AS" for item in result.edges))
        self.assertEqual(result.summary["excluded_semantically_invalid_branches"], 1)

    def test_non_finite_configuration_is_rejected(self):
        with self.assertRaises(ValueError):
            build_uncertain_graph([], [], alpha=float("nan"))
        with self.assertRaises(ValueError):
            build_uncertain_graph([], [], risk_weights=(0.5, float("nan"), 0.5))

    def test_incomplete_planned_consensus_is_not_graphed(self):
        orders = [WorkOrder("F1", fault_name="起动机损坏")]
        candidates = [{
            "work_order_id": "F1",
            "record_type": "FAULT",
            "support_source": "planned_independent_run_frequency",
            "formal_eligible": False,
            "canonical_paths": [
                path("STARTER", "起动机", "FUNCTION_FAILED", "功能失效/损坏")
            ],
        }]
        result = build_uncertain_graph(orders, candidates)
        self.assertEqual(result.path_ranking, [])
        self.assertEqual(result.summary["excluded_incomplete_consensus_orders"], 1)


if __name__ == "__main__":
    unittest.main()
