import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

from engine_fault_nlp.graph import build_fault_graph, export_fault_graph
from engine_fault_nlp.schemas import WorkOrder


def chain(component_id, component_name, mode_id, mode_name, cause_text=""):
    return {
        "subsystem_id": "IGNITION",
        "subsystem_name": "点火系统",
        "component_id": component_id,
        "component_name": component_name,
        "failure_mode_id": mode_id,
        "failure_mode_name": mode_name,
        "cause_id": "UNKNOWN",
        "cause_name": "未说明",
        "cause_text": cause_text,
    }


class GraphRankingTest(unittest.TestCase):
    def test_layering_cost_conservation_and_ranking(self):
        orders = [
            WorkOrder("G1", total_cost=100.0),
            WorkOrder("G2", total_cost=300.0),
            WorkOrder("M1", total_cost=500.0),
        ]
        path_a = chain("C1", "点火线圈", "FAILED", "功能失效", "点火线圈质量问题")
        path_b = chain("C2", "火花塞", "WORN", "磨损", "异常磨损")
        predictions = [
            {"work_order_id": "G1", "record_type": "FAULT", "fault_chains": [path_a, path_a]},
            {"work_order_id": "G2", "record_type": "FAULT", "fault_chains": [path_a, path_b]},
            {"work_order_id": "M1", "record_type": "MAINTENANCE", "fault_chains": [path_a]},
        ]
        result = build_fault_graph(orders, predictions)
        self.assertEqual(result.summary["fault_work_orders"], 2)
        self.assertAlmostEqual(result.summary["input_fault_cost"], 400.0)
        self.assertAlmostEqual(result.summary["allocated_fault_cost"], 400.0)
        self.assertAlmostEqual(result.summary["cost_conservation_error"], 0.0)
        self.assertTrue(all(edge.target_layer == edge.source_layer + 1 for edge in result.edges.values()))
        self.assertEqual(result.ranking[0]["component"], "点火线圈")
        self.assertEqual(result.ranking[0]["unique_work_orders"], 2)
        self.assertAlmostEqual(result.ranking[0]["allocated_total_cost"], 250.0)
        self.assertAlmostEqual(result.summary["frequency_weight"] + result.summary["cost_weight"], 1.0)

    def test_exports_graph_files(self):
        orders = [WorkOrder("G1", total_cost=100.0)]
        predictions = [{"work_order_id": "G1", "record_type": "FAULT", "fault_chains": [chain("C1", "点火线圈", "FAILED", "功能失效")]}]
        result = build_fault_graph(orders, predictions)
        with tempfile.TemporaryDirectory() as directory:
            export_fault_graph(result, directory)
            for name in (
                "fault_graph_nodes.csv", "fault_graph_edges.csv", "fault_path_ranking.csv",
                "fault_graph_summary.json", "fault_graph.graphml",
            ):
                self.assertTrue((Path(directory) / name).exists())

    def test_non_failure_fluids_are_excluded(self):
        orders = [WorkOrder("G1", total_cost=200.0)]
        fluid = chain("OIL", "康明斯天然气专用油10W-30-18L", "BROKEN", "断裂")
        actual = chain("C1", "气门弹簧", "BROKEN", "断裂")
        predictions = [{"work_order_id": "G1", "record_type": "FAULT", "fault_chains": [fluid, actual]}]
        result = build_fault_graph(orders, predictions)
        self.assertEqual(result.summary["excluded_non_failure_material_chains"], 1)
        self.assertEqual(len(result.ranking), 1)
        self.assertEqual(result.ranking[0]["component"], "气门弹簧")
        self.assertAlmostEqual(result.ranking[0]["allocated_total_cost"], 200.0)


if __name__ == "__main__":
    unittest.main()
