"""Scientific invariants for the current four-layer computation."""
from pathlib import Path
import tempfile
import unittest

from engine_fault_nlp.paper_ranking import build_ranking, export_ranking, empirical_percentiles


def chain(component="C1", mode="LEAK"):
    return {"subsystem_id": "S1", "component_id": component, "failure_mode_id": mode}


class PaperRankingTest(unittest.TestCase):
    def test_unique_support_cost_conservation_and_layers(self):
        orders = [{"work_order_id": "a", "total_cost": 100}, {"work_order_id": "b", "total_cost": 300},
                  {"work_order_id": "c", "total_cost": 90}]
        predictions = [{"work_order_id": "a", "fault_chains": [chain(), chain()]},
                       {"document_id": "b", "fault_chains": [chain(), chain("C2")]},
                       {"document_id": "c", "predicted": {**chain(), "failure_mode_id": None}}]
        result = build_ranking(orders, predictions)
        self.assertEqual(result.summary["frequency_denominator"], 3)
        self.assertEqual(result.summary["complete_path_orders"], 2)
        self.assertEqual(result.summary["cost_conservation_error"], 0)
        self.assertEqual(result.summary["allocated_cost_cny"], 400)
        rows = {row["component_id"]: row for row in result.ranking}
        self.assertEqual(rows["C1"]["unique_work_orders"], 2)
        self.assertAlmostEqual(rows["C1"]["frequency_rate"], 2 / 3)
        self.assertEqual(rows["C1"]["economic_burden_cny"], 250)
        self.assertTrue(all(edge.target_layer == edge.source_layer + 1 for edge in result.edges.values()))
        self.assertEqual(set(node.layer for node in result.nodes.values()), {0, 1, 2, 3})

    def test_threshold_and_prediction_schemas(self):
        orders = [{"work_order_id": "a", "total_cost": 10}, {"work_order_id": "b", "total_cost": 20}]
        current = {"document_id": "a", "predicted": chain(), "confidence": {"component_id": 0.49}}
        legacy = {"work_order_id": "b", "classifications": {field: [{"label": label, "confidence": 0.9}]
                                                              for field, label in chain().items()}}
        result = build_ranking(orders, [current, legacy])
        self.assertEqual(result.summary["complete_path_orders"], 1)
        self.assertEqual(result.ranking[0]["economic_burden_cny"], 20)

    def test_tied_ranks_and_empty_cohort(self):
        self.assertEqual(empirical_percentiles([10, 10, 20]), [0.25, 0.25, 1.0])
        self.assertEqual(empirical_percentiles([10]), [1.0])
        self.assertEqual(build_ranking([], []).summary["paths"], 0)

    def test_missing_and_zero_cost(self):
        orders = [{"work_order_id": "a", "total_cost": 0}, {"work_order_id": "b", "total_cost": None}]
        result = build_ranking(orders, [{"document_id": row["work_order_id"], "predicted": chain()} for row in orders])
        self.assertEqual(result.ranking[0]["shrunk_cost_severity_cny"], 0)
        self.assertEqual(result.ranking[0]["known_cost_observations"], 1)
        self.assertEqual(result.ranking[0]["positive_cost_observations"], 0)

    def test_invalid_cost_and_unjoinable_ids(self):
        for cost in (-1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                build_ranking([{"work_order_id": "a", "total_cost": cost}], [])
        with self.assertRaises(ValueError):
            build_ranking([], [], cohort_ids=["missing"])

    def test_export_contains_no_figures(self):
        result = build_ranking([{"work_order_id": "a", "total_cost": 10}], [{"document_id": "a", "predicted": chain()}])
        with tempfile.TemporaryDirectory() as folder:
            export_ranking(result, folder)
            self.assertEqual({path.name for path in Path(folder).iterdir()},
                             {"fault_path_ranking.csv", "fault_graph_nodes.csv", "fault_graph_edges.csv",
                              "summary.json", "four_layer_fault_graph.graphml"})


if __name__ == "__main__":
    unittest.main()
