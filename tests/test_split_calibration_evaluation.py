import unittest

from engine_fault_nlp.calibration import select_threshold, wilson_lower_bound
from engine_fault_nlp.evaluation import evaluate_records
from engine_fault_nlp.schemas import WorkOrder
from engine_fault_nlp.split import grouped_split, grouping_audit


class SplitCalibrationEvaluationTest(unittest.TestCase):
    def test_grouped_split_has_no_chassis_template_leakage(self):
        orders = [
            WorkOrder("G{}".format(i), chassis_id="C{}".format(i // 2), fault_name="传感器故障 {}".format(i // 2))
            for i in range(20)
        ]
        splits = grouped_split(orders, seed=7)
        locations = {}
        for split_name, values in splits.items():
            for order in values:
                locations.setdefault(order.chassis_id, set()).add(split_name)
        self.assertTrue(all(len(value) == 1 for value in locations.values()))

    def test_identity_and_template_constraints_are_independent(self):
        orders = [
            WorkOrder("A", chassis_id="SAME", engine_id="E1", fault_name="起动机损坏"),
            WorkOrder("B", chassis_id="SAME", engine_id="E2", fault_name="水泵泄漏"),
            WorkOrder("C", chassis_id="C3", engine_id="E3", fault_name="线束损坏"),
            WorkOrder("D", chassis_id="C4", engine_id="E4", fault_name="线束损坏"),
        ]
        splits = grouped_split(orders, train_ratio=0.5, validation_ratio=0.25, seed=3)
        locations = {
            order.work_order_id: split_name
            for split_name, values in splits.items()
            for order in values
        }
        self.assertEqual(locations["A"], locations["B"])
        self.assertEqual(locations["C"], locations["D"])
        audit = grouping_audit(orders)
        self.assertEqual(audit["largest_connected_group"], 2)
        self.assertEqual(audit["connected_groups"], 2)

    def test_wilson_and_threshold(self):
        self.assertLess(wilson_lower_bound(95, 100), 0.95)
        scored = [(0.99 - i * 0.001, True) for i in range(100)] + [(0.5, False)]
        selected = select_threshold(scored, 0.94, minimum_accepted=80)
        self.assertGreaterEqual(selected["accepted"], 80)
        self.assertGreaterEqual(selected["precision_lower_bound"], 0.94)

    def test_evaluation(self):
        gold = [{"work_order_id": "G1", "fault_chains": [{"component_id": "C1"}]}]
        predictions = [{"work_order_id": "G1", "fault_chains": [{"component_id": "C1"}]}]
        metrics = evaluate_records(predictions, gold)
        self.assertEqual(metrics["component_id"]["f1"], 1.0)
        self.assertEqual(metrics["chain_exact_match"], 1.0)


if __name__ == "__main__":
    unittest.main()
