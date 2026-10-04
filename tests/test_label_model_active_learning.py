import unittest

from engine_fault_nlp.active_learning import select_for_annotation
from engine_fault_nlp.export import flatten_predictions
from engine_fault_nlp.label_model import GenerativeLabelModel
from engine_fault_nlp.schemas import WorkOrder


class LabelModelActiveLearningTest(unittest.TestCase):
    def test_label_model_prefers_agreement(self):
        rows = [
            {"part": "C1", "dtc": "C1", "text": "C1"},
            {"part": "C1", "dtc": "C1", "text": "C2"},
            {"part": "C2", "dtc": "C2", "text": "C2"},
            {"part": "C2", "dtc": "C2", "text": "C1"},
        ]
        model = GenerativeLabelModel().fit(rows)
        prediction = model.predict({"part": "C1", "dtc": "C1", "text": "C2"})
        self.assertEqual(prediction["label"], "C1")

    def test_active_learning_and_csv_flatten(self):
        orders = [WorkOrder("G1", fault_name="启动困难"), WorkOrder("G2", fault_name="动力不足")]
        predictions = [
            {"work_order_id": "G1", "fault_chains": [{"confidence": {"component": 0.99, "subsystem": 0.99, "failure_mode": 0.99, "cause": 0.99}, "decision": "ACCEPT"}]},
            {"work_order_id": "G2", "fault_chains": [{"confidence": {"component": 0.4, "subsystem": 0.8, "failure_mode": 0.3, "cause": 0.2}, "decision": "REVIEW"}]},
        ]
        selected = select_for_annotation(orders, predictions, limit=1)
        self.assertEqual(selected[0].work_order_id, "G2")
        flat = list(flatten_predictions(predictions))
        self.assertEqual(len(flat), 2)


if __name__ == "__main__":
    unittest.main()

