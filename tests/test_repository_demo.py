import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("demo_pipeline", ROOT / "scripts/demo_pipeline.py")
DEMO = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(DEMO)


class RepositoryDemoTest(unittest.TestCase):
    def test_annotations_offsets_and_no_cost_input(self):
        rows = DEMO.make_samples()
        self.assertEqual(len({order.work_order_id for order, _ in rows}), 12)
        for order, annotation in rows:
            for entity in annotation["entities"]:
                self.assertEqual(annotation["text"][entity["start"]:entity["end"]], entity["text"])
            original_text = order.model_text()
            order.total_cost = 999999999.99
            self.assertEqual(order.model_text(), original_text)

    def test_offline_demo_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            DEMO.main(["--output-dir", directory])
            out = Path(directory)
            summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
            self.assertFalse(summary["is_paper_result"])
            self.assertEqual(summary["prediction_documents"], 3)
            train = [json.loads(x) for x in (out / "train_annotations.jsonl").read_text(encoding="utf-8").splitlines()]
            test = [json.loads(x) for x in (out / "test_annotations.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertFalse({r["document_id"] for r in train} & {r["document_id"] for r in test})
            self.assertTrue((out / "predictions.csv").is_file())


if __name__ == "__main__":
    unittest.main()
