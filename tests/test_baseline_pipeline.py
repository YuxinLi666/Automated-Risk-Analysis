import tempfile
import unittest
from pathlib import Path

from engine_fault_nlp.baseline import BaselineModel
from engine_fault_nlp.calibration import ThresholdPolicy
from engine_fault_nlp.inference import HybridExtractor
from engine_fault_nlp.ontology import ComponentNode, KnowledgeBase
from engine_fault_nlp.schemas import PartMention, WorkOrder


class BaselinePipelineTest(unittest.TestCase):
    def test_train_save_load_and_hybrid_predict(self):
        component = ComponentNode(
            id="COMP_STARTER",
            name="起动机",
            subsystem_id="START_CHARGE",
            aliases=["启动机"],
            part_numbers=["S1"],
        )
        kb = KnowledgeBase(components={component.id: component})
        documents = ["起动机损坏无法启动", "启动机故障启动困难", "起动机损坏"]
        labels = [[{
            "subsystem_id": "START_CHARGE",
            "component_id": "COMP_STARTER",
            "failure_mode_id": "FUNCTION_FAILED",
            "cause_id": "UNKNOWN",
        }]] * 3
        model = BaselineModel().fit(documents, labels)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.json"
            model.save(str(path))
            loaded = BaselineModel.load(str(path))
            scores = loaded.predict("车辆起动机故障，无法启动")
            self.assertEqual(scores["component_id"][0][0], "COMP_STARTER")
            extractor = HybridExtractor(kb, loaded, ThresholdPolicy())
            order = WorkOrder(
                work_order_id="G3",
                fault_name="无法启动",
                diagnosis="起动机损坏",
                parts=[PartMention("S1", "起动机", True)],
            )
            result = extractor.predict(order)
            self.assertEqual(result.fault_chains[0].component_id, "COMP_STARTER")


if __name__ == "__main__":
    unittest.main()

