"""Offline synthetic TF-IDF smoke demo, NOT a MacBERT experiment."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from engine_fault_nlp.baseline import BaselineModel
from engine_fault_nlp.io_utils import write_csv, write_json, write_jsonl
from engine_fault_nlp.schemas import PartMention, WorkOrder


def make_samples():
    specs = [
        ("START_CHARGE", "COMP_STARTER", "起动机", "FUNCTION_FAILED", "无法启动", "检查起动机失效，更换后启动正常"),
        ("COOLING", "COMP_WATER_PUMP", "水泵", "LEAK", "冷却液泄漏", "检查水泵漏水，更换水泵后恢复正常"),
        ("IGNITION", "COMP_IGNITION_COIL", "点火线圈", "FUNCTION_FAILED", "点火异常", "检查点火线圈失效，更换后恢复正常"),
    ]
    rows = []
    for index, (subsystem, component, name, mode, fault, diagnosis) in enumerate(specs):
        for repeat in range(4):
            split = "test" if repeat == 3 else "train"
            order = WorkOrder(
                work_order_id=f"DEMO-{index + 1}-{repeat + 1}",
                fault_name=fault, diagnosis=diagnosis,
                fault_description=("模拟行驶后出现故障" if split == "test" else "模拟维修工况"),
                total_cost=float(100 + 50 * index + repeat),
                parts=[PartMention(f"DEMO-P{index + 1}", name, True)],
            )
            text = order.model_text()
            start = text.index(name)
            rows.append((order, {
                "document_id": order.work_order_id, "text": text, "split": split,
                "entities": [{"entity_id": "e1", "label": "COMPONENT", "start": start,
                              "end": start + len(name), "text": name, "canonical_id": component}],
                "relations": [],
                "chains": [{"subsystem_id": subsystem, "component_id": component,
                            "failure_mode_id": mode}],
            }))
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "demo")
    args = parser.parse_args(argv)
    out = args.output_dir.resolve()
    rows = make_samples()
    train = [annotation for _, annotation in rows if annotation["split"] == "train"]
    test = [annotation for _, annotation in rows if annotation["split"] == "test"]
    model = BaselineModel().fit([row["text"] for row in train], [row["chains"] for row in train])
    model.save(str(out / "tfidf_demo_model.json"))
    model = BaselineModel.load(str(out / "tfidf_demo_model.json"))
    predictions = []
    fields = ("subsystem_id", "component_id", "failure_mode_id")
    for row in test:
        scores = model.predict(row["text"])
        prediction = {"document_id": row["document_id"], "engine_root": "E0"}
        for field in fields:
            prediction[field] = scores[field][0][0] if scores[field] else None
        predictions.append(prediction)
    write_jsonl(str(out / "work_orders.jsonl"), [order.to_dict() for order, _ in rows])
    write_jsonl(str(out / "train_annotations.jsonl"), train)
    write_jsonl(str(out / "test_annotations.jsonl"), test)
    write_jsonl(str(out / "predictions.jsonl"), predictions)
    write_csv(str(out / "predictions.csv"), ["document_id", "engine_root", *fields], predictions)
    write_json(str(out / "summary.json"), {
        "data_source": "synthetic", "model": "character TF-IDF centroid (demo only)",
        "training_documents": len(train), "test_documents": len(test),
        "prediction_documents": len(predictions), "is_paper_result": False,
        "cost_used_as_model_input": False,
    })
    print(f"Synthetic demo complete: {out}")


if __name__ == "__main__":
    main()
