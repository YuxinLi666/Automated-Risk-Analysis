from __future__ import annotations

import argparse
import json
from pathlib import Path

from engine_fault_nlp.evaluation import evaluate_records
from engine_fault_nlp.io_utils import write_json, write_jsonl


FIELDS = ("subsystem_id", "component_id", "failure_mode_id", "cause_id")


def read_jsonl(path: str):
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def transformer_row_to_fault_chain(row: dict) -> dict:
    classifications = row.get("classifications", {})
    chain = {"confidence": {}, "sources": ["transformer_classification_head"]}
    for field in FIELDS:
        ranked = classifications.get(field) or []
        if ranked:
            best = ranked[0]
            chain[field] = best.get("label")
            chain["confidence"][field.replace("_id", "")] = best.get("confidence", 0.0)
    return {
        "work_order_id": row["work_order_id"],
        "fault_chains": [chain] if any(chain.get(field) for field in FIELDS) else [],
        "model_version": "transformer-classification-head-top1",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--silver-labels", required=True)
    parser.add_argument("--record-type", default="FAULT")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    silver = {
        row["work_order_id"]: row
        for row in read_jsonl(args.silver_labels)
        if row.get("record_type") == args.record_type
    }
    transformed = []
    gold = []
    for row in read_jsonl(args.predictions):
        order_id = row.get("work_order_id")
        if order_id not in silver:
            continue
        transformed.append(transformer_row_to_fault_chain(row))
        gold.append(
            {
                "work_order_id": order_id,
                "fault_chains": silver[order_id].get("fault_chains", []),
            }
        )

    metrics = evaluate_records(transformed, gold)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(str(output), metrics)
    write_jsonl(str(output.with_suffix(".predictions.jsonl")), transformed)
    write_jsonl(str(output.with_suffix(".silver_gold.jsonl")), gold)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

