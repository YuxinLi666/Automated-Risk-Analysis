from __future__ import annotations

import argparse
import json
from pathlib import Path


def read_jsonl(path: str):
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path: str, rows) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def mutate(row: dict, variant: str) -> dict:
    row = dict(row)
    if variant == "full":
        return row
    if variant == "no_parts":
        row["parts"] = []
        return row
    if variant == "no_dtc":
        row["dtc_raw"] = ""
        row["dtc_codes"] = []
        return row
    if variant == "no_diagnosis_solution":
        row["diagnosis"] = ""
        row["solution"] = ""
        return row
    if variant == "complaint_only":
        row["diagnosis"] = ""
        row["solution"] = ""
        row["dtc_raw"] = ""
        row["dtc_codes"] = []
        row["parts"] = []
        return row
    raise ValueError(f"Unknown variant: {variant}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    rows = list(read_jsonl(args.records))
    variants = ["full", "no_parts", "no_dtc", "no_diagnosis_solution", "complaint_only"]
    summary = {}
    for variant in variants:
        output = Path(args.output_dir) / f"work_orders_{variant}.jsonl"
        write_jsonl(str(output), (mutate(row, variant) for row in rows))
        summary[variant] = str(output)
    print(json.dumps({"rows": len(rows), "outputs": summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
