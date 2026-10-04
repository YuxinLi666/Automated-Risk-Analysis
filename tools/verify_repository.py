"""Check release contents and data integrity without printing private values."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
DIRECTORIES = {"src", "scripts", "tests", "tools", "data", "docs", "config", "examples", ".github"}
TOP_FILES = {"README.md", "LICENSE", "CONTRIBUTING.md", "DATA_AVAILABILITY.md", "pyproject.toml",
             "requirements.txt", "requirements-ml.txt", "run.py", ".gitignore", ".gitattributes"}
IGNORED_PARTS = {"__pycache__", ".pytest_cache", ".ruff_cache", ".git"}
ALLOWED_SUFFIXES = {".py", ".md", ".json", ".jsonl", ".csv", ".toml", ".txt", ".yml", ".yaml"}
SECRET_PATTERNS = (
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{30,}\b"),
    re.compile(r"\bgithub_pat_[A-Za-z0-9_]{30,}\b"),
    re.compile(r"\bsk-[A-Za-z0-9_-]{30,}\b"),
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
)


def release_files():
    for path in sorted(ROOT.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(ROOT)
        if any(part in IGNORED_PARTS or part.endswith(".egg-info") for part in relative.parts):
            continue
        if len(relative.parts) == 1 and relative.name in TOP_FILES:
            yield path
        elif relative.parts[0] in DIRECTORIES:
            yield path


def rows(path):
    with path.open(encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def audit():
    errors = []
    files = list(release_files())
    for path in files:
        relative = path.relative_to(ROOT).as_posix()
        if path.suffix not in ALLOWED_SUFFIXES and path.name not in TOP_FILES:
            errors.append(f"Unexpected release file: {relative}")
        if path.stat().st_size >= 50 * 1024 * 1024:
            errors.append(f"Oversized release file: {relative}")
        try:
            text = path.read_text(encoding="utf-8-sig")
        except UnicodeDecodeError:
            errors.append(f"Non-text release file: {relative}")
            continue
        if any(pattern.search(text) for pattern in SECRET_PATTERNS):
            errors.append(f"Possible credential material in {relative}; inspect locally, do not upload.")
        if path.suffix == ".py" and re.search(r"(?m)^\s*(?:from|import)\s+(?:matplotlib|seaborn|plotly|graphviz)(?:\.|\s|$)", text):
            errors.append(f"Plotting import found in {relative}")

    manifest_path = ROOT / "docs/data_manifest.json"
    if not manifest_path.is_file():
        errors.append("Missing docs/data_manifest.json")
        return errors, {"release_files": len(files)}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    entries = manifest["files"]
    approved = {entry["path"] for entry in entries}
    for entry in entries:
        relative = Path(entry["path"])
        path = (ROOT / relative).resolve()
        if not path.is_relative_to(ROOT) or not path.is_file():
            errors.append(f"Missing or invalid data path: {relative.as_posix()}")
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest.lower() != entry["sha256"].lower() or path.stat().st_size != entry["bytes"]:
            errors.append(f"Research-input bytes changed: {relative.as_posix()}")
    for path in files:
        relative = path.relative_to(ROOT).as_posix()
        if relative.startswith("data/") and relative != "data/README.md" and relative not in approved:
            errors.append(f"Unlisted research input: {relative}")

    work_orders = rows(ROOT / "data/work_orders.jsonl")
    order_ids = [str(row["work_order_id"]) for row in work_orders]
    if len(order_ids) != len(set(order_ids)):
        errors.append("Duplicate work-order identifiers in the main input.")
    split_sets = {}
    expected = {"train": 800, "validation": 100, "test": 100}
    for split, count in expected.items():
        records = rows(ROOT / f"data/seed/{split}.jsonl")
        ids = [str(row["document_id"]) for row in records]
        split_sets[split] = set(ids)
        if len(ids) != count or len(set(ids)) != count:
            errors.append(f"Unexpected or duplicate {split} records.")
    if any(split_sets[a] & split_sets[b] for a, b in (("train", "validation"), ("train", "test"), ("validation", "test"))):
        errors.append("Document overlap between development splits.")
    unseen = rows(ROOT / "data/unseen/labels.jsonl")
    unseen_ids = {str(row["work_order_id"]) for row in unseen}
    seed_ids = set().union(*split_sets.values())
    if len(unseen) != 727 or len(unseen_ids) != 727 or unseen_ids & seed_ids:
        errors.append("Unexpected unseen-cohort size or seed overlap.")
    if not (unseen_ids | seed_ids) <= set(order_ids):
        errors.append("Some supervised records cannot be joined to the work orders.")
    summary = {
        "release_files": len(files),
        "release_bytes": sum(path.stat().st_size for path in files),
        "preserved_data_files": len(entries),
        "work_orders": len(work_orders),
        "seed_splits": {key: len(value) for key, value in split_sets.items()},
        "unseen_labels": len(unseen),
        "input_hashes_verified": not any("bytes changed" in error for error in errors),
        "data_visibility_requirement": "private; records are not anonymized",
        "credential_scan_scope": "common token/private-key patterns only, not a security certification",
    }
    return errors, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", type=Path, help="Create a PRIVATE release archive after a successful audit.")
    args = parser.parse_args()
    errors, summary = audit()
    print(json.dumps({"status": "FAIL" if errors else "PASS", **summary, "errors": errors}, indent=2))
    if errors:
        return 1
    if args.zip:
        archive = args.zip.resolve()
        if archive.is_relative_to(ROOT):
            raise SystemExit("Keep release archives outside the repository.")
        if archive.exists():
            raise SystemExit("Archive already exists; choose a new path.")
        with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as output:
            for path in release_files():
                output.write(path, Path(ROOT.name) / path.relative_to(ROOT))
        print("Private archive created successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
