from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Dict, List

from .annotation import ANNOTATION_GUIDE, export_annotation_documents
from .active_learning import select_for_annotation
from .baseline import BaselineModel
from .calibration import ThresholdPolicy, select_threshold
from .consensus import build_consensus_records, predictions_to_candidate_records
from .evaluation import evaluate_records
from .export import flatten_predictions
from .ingest import audit_workbook, load_work_orders
from .inference import HybridExtractor
from .graph import build_fault_graph, export_fault_graph
from .io_utils import read_jsonl, write_csv, write_json, write_jsonl
from .label_model import fuse_vote_records
from .ontology import KnowledgeBase, build_knowledge_base_from_workbook, merge_knowledge_bases
from .schemas import WorkOrder
from .split import grouped_split, grouping_audit
from .teacher import OpenAICompatibleTeacher
from .teacher import PROMPT_VERSION
from .uncertain_graph import build_uncertain_graph, export_uncertain_graph
from .weak_supervision import WeakSupervisor


def _load_orders(path: str) -> List[WorkOrder]:
    return [WorkOrder.from_dict(row) for row in read_jsonl(path)]


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def cmd_audit(args):
    print(json.dumps(audit_workbook(args.input, args.sheet), ensure_ascii=False, indent=2))


def cmd_prepare(args):
    orders = load_work_orders(args.input, args.sheet)
    write_jsonl(args.output, (order.to_dict() for order in orders))
    audit = audit_workbook(args.input, args.sheet)
    manifest_path = str(Path(args.output).with_name(Path(args.output).name + ".manifest.json"))
    write_json(
        manifest_path,
        {
            "source_path": str(Path(args.input).resolve()),
            "source_sha256": _sha256(args.input),
            "sheet": args.sheet,
            **audit,
            "output_records": len(orders),
        },
    )
    print("prepared {} unique work orders -> {}".format(len(orders), args.output))
    print("data manifest -> {}".format(manifest_path))


def cmd_build_kb(args):
    kb = build_knowledge_base_from_workbook(args.input, args.sheet)
    if args.base_kb:
        kb = merge_knowledge_bases(KnowledgeBase.load(args.base_kb), kb)
    kb.save(args.output)
    print("knowledge base: {} components, {} dtc -> {}".format(len(kb.components), len(kb.dtc), args.output))


def cmd_split(args):
    orders = _load_orders(args.records)
    splits = grouped_split(orders, args.train_ratio, args.validation_ratio, args.seed)
    output = Path(args.output_dir)
    for name, values in splits.items():
        write_jsonl(str(output / (name + ".jsonl")), (order.to_dict() for order in values))
        print("{}: {}".format(name, len(values)))
    audit = grouping_audit(orders)
    write_json(
        str(output / "split_manifest.json"),
        {
            "records_sha256": _sha256(args.records),
            "seed": args.seed,
            "requested_ratios": {
                "train": args.train_ratio,
                "validation": args.validation_ratio,
                "test": 1.0 - args.train_ratio - args.validation_ratio,
            },
            "actual_counts": {name: len(values) for name, values in splits.items()},
            "actual_ratios": {
                name: len(values) / len(orders) if orders else 0.0
                for name, values in splits.items()
            },
            **audit,
        },
    )
    print("split manifest -> {}".format(output / "split_manifest.json"))


def cmd_export_annotation(args):
    orders = _load_orders(args.records)
    write_jsonl(args.output, export_annotation_documents(orders))
    guide = str(Path(args.output).with_suffix(".guide.json"))
    write_json(guide, ANNOTATION_GUIDE)
    print("annotation tasks: {} -> {}".format(len(orders), args.output))


def cmd_weak_label(args):
    kb = KnowledgeBase.load(args.kb)
    supervisor = WeakSupervisor(kb)
    orders = _load_orders(args.records)
    write_jsonl(args.output, (supervisor.label(order).to_dict() for order in orders))
    print("weak labels: {} -> {}".format(len(orders), args.output))


def cmd_weak_votes(args):
    kb = KnowledgeBase.load(args.kb)
    supervisor = WeakSupervisor(kb)
    rows = []
    for order in _load_orders(args.records):
        result = supervisor.label(order)
        votes = {field: {} for field in ("subsystem_id", "component_id", "failure_mode_id", "cause_id")}
        for chain in result.fault_chains:
            source = "+".join(chain.sources) or "weak_supervisor"
            for field in votes:
                value = getattr(chain, field)
                if value:
                    votes[field][source] = value
        rows.extend({"work_order_id": order.work_order_id, "field": field, "votes": field_votes} for field, field_votes in votes.items())
    write_jsonl(args.output, rows)
    print("weak vote records: {} -> {}".format(len(rows), args.output))


def cmd_fit_label_model(args):
    rows = list(read_jsonl(args.votes))
    output = fuse_vote_records(rows)
    write_jsonl(args.output, output)
    print("fused weak votes: {} -> {}".format(len(output), args.output))


def cmd_train_baseline(args):
    orders = _load_orders(args.records)
    label_map = {row["work_order_id"]: row.get("fault_chains", row.get("chains", [])) for row in read_jsonl(args.labels)}
    documents, labels = [], []
    for order in orders:
        if order.work_order_id in label_map:
            documents.append(order.model_text())
            labels.append(label_map[order.work_order_id])
    if not documents:
        raise SystemExit("No matching labels found")
    model = BaselineModel().fit(documents, labels)
    model.save(args.output)
    print("trained baseline on {} documents -> {}".format(len(documents), args.output))


def cmd_predict(args):
    kb = KnowledgeBase.load(args.kb)
    model = BaselineModel.load(args.model) if args.model else None
    policy = ThresholdPolicy.load(args.policy) if args.policy else ThresholdPolicy()
    extractor = HybridExtractor(kb, model, policy)
    orders = load_work_orders(args.input, args.sheet) if args.input.lower().endswith(".xlsx") else _load_orders(args.input)
    write_jsonl(args.output, (extractor.predict(order).to_dict() for order in orders))
    print("predicted {} work orders -> {}".format(len(orders), args.output))


def cmd_export_csv(args):
    fieldnames = [
        "工单号", "故障链序号", "工单类型", "子系统ID", "子系统", "部件ID", "零部件", "故障模式ID", "故障模式",
        "原因ID", "失效原因", "故障原因原文", "原因抽取方法", "根因分类ID", "根因分类",
        "症状", "故障码", "部件置信度", "模式置信度", "原因置信度", "原因文本置信度",
        "故障链置信度", "决策", "证据", "警告",
    ]
    write_csv(args.output, fieldnames, flatten_predictions(read_jsonl(args.predictions)))
    print("CSV export -> {}".format(args.output))


def cmd_active_learn(args):
    orders = _load_orders(args.records)
    selected = select_for_annotation(orders, read_jsonl(args.predictions), args.limit)
    write_jsonl(args.output, export_annotation_documents(selected))
    print("active-learning tasks: {} -> {}".format(len(selected), args.output))


def cmd_build_graph_rank(args):
    print("WARNING: build-graph-rank is the legacy five-layer aggregator; use build-uncertain-graph for the Word method.")
    orders = _load_orders(args.records)
    result = build_fault_graph(
        orders,
        read_jsonl(args.predictions),
        engine_name=args.engine_name,
        shrinkage_strength=args.shrinkage_strength,
        entropy_strength=args.entropy_strength,
    )
    export_fault_graph(result, args.output_dir)
    print(json.dumps(result.summary, ensure_ascii=False, indent=2))


def cmd_evaluate(args):
    metrics = evaluate_records(read_jsonl(args.predictions), read_jsonl(args.gold))
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    if args.output:
        write_json(args.output, metrics)


def cmd_calibrate(args):
    predictions = {row["work_order_id"]: row for row in read_jsonl(args.predictions)}
    gold = {str(row.get("work_order_id", row.get("document_id"))): row for row in read_jsonl(args.gold)}
    field_map = {"subsystem": "subsystem_id", "component": "component_id", "failure_mode": "failure_mode_id", "cause": "cause_id"}
    thresholds = {}
    details = {}
    for short, field in field_map.items():
        scored = []
        for order_id, gold_row in gold.items():
            pred_chains = predictions.get(order_id, {}).get("fault_chains", [])
            gold_chains = gold_row.get("fault_chains", gold_row.get("chains", []))
            expected = {chain.get(field) for chain in gold_chains if chain.get(field)}
            for chain in pred_chains:
                if chain.get(field):
                    scored.append((float(chain.get("confidence", {}).get(short, 0.0)), chain.get(field) in expected))
        selected = select_threshold(scored, args.target_precision, args.minimum_accepted)
        thresholds[short] = selected["threshold"]
        details[short] = selected
    thresholds["chain"] = max(thresholds.get("component", 1.0), thresholds.get("failure_mode", 1.0))
    ThresholdPolicy(thresholds).save(args.output)
    print(json.dumps(details, ensure_ascii=False, indent=2))


def cmd_teacher_label(args):
    kb = KnowledgeBase.load(args.kb)
    teacher = OpenAICompatibleTeacher(args.endpoint, args.model, args.api_key)
    orders = _load_orders(args.records)
    rows = []
    for order in orders:
        result = teacher.label(order, kb)
        result.setdefault("work_order_id", order.work_order_id)
        rows.append(result)
    write_jsonl(args.output, rows)
    print("teacher labels: {} -> {}".format(len(rows), args.output))


def cmd_repeat_teacher(args):
    """Run independent teacher calls and retain every successful/failed slot."""

    if args.runs < 1:
        raise SystemExit("--runs must be at least 1")
    if args.retries < 0:
        raise SystemExit("--retries cannot be negative")
    if not math.isfinite(args.minimum_support) or not 0.0 <= args.minimum_support <= 1.0:
        raise SystemExit("--minimum-support must be between 0 and 1")
    if (
        not math.isfinite(args.minimum_component_score)
        or not 0.0 <= args.minimum_component_score <= 1.0
    ):
        raise SystemExit("--minimum-component-score must be between 0 and 1")
    if not math.isfinite(args.temperature) or not 0.0 <= args.temperature <= 2.0:
        raise SystemExit("--temperature must be between 0 and 2")
    if args.timeout <= 0:
        raise SystemExit("--timeout must be positive")
    if args.limit is not None and args.limit < 0:
        raise SystemExit("--limit cannot be negative")
    kb = KnowledgeBase.load(args.kb)
    teacher = OpenAICompatibleTeacher(args.endpoint, args.model, args.api_key, args.timeout)
    orders = _load_orders(args.records)
    if args.limit is not None:
        orders = orders[: args.limit]
    rows = []
    for order_position, order in enumerate(orders):
        for run_index in range(args.runs):
            failures = []
            completed = None
            for attempt in range(args.retries + 1):
                seed = args.seed + order_position * 100003 + run_index * 101 + attempt
                try:
                    extraction = teacher.label(
                        order, kb, temperature=args.temperature, seed=seed
                    )
                    completed = {
                        "work_order_id": order.work_order_id,
                        "run_index": run_index,
                        "ok": True,
                        "attempts": attempt + 1,
                        "seed": seed,
                        "model": args.model,
                        "prompt_version": PROMPT_VERSION,
                        "temperature": args.temperature,
                        "extraction": extraction,
                    }
                    break
                except (RuntimeError, OSError, ValueError) as error:
                    failures.append(str(error))
            if completed is None:
                completed = {
                    "work_order_id": order.work_order_id,
                    "run_index": run_index,
                    "ok": False,
                    "attempts": args.retries + 1,
                    "model": args.model,
                    "prompt_version": PROMPT_VERSION,
                    "temperature": args.temperature,
                    "error": " | ".join(failures),
                }
            rows.append(completed)
    write_jsonl(args.output, rows)
    successful = sum(bool(row.get("ok")) for row in rows)
    complete = successful == len(rows)
    formal_eligible = bool(rows) and complete
    manifest_path = str(Path(args.output).with_name(Path(args.output).name + ".manifest.json"))
    write_json(
        manifest_path,
        {
            "records_sha256": _sha256(args.records),
            "knowledge_base_sha256": _sha256(args.kb),
            "model": args.model,
            "prompt_version": PROMPT_VERSION,
            "runs_per_order": args.runs,
            "temperature": args.temperature,
            "base_seed": args.seed,
            "retries_per_slot": args.retries,
            "orders": len(orders),
            "successful_slots": successful,
            "total_slots": len(rows),
            "complete": complete,
            "formal_eligible": formal_eligible,
            "minimum_support": args.minimum_support,
            "minimum_component_score": args.minimum_component_score,
        },
    )
    print("repeated extraction slots: {}/{} successful -> {}".format(successful, len(rows), args.output))
    print("run manifest -> {}".format(manifest_path))
    if not complete and not args.allow_incomplete:
        raise SystemExit(
            "{} extraction slots failed; raw runs and manifest were saved, but no "
            "consensus was written. Use --allow-incomplete only for diagnostic runs.".format(
                len(rows) - successful
            )
        )
    if args.consensus_output:
        consensus = build_consensus_records(
            orders,
            rows,
            kb,
            planned_runs=args.runs,
            minimum_support=args.minimum_support,
            minimum_component_score=args.minimum_component_score,
            include_orders_without_runs=True,
        )
        for record in consensus:
            record["experiment_complete"] = complete
            if not formal_eligible:
                record["formal_eligible"] = False
        write_jsonl(args.consensus_output, consensus)
        print("canonical consensus: {} -> {}".format(len(consensus), args.consensus_output))


def cmd_build_consensus(args):
    if args.planned_runs < 1:
        raise SystemExit("--planned-runs must be at least 1")
    if not math.isfinite(args.minimum_support) or not 0.0 <= args.minimum_support <= 1.0:
        raise SystemExit("--minimum-support must be between 0 and 1")
    if (
        not math.isfinite(args.minimum_component_score)
        or not 0.0 <= args.minimum_component_score <= 1.0
    ):
        raise SystemExit("--minimum-component-score must be between 0 and 1")
    orders = _load_orders(args.records)
    kb = KnowledgeBase.load(args.kb)
    rows = build_consensus_records(
        orders,
        read_jsonl(args.runs),
        kb,
        planned_runs=args.planned_runs,
        minimum_support=args.minimum_support,
        minimum_component_score=args.minimum_component_score,
        include_orders_without_runs=True,
    )
    incomplete_ids = [
        row["work_order_id"] for row in rows if not row.get("formal_eligible", False)
    ]
    experiment_complete = bool(rows) and not incomplete_ids
    for row in rows:
        row["experiment_complete"] = experiment_complete
        if not experiment_complete:
            row["formal_eligible"] = False
    if incomplete_ids and not args.allow_incomplete:
        raise SystemExit(
            "{} work orders do not contain all planned successful run slots; no consensus "
            "was written. Use --allow-incomplete only for diagnostic output.".format(
                len(incomplete_ids)
            )
        )
    write_jsonl(args.output, rows)
    print("canonical consensus: {} -> {}".format(len(rows), args.output))


def cmd_prediction_candidates(args):
    if (
        not math.isfinite(args.minimum_component_score)
        or not 0.0 <= args.minimum_component_score <= 1.0
    ):
        raise SystemExit("--minimum-component-score must be between 0 and 1")
    orders = _load_orders(args.records)
    kb = KnowledgeBase.load(args.kb)
    rows = predictions_to_candidate_records(
        orders,
        read_jsonl(args.predictions),
        kb,
        minimum_component_score=args.minimum_component_score,
    )
    write_jsonl(args.output, rows)
    print("single-model candidate adapter: {} -> {}".format(len(rows), args.output))


def cmd_build_uncertain_graph(args):
    orders = _load_orders(args.records)
    kb = KnowledgeBase.load(args.kb)
    result = build_uncertain_graph(
        orders,
        read_jsonl(args.candidates),
        engine_name=args.engine_name,
        minimum_path_support=args.minimum_path_support,
        include_review=args.include_review,
        alpha=args.alpha,
        fusion_method=args.fusion_method,
        shrinkage_strength=args.shrinkage_strength,
        risk_weights=(args.occurrence_weight, args.severity_weight, args.burden_weight),
        kb=kb,
    )
    result.summary["records_sha256"] = _sha256(args.records)
    result.summary["candidates_sha256"] = _sha256(args.candidates)
    result.summary["knowledge_base_sha256"] = _sha256(args.kb)
    export_uncertain_graph(result, args.output_dir)
    print(json.dumps(result.summary, ensure_ascii=False, indent=2))


def cmd_train_transformer(args):
    from .transformer import train_from_jsonl

    train_from_jsonl(
        annotations_path=args.annotations,
        output_dir=args.output_dir,
        pretrained_model=args.pretrained_model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_length=args.max_length,
    )


def cmd_predict_transformer(args):
    from .transformer import TransformerExtractor

    extractor = TransformerExtractor(args.model_dir)
    orders = _load_orders(args.records)
    rows = []
    for order in orders:
        result = extractor.predict(order.model_text(), args.entity_threshold, args.class_threshold)
        result["work_order_id"] = order.work_order_id
        rows.append(result)
    write_jsonl(args.output, rows)
    print("transformer predictions: {} -> {}".format(len(rows), args.output))


def cmd_serve(args):
    try:
        import uvicorn
    except ImportError as error:
        raise SystemExit("Install API dependencies: pip install -e '.[api]'") from error
    from .api import create_app

    app = create_app(args.kb, args.model, args.policy)
    uvicorn.run(app, host=args.host, port=args.port)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="engine-fault-nlp")
    sub = parser.add_subparsers(dest="command", required=True)

    item = sub.add_parser("audit", help="Audit source workbook")
    item.add_argument("--input", required=True); item.add_argument("--sheet", default="Report"); item.set_defaults(func=cmd_audit)
    item = sub.add_parser("prepare", help="Aggregate workbook rows to work-order JSONL")
    item.add_argument("--input", required=True); item.add_argument("--sheet", default="Report"); item.add_argument("--output", required=True); item.set_defaults(func=cmd_prepare)
    item = sub.add_parser("build-kb", help="Build part/DTC/symptom knowledge base")
    item.add_argument("--input", required=True); item.add_argument("--sheet", default="Report"); item.add_argument("--base-kb"); item.add_argument("--output", required=True); item.set_defaults(func=cmd_build_kb)
    item = sub.add_parser("split", help="Leakage-safe grouped train/validation/test split")
    item.add_argument("--records", required=True); item.add_argument("--output-dir", required=True); item.add_argument("--train-ratio", type=float, default=0.70); item.add_argument("--validation-ratio", type=float, default=0.15); item.add_argument("--seed", type=int, default=42); item.set_defaults(func=cmd_split)
    item = sub.add_parser("export-annotation", help="Export span/relation annotation tasks")
    item.add_argument("--records", required=True); item.add_argument("--output", required=True); item.set_defaults(func=cmd_export_annotation)
    item = sub.add_parser("weak-label", help="Generate denoised rule-based weak labels")
    item.add_argument("--records", required=True); item.add_argument("--kb", required=True); item.add_argument("--output", required=True); item.set_defaults(func=cmd_weak_label)
    item = sub.add_parser("weak-votes", help="Export independent weak labeling-function votes")
    item.add_argument("--records", required=True); item.add_argument("--kb", required=True); item.add_argument("--output", required=True); item.set_defaults(func=cmd_weak_votes)
    item = sub.add_parser("fit-label-model", help="Fit a generative label model over weak votes")
    item.add_argument("--votes", required=True); item.add_argument("--output", required=True); item.set_defaults(func=cmd_fit_label_model)
    item = sub.add_parser("train-baseline", help="Train dependency-free character TF-IDF centroid model")
    item.add_argument("--records", required=True); item.add_argument("--labels", required=True); item.add_argument("--output", required=True); item.set_defaults(func=cmd_train_baseline)
    item = sub.add_parser("train-transformer", help="Train multi-task GlobalPointer transformer")
    item.add_argument("--annotations", required=True); item.add_argument("--output-dir", required=True); item.add_argument("--pretrained-model", default="hfl/chinese-macbert-base"); item.add_argument("--epochs", type=int, default=8); item.add_argument("--batch-size", type=int, default=8); item.add_argument("--learning-rate", type=float, default=2e-5); item.add_argument("--max-length", type=int, default=512); item.set_defaults(func=cmd_train_transformer)
    item = sub.add_parser("predict-transformer", help="Run a trained GlobalPointer transformer")
    item.add_argument("--records", required=True); item.add_argument("--model-dir", required=True); item.add_argument("--entity-threshold", type=float, default=0.5); item.add_argument("--class-threshold", type=float, default=0.5); item.add_argument("--output", required=True); item.set_defaults(func=cmd_predict_transformer)
    item = sub.add_parser("predict", help="Hybrid batch prediction")
    item.add_argument("--input", required=True); item.add_argument("--sheet", default="Report"); item.add_argument("--kb", required=True); item.add_argument("--model"); item.add_argument("--policy"); item.add_argument("--output", required=True); item.set_defaults(func=cmd_predict)
    item = sub.add_parser("export-csv", help="Flatten JSONL predictions to analyst-friendly CSV")
    item.add_argument("--predictions", required=True); item.add_argument("--output", required=True); item.set_defaults(func=cmd_export_csv)
    item = sub.add_parser("active-learn", help="Select uncertain and diverse records for expert annotation")
    item.add_argument("--records", required=True); item.add_argument("--predictions", required=True); item.add_argument("--limit", type=int, default=100); item.add_argument("--output", required=True); item.set_defaults(func=cmd_active_learn)
    item = sub.add_parser("build-graph-rank", help="Legacy five-layer graph (no repeated-run support)")
    item.add_argument("--records", required=True); item.add_argument("--predictions", required=True); item.add_argument("--engine-name", default="15N发动机"); item.add_argument("--shrinkage-strength", type=float, default=5.0); item.add_argument("--entropy-strength", type=float, default=0.5); item.add_argument("--output-dir", required=True); item.set_defaults(func=cmd_build_graph_rank)
    item = sub.add_parser("evaluate", help="Evaluate predictions against gold JSONL")
    item.add_argument("--predictions", required=True); item.add_argument("--gold", required=True); item.add_argument("--output"); item.set_defaults(func=cmd_evaluate)
    item = sub.add_parser("calibrate", help="Select thresholds using Wilson precision lower bounds")
    item.add_argument("--predictions", required=True); item.add_argument("--gold", required=True); item.add_argument("--target-precision", type=float, default=0.95); item.add_argument("--minimum-accepted", type=int, default=30); item.add_argument("--output", required=True); item.set_defaults(func=cmd_calibrate)
    item = sub.add_parser("teacher-label", help="Label with an OpenAI-compatible local/cloud LLM")
    item.add_argument("--records", required=True); item.add_argument("--kb", required=True); item.add_argument("--endpoint", required=True); item.add_argument("--model", required=True); item.add_argument("--api-key", default=""); item.add_argument("--output", required=True); item.set_defaults(func=cmd_teacher_label)
    item = sub.add_parser("repeat-teacher", help="Run K independent LLM candidate extractions and optionally build consensus")
    item.add_argument("--records", required=True); item.add_argument("--kb", required=True)
    item.add_argument("--endpoint", required=True); item.add_argument("--model", required=True)
    item.add_argument("--api-key", default=""); item.add_argument("--runs", type=int, default=5)
    item.add_argument("--temperature", type=float, default=0.3); item.add_argument("--seed", type=int, default=42)
    item.add_argument("--retries", type=int, default=2); item.add_argument("--timeout", type=int, default=120)
    item.add_argument("--allow-incomplete", action="store_true")
    item.add_argument("--limit", type=int); item.add_argument("--minimum-support", type=float, default=0.6)
    item.add_argument("--minimum-component-score", type=float, default=0.25)
    item.add_argument("--output", required=True); item.add_argument("--consensus-output")
    item.set_defaults(func=cmd_repeat_teacher)
    item = sub.add_parser("build-consensus", help="Canonicalise cached repeated runs and compute n_e/K support")
    item.add_argument("--records", required=True); item.add_argument("--runs", required=True)
    item.add_argument("--kb", required=True); item.add_argument("--planned-runs", type=int, required=True)
    item.add_argument("--minimum-support", type=float, default=0.6)
    item.add_argument("--minimum-component-score", type=float, default=0.25)
    item.add_argument("--allow-incomplete", action="store_true")
    item.add_argument("--output", required=True); item.set_defaults(func=cmd_build_consensus)
    item = sub.add_parser("prediction-candidates", help="Adapt existing single-model outputs without claiming repeated-run support")
    item.add_argument("--records", required=True); item.add_argument("--predictions", required=True)
    item.add_argument("--kb", required=True); item.add_argument("--minimum-component-score", type=float, default=0.25)
    item.add_argument("--output", required=True); item.set_defaults(func=cmd_prediction_candidates)
    item = sub.add_parser("build-uncertain-graph", help="Fuse evidence and rank association-risk paths/key nodes")
    item.add_argument("--records", required=True); item.add_argument("--candidates", required=True)
    item.add_argument("--kb", required=True)
    item.add_argument("--engine-name", default="15N发动机")
    item.add_argument("--minimum-path-support", type=float, default=0.6)
    item.add_argument("--include-review", action="store_true")
    item.add_argument("--alpha", type=float, default=0.5)
    item.add_argument("--fusion-method", choices=("geometric", "linear"), default="geometric")
    item.add_argument("--shrinkage-strength", type=float, default=5.0)
    item.add_argument("--occurrence-weight", type=float, default=0.45)
    item.add_argument("--severity-weight", type=float, default=0.35)
    item.add_argument("--burden-weight", type=float, default=0.20)
    item.add_argument("--output-dir", required=True); item.set_defaults(func=cmd_build_uncertain_graph)
    item = sub.add_parser("serve", help="Serve prediction API")
    item.add_argument("--kb", required=True); item.add_argument("--model"); item.add_argument("--policy"); item.add_argument("--host", default="127.0.0.1"); item.add_argument("--port", type=int, default=8000); item.set_defaults(func=cmd_serve)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
