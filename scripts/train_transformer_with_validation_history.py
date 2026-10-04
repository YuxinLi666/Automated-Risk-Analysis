"""Train MacBERT-MGP with reproducible epoch-wise train/validation monitoring.

This script keeps the original multi-task objective, evaluates the frozen model
state after every epoch, and stores the checkpoint with minimum validation loss.
Top-1 document accuracy is reported for the three four-layer path heads used in
the paper: subsystem, component, and failure mode.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import csv
import json
import random
from pathlib import Path
import sys
import time

import numpy as np


REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from engine_fault_nlp.io_utils import read_jsonl, write_json  # noqa: E402
from engine_fault_nlp.transformer import (  # noqa: E402
    AnnotationDataset,
    BatchCollator,
    CLASSIFICATION_FIELDS,
    ENTITY_TYPES,
    _move_to_device,
    _require_ml,
    build_label_maps,
    create_model,
)


PAPER_FIELDS = ("subsystem_id", "component_id", "failure_mode_id")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train",
        default="data/seed/train.jsonl",
    )
    parser.add_argument(
        "--validation",
        default="data/seed/validation.jsonl",
    )
    parser.add_argument("--output-dir", default="outputs/legacy_transformer/checkpoint")
    parser.add_argument(
        "--history",
        default="outputs/legacy_transformer/history.json",
    )
    parser.add_argument("--pretrained-model", default="hfl/chinese-macbert-base")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mixed-precision", action="store_true")
    return parser.parse_args()


def set_seed(torch, seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def gold_values(row: dict, field: str) -> set[str]:
    return {
        str(chain[field])
        for chain in row.get("chains", row.get("fault_chains", []))
        if chain.get(field)
    }


def evaluate_split(
    model,
    loader,
    rows: list[dict],
    inverse_maps: dict,
    device,
    torch,
    mixed_precision: bool = False,
) -> dict:
    model.eval()
    weighted_total_loss = 0.0
    weighted_task_loss = {field: 0.0 for field in CLASSIFICATION_FIELDS}
    evaluated_documents = 0
    correct = {field: 0 for field in PAPER_FIELDS}
    support = {field: 0 for field in PAPER_FIELDS}
    offset = 0

    with torch.no_grad():
        for batch in loader:
            batch_size = int(batch["input_ids"].shape[0])
            batch_rows = rows[offset : offset + batch_size]
            offset += batch_size
            batch = _move_to_device(batch, device)
            amp_context = (
                torch.amp.autocast(device_type="cuda", dtype=torch.float16)
                if mixed_precision and device.type == "cuda"
                else nullcontext()
            )
            with amp_context:
                output = model(**batch)
            if output["loss"] is not None:
                weighted_total_loss += float(output["loss"].detach().cpu()) * batch_size
            for field in CLASSIFICATION_FIELDS:
                task_loss = output["losses"].get(field)
                if task_loss is not None:
                    weighted_task_loss[field] += float(task_loss.detach().cpu()) * batch_size

            for field in PAPER_FIELDS:
                predicted_indices = output["class_logits"][field].argmax(dim=1).detach().cpu().tolist()
                for row, predicted_index in zip(batch_rows, predicted_indices):
                    gold = gold_values(row, field)
                    if not gold:
                        continue
                    support[field] += 1
                    predicted = inverse_maps[field].get(int(predicted_index))
                    correct[field] += int(predicted in gold)
            evaluated_documents += batch_size

    result = {
        "loss": weighted_total_loss / max(1, evaluated_documents),
        "documents": evaluated_documents,
    }
    for field in CLASSIFICATION_FIELDS:
        result[f"{field}_loss"] = weighted_task_loss[field] / max(1, evaluated_documents)
    for field in PAPER_FIELDS:
        result[f"{field}_accuracy"] = correct[field] / max(1, support[field])
        result[f"{field}_support"] = support[field]
    return result


def save_history(history_path: Path, payload: dict) -> None:
    history_path.parent.mkdir(parents=True, exist_ok=True)
    history_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    csv_path = history_path.with_suffix(".csv")
    rows = payload["epochs"]
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    torch, _, _, _, AutoTokenizer = _require_ml()
    set_seed(torch, args.seed)

    train_rows = list(read_jsonl(args.train))
    validation_rows = list(read_jsonl(args.validation))
    label_maps = build_label_maps(train_rows)
    inverse_maps = {
        field: {int(index): label for label, index in label_maps[field].items()}
        for field in CLASSIFICATION_FIELDS
    }
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model, use_fast=True)
    collator = BatchCollator(tokenizer, label_maps)
    train_dataset = AnnotationDataset(train_rows, tokenizer, label_maps, args.max_length)
    validation_dataset = AnnotationDataset(validation_rows, tokenizer, label_maps, args.max_length)

    generator = torch.Generator()
    generator.manual_seed(args.seed)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        collate_fn=collator,
    )
    validation_loader = torch.utils.data.DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collator,
    )

    model = create_model(args.pretrained_model, label_maps)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    scaler = torch.amp.GradScaler("cuda", enabled=args.mixed_precision and device.type == "cuda")

    target = Path(args.output_dir)
    target.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(str(target))
    model.encoder.config.save_pretrained(str(target))
    write_json(str(target / "label_maps.json"), label_maps)

    epoch_rows: list[dict] = []
    best_validation_loss = float("inf")
    best_epoch = 0
    start_time = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        model.train()
        optimization_loss = 0.0
        optimized_documents = 0
        online_correct = {field: 0 for field in PAPER_FIELDS}
        online_support = {field: 0 for field in PAPER_FIELDS}
        for batch in train_loader:
            batch_size = int(batch["input_ids"].shape[0])
            batch = _move_to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            amp_context = (
                torch.amp.autocast(device_type="cuda", dtype=torch.float16)
                if args.mixed_precision and device.type == "cuda"
                else nullcontext()
            )
            with amp_context:
                output = model(**batch)
            if output["loss"] is None:
                continue
            scaler.scale(output["loss"]).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimization_loss += float(output["loss"].detach().cpu()) * batch_size
            optimized_documents += batch_size

            for field in PAPER_FIELDS:
                targets, mask = batch["class_labels"][field]
                if not mask.any():
                    continue
                predictions = output["class_logits"][field][mask].argmax(dim=1)
                selected_targets = targets[mask]
                row_indices = torch.arange(len(predictions), device=device)
                online_correct[field] += int(selected_targets[row_indices, predictions].sum().detach().cpu())
                online_support[field] += int(mask.sum().detach().cpu())

        validation_metrics = evaluate_split(
            model,
            validation_loader,
            validation_rows,
            inverse_maps,
            device,
            torch,
            mixed_precision=args.mixed_precision,
        )
        elapsed = time.perf_counter() - start_time
        row = {
            "epoch": epoch,
            "optimization_loss": optimization_loss / max(1, optimized_documents),
            "train_loss": optimization_loss / max(1, optimized_documents),
            "validation_loss": validation_metrics["loss"],
            "train_subsystem_accuracy": online_correct["subsystem_id"] / max(1, online_support["subsystem_id"]),
            "validation_subsystem_accuracy": validation_metrics["subsystem_id_accuracy"],
            "train_component_accuracy": online_correct["component_id"] / max(1, online_support["component_id"]),
            "validation_component_accuracy": validation_metrics["component_id_accuracy"],
            "train_failure_mode_accuracy": online_correct["failure_mode_id"] / max(1, online_support["failure_mode_id"]),
            "validation_failure_mode_accuracy": validation_metrics["failure_mode_id_accuracy"],
            "validation_subsystem_support": validation_metrics["subsystem_id_support"],
            "validation_component_support": validation_metrics["component_id_support"],
            "validation_failure_mode_support": validation_metrics["failure_mode_id_support"],
            "elapsed_seconds": elapsed,
        }
        epoch_rows.append(row)

        if validation_metrics["loss"] < best_validation_loss:
            best_validation_loss = validation_metrics["loss"]
            best_epoch = epoch
            torch.save(model.state_dict(), target / "model.pt")

        print(
            "epoch {}/{} train_loss={:.6f} val_loss={:.6f} "
            "val_acc(sub/component/mode)={:.3f}/{:.3f}/{:.3f} best_epoch={}".format(
                epoch,
                args.epochs,
                row["train_loss"],
                validation_metrics["loss"],
                validation_metrics["subsystem_id_accuracy"],
                validation_metrics["component_id_accuracy"],
                validation_metrics["failure_mode_id_accuracy"],
                best_epoch,
            ),
            flush=True,
        )

        payload = {
            "description": "Epoch-wise monitored MacBERT-MGP training run",
            "label_status": "PREANNOTATED_NEEDS_EXPERT_REVIEW",
            "metric_definition": "Top-1 document accuracy; prediction is correct when it matches any reference label for the field.",
            "train_records": len(train_rows),
            "validation_records": len(validation_rows),
            "pretrained_model": args.pretrained_model,
            "epochs_requested": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "max_length": args.max_length,
            "seed": args.seed,
            "device": str(device),
            "mixed_precision": bool(args.mixed_precision and device.type == "cuda"),
            "train_accuracy_definition": "Online mini-batch top-1 accuracy accumulated during optimization; validation accuracy is evaluated after each epoch with the model in eval mode.",
            "best_epoch": best_epoch,
            "best_validation_loss": best_validation_loss,
            "epochs": epoch_rows,
        }
        save_history(Path(args.history), payload)

    torch.save(model.state_dict(), target / "final_model.pt")
    write_json(
        str(target / "training_config.json"),
        {
            "pretrained_model": args.pretrained_model,
            "entity_types": ENTITY_TYPES,
            "max_length": args.max_length,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "seed": args.seed,
            "best_epoch": best_epoch,
            "checkpoint_selection": "minimum_validation_loss",
            "mixed_precision": bool(args.mixed_precision and device.type == "cuda"),
            "history": str(Path(args.history)),
        },
    )
    print(json.dumps({"output_dir": str(target), "history": args.history, "best_epoch": best_epoch}, indent=2))


if __name__ == "__main__":
    main()
