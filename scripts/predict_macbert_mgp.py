"""Run offline, three-head MacBERT-MGP inference from an explicitly saved checkpoint.

Input may be prepared work orders or documents with an explicit text field.
Output follows the module-ablation schema: document_id, predicted, confidence.
No rule-based labels, cause classification, or reference annotations are used.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from run_macbert_mgp_module_ablation import (
    PAPER_FIELDS,
    PaperCollator,
    PaperDataset,
    WorkOrder,
    create_model,
    predict,
    read_jsonl,
    resolve,
    validate_checkpoint_metadata,
    write_jsonl,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Directory saved with --save-checkpoints.")
    parser.add_argument("--records", default="data/work_orders.jsonl", help="Input work-order JSONL.")
    parser.add_argument("--output", default="outputs/macbert_mgp_predictions.jsonl", help="Output JSONL.")
    parser.add_argument("--batch-size", type=int, default=4, help="Inference batch size.")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--mixed-precision", action="store_true", help="Use float16 autocast on CUDA only.")
    args = parser.parse_args(argv)
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    return args


def build_inference_rows(records):
    """Use training-compatible text construction while dropping all supervision."""
    rows = []
    seen = set()
    for index, record in enumerate(records, start=1):
        document_id = record.get("document_id", record.get("work_order_id"))
        if document_id is None or not str(document_id).strip():
            raise ValueError(f"Missing document_id/work_order_id in record {index}")
        document_id = str(document_id)
        if document_id in seen:
            raise ValueError(f"Duplicate document ID: {document_id}")
        seen.add(document_id)
        if "text" in record:
            text = record["text"]
            if not isinstance(text, str):
                raise ValueError(f"Text must be a string for document {document_id}")
        else:
            text = WorkOrder.from_dict(record).model_text()
        rows.append({"document_id": document_id, "text": text,
                     "entities": [], "relations": [], "chains": []})
    return rows


def load_checkpoint(checkpoint, device="cpu"):
    """Rebuild from local config and load safetensors with strict key checking.

    Neither remote model code nor Python-object pickle loading is permitted.
    Model weights, tokenizer assets, and config must already exist locally.
    """
    import torch
    from safetensors.torch import load_file
    from transformers import AutoConfig, AutoTokenizer

    checkpoint = Path(checkpoint)
    metadata = json.loads((checkpoint / "checkpoint.json").read_text(encoding="utf-8"))
    validate_checkpoint_metadata(metadata)
    config = AutoConfig.from_pretrained(
        str(checkpoint), local_files_only=True, trust_remote_code=False,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        str(checkpoint), use_fast=True, local_files_only=True, trust_remote_code=False,
    )
    if not tokenizer.is_fast:
        raise ValueError("MacBERT-MGP requires a fast tokenizer for offset mappings")
    model = create_model(
        None, metadata["label_maps"], metadata["weights"], metadata["flags"],
        torch, torch.nn, encoder_config=config,
    )
    model.load_state_dict(load_file(str(checkpoint / "model.safetensors"), device="cpu"), strict=True)
    model.to(torch.device(device)).eval()
    return model, tokenizer, metadata


def predict_rows(model, tokenizer, metadata, rows, *, batch_size=4, device="cpu", mixed_precision=False):
    """Reuse the training dataset, collation, threshold, and prediction function."""
    import torch

    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    device = torch.device(device)
    model.to(device)
    dataset = PaperDataset(rows, tokenizer, metadata["label_maps"], metadata["max_length"])
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        collate_fn=PaperCollator(tokenizer, metadata["label_maps"], torch),
    )
    inverse_maps = {
        field: {index: label for label, index in metadata["label_maps"][field].items()}
        for field in PAPER_FIELDS
    }
    return predict(model, loader, rows, inverse_maps, device, torch,
                   bool(mixed_precision and device.type == "cuda"))


def main(argv=None):
    args = parse_args(argv)
    import torch

    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; use --device cpu")
    source, target = resolve(args.records), resolve(args.output)
    if source.resolve() == target.resolve():
        raise ValueError("Output must not overwrite the input records")
    rows = build_inference_rows(read_jsonl(str(source)))
    model, tokenizer, metadata = load_checkpoint(resolve(args.checkpoint), device)
    predictions = predict_rows(
        model, tokenizer, metadata, rows, batch_size=args.batch_size,
        device=device, mixed_precision=args.mixed_precision,
    )
    write_jsonl(target, predictions)
    print(json.dumps({"records": len(predictions), "output": str(target),
                      "device": device, "checkpoint_selection": metadata["checkpoint_selection"]}, indent=2))


if __name__ == "__main__":
    main()
