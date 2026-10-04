"""Run controlled module ablations for the four-layer MacBERT--MGP extractor.

The paper output is Engine root -> subsystem -> component -> failure mode.  To
match that definition, this experiment deliberately excludes cause labels,
CAUSE spans, and CAUSED_BY relations from every variant.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from collections.abc import Mapping
from contextlib import nullcontext
import copy
import json
import math
from pathlib import Path
import random
import sys
import time

import numpy as np


REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from engine_fault_nlp.io_utils import read_jsonl  # noqa: E402
from engine_fault_nlp.schemas import WorkOrder  # noqa: E402


PAPER_FIELDS = ("subsystem_id", "component_id", "failure_mode_id")
PAPER_ENTITY_TYPES = ("COMPONENT", "FAILURE_MODE", "SYMPTOM", "DTC")
CLASS_THRESHOLD = 0.5

VARIANTS = {
    "full": {
        "display_name": "MacBERT--MGP",
        "use_span": True,
        "use_relation": True,
        "use_class_weights": True,
        "use_mean_pooling": True,
    },
    "no_globalpointer": {
        "display_name": "w/o GlobalPointer",
        "use_span": False,
        "use_relation": True,
        "use_class_weights": True,
        "use_mean_pooling": True,
    },
    "no_relation": {
        "display_name": "w/o relation learning",
        "use_span": True,
        "use_relation": False,
        "use_class_weights": True,
        "use_mean_pooling": True,
    },
    "no_auxiliary": {
        "display_name": "w/o auxiliary tasks",
        "use_span": False,
        "use_relation": False,
        "use_class_weights": True,
        "use_mean_pooling": True,
    },
    "no_class_reweighting": {
        "display_name": "w/o class reweighting",
        "use_span": True,
        "use_relation": True,
        "use_class_weights": False,
        "use_mean_pooling": True,
    },
    "no_mean_pooling": {
        "display_name": "w/o mean pooling",
        "use_span": True,
        "use_relation": True,
        "use_class_weights": True,
        "use_mean_pooling": False,
    },
}


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train",
        default="data/seed/train.jsonl",
    )
    parser.add_argument(
        "--validation",
        default="data/seed/validation.jsonl",
    )
    parser.add_argument(
        "--test",
        default="data/seed/test.jsonl",
    )
    parser.add_argument("--work-orders", default="data/work_orders.jsonl")
    parser.add_argument(
        "--unseen-labels",
        default="data/unseen/labels.jsonl",
    )
    parser.add_argument(
        "--output-dir", default="outputs/module_ablation"
    )
    parser.add_argument("--pretrained-model", default="hfl/chinese-macbert-base")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--seeds", default="42,52,62")
    parser.add_argument("--variants", default=",".join(VARIANTS))
    parser.add_argument("--mixed-precision", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--save-checkpoints", action="store_true",
        help="Save each selected best-epoch model as safetensors for offline inference.",
    )
    return parser.parse_args(argv)


def resolve(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else REPO / candidate


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def set_seed(torch, seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def clean_annotated_row(row: dict) -> dict:
    """Remove the discarded cause layer without changing surviving offsets."""
    cleaned = copy.deepcopy(row)
    entities = [
        entity
        for entity in cleaned.get("entities", [])
        if entity.get("label") in PAPER_ENTITY_TYPES
    ]
    kept_ids = {entity.get("entity_id") for entity in entities}
    relations = [
        relation
        for relation in cleaned.get("relations", [])
        if relation.get("head_id") in kept_ids
        and relation.get("tail_id") in kept_ids
        and relation.get("relation") != "CAUSED_BY"
    ]
    chains = []
    for source in cleaned.get("chains", cleaned.get("fault_chains", [])):
        chain = {field: source.get(field) for field in PAPER_FIELDS if source.get(field)}
        if chain:
            chains.append(chain)
    cleaned["entities"] = entities
    cleaned["relations"] = relations
    cleaned["chains"] = chains
    cleaned.pop("fault_chains", None)
    return cleaned


def build_unseen_rows(work_orders_path: Path, labels_path: Path) -> list[dict]:
    orders = {
        str(row["work_order_id"]): WorkOrder.from_dict(row)
        for row in read_jsonl(str(work_orders_path))
    }
    rows = []
    for label_row in read_jsonl(str(labels_path)):
        order_id = str(label_row["work_order_id"])
        if order_id not in orders:
            continue
        chains = []
        for source in label_row.get("fault_chains", []):
            chain = {
                field: source.get(field)
                for field in PAPER_FIELDS
                if source.get(field)
            }
            if chain:
                chains.append(chain)
        rows.append(
            {
                "document_id": order_id,
                "text": orders[order_id].model_text(),
                "entities": [],
                "relations": [],
                "chains": chains,
            }
        )
    return rows


def build_label_maps(rows: list[dict]) -> dict:
    values = {field: set() for field in PAPER_FIELDS}
    relations = set()
    for row in rows:
        for chain in row.get("chains", []):
            for field in PAPER_FIELDS:
                if chain.get(field):
                    values[field].add(str(chain[field]))
        for relation in row.get("relations", []):
            if relation.get("relation"):
                relations.add(str(relation["relation"]))
    maps = {
        field: {label: index for index, label in enumerate(sorted(labels))}
        for field, labels in values.items()
    }
    maps["relations"] = {
        label: index for index, label in enumerate(sorted(relations))
    }
    return maps


def class_weights(rows: list[dict], label_maps: dict) -> dict[str, list[float]]:
    total = max(1, len(rows))
    result: dict[str, list[float]] = {}
    for field in PAPER_FIELDS:
        counts = [0] * len(label_maps[field])
        for row in rows:
            present = {
                str(chain[field])
                for chain in row.get("chains", [])
                if chain.get(field) and str(chain[field]) in label_maps[field]
            }
            for label in present:
                counts[label_maps[field][label]] += 1
        result[field] = [
            min(20.0, max(1.0, math.sqrt(max(1, total - count) / count)))
            if count > 0
            else 1.0
            for count in counts
        ]
    return result


def char_span_to_tokens(offsets, start: int, end: int):
    token_start = next(
        (i for i, (left, right) in enumerate(offsets) if left <= start < right),
        None,
    )
    token_end = next(
        (
            i
            for i, (left, right) in reversed(list(enumerate(offsets)))
            if left < end <= right
        ),
        None,
    )
    if token_start is None or token_end is None or token_end < token_start:
        return None
    return token_start, token_end


class PaperDataset:
    def __init__(self, rows, tokenizer, label_maps, max_length):
        self.rows = rows
        self.tokenizer = tokenizer
        self.label_maps = label_maps
        self.max_length = max_length
        # Tokenization and character-to-token alignment are deterministic for
        # a fixed tokenizer. Cache them once so eight epochs do not repeat the
        # same CPU preprocessing.
        self.items = [self._encode(row) for row in rows]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]

    def _encode(self, row):
        encoded = self.tokenizer(
            row["text"],
            truncation=True,
            max_length=self.max_length,
            return_offsets_mapping=True,
        )
        offsets = encoded.pop("offset_mapping")
        spans = []
        token_entities = {}
        for entity in row.get("entities", []):
            if entity.get("label") not in PAPER_ENTITY_TYPES:
                continue
            token_span = char_span_to_tokens(
                offsets, int(entity["start"]), int(entity["end"])
            )
            if token_span:
                label_index = PAPER_ENTITY_TYPES.index(entity["label"])
                spans.append((label_index, token_span[0], token_span[1]))
                token_entities[entity["entity_id"]] = token_span
        classifications = {}
        for field in PAPER_FIELDS:
            classifications[field] = [
                self.label_maps[field][str(chain[field])]
                for chain in row.get("chains", [])
                if chain.get(field)
                and str(chain[field]) in self.label_maps[field]
            ]
        relation_pairs = []
        for relation in row.get("relations", []):
            head = token_entities.get(relation.get("head_id"))
            tail = token_entities.get(relation.get("tail_id"))
            label = self.label_maps["relations"].get(str(relation.get("relation")))
            if head and tail and label is not None:
                relation_pairs.append((head[0], head[1], tail[0], tail[1], label))
        return {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "token_type_ids": encoded.get("token_type_ids"),
            "spans": spans,
            "classifications": classifications,
            "relation_pairs": relation_pairs,
        }


class PaperCollator:
    def __init__(self, tokenizer, label_maps, torch):
        self.tokenizer = tokenizer
        self.label_maps = label_maps
        self.torch = torch

    def __call__(self, items):
        torch = self.torch
        padded = self.tokenizer.pad(
            [
                {
                    key: value
                    for key, value in item.items()
                    if key in ("input_ids", "attention_mask", "token_type_ids")
                    and value is not None
                }
                for item in items
            ],
            return_tensors="pt",
        )
        batch_size, length = padded["input_ids"].shape
        entity_labels = torch.zeros(
            (batch_size, len(PAPER_ENTITY_TYPES), length, length),
            dtype=torch.float32,
        )
        for batch_index, item in enumerate(items):
            for label, start, end in item["spans"]:
                if start < length and end < length:
                    entity_labels[batch_index, label, start, end] = 1.0
        labels = {}
        for field in PAPER_FIELDS:
            target = torch.zeros(
                (batch_size, len(self.label_maps[field])), dtype=torch.float32
            )
            mask = torch.zeros(batch_size, dtype=torch.bool)
            for batch_index, item in enumerate(items):
                for label in item["classifications"][field]:
                    target[batch_index, label] = 1.0
                    mask[batch_index] = True
            labels[field] = (target, mask)
        max_pairs = max((len(item["relation_pairs"]) for item in items), default=0)
        pairs = torch.full((batch_size, max_pairs, 4), -1, dtype=torch.long)
        relation_labels = torch.full(
            (batch_size, max_pairs), -100, dtype=torch.long
        )
        for batch_index, item in enumerate(items):
            for pair_index, pair in enumerate(item["relation_pairs"]):
                pairs[batch_index, pair_index] = torch.tensor(pair[:4])
                relation_labels[batch_index, pair_index] = pair[4]
        padded.update(
            {
                "entity_labels": entity_labels,
                "class_labels": labels,
                "relation_pairs": pairs,
                "relation_labels": relation_labels,
            }
        )
        return padded


def move_to_device(value, device, torch):
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, Mapping):
        return {key: move_to_device(item, device, torch) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(move_to_device(item, device, torch) for item in value)
    if isinstance(value, list):
        return [move_to_device(item, device, torch) for item in value]
    return value


def create_model(pretrained_model, label_maps, weights, flags, torch, nn, *, encoder_config=None):
    """Build the shared architecture from pretrained weights or an offline config."""
    from transformers import AutoModel

    class AblationModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.flags = flags
            self.encoder = (
                AutoModel.from_pretrained(pretrained_model)
                if encoder_config is None
                else AutoModel.from_config(encoder_config, trust_remote_code=False)
            )
            hidden = self.encoder.config.hidden_size
            self.head_size = 64
            if flags["use_span"]:
                self.entity_projection = nn.Linear(
                    hidden, len(PAPER_ENTITY_TYPES) * self.head_size * 2
                )
            pooled_width = hidden * 2 if flags["use_mean_pooling"] else hidden
            self.classifiers = nn.ModuleDict(
                {
                    field: nn.Sequential(
                        nn.Linear(pooled_width, hidden),
                        nn.GELU(),
                        nn.Dropout(0.1),
                        nn.Linear(hidden, len(label_maps[field])),
                    )
                    for field in PAPER_FIELDS
                }
            )
            for field in PAPER_FIELDS:
                values = weights[field] if flags["use_class_weights"] else [1.0] * len(label_maps[field])
                self.register_buffer(
                    field + "_pos_weight", torch.tensor(values, dtype=torch.float32)
                )
            if flags["use_relation"]:
                self.relation_classifier = nn.Sequential(
                    nn.Linear(hidden * 4, hidden),
                    nn.GELU(),
                    nn.Dropout(0.1),
                    nn.Linear(hidden, max(1, len(label_maps["relations"]))),
                )

        def span_logits(self, hidden, attention_mask):
            batch, length, _ = hidden.shape
            projected = self.entity_projection(hidden).view(
                batch, length, len(PAPER_ENTITY_TYPES), 2, self.head_size
            )
            query, key = projected[..., 0, :], projected[..., 1, :]
            logits = torch.einsum("blth,bmth->btlm", query, key) / math.sqrt(
                self.head_size
            )
            valid = attention_mask[:, None, :, None] * attention_mask[:, None, None, :]
            triangle = torch.triu(
                torch.ones((length, length), device=hidden.device, dtype=torch.bool)
            )
            return logits.masked_fill(
                ~(valid.bool() & triangle[None, None]), -1e4
            )

        def forward(
            self,
            input_ids,
            attention_mask,
            token_type_ids=None,
            entity_labels=None,
            class_labels=None,
            relation_pairs=None,
            relation_labels=None,
        ):
            kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
            if token_type_ids is not None:
                kwargs["token_type_ids"] = token_type_ids
            hidden = self.encoder(**kwargs).last_hidden_state
            losses = {}
            entity_logits = None
            if self.flags["use_span"]:
                entity_logits = self.span_logits(hidden, attention_mask)
                if entity_labels is not None:
                    losses["span"] = nn.functional.binary_cross_entropy_with_logits(
                        entity_logits,
                        entity_labels,
                        pos_weight=torch.tensor(20.0, device=hidden.device),
                    )
            if self.flags["use_mean_pooling"]:
                mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
                mean_pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
                pooled = torch.cat([hidden[:, 0], mean_pooled], dim=-1)
            else:
                pooled = hidden[:, 0]
            class_logits = {
                field: head(pooled) for field, head in self.classifiers.items()
            }
            if class_labels is not None:
                for field, (targets, mask) in class_labels.items():
                    if mask.any():
                        losses[field] = nn.functional.binary_cross_entropy_with_logits(
                            class_logits[field][mask],
                            targets[mask],
                            pos_weight=getattr(self, field + "_pos_weight"),
                        )
            relation_logits = None
            if (
                self.flags["use_relation"]
                and relation_pairs is not None
                and relation_pairs.shape[1] > 0
            ):
                safe_pairs = relation_pairs.clamp(min=0)
                batch_ids = torch.arange(hidden.shape[0], device=hidden.device)[:, None]
                representations = [
                    hidden[batch_ids, safe_pairs[:, :, index]] for index in range(4)
                ]
                relation_logits = self.relation_classifier(
                    torch.cat(representations, dim=-1)
                )
                if relation_labels is not None and (relation_labels >= 0).any():
                    losses["relation"] = nn.functional.cross_entropy(
                        relation_logits.view(-1, relation_logits.shape[-1]),
                        relation_labels.view(-1),
                        ignore_index=-100,
                    )
            return {
                "loss": sum(losses.values()) if losses else None,
                "losses": losses,
                "class_logits": class_logits,
                "entity_logits": entity_logits,
                "relation_logits": relation_logits,
            }

    return AblationModel()


def validate_checkpoint_metadata(metadata: dict) -> None:
    """Reject incompatible label layouts before reconstructing a saved model."""
    if metadata.get("format_version") != 1:
        raise ValueError("Unsupported MacBERT-MGP checkpoint format version")
    if metadata.get("paper_fields") != list(PAPER_FIELDS):
        raise ValueError("Checkpoint must contain exactly the three paper classification heads")
    if metadata.get("entity_types") != list(PAPER_ENTITY_TYPES):
        raise ValueError("Checkpoint entity types do not match this implementation")
    if metadata.get("class_threshold") != CLASS_THRESHOLD:
        raise ValueError("Checkpoint classification threshold does not match this implementation")
    length = metadata.get("max_length")
    if type(length) is not int or length < 2:
        raise ValueError("Checkpoint max_length must be an integer of at least 2")
    label_maps = metadata.get("label_maps", {})
    if set(label_maps) != set(PAPER_FIELDS) | {"relations"}:
        raise ValueError("Checkpoint label-map fields do not match the paper model")
    for field, mapping in label_maps.items():
        if not isinstance(mapping, dict) or any(
            not isinstance(label, str) or type(index) is not int
            for label, index in mapping.items()
        ):
            raise ValueError(f"Invalid label map for {field}")
        if sorted(mapping.values()) != list(range(len(mapping))):
            raise ValueError(f"Label indices must be contiguous for {field}")
        if field in PAPER_FIELDS and not mapping:
            raise ValueError(f"Empty classification head: {field}")
    for field in PAPER_FIELDS:
        values = metadata.get("weights", {}).get(field, [])
        if len(values) != len(label_maps[field]) or any(
            not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0
            for value in values
        ):
            raise ValueError(f"Invalid class weights for {field}")
    for name in ("use_span", "use_relation", "use_class_weights", "use_mean_pooling"):
        if type(metadata.get("flags", {}).get(name)) is not bool:
            raise ValueError(f"Missing or invalid architecture flag: {name}")


def save_checkpoint(target, model, tokenizer, label_maps, weights, flags, *,
                    max_length, variant, seed, best_epoch, best_validation_loss):
    """Save tensor-only weights and local reconstruction assets, never a pickled model.

    The caller must first restore the best validation-loss state. A metadata file
    written last acts as the completion marker for the checkpoint directory.
    """
    from safetensors.torch import save_file

    target = Path(target)
    metadata = {
        "format_version": 1,
        "architecture": "MacBERT-MGP-paper",
        "paper_fields": list(PAPER_FIELDS),
        "entity_types": list(PAPER_ENTITY_TYPES),
        "class_threshold": CLASS_THRESHOLD,
        "label_maps": label_maps,
        "weights": weights,
        "flags": flags,
        "max_length": max_length,
        "variant": variant,
        "seed": seed,
        "best_epoch": best_epoch,
        "best_validation_loss": best_validation_loss,
        "checkpoint_selection": "minimum_validation_loss",
    }
    validate_checkpoint_metadata(metadata)
    target.mkdir(parents=True, exist_ok=True)
    completion_marker = target / "checkpoint.json"
    completion_marker.unlink(missing_ok=True)
    tokenizer.save_pretrained(str(target))
    model.encoder.config.save_pretrained(str(target))
    # Clone shared storage as well as moving to CPU: safetensors stores tensors,
    # not Python objects, and requires independent contiguous tensor buffers.
    state = {
        key: value.detach().cpu().clone().contiguous()
        for key, value in model.state_dict().items()
    }
    save_file(state, str(target / "model.safetensors"))
    write_json(completion_marker, metadata)
    return target


def evaluate_loss(model, loader, device, torch, mixed_precision):
    model.eval()
    total_loss = 0.0
    documents = 0
    with torch.no_grad():
        for batch in loader:
            batch_size = int(batch["input_ids"].shape[0])
            batch = move_to_device(batch, device, torch)
            amp = (
                torch.amp.autocast("cuda", dtype=torch.float16)
                if mixed_precision and device.type == "cuda"
                else nullcontext()
            )
            with amp:
                output = model(**batch)
            if output["loss"] is not None:
                total_loss += float(output["loss"].detach().cpu()) * batch_size
            documents += batch_size
    return total_loss / max(1, documents)


def predict(model, loader, rows, inverse_maps, device, torch, mixed_precision):
    model.eval()
    predictions = []
    offset = 0
    with torch.no_grad():
        for batch in loader:
            batch_size = int(batch["input_ids"].shape[0])
            batch_rows = rows[offset : offset + batch_size]
            offset += batch_size
            batch = move_to_device(batch, device, torch)
            amp = (
                torch.amp.autocast("cuda", dtype=torch.float16)
                if mixed_precision and device.type == "cuda"
                else nullcontext()
            )
            with amp:
                output = model(**batch)
            probabilities = {
                field: logits.sigmoid().detach().cpu().numpy()
                for field, logits in output["class_logits"].items()
            }
            for index, row in enumerate(batch_rows):
                labels = {}
                confidence = {}
                for field in PAPER_FIELDS:
                    values = probabilities[field][index]
                    best = int(values.argmax())
                    score = float(values[best])
                    labels[field] = (
                        inverse_maps[field][best]
                        if score >= CLASS_THRESHOLD
                        else None
                    )
                    confidence[field] = score
                predictions.append(
                    {
                        "document_id": str(row.get("document_id", row.get("work_order_id"))),
                        "predicted": labels,
                        "confidence": confidence,
                    }
                )
    return predictions


def prf(tp: int, fp: int, fn: int) -> dict:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support": tp + fn,
    }


def calculate_metrics(rows, predictions, train_support):
    pred_map = {row["document_id"]: row for row in predictions}
    counts = {field: Counter() for field in PAPER_FIELDS}
    class_counts = {field: defaultdict(Counter) for field in PAPER_FIELDS}
    exact = 0
    complete = 0
    total = 0
    for row in rows:
        document_id = str(row.get("document_id", row.get("work_order_id")))
        predicted = pred_map[document_id]["predicted"]
        gold_chains = row.get("chains", [])
        gold_tuples = {
            tuple(chain.get(field) for field in PAPER_FIELDS) for chain in gold_chains
        }
        if all(predicted.get(field) for field in PAPER_FIELDS):
            pred_tuples = {tuple(predicted[field] for field in PAPER_FIELDS)}
            complete += 1
        else:
            pred_tuples = set()
        exact += int(pred_tuples == gold_tuples)
        total += 1
        for field in PAPER_FIELDS:
            expected = {str(chain[field]) for chain in gold_chains if chain.get(field)}
            actual = {str(predicted[field])} if predicted.get(field) else set()
            counts[field]["tp"] += len(expected & actual)
            counts[field]["fp"] += len(actual - expected)
            counts[field]["fn"] += len(expected - actual)
            labels = expected | actual
            for label in labels:
                class_counts[field][label]["tp"] += int(label in expected and label in actual)
                class_counts[field][label]["fp"] += int(label not in expected and label in actual)
                class_counts[field][label]["fn"] += int(label in expected and label not in actual)
    metrics = {
        field: prf(counts[field]["tp"], counts[field]["fp"], counts[field]["fn"])
        for field in PAPER_FIELDS
    }
    metrics["mean_node_f1"] = float(
        np.mean([metrics[field]["f1"] for field in PAPER_FIELDS])
    )
    metrics["four_layer_exact_match"] = exact / max(1, total)
    metrics["complete_prediction_rate"] = complete / max(1, total)
    metrics["documents"] = total
    per_class = {}
    for field in PAPER_FIELDS:
        rows_out = []
        for label, value in sorted(class_counts[field].items()):
            result = prf(value["tp"], value["fp"], value["fn"])
            result.update(
                {
                    "label": label,
                    "train_support": int(train_support[field][label]),
                }
            )
            rows_out.append(result)
        per_class[field] = rows_out
        supported = [item["f1"] for item in rows_out if item["support"] > 0]
        metrics[field]["macro_f1"] = float(np.mean(supported)) if supported else 0.0
    return metrics, per_class


def support_counts(rows):
    result = {field: Counter() for field in PAPER_FIELDS}
    for row in rows:
        for field in PAPER_FIELDS:
            result[field].update(
                {
                    str(chain[field])
                    for chain in row.get("chains", [])
                    if chain.get(field)
                }
            )
    return result


def train_one(
    variant_name,
    flags,
    seed,
    args,
    data,
    tokenizer,
    label_maps,
    weights,
    torch,
    nn,
    device,
    output_dir,
):
    run_dir = output_dir / "runs" / variant_name / f"seed_{seed}"
    metrics_path = run_dir / "metrics.json"
    if metrics_path.exists() and not args.force:
        if args.save_checkpoints and not all(
            (run_dir / "checkpoint" / name).is_file()
            for name in ("checkpoint.json", "model.safetensors", "config.json", "tokenizer_config.json")
        ):
            raise RuntimeError(
                f"Completed run has no saved checkpoint: {run_dir}. "
                "Use --force --save-checkpoints to retrain and save it."
            )
        print(f"SKIP completed {variant_name} seed={seed}", flush=True)
        return json.loads(metrics_path.read_text(encoding="utf-8"))

    set_seed(torch, seed)
    collator = PaperCollator(tokenizer, label_maps, torch)
    datasets = {
        split: PaperDataset(rows, tokenizer, label_maps, args.max_length)
        for split, rows in data.items()
    }
    generator = torch.Generator()
    generator.manual_seed(seed)
    loaders = {
        "train": torch.utils.data.DataLoader(
            datasets["train"],
            batch_size=args.batch_size,
            shuffle=True,
            generator=generator,
            collate_fn=collator,
        ),
        **{
            split: torch.utils.data.DataLoader(
                datasets[split],
                batch_size=args.batch_size,
                shuffle=False,
                collate_fn=collator,
            )
            for split in ("validation", "test", "unseen")
        },
    }
    model = create_model(
        args.pretrained_model, label_maps, weights, flags, torch, nn
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=0.01
    )
    use_amp = bool(args.mixed_precision and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    best_loss = float("inf")
    best_epoch = 0
    best_state = None
    history = []
    started = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_total = 0.0
        documents = 0
        for batch in loaders["train"]:
            batch_size = int(batch["input_ids"].shape[0])
            batch = move_to_device(batch, device, torch)
            optimizer.zero_grad(set_to_none=True)
            amp = (
                torch.amp.autocast("cuda", dtype=torch.float16)
                if use_amp
                else nullcontext()
            )
            with amp:
                output = model(**batch)
            scaler.scale(output["loss"]).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            loss_total += float(output["loss"].detach().cpu()) * batch_size
            documents += batch_size
        train_loss = loss_total / max(1, documents)
        validation_loss = evaluate_loss(
            model, loaders["validation"], device, torch, use_amp
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_loss": validation_loss,
            }
        )
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
        print(
            f"{variant_name} seed={seed} epoch={epoch}/{args.epochs} "
            f"train={train_loss:.5f} val={validation_loss:.5f} best={best_epoch}",
            flush=True,
        )

    model.load_state_dict(best_state)
    checkpoint = None
    if args.save_checkpoints:
        checkpoint = save_checkpoint(
            run_dir / "checkpoint", model, tokenizer, label_maps, weights, flags,
            max_length=args.max_length, variant=variant_name, seed=seed,
            best_epoch=best_epoch, best_validation_loss=best_loss,
        )
    inverse_maps = {
        field: {index: label for label, index in label_maps[field].items()}
        for field in PAPER_FIELDS
    }
    train_support = support_counts(data["train"])
    split_metrics = {}
    for split in ("validation", "test", "unseen"):
        predictions = predict(
            model,
            loaders[split],
            data[split],
            inverse_maps,
            device,
            torch,
            use_amp,
        )
        metrics, per_class = calculate_metrics(
            data[split], predictions, train_support
        )
        split_metrics[split] = metrics
        write_jsonl(run_dir / f"{split}_predictions.jsonl", predictions)
        write_json(run_dir / f"{split}_per_class.json", per_class)

    result = {
        "variant": variant_name,
        "display_name": flags["display_name"],
        "flags": flags,
        "seed": seed,
        "best_epoch": best_epoch,
        "best_validation_loss": best_loss,
        "elapsed_seconds": time.perf_counter() - started,
        "trainable_parameters": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
        "history": history,
        "metrics": split_metrics,
    }
    if checkpoint is not None:
        result["checkpoint"] = str(checkpoint.relative_to(output_dir))
    write_json(metrics_path, result)
    del best_state, model, optimizer, scaler
    torch.cuda.empty_cache()
    return result


def main() -> None:
    args = parse_args()
    import torch
    from torch import nn
    from transformers import AutoTokenizer

    requested_variants = [value.strip() for value in args.variants.split(",") if value.strip()]
    unknown = set(requested_variants) - set(VARIANTS)
    if unknown:
        raise ValueError(f"Unknown variants: {sorted(unknown)}")
    seeds = [int(value.strip()) for value in args.seeds.split(",") if value.strip()]
    output_dir = resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data = {
        "train": [clean_annotated_row(row) for row in read_jsonl(str(resolve(args.train)))],
        "validation": [clean_annotated_row(row) for row in read_jsonl(str(resolve(args.validation)))],
        "test": [clean_annotated_row(row) for row in read_jsonl(str(resolve(args.test)))],
        "unseen": build_unseen_rows(resolve(args.work_orders), resolve(args.unseen_labels)),
    }
    if {key: len(value) for key, value in data.items()} != {
        "train": 800,
        "validation": 100,
        "test": 100,
        "unseen": 727,
    }:
        raise RuntimeError(
            f"Unexpected split sizes: { {key: len(value) for key, value in data.items()} }"
        )

    label_maps = build_label_maps(data["train"])
    weights = class_weights(data["train"], label_maps)
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model, use_fast=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA GPU is required for the full ablation run")

    experiment_config = {
        "description": "Controlled four-layer MacBERT-MGP module ablation",
        "paper_path": "engine root -> subsystem -> component -> failure mode",
        "discarded_supervision": ["cause_id", "CAUSE spans", "CAUSED_BY relations"],
        "splits": {key: len(value) for key, value in data.items()},
        "pretrained_model": args.pretrained_model,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "max_length": args.max_length,
        "class_threshold": CLASS_THRESHOLD,
        "seeds": seeds,
        "variants": {name: VARIANTS[name] for name in requested_variants},
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "mixed_precision": bool(args.mixed_precision),
        "save_checkpoints": bool(args.save_checkpoints),
        "label_counts": {field: len(label_maps[field]) for field in PAPER_FIELDS},
        "entity_types": list(PAPER_ENTITY_TYPES),
        "relation_types": sorted(label_maps["relations"]),
    }
    write_json(output_dir / "experiment_config.json", experiment_config)
    write_json(output_dir / "label_maps.json", label_maps)

    all_results = []
    for variant_name in requested_variants:
        for seed in seeds:
            all_results.append(
                train_one(
                    variant_name,
                    VARIANTS[variant_name],
                    seed,
                    args,
                    data,
                    tokenizer,
                    label_maps,
                    weights,
                    torch,
                    nn,
                    device,
                    output_dir,
                )
            )
            write_json(output_dir / "all_run_results.json", all_results)
    print(json.dumps({"output_dir": str(output_dir), "runs": len(all_results)}, indent=2))


if __name__ == "__main__":
    main()
