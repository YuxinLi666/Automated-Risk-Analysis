"""Optional multi-task Transformer + GlobalPointer implementation.

This module is imported only by the Transformer CLI commands. The core
pipeline remains usable without PyTorch or Hugging Face Transformers.
"""

from __future__ import annotations

from collections.abc import Mapping
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .io_utils import read_jsonl, write_json


ENTITY_TYPES = ["COMPONENT", "FAILURE_MODE", "CAUSE", "SYMPTOM", "DTC"]
CLASSIFICATION_FIELDS = ["subsystem_id", "component_id", "failure_mode_id", "cause_id"]


def _require_ml():
    try:
        import torch
        from torch import nn
        from transformers import AutoConfig, AutoModel, AutoTokenizer
    except ImportError as error:
        raise RuntimeError(
            "Transformer dependencies missing. Use Python 3.10+ and install: pip install -e '.[ml]'"
        ) from error
    return torch, nn, AutoConfig, AutoModel, AutoTokenizer


def _char_span_to_tokens(offsets: List[Tuple[int, int]], start: int, end: int) -> Optional[Tuple[int, int]]:
    token_start = next((i for i, (left, right) in enumerate(offsets) if left <= start < right), None)
    token_end = next((i for i, (left, right) in reversed(list(enumerate(offsets))) if left < end <= right), None)
    if token_start is None or token_end is None or token_end < token_start:
        return None
    return token_start, token_end


class AnnotationDataset:
    def __init__(self, rows, tokenizer, label_maps, max_length=512):
        self.rows = rows
        self.tokenizer = tokenizer
        self.label_maps = label_maps
        self.max_length = max_length

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
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
            if entity.get("label") not in ENTITY_TYPES:
                continue
            token_span = _char_span_to_tokens(offsets, int(entity["start"]), int(entity["end"]))
            if token_span:
                label_index = ENTITY_TYPES.index(entity["label"])
                spans.append((label_index, token_span[0], token_span[1]))
                token_entities[entity["entity_id"]] = token_span
        classifications = {}
        chains = row.get("chains", row.get("fault_chains", []))
        for field in CLASSIFICATION_FIELDS:
            classifications[field] = [
                self.label_maps[field][str(chain[field])]
                for chain in chains
                if chain.get(field) and str(chain[field]) in self.label_maps[field]
            ]
        relation_pairs = []
        relation_map = self.label_maps["relations"]
        for relation in row.get("relations", []):
            head = token_entities.get(relation.get("head_id"))
            tail = token_entities.get(relation.get("tail_id"))
            label = relation_map.get(str(relation.get("relation")))
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


class BatchCollator:
    def __init__(self, tokenizer, label_maps):
        self.tokenizer = tokenizer
        self.label_maps = label_maps

    def __call__(self, items):
        torch, _, _, _, _ = _require_ml()
        padded = self.tokenizer.pad(
            [{k: v for k, v in item.items() if k in ("input_ids", "attention_mask", "token_type_ids") and v is not None} for item in items],
            return_tensors="pt",
        )
        batch_size, length = padded["input_ids"].shape
        entity_labels = torch.zeros((batch_size, len(ENTITY_TYPES), length, length), dtype=torch.float32)
        for batch_index, item in enumerate(items):
            for label, start, end in item["spans"]:
                if start < length and end < length:
                    entity_labels[batch_index, label, start, end] = 1.0
        class_labels = {}
        for field in CLASSIFICATION_FIELDS:
            width = len(self.label_maps[field])
            target = torch.zeros((batch_size, width), dtype=torch.float32)
            mask = torch.zeros(batch_size, dtype=torch.bool)
            for batch_index, item in enumerate(items):
                for label in item["classifications"][field]:
                    target[batch_index, label] = 1.0
                    mask[batch_index] = True
            class_labels[field] = (target, mask)
        max_pairs = max((len(item["relation_pairs"]) for item in items), default=0)
        pair_tensor = torch.full((batch_size, max_pairs, 4), -1, dtype=torch.long)
        relation_labels = torch.full((batch_size, max_pairs), -100, dtype=torch.long)
        for batch_index, item in enumerate(items):
            for pair_index, pair in enumerate(item["relation_pairs"]):
                pair_tensor[batch_index, pair_index] = torch.tensor(pair[:4])
                relation_labels[batch_index, pair_index] = pair[4]
        padded.update({
            "entity_labels": entity_labels,
            "class_labels": class_labels,
            "relation_pairs": pair_tensor,
            "relation_labels": relation_labels,
        })
        return padded


def build_label_maps(rows) -> Dict[str, Dict[str, int]]:
    values = {field: set() for field in CLASSIFICATION_FIELDS}
    relations = set()
    for row in rows:
        for chain in row.get("chains", row.get("fault_chains", [])):
            for field in CLASSIFICATION_FIELDS:
                if chain.get(field):
                    values[field].add(str(chain[field]))
        relations.update(str(relation["relation"]) for relation in row.get("relations", []) if relation.get("relation"))
    maps = {field: {label: index for index, label in enumerate(sorted(labels))} for field, labels in values.items()}
    maps["relations"] = {label: index for index, label in enumerate(sorted(relations))}
    maps["_class_pos_weights"] = build_class_pos_weights(rows, maps)
    return maps


def build_class_pos_weights(rows, label_maps) -> Dict[str, List[float]]:
    """Compute balanced positive weights for multi-label classification heads.

    The component and subsystem labels are highly imbalanced in service work orders.
    Without positive weighting, BCE is dominated by negative labels and the model
    learns frequent/common labels only. We use a square-root inverse-frequency
    weight, clipped to keep training stable on noisy silver labels.
    """
    weights: Dict[str, List[float]] = {}
    total = max(1, len(rows))
    for field in CLASSIFICATION_FIELDS:
        counts = [0 for _ in range(len(label_maps.get(field, {})))]
        for row in rows:
            present = {
                str(chain[field])
                for chain in row.get("chains", row.get("fault_chains", []))
                if chain.get(field) and str(chain[field]) in label_maps.get(field, {})
            }
            for label in present:
                counts[label_maps[field][label]] += 1
        field_weights = []
        for positive in counts:
            if positive <= 0:
                field_weights.append(1.0)
                continue
            negative = max(1, total - positive)
            field_weights.append(min(20.0, max(1.0, math.sqrt(negative / positive))))
        weights[field] = field_weights
    return weights


def create_model(pretrained_model: str, label_maps, head_size: int = 64, config_dir: Optional[str] = None):
    torch, nn, AutoConfig, AutoModel, _ = _require_ml()

    class MultiTaskGlobalPointer(nn.Module):
        def __init__(self):
            super().__init__()
            if config_dir:
                self.encoder = AutoModel.from_config(AutoConfig.from_pretrained(config_dir))
            else:
                self.encoder = AutoModel.from_pretrained(pretrained_model)
            hidden = self.encoder.config.hidden_size
            self.head_size = head_size
            self.entity_projection = nn.Linear(hidden, len(ENTITY_TYPES) * head_size * 2)
            pooled_width = hidden * 2
            self.classifiers = nn.ModuleDict({
                field: nn.Sequential(
                    nn.Linear(pooled_width, hidden),
                    nn.GELU(),
                    nn.Dropout(0.1),
                    nn.Linear(hidden, max(1, len(label_maps[field]))),
                )
                for field in CLASSIFICATION_FIELDS
            })
            for field in CLASSIFICATION_FIELDS:
                weights = label_maps.get("_class_pos_weights", {}).get(field)
                width = max(1, len(label_maps[field]))
                if not weights or len(weights) != width:
                    weights = [1.0] * width
                self.register_buffer(field + "_pos_weight", torch.tensor(weights, dtype=torch.float32))
            relation_count = max(1, len(label_maps["relations"]))
            self.relation_classifier = nn.Sequential(
                nn.Linear(hidden * 4, hidden), nn.GELU(), nn.Dropout(0.1), nn.Linear(hidden, relation_count)
            )

        def entity_logits(self, hidden, attention_mask):
            batch, length, _ = hidden.shape
            projected = self.entity_projection(hidden).view(batch, length, len(ENTITY_TYPES), 2, self.head_size)
            query, key = projected[..., 0, :], projected[..., 1, :]
            logits = torch.einsum("blth,bmth->btlm", query, key) / math.sqrt(self.head_size)
            valid = attention_mask[:, None, :, None] * attention_mask[:, None, None, :]
            triangle = torch.triu(torch.ones((length, length), device=hidden.device, dtype=torch.bool))
            return logits.masked_fill(~(valid.bool() & triangle[None, None]), -1e4)

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
            entity_logits = self.entity_logits(hidden, attention_mask)
            mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
            mean_pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
            pooled = torch.cat([hidden[:, 0], mean_pooled], dim=-1)
            class_logits = {field: head(pooled) for field, head in self.classifiers.items()}
            total_loss = None
            losses = {}
            if entity_labels is not None:
                positive_weight = torch.tensor(20.0, device=hidden.device)
                losses["entity"] = nn.functional.binary_cross_entropy_with_logits(
                    entity_logits, entity_labels, pos_weight=positive_weight
                )
            if class_labels is not None:
                for field, (targets, mask) in class_labels.items():
                    if mask.any() and targets.shape[1] == class_logits[field].shape[1]:
                        losses[field] = nn.functional.binary_cross_entropy_with_logits(
                            class_logits[field][mask],
                            targets[mask],
                            pos_weight=getattr(self, field + "_pos_weight").to(hidden.device),
                        )
            relation_logits = None
            if relation_pairs is not None and relation_pairs.shape[1] > 0:
                safe_pairs = relation_pairs.clamp(min=0)
                batch_ids = torch.arange(hidden.shape[0], device=hidden.device)[:, None]
                representations = [hidden[batch_ids, safe_pairs[:, :, idx]] for idx in range(4)]
                relation_logits = self.relation_classifier(torch.cat(representations, dim=-1))
                if relation_labels is not None and (relation_labels >= 0).any():
                    losses["relation"] = nn.functional.cross_entropy(
                        relation_logits.view(-1, relation_logits.shape[-1]), relation_labels.view(-1), ignore_index=-100
                    )
            if losses:
                total_loss = sum(losses.values())
            return {
                "loss": total_loss,
                "losses": losses,
                "entity_logits": entity_logits,
                "class_logits": class_logits,
                "relation_logits": relation_logits,
            }

    return MultiTaskGlobalPointer()


def train_from_jsonl(
    annotations_path: str,
    output_dir: str,
    pretrained_model: str,
    epochs: int = 8,
    batch_size: int = 8,
    learning_rate: float = 2e-5,
    max_length: int = 512,
) -> None:
    torch, _, _, _, AutoTokenizer = _require_ml()
    rows = list(read_jsonl(annotations_path))
    rows = [row for row in rows if row.get("entities") or row.get("chains") or row.get("fault_chains")]
    if not rows:
        raise ValueError("No annotated rows found. Complete the annotation JSONL before training.")
    label_maps = build_label_maps(rows)
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model, use_fast=True)
    dataset = AnnotationDataset(rows, tokenizer, label_maps, max_length)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, collate_fn=BatchCollator(tokenizer, label_maps)
    )
    model = create_model(pretrained_model, label_maps)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch in loader:
            batch = _move_to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            output = model(**batch)
            if output["loss"] is None:
                continue
            output["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += float(output["loss"].detach().cpu())
        print("epoch {}/{} loss={:.6f}".format(epoch + 1, epochs, epoch_loss / max(1, len(loader))))
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(str(target))
    model.encoder.config.save_pretrained(str(target))
    torch.save(model.state_dict(), target / "model.pt")
    write_json(str(target / "label_maps.json"), label_maps)
    write_json(str(target / "training_config.json"), {
        "pretrained_model": pretrained_model,
        "entity_types": ENTITY_TYPES,
        "max_length": max_length,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
    })


def _move_to_device(value, device):
    torch, _, _, _, _ = _require_ml()
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, Mapping):
        return {key: _move_to_device(item, device) for key, item in value.items()}
    if hasattr(value, "items"):
        return {key: _move_to_device(item, device) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_move_to_device(item, device) for item in value)
    if isinstance(value, list):
        return [_move_to_device(item, device) for item in value]
    return value


class TransformerExtractor:
    def __init__(self, model_dir: str):
        torch, _, _, _, AutoTokenizer = _require_ml()
        root = Path(model_dir)
        with (root / "label_maps.json").open("r", encoding="utf-8") as handle:
            self.label_maps = json.load(handle)
        with (root / "training_config.json").open("r", encoding="utf-8") as handle:
            self.config = json.load(handle)
        self.tokenizer = AutoTokenizer.from_pretrained(str(root), use_fast=True)
        self.model = create_model(self.config["pretrained_model"], self.label_maps, config_dir=str(root))
        self.model.load_state_dict(torch.load(root / "model.pt", map_location="cpu"))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device).eval()
        self.inverse_maps = {
            field: {index: label for label, index in self.label_maps[field].items()}
            for field in CLASSIFICATION_FIELDS
        }

    def predict(self, text: str, entity_threshold: float = 0.5, class_threshold: float = 0.5):
        torch, _, _, _, _ = _require_ml()
        encoded = self.tokenizer(
            text,
            truncation=True,
            max_length=self.config.get("max_length", 512),
            return_offsets_mapping=True,
            return_tensors="pt",
        )
        offsets = encoded.pop("offset_mapping")[0].tolist()
        batch = {key: value.to(self.device) for key, value in encoded.items()}
        with torch.no_grad():
            output = self.model(**batch)
        probabilities = output["entity_logits"].sigmoid()[0].cpu()
        entities = []
        for label_idx, start, end in (probabilities >= entity_threshold).nonzero(as_tuple=False).tolist():
            char_start, char_end = offsets[start][0], offsets[end][1]
            if char_end > char_start:
                entities.append({
                    "label": ENTITY_TYPES[label_idx],
                    "start": char_start,
                    "end": char_end,
                    "text": text[char_start:char_end],
                    "confidence": float(probabilities[label_idx, start, end]),
                })
        classifications = {}
        for field, logits in output["class_logits"].items():
            probs = logits.sigmoid()[0].cpu().tolist()
            classifications[field] = sorted(
                [
                    {"label": self.inverse_maps[field][idx], "confidence": score}
                    for idx, score in enumerate(probs)
                    if score >= class_threshold and idx in self.inverse_maps[field]
                ],
                key=lambda item: item["confidence"],
                reverse=True,
            )
        return {"entities": entities, "classifications": classifications}
