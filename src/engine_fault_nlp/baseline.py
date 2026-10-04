from __future__ import annotations

import hashlib
import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .io_utils import read_json, write_json
from .text import character_ngrams


SparseVector = Dict[int, float]


class HashingTfidfVectorizer:
    def __init__(self, dimensions: int = 65536, ngram_sizes: Sequence[int] = (2, 3, 4, 5)):
        self.dimensions = dimensions
        self.ngram_sizes = tuple(ngram_sizes)
        self.idf: Dict[int, float] = {}
        self.document_count = 0

    def _index(self, token: str) -> int:
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        return int.from_bytes(digest, "big") % self.dimensions

    def fit(self, documents: Iterable[str]) -> "HashingTfidfVectorizer":
        df: Counter = Counter()
        count = 0
        for document in documents:
            count += 1
            df.update({self._index(token) for token in character_ngrams(document, self.ngram_sizes)})
        self.document_count = count
        self.idf = {idx: math.log((count + 1) / (freq + 1)) + 1.0 for idx, freq in df.items()}
        return self

    def transform_one(self, document: str) -> SparseVector:
        counts = Counter(self._index(token) for token in character_ngrams(document, self.ngram_sizes))
        vector = {idx: (1.0 + math.log(freq)) * self.idf.get(idx, 1.0) for idx, freq in counts.items()}
        norm = math.sqrt(sum(value * value for value in vector.values())) or 1.0
        return {idx: value / norm for idx, value in vector.items()}

    def to_dict(self) -> Dict[str, object]:
        return {
            "dimensions": self.dimensions,
            "ngram_sizes": list(self.ngram_sizes),
            "idf": {str(key): value for key, value in self.idf.items()},
            "document_count": self.document_count,
        }

    @classmethod
    def from_dict(cls, value: Dict[str, object]) -> "HashingTfidfVectorizer":
        instance = cls(int(value["dimensions"]), value["ngram_sizes"])
        instance.idf = {int(key): float(score) for key, score in value.get("idf", {}).items()}
        instance.document_count = int(value.get("document_count", 0))
        return instance


def cosine(left: SparseVector, right: SparseVector) -> float:
    if len(left) > len(right):
        left, right = right, left
    return sum(value * right.get(idx, 0.0) for idx, value in left.items())


@dataclass
class CentroidHead:
    labels: List[str] = field(default_factory=list)
    centroids: Dict[str, SparseVector] = field(default_factory=dict)
    priors: Dict[str, float] = field(default_factory=dict)

    def fit(self, vectors: List[SparseVector], targets: List[List[str]]) -> "CentroidHead":
        sums: Dict[str, Counter] = defaultdict(Counter)
        counts: Counter = Counter()
        for vector, labels in zip(vectors, targets):
            for label in set(label for label in labels if label):
                sums[label].update(vector)
                counts[label] += 1
        self.labels = sorted(counts)
        total = sum(counts.values()) or 1
        for label in self.labels:
            centroid = {idx: value / counts[label] for idx, value in sums[label].items()}
            norm = math.sqrt(sum(value * value for value in centroid.values())) or 1.0
            self.centroids[label] = {idx: value / norm for idx, value in centroid.items()}
            self.priors[label] = counts[label] / total
        return self

    def predict(self, vector: SparseVector, top_k: int = 5) -> List[Tuple[str, float]]:
        scores = []
        for label in self.labels:
            similarity = cosine(vector, self.centroids[label])
            prior = math.log(self.priors.get(label, 1e-9) + 1e-9)
            logit = 7.0 * similarity + 0.15 * prior
            scores.append((label, logit))
        if not scores:
            return []
        max_logit = max(score for _, score in scores)
        probabilities = [(label, math.exp(score - max_logit)) for label, score in scores]
        denominator = sum(score for _, score in probabilities) or 1.0
        ranked = sorted(((label, score / denominator) for label, score in probabilities), key=lambda pair: pair[1], reverse=True)
        return ranked[:top_k]

    def to_dict(self) -> Dict[str, object]:
        return {
            "labels": self.labels,
            "centroids": {label: {str(i): v for i, v in vec.items()} for label, vec in self.centroids.items()},
            "priors": self.priors,
        }

    @classmethod
    def from_dict(cls, value: Dict[str, object]) -> "CentroidHead":
        return cls(
            labels=list(value.get("labels", [])),
            centroids={label: {int(i): float(v) for i, v in vec.items()} for label, vec in value.get("centroids", {}).items()},
            priors={label: float(v) for label, v in value.get("priors", {}).items()},
        )


class BaselineModel:
    FIELDS = ("subsystem_id", "component_id", "failure_mode_id", "cause_id")

    def __init__(self, vectorizer: Optional[HashingTfidfVectorizer] = None):
        self.vectorizer = vectorizer or HashingTfidfVectorizer()
        self.heads = {field: CentroidHead() for field in self.FIELDS}

    def fit(self, documents: List[str], labels: List[List[Dict[str, object]]]) -> "BaselineModel":
        self.vectorizer.fit(documents)
        vectors = [self.vectorizer.transform_one(document) for document in documents]
        for field in self.FIELDS:
            targets = []
            for chains in labels:
                targets.append([str(chain[field]) for chain in chains if chain.get(field)])
            self.heads[field].fit(vectors, targets)
        return self

    def predict(self, document: str, top_k: int = 3) -> Dict[str, List[Tuple[str, float]]]:
        vector = self.vectorizer.transform_one(document)
        return {field: head.predict(vector, top_k) for field, head in self.heads.items()}

    def save(self, path: str) -> None:
        write_json(path, {"vectorizer": self.vectorizer.to_dict(), "heads": {k: v.to_dict() for k, v in self.heads.items()}})

    @classmethod
    def load(cls, path: str) -> "BaselineModel":
        raw = read_json(path)
        model = cls(HashingTfidfVectorizer.from_dict(raw["vectorizer"]))
        model.heads = {field: CentroidHead.from_dict(value) for field, value in raw["heads"].items()}
        return model
