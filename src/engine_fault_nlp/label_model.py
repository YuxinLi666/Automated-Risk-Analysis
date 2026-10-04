from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional


@dataclass
class GenerativeLabelModel:
    """Small Snorkel-style conditional-independence label model.

    Each training row is a mapping of labeling-function name to a class label;
    missing or empty values are abstentions. EM estimates LF accuracies and
    returns a posterior without requiring gold labels.
    """

    classes: List[str] = field(default_factory=list)
    accuracies: Dict[str, float] = field(default_factory=dict)
    priors: Dict[str, float] = field(default_factory=dict)

    def fit(self, rows: Iterable[Dict[str, str]], iterations: int = 30) -> "GenerativeLabelModel":
        rows = [{key: value for key, value in row.items() if value} for row in rows]
        self.classes = sorted({value for row in rows for value in row.values()})
        functions = sorted({key for row in rows for key in row})
        if not self.classes:
            return self
        self.accuracies = {name: 0.70 for name in functions}
        self.priors = {label: 1.0 / len(self.classes) for label in self.classes}
        for _ in range(iterations):
            posteriors = [self.predict_proba(row) for row in rows]
            prior_counts = Counter()
            accuracy_numerators = Counter()
            accuracy_denominators = Counter()
            for row, posterior in zip(rows, posteriors):
                prior_counts.update(posterior)
                for function, vote in row.items():
                    accuracy_numerators[function] += posterior.get(vote, 0.0)
                    accuracy_denominators[function] += 1.0
            total = sum(prior_counts.values()) or 1.0
            self.priors = {label: max(1e-6, prior_counts[label] / total) for label in self.classes}
            self.accuracies = {
                function: min(0.995, max(0.501, accuracy_numerators[function] / max(1.0, accuracy_denominators[function])))
                for function in functions
            }
        return self

    def predict_proba(self, votes: Dict[str, str]) -> Dict[str, float]:
        if not self.classes:
            return {}
        class_count = len(self.classes)
        logits = {}
        for label in self.classes:
            score = math.log(self.priors.get(label, 1.0 / class_count) + 1e-12)
            for function, vote in votes.items():
                if not vote or function not in self.accuracies:
                    continue
                accuracy = self.accuracies[function]
                score += math.log(accuracy + 1e-12) if vote == label else math.log((1 - accuracy) / max(1, class_count - 1) + 1e-12)
            logits[label] = score
        peak = max(logits.values())
        probabilities = {label: math.exp(score - peak) for label, score in logits.items()}
        denominator = sum(probabilities.values()) or 1.0
        return {label: score / denominator for label, score in probabilities.items()}

    def predict(self, votes: Dict[str, str]) -> Optional[Dict[str, object]]:
        probabilities = self.predict_proba(votes)
        if not probabilities:
            return None
        label = max(probabilities, key=probabilities.get)
        return {"label": label, "confidence": probabilities[label], "probabilities": probabilities}


def fuse_vote_records(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    by_field = {}
    for row in rows:
        by_field.setdefault(str(row["field"]), []).append(row)
    output = []
    for field, items in by_field.items():
        model = GenerativeLabelModel().fit([item.get("votes", {}) for item in items])
        for item in items:
            prediction = model.predict(item.get("votes", {}))
            output.append({
                "work_order_id": item["work_order_id"],
                "field": field,
                "prediction": prediction,
                "lf_accuracies": model.accuracies,
            })
    return output

