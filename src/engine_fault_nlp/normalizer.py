from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .ontology import ComponentNode, KnowledgeBase
from .text import character_ngrams, normalize_text


@dataclass
class NormalizationCandidate:
    component: ComponentNode
    score: float
    source: str


class ComponentNormalizer:
    def __init__(self, kb: KnowledgeBase):
        self.kb = kb
        self.alias_index: Dict[str, List[ComponentNode]] = {}
        self.documents: Dict[str, Counter] = {}
        self.document_frequency: Counter = Counter()
        for component in kb.components.values():
            aliases = set(component.aliases + component.part_numbers + [component.name])
            for alias in aliases:
                normalized = normalize_text(alias).lower()
                if normalized:
                    values = self.alias_index.setdefault(normalized, [])
                    if all(value.id != component.id for value in values):
                        values.append(component)
            grams = Counter(character_ngrams(" ".join(aliases), (2, 3)))
            self.documents[component.id] = grams
            self.document_frequency.update(grams.keys())

    def retrieve(self, mention: str, top_k: int = 5) -> List[NormalizationCandidate]:
        query = normalize_text(mention).lower()
        if not query:
            return []
        if query in self.alias_index:
            return [
                NormalizationCandidate(component, 1.0, "exact")
                for component in sorted(self.alias_index[query], key=lambda value: value.id)
            ][:top_k]
        query_grams = Counter(character_ngrams(query, (2, 3)))
        if not query_grams:
            return []
        total_docs = max(1, len(self.documents))
        scored: List[Tuple[float, ComponentNode]] = []
        for component_id, document in self.documents.items():
            score = 0.0
            for gram, qtf in query_grams.items():
                if gram not in document:
                    continue
                idf = math.log((total_docs + 1) / (self.document_frequency[gram] + 1)) + 1
                score += min(qtf, document[gram]) * idf
            denom = math.sqrt(sum(query_grams.values()) * max(1, sum(document.values())))
            if score > 0:
                scored.append((score / denom, self.kb.components[component_id]))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [NormalizationCandidate(component, min(0.95, score), "char_bm25") for score, component in scored[:top_k]]

    def best(self, mention: str, minimum_score: float = 0.15) -> Optional[NormalizationCandidate]:
        candidates = self.retrieve(mention, top_k=1)
        return candidates[0] if candidates and candidates[0].score >= minimum_score else None

    def best_unique(
        self,
        mention: str,
        minimum_score: float = 0.15,
        minimum_margin: float = 0.05,
    ) -> Optional[NormalizationCandidate]:
        """Return a match only when the best canonical component is unambiguous."""

        candidates = self.retrieve(mention, top_k=2)
        if not candidates or candidates[0].score < minimum_score:
            return None
        if len(candidates) > 1 and candidates[0].score - candidates[1].score < minimum_margin:
            return None
        return candidates[0]
