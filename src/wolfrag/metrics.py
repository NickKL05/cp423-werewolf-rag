"""Retrieval and generation metrics.

Retrieval uses binary relevance: a retrieved chunk is relevant when it appears
in the question's gold chunk set. Because multi-hop questions have more than one
gold chunk, rank-aware measures that reward finding all of them (MAP, nDCG)
matter as much as the top-heavy ones (Precision@1, MRR).

Everything is also computed at document level. Chunk level answers "did we find
the right passage", document level answers "did we find the right page", and the
gap between them is informative when a page is split across several chunks.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Iterable, Sequence

# ---------------------------------------------------------------------------
# Retrieval metrics
# ---------------------------------------------------------------------------


def precision_at_k(ranked: Sequence[str], gold: Iterable[str], k: int) -> float:
    gold_set = set(gold)
    if k <= 0:
        return 0.0
    top = ranked[:k]
    if not top:
        return 0.0
    return sum(1 for item in top if item in gold_set) / k


def recall_at_k(ranked: Sequence[str], gold: Iterable[str], k: int) -> float:
    gold_set = set(gold)
    if not gold_set:
        return 0.0
    top = set(ranked[:k])
    return len(top & gold_set) / len(gold_set)


def average_precision(ranked: Sequence[str], gold: Iterable[str]) -> float:
    """Average precision over the full ranked list."""
    gold_set = set(gold)
    if not gold_set:
        return 0.0
    hits = 0
    precision_sum = 0.0
    for index, item in enumerate(ranked, start=1):
        if item in gold_set:
            hits += 1
            precision_sum += hits / index
    return precision_sum / len(gold_set)


def reciprocal_rank(ranked: Sequence[str], gold: Iterable[str]) -> float:
    gold_set = set(gold)
    for index, item in enumerate(ranked, start=1):
        if item in gold_set:
            return 1.0 / index
    return 0.0


def ndcg_at_k(ranked: Sequence[str], gold: Iterable[str], k: int) -> float:
    """Normalised discounted cumulative gain with binary gains."""
    gold_set = set(gold)
    if not gold_set:
        return 0.0
    dcg = 0.0
    for index, item in enumerate(ranked[:k], start=1):
        if item in gold_set:
            dcg += 1.0 / math.log2(index + 1)
    ideal_hits = min(len(gold_set), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0


# ---------------------------------------------------------------------------
# Generation metrics
# ---------------------------------------------------------------------------

_ARTICLES_RE = re.compile(r"\b(a|an|the)\b", re.IGNORECASE)
_PUNCT_RE = re.compile(r"[^\w\s]")
_WS_RE = re.compile(r"\s+")


def normalise_answer(text: str) -> str:
    """SQuAD style normalisation: lowercase, drop articles and punctuation."""
    text = text.lower()
    text = _PUNCT_RE.sub(" ", text)
    text = _ARTICLES_RE.sub(" ", text)
    return _WS_RE.sub(" ", text).strip()


def strip_citations(text: str) -> str:
    """Remove inline [C00123] markers before scoring answer text."""
    return re.sub(r"\[C\d{5}\]", "", text)


def token_f1(prediction: str, reference: str) -> float:
    """Token overlap F1 between a generated answer and its reference."""
    pred_tokens = normalise_answer(strip_citations(prediction)).split()
    ref_tokens = normalise_answer(reference).split()
    if not pred_tokens or not ref_tokens:
        return float(pred_tokens == ref_tokens)

    common = Counter(pred_tokens) & Counter(ref_tokens)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def exact_match(prediction: str, reference: str) -> float:
    return float(
        normalise_answer(strip_citations(prediction)) == normalise_answer(reference)
    )


def rouge_l(prediction: str, reference: str) -> float:
    """ROUGE-L F-measure, computed with the rouge_score package."""
    from rouge_score import rouge_scorer

    global _ROUGE_SCORER
    try:
        scorer = _ROUGE_SCORER
    except NameError:
        scorer = None
    if scorer is None:
        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        _ROUGE_SCORER = scorer

    scores = scorer.score(reference, strip_citations(prediction))
    return scores["rougeL"].fmeasure


_ROUGE_SCORER = None


# ---------------------------------------------------------------------------
# Citation metrics
# ---------------------------------------------------------------------------


def citation_precision(cited: Sequence[str], gold: Iterable[str]) -> float:
    """Share of cited chunks that are actually gold chunks for the question."""
    gold_set = set(gold)
    if not cited:
        return 0.0
    return sum(1 for c in cited if c in gold_set) / len(cited)


def citation_recall(cited: Sequence[str], gold: Iterable[str]) -> float:
    gold_set = set(gold)
    if not gold_set:
        return 0.0
    return len(set(cited) & gold_set) / len(gold_set)


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0
