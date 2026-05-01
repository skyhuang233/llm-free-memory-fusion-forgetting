"""Deterministic F1 partial-match scorer following LoCoMo Section 4.1 protocol.
Replicates the exact scoring from external/locomo/task_eval/evaluation.py:
- normalize_answer: lowercase, remove commas, remove articles (a/an/the/and),
  remove punctuation, whitespace fix
- f1_score: Porter stemming on both, token-level precision/recall/F1
- Multi-hop (cat=1): split on commas, partial F1 per sub-answer
- Temporal/open-domain/single-hop (cat=2,3,4): direct f1_score
- Adversarial (cat=5): keyword match for "no information available"/"not mentioned"
"""

import string
from collections import Counter

import numpy as np
from nltk.stem import PorterStemmer

_ps = PorterStemmer()

CATEGORY_NAMES = {
    1: "multi-hop",
    2: "temporal",
    3: "open-domain",
    4: "single-hop",
    5: "adversarial",
}


def normalize_answer(s: str) -> str:
    s = s.replace(",", "")
    s = s.lower()
    exclude = set(string.punctuation)
    s = "".join(ch for ch in s if ch not in exclude)
    import re
    s = re.sub(r"\b(a|an|the|and)\b", " ", s)
    s = " ".join(s.split())
    return s


def _f1_score_single(prediction: str, ground_truth: str) -> float:
    pred_tokens = [_ps.stem(w) for w in normalize_answer(prediction).split()]
    gt_tokens = [_ps.stem(w) for w in normalize_answer(ground_truth).split()]
    common = Counter(pred_tokens) & Counter(gt_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gt_tokens)
    return (2 * precision * recall) / (precision + recall)


def _f1_multi_answer(prediction: str, ground_truth: str) -> float:
    predictions = [p.strip() for p in prediction.split(",")]
    ground_truths = [g.strip() for g in ground_truth.split(",")]
    return float(np.mean([
        max(_f1_score_single(p, gt) for p in predictions)
        for gt in ground_truths
    ]))


def score_qa(prediction: str, answer: str, category: int) -> float:
    if category == 5:
        lower = prediction.lower()
        if "no information available" in lower or "not mentioned" in lower:
            return 1.0
        return 0.0
    if category == 1:
        return _f1_multi_answer(prediction, answer)
    return _f1_score_single(prediction, answer)


def aggregate_scores(results: list[dict]) -> dict:
    by_cat: dict[int, list[float]] = {}
    all_scores = []
    for r in results:
        cat = r["category"]
        s = r["f1"]
        by_cat.setdefault(cat, []).append(s)
        all_scores.append(s)

    agg = {"overall_f1": float(np.mean(all_scores)) if all_scores else 0.0}
    for cat, scores in sorted(by_cat.items()):
        name = CATEGORY_NAMES.get(cat, f"cat_{cat}")
        agg[f"{name}_f1"] = float(np.mean(scores))
        agg[f"{name}_count"] = len(scores)
    agg["total_count"] = len(all_scores)
    return agg
