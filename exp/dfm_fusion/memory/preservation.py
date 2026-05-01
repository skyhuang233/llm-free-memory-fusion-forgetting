"""Preservation checks for memory fusion.
- LLM-based: prompt gpt-4o-mini to score information preservation [0-1]
- Deterministic: salient-token coverage recall via numbers, entities, TF-IDF
"""

import json
import logging
import os
import re
import time

import numpy as np
from openai import OpenAI
from sklearn.feature_extraction.text import TfidfVectorizer

log = logging.getLogger(__name__)

PRESERVATION_PROMPT = """You are an information preservation evaluator.

Given original memory texts and a fused summary, score how well the fused text preserves the unique information from the originals.

Consider:
1. Are all key facts, names, dates, and numbers preserved?
2. Is the temporal progression maintained?
3. Are causal relationships kept intact?

Original texts:
{originals}

Fused text:
{fused}

Respond with ONLY a JSON object: {{"score": <float between 0.0 and 1.0>}}
A score of 1.0 means perfect preservation. A score below {threshold} means unacceptable loss."""


class LLMPreservationChecker:
    def __init__(self, model: str = "gpt-4o-mini", threshold: float = 0.7):
        self.model = model
        self.threshold = threshold
        self._client = OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY", os.environ.get("LEMMA_MAAS_API_KEY", "")),
            base_url=os.environ.get("OPENAI_BASE_URL", None),
        )
        self.call_count = 0

    def check(self, original_texts: list[str], fused_text: str) -> tuple[bool, float]:
        originals_str = "\n---\n".join(original_texts)
        prompt = PRESERVATION_PROMPT.format(
            originals=originals_str,
            fused=fused_text,
            threshold=self.threshold,
        )

        for attempt in range(3):
            try:
                resp = self._client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0,
                    max_tokens=50,
                )
                self.call_count += 1
                content = resp.choices[0].message.content.strip()
                content = content.strip("`").strip()
                if content.startswith("json"):
                    content = content[4:].strip()
                result = json.loads(content)
                score = float(result["score"])
                return score >= self.threshold, score
            except Exception as e:
                log.warning(f"Preservation check attempt {attempt+1} failed: {e}")
                if attempt < 2:
                    time.sleep(2 ** attempt)
        return False, 0.0


class DeterministicPreservationChecker:
    def __init__(self, coverage_threshold: float = 0.85, top_k_tfidf: int = 20,
                 corpus_texts: list[str] | None = None):
        self.coverage_threshold = coverage_threshold
        self.top_k = top_k_tfidf
        self.check_count = 0
        self.accept_count = 0
        self.reject_count = 0
        self._tfidf = None
        self._vocab = None
        if corpus_texts:
            self.fit_tfidf(corpus_texts)

    def fit_tfidf(self, corpus_texts: list[str]):
        self._tfidf = TfidfVectorizer(
            token_pattern=r"(?u)\b\w+\b",
            lowercase=True,
            max_features=5000,
        )
        self._tfidf.fit(corpus_texts)
        self._vocab = self._tfidf.get_feature_names_out()

    def _extract_numeric_tokens(self, text: str) -> set[str]:
        return set(re.findall(r"\b\d[\d,./:%-]*\b", text))

    def _extract_capitalized_tokens(self, text: str) -> set[str]:
        return {m for m in re.findall(r"\b[A-Z][a-zA-Z]+\b", text) if len(m) >= 2}

    def _extract_tfidf_tokens(self, text: str) -> set[str]:
        if self._tfidf is None or self._vocab is None:
            return set()
        vec = self._tfidf.transform([text])
        scores = vec.toarray().flatten()
        if len(scores) == 0:
            return set()
        top_indices = np.argsort(scores)[::-1][:self.top_k]
        return {self._vocab[i] for i in top_indices if scores[i] > 0}

    def extract_salient_tokens(self, text: str) -> set[str]:
        numeric = self._extract_numeric_tokens(text)
        capitalized = {t.lower() for t in self._extract_capitalized_tokens(text)}
        tfidf = self._extract_tfidf_tokens(text)
        return numeric | capitalized | tfidf

    def check(self, original_texts: list[str], fused_text: str) -> tuple[bool, float]:
        self.check_count += 1
        combined = " ".join(original_texts)
        salient = self.extract_salient_tokens(combined)
        if not salient:
            self.accept_count += 1
            return True, 1.0

        fused_lower = fused_text.lower()
        fused_tokens = set(re.findall(r"(?u)\b[\w,./:%-]+\b", fused_lower))
        fused_tokens |= set(re.findall(r"\b\d[\d,./:%-]*\b", fused_text))

        covered = salient & fused_tokens
        recall = len(covered) / len(salient)

        passed = recall >= self.coverage_threshold
        if passed:
            self.accept_count += 1
        else:
            self.reject_count += 1
        return passed, recall
