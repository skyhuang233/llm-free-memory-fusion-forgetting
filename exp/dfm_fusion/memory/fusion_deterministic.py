# Deterministic quote-preserving fusion operator (DFM-Fusion).
# Pipeline: sentence segmentation -> near-duplicate removal -> budgeted MMR packing.
# Uses DeterministicPreservationChecker for salient-token coverage gating.

import logging
import math

import nltk
import numpy as np
import tiktoken

from dfm_fusion.memory.embeddings import EmbeddingManager
from dfm_fusion.memory.memory_store import MemoryItem, MemoryStore
from dfm_fusion.memory.preservation import DeterministicPreservationChecker

log = logging.getLogger(__name__)

try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab", quiet=True)


class DeterministicFusionOperator:
    def __init__(
        self,
        embedding_manager: EmbeddingManager,
        dedup_threshold: float = 0.90,
        mmr_lambda: float = 0.7,
        token_budget: int = 512,
        coverage_threshold: float = 0.85,
        top_k_tfidf: int = 20,
        epsilon: float = 0.1,
        lambda_base: float = 0.1,
        corpus_texts: list[str] | None = None,
        skip_coverage_check: bool = False,
    ):
        self.emb = embedding_manager
        self.theta_dup = dedup_threshold
        self.mmr_lambda = mmr_lambda
        self.budget = token_budget
        self.epsilon = epsilon
        self.lambda_base = lambda_base
        self.skip_coverage_check = skip_coverage_check
        self._enc = tiktoken.encoding_for_model("gpt-4o-mini")

        self._checker = DeterministicPreservationChecker(
            coverage_threshold=coverage_threshold,
            top_k_tfidf=top_k_tfidf,
            corpus_texts=corpus_texts,
        )

        self.accepted_count = 0
        self.rejected_count = 0
        self.fusion_events = 0
        self.would_have_rejected_count = 0
        self.coverage_recall_values: list[float] = []

    def _segment_sentences(self, cluster: list[MemoryItem]):
        sentences = []
        for mem in cluster:
            sents = nltk.sent_tokenize(mem.text)
            for s in sents:
                s = s.strip()
                if not s:
                    continue
                sentences.append({
                    "text": s,
                    "strength": mem.strength,
                    "timestamp": mem.timestamp,
                    "mid": mem.mid,
                })
        return sentences

    def _dedup_sentences(self, sentences: list[dict]) -> list[dict]:
        if len(sentences) <= 1:
            return sentences

        texts = [s["text"] for s in sentences]
        embs = self.emb.embed_batch(texts)

        kept = [True] * len(sentences)
        for i in range(len(sentences)):
            if not kept[i]:
                continue
            for j in range(i + 1, len(sentences)):
                if not kept[j]:
                    continue
                sim = float(np.dot(embs[i], embs[j]))
                if sim > self.theta_dup:
                    si, sj = sentences[i], sentences[j]
                    if sj["strength"] > si["strength"]:
                        kept[i] = False
                        break
                    elif sj["strength"] == si["strength"]:
                        if sj["timestamp"] > si["timestamp"]:
                            kept[i] = False
                            break
                        else:
                            kept[j] = False
                    else:
                        kept[j] = False

        return [s for s, k in zip(sentences, kept) if k]

    def _mmr_select(self, sentences: list[dict]) -> list[dict]:
        if not sentences:
            return []

        texts = [s["text"] for s in sentences]
        embs = self.emb.embed_batch(texts)

        strengths = np.array([s["strength"] for s in sentences])
        s_max = strengths.max() if strengths.max() > 0 else 1.0
        norm_strengths = strengths / s_max

        selected_idx = []
        remaining = list(range(len(sentences)))
        total_tokens = 0

        while remaining:
            best_idx = None
            best_score = -float("inf")

            for idx in remaining:
                relevance = norm_strengths[idx]

                if selected_idx:
                    max_sim = max(
                        float(np.dot(embs[idx], embs[si]))
                        for si in selected_idx
                    )
                else:
                    max_sim = 0.0

                mmr = self.mmr_lambda * relevance - (1 - self.mmr_lambda) * max_sim
                if mmr > best_score:
                    best_score = mmr
                    best_idx = idx

            if best_idx is None:
                break

            sent_tokens = len(self._enc.encode(sentences[best_idx]["text"]))
            if total_tokens + sent_tokens > self.budget and selected_idx:
                break

            selected_idx.append(best_idx)
            remaining.remove(best_idx)
            total_tokens += sent_tokens

        selected_idx.sort(key=lambda i: sentences[i]["timestamp"])
        return [sentences[i] for i in selected_idx]

    def _build_fallback(self, cluster: list[MemoryItem]) -> str:
        sorted_mems = sorted(cluster, key=lambda m: m.strength, reverse=True)
        parts = []
        total_tokens = 0
        for mem in sorted_mems:
            mem_tokens = self._enc.encode(mem.text)
            if total_tokens + len(mem_tokens) <= self.budget:
                parts.append(mem.text)
                total_tokens += len(mem_tokens)
            else:
                remaining = self.budget - total_tokens
                if remaining > 0:
                    parts.append(self._enc.decode(mem_tokens[:remaining]))
                    total_tokens += remaining
                break
        return "\n".join(parts)

    def fuse_cluster(
        self,
        cluster: list[MemoryItem],
        store: MemoryStore,
        current_time,
    ) -> dict:
        self.fusion_events += 1
        texts = [m.text for m in cluster]
        mids = [m.mid for m in cluster]

        sentences = self._segment_sentences(cluster)
        if not sentences:
            self.rejected_count += 1
            return {"accepted": False, "reason": "no_sentences"}

        deduped = self._dedup_sentences(sentences)
        selected = self._mmr_select(deduped)

        if not selected:
            self.rejected_count += 1
            return {"accepted": False, "reason": "no_selection"}

        fused_text = " | ".join(s["text"] for s in selected)

        passed, recall = self._checker.check(texts, fused_text)
        self.coverage_recall_values.append(recall)
        if not passed:
            if self.skip_coverage_check:
                self.would_have_rejected_count += 1
                log.debug(f"Coverage check would reject (recall={recall:.3f}), but skip_coverage_check=True")
            else:
                fused_text = self._build_fallback(cluster)
                log.debug(f"Coverage check failed (recall={recall:.3f}), using fallback")

        strengths = [m.strength for m in cluster]
        v_fused = max(strengths) + self.epsilon * np.var(strengths)
        v_fused = min(max(v_fused, 0.0), 1.0)

        cluster_size = len(cluster)
        decay_factor = 1.0 / (1.0 + math.log(cluster_size))

        fused_emb = self.emb.embed(fused_text)

        ts = max(m.timestamp for m in cluster)

        new_mid = store.add_fused_memory(
            text=fused_text,
            embedding=fused_emb,
            timestamp=ts,
            strength=v_fused,
            decay_rate_factor=decay_factor,
            constituent_ids=mids,
        )

        for m in cluster:
            store.deactivate(m.mid)

        self.accepted_count += 1
        return {
            "accepted": True,
            "new_mid": new_mid,
            "cluster_size": cluster_size,
            "fused_strength": v_fused,
            "preservation_recall": recall,
            "preservation_passed": passed,
            "num_sentences_in": len(sentences),
            "num_sentences_deduped": len(deduped),
            "num_sentences_selected": len(selected),
        }

    def run_fusion(
        self,
        store: MemoryStore,
        current_time,
        min_cluster_size: int = 3,
    ) -> list[dict]:
        clusters = store.get_fusion_candidates(current_time, min_cluster_size)
        results = []
        for cluster in clusters:
            result = self.fuse_cluster(cluster, store, current_time)
            results.append(result)
        return results

    def stats(self) -> dict:
        result = {
            "fusion_llm_calls": 0,
            "preservation_llm_calls": 0,
            "total_llm_calls": 0,
            "accepted": self.accepted_count,
            "rejected": self.rejected_count,
            "fusion_events": self.fusion_events,
            "preservation_checks": self._checker.check_count,
            "preservation_accepts": self._checker.accept_count,
            "preservation_rejects": self._checker.reject_count,
            "would_have_rejected": self.would_have_rejected_count,
            "coverage_recall_mean": float(np.mean(self.coverage_recall_values)) if self.coverage_recall_values else 0.0,
            "coverage_recall_values": [float(v) for v in self.coverage_recall_values],
        }
        return result
