"""LLM-guided fusion operator replicating FadeMem Section 2.4.
For each cluster of temporally-semantically similar memories, invokes
gpt-4o-mini to produce a fused text, then validates with LLM preservation check.
"""

import logging
import math
import os
import time

import numpy as np
from openai import OpenAI

from dfm_fusion.memory.embeddings import EmbeddingManager
from dfm_fusion.memory.memory_store import MemoryItem, MemoryStore
from dfm_fusion.memory.preservation import LLMPreservationChecker

log = logging.getLogger(__name__)

FUSION_PROMPT = """You are a memory consolidation assistant. Fuse the following related memory entries into a single coherent summary.

Requirements:
1. Preserve ALL unique information from each entry
2. Maintain temporal progression and chronological order
3. Preserve causal relationships between events
4. Keep specific names, dates, numbers, and facts
5. Be concise but comprehensive

Memory entries to fuse:
{entries}

Produce ONLY the fused memory text, no commentary."""


class LLMFusionOperator:
    def __init__(
        self,
        embedding_manager: EmbeddingManager,
        model: str = "gpt-4o-mini",
        preservation_threshold: float = 0.7,
        epsilon: float = 0.1,
        lambda_base: float = 0.1,
    ):
        self.emb = embedding_manager
        self.model = model
        self.epsilon = epsilon
        self.lambda_base = lambda_base
        self._client = OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY", os.environ.get("LEMMA_MAAS_API_KEY", "")),
            base_url=os.environ.get("OPENAI_BASE_URL", None),
        )
        self._checker = LLMPreservationChecker(model=model, threshold=preservation_threshold)
        self.fusion_call_count = 0
        self.accepted_count = 0
        self.rejected_count = 0

    def _call_llm_fuse(self, texts: list[str]) -> str | None:
        entries = "\n---\n".join(f"[Entry {i+1}]\n{t}" for i, t in enumerate(texts))
        prompt = FUSION_PROMPT.format(entries=entries)

        for attempt in range(3):
            try:
                resp = self._client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0,
                    max_tokens=1024,
                )
                self.fusion_call_count += 1
                return resp.choices[0].message.content.strip()
            except Exception as e:
                log.warning(f"Fusion LLM call attempt {attempt+1} failed: {e}")
                if attempt < 2:
                    time.sleep(2 ** attempt)
        return None

    def fuse_cluster(
        self,
        cluster: list[MemoryItem],
        store: MemoryStore,
        current_time,
    ) -> dict:
        texts = [m.text for m in cluster]
        mids = [m.mid for m in cluster]

        fused_text = self._call_llm_fuse(texts)
        if fused_text is None:
            self.rejected_count += 1
            return {"accepted": False, "reason": "llm_call_failed"}

        passed, score = self._checker.check(texts, fused_text)
        if not passed:
            self.rejected_count += 1
            return {"accepted": False, "reason": "preservation_failed", "score": score}

        strengths = [m.strength for m in cluster]
        v_fused = max(strengths) + self.epsilon * np.var(strengths)
        v_fused = min(max(v_fused, 0.0), 1.0)

        cluster_size = len(cluster)
        decay_factor = 1.0 / (1.0 + math.log(cluster_size))

        embeddings = [m.embedding for m in cluster]
        weights = [m.strength for m in cluster]
        fused_emb = self.emb.fused_embedding(embeddings, weights)

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
            "preservation_score": score,
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
        return {
            "fusion_llm_calls": self.fusion_call_count,
            "preservation_llm_calls": self._checker.call_count,
            "total_llm_calls": self.fusion_call_count + self._checker.call_count,
            "accepted": self.accepted_count,
            "rejected": self.rejected_count,
        }
