"""Core dual-layer memory store implementing FadeMem's memory architecture.
LML (Long-term Memory Layer) and SML (Short-term Memory Layer) with
biologically-inspired decay, importance scoring, consolidation, pruning,
and retrieval with per-entry token budget control.
"""

import math
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import tiktoken


@dataclass
class MemoryItem:
    mid: str
    text: str
    embedding: np.ndarray
    strength: float
    timestamp: datetime
    access_freq: float
    layer: str  # "LML" or "SML"
    last_access: datetime | None = None
    access_count: int = 0
    constituent_ids: list[str] = field(default_factory=list)
    is_fused: bool = False
    active: bool = True


class MemoryStore:
    def __init__(self, config: dict):
        decay_cfg = config.get("decay", {})
        mem_cfg = config.get("memory", {})

        self.lambda_base = decay_cfg.get("lambda_base", 0.1)
        self.theta_promote = decay_cfg.get("theta_promote", 0.7)
        self.theta_demote = decay_cfg.get("theta_demote", 0.3)
        self.theta_fusion = decay_cfg.get("theta_fusion", 0.75)
        self.t_window = decay_cfg.get("t_window", 10)

        self.lml_capacity = mem_cfg.get("lml_capacity", 1000)
        self.sml_capacity = mem_cfg.get("sml_capacity", 500)

        self.top_k = config.get("retrieval_top_k", 10)
        self.token_budget = config.get("retrieval_token_budget", 4000)

        self.alpha = 0.4
        self.beta = 0.3
        self.gamma = 0.3
        self.delta = 0.1
        self.mu = 2.0
        self.dv = 0.15
        self.N = 5.0
        self.epsilon_prune = 0.005
        self.t_max_days = 365.0
        self._last_decay_time: dict[str, datetime] = {}

        self.skip_per_entry_truncation = config.get("ablation_skip_per_entry_truncation", False)
        self.length_dominance_count = 0
        self.total_retrieve_calls = 0

        self.memories: dict[str, MemoryItem] = {}
        self._next_id = 0
        self._enc = tiktoken.encoding_for_model("gpt-4o-mini")
        self._context_emb: np.ndarray | None = None

    def _gen_id(self) -> str:
        mid = f"m_{self._next_id}"
        self._next_id += 1
        return mid

    def _days_between(self, t1: datetime, t2: datetime) -> float:
        return abs((t2 - t1).total_seconds()) / 86400.0

    def update_context(self, context_emb: np.ndarray):
        self._context_emb = context_emb

    def _importance(self, mem: MemoryItem, current_time: datetime, query_emb: np.ndarray | None = None) -> float:
        q = query_emb if query_emb is not None else self._context_emb
        if q is not None:
            rel = max(0.0, float(np.dot(mem.embedding, q)))
        else:
            rel = 0.5
        freq_term = mem.access_freq / (1.0 + mem.access_freq)
        dt = self._days_between(mem.timestamp, current_time)
        recency = math.exp(-self.delta * dt)
        return self.alpha * rel + self.beta * freq_term + self.gamma * recency

    def _decay_strength(self, mem: MemoryItem, current_time: datetime) -> float:
        last = self._last_decay_time.get(mem.mid, mem.timestamp)
        dt = self._days_between(last, current_time)
        if dt <= 0:
            return mem.strength
        importance = self._importance(mem, current_time)
        lam = self.lambda_base * math.exp(-self.mu * importance)
        beta_exp = 0.8 if mem.layer == "LML" else 1.2
        return mem.strength * math.exp(-lam * (dt ** beta_exp))

    def _assign_layer(self, mem: MemoryItem, current_time: datetime):
        importance = self._importance(mem, current_time)
        if mem.layer == "SML" and importance >= self.theta_promote:
            mem.layer = "LML"
        elif mem.layer == "LML" and importance < self.theta_demote:
            mem.layer = "SML"

    def add_memory(self, text: str, embedding: np.ndarray, timestamp: datetime, mid: str | None = None) -> str:
        if mid is None:
            mid = self._gen_id()
        mem = MemoryItem(
            mid=mid,
            text=text,
            embedding=embedding,
            strength=1.0,
            timestamp=timestamp,
            access_freq=0.0,
            layer="SML",
            last_access=timestamp,
        )
        self.memories[mid] = mem
        self._assign_layer(mem, timestamp)
        return mid

    def add_fused_memory(
        self,
        text: str,
        embedding: np.ndarray,
        timestamp: datetime,
        strength: float,
        decay_rate_factor: float,
        constituent_ids: list[str],
    ) -> str:
        mid = self._gen_id()
        mem = MemoryItem(
            mid=mid,
            text=text,
            embedding=embedding,
            strength=min(max(strength, 0.0), 1.0),
            timestamp=timestamp,
            access_freq=1.0,
            layer="LML",
            last_access=timestamp,
            constituent_ids=constituent_ids,
            is_fused=True,
        )
        self.memories[mid] = mem
        return mid

    def deactivate(self, mid: str):
        if mid in self.memories:
            self.memories[mid].active = False

    def apply_decay(self, current_time: datetime):
        for mem in self._active_memories():
            mem.strength = self._decay_strength(mem, current_time)
            self._last_decay_time[mem.mid] = current_time
            self._assign_layer(mem, current_time)

    def prune(self, current_time: datetime, soft: bool = False) -> list[str]:
        if soft:
            self._enforce_capacity()
            return []
        pruned = []
        for mem in self._active_memories():
            dt = self._days_between(mem.last_access or mem.timestamp, current_time)
            if mem.strength < self.epsilon_prune or dt > self.t_max_days:
                mem.active = False
                pruned.append(mem.mid)
        self._enforce_capacity()
        return pruned

    def _enforce_capacity(self):
        lml = [m for m in self._active_memories() if m.layer == "LML"]
        sml = [m for m in self._active_memories() if m.layer == "SML"]
        if len(lml) > self.lml_capacity:
            lml.sort(key=lambda m: m.strength)
            for m in lml[: len(lml) - self.lml_capacity]:
                m.active = False
        if len(sml) > self.sml_capacity:
            sml.sort(key=lambda m: m.strength)
            for m in sml[: len(sml) - self.sml_capacity]:
                m.active = False

    def consolidate_on_access(self, mid: str, current_time: datetime):
        mem = self.memories.get(mid)
        if mem is None or not mem.active:
            return
        mem.access_count += 1
        mem.access_freq += 1.0
        mem.last_access = current_time
        mem.strength = mem.strength + self.dv * (1.0 - mem.strength) * math.exp(-mem.access_count / self.N)

    def retrieve(self, query_embedding: np.ndarray, current_time: datetime, top_k: int | None = None, budget: int | None = None) -> list[dict]:
        top_k = top_k or self.top_k
        budget = budget or self.token_budget
        active = self._active_memories()
        if not active:
            return []

        embs = np.array([m.embedding for m in active], dtype=np.float32)
        sims = embs @ query_embedding.astype(np.float32)
        scores = sims

        k = min(top_k, len(active))
        top_indices = np.argsort(scores)[::-1][:k]

        self.total_retrieve_calls += 1

        if self.skip_per_entry_truncation:
            results = []
            for idx in top_indices:
                mem = active[idx]
                self.consolidate_on_access(mem.mid, current_time)
                results.append({
                    "mid": mem.mid,
                    "text": mem.text,
                    "similarity": float(sims[idx]),
                    "score": float(scores[idx]),
                    "strength": mem.strength,
                    "truncated": False,
                    "dia_ids": mem.constituent_ids if mem.is_fused else [mem.mid],
                })
            if results:
                first_tokens = len(self._enc.encode(results[0]["text"]))
                if first_tokens > budget * 0.5:
                    self.length_dominance_count += 1
            concat = "\n\n".join(r["text"] for r in results)
            concat_tokens = self._enc.encode(concat)
            if len(concat_tokens) > budget:
                concat = self._enc.decode(concat_tokens[:budget])
                parts = concat.split("\n\n")
                cum_tokens = 0
                for i, r in enumerate(results):
                    if i < len(parts):
                        r["text"] = parts[i]
                        r_tokens = len(self._enc.encode(parts[i]))
                        cum_tokens += r_tokens
                        r["truncated"] = cum_tokens > budget
                    else:
                        r["text"] = ""
                        r["truncated"] = True
            return results

        per_item_budget = budget // k if k > 0 else budget
        results = []
        for idx in top_indices:
            mem = active[idx]
            self.consolidate_on_access(mem.mid, current_time)
            text = mem.text
            tokens = self._enc.encode(text)
            truncated = len(tokens) > per_item_budget
            if truncated:
                tokens = tokens[:per_item_budget]
                text = self._enc.decode(tokens)
            results.append({
                "mid": mem.mid,
                "text": text,
                "similarity": float(sims[idx]),
                "score": float(scores[idx]),
                "strength": mem.strength,
                "truncated": truncated,
                "dia_ids": mem.constituent_ids if mem.is_fused else [mem.mid],
            })
        return results

    def retrieval_stats(self) -> dict:
        frac = self.length_dominance_count / max(self.total_retrieve_calls, 1)
        return {
            "length_dominance_count": self.length_dominance_count,
            "total_retrieve_calls": self.total_retrieve_calls,
            "length_dominance_fraction": float(frac),
        }

    def get_fusion_candidates(self, current_time: datetime, min_cluster_size: int = 3) -> list[list[MemoryItem]]:
        active = self._active_memories()
        if len(active) < min_cluster_size:
            return []

        embs = np.array([m.embedding for m in active], dtype=np.float32)
        sim_matrix = embs @ embs.T

        visited = set()
        clusters = []
        for i, mem_i in enumerate(active):
            if mem_i.mid in visited:
                continue
            cluster = [mem_i]
            visited.add(mem_i.mid)
            for j, mem_j in enumerate(active):
                if mem_j.mid in visited:
                    continue
                if sim_matrix[i, j] > self.theta_fusion:
                    dt = self._days_between(mem_i.timestamp, mem_j.timestamp)
                    if dt <= self.t_window:
                        cluster.append(mem_j)
                        visited.add(mem_j.mid)
            if len(cluster) >= min_cluster_size:
                clusters.append(cluster)

        return clusters

    def _active_memories(self) -> list[MemoryItem]:
        return [m for m in self.memories.values() if m.active]

    def active_count(self) -> int:
        return len(self._active_memories())

    def layer_counts(self) -> dict[str, int]:
        lml = sum(1 for m in self._active_memories() if m.layer == "LML")
        sml = sum(1 for m in self._active_memories() if m.layer == "SML")
        return {"LML": lml, "SML": sml}
