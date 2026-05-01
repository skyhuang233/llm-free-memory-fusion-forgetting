# No-fusion passthrough operator (ablation baseline).
# Provides the same interface as LLMFusionOperator but never fuses --
# memories are stored and retrieved individually. Used to measure the
# fusion gap (LLM-Fusion vs No-Fusion).

import logging

from dfm_fusion.memory.memory_store import MemoryStore

log = logging.getLogger(__name__)


class NoFusionOperator:
    def __init__(self, **kwargs):
        self.fusion_call_count = 0
        self.accepted_count = 0
        self.rejected_count = 0

    def run_fusion(self, store: MemoryStore, current_time, min_cluster_size: int = 3) -> list[dict]:
        return []

    def stats(self) -> dict:
        return {
            "fusion_llm_calls": 0,
            "preservation_llm_calls": 0,
            "total_llm_calls": 0,
            "accepted": 0,
            "rejected": 0,
        }
