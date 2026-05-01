"""Embedding computation via local sentence-transformers (all-MiniLM-L6-v2).
Provides batch embedding, fused-item embedding (L2-normalized mean), and
disk caching to avoid recomputation across runs.
"""

import hashlib
import pickle
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


_CACHE_DIR = Path(__file__).resolve().parent.parent / "results" / ".embed_cache"


class EmbeddingManager:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", cache_dir: Path | None = None):
        self._model = SentenceTransformer(model_name)
        self._cache_dir = cache_dir or _CACHE_DIR
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._cache_file = self._cache_dir / f"{model_name.replace('/', '_')}.pkl"
        self._mem_cache: dict[str, np.ndarray] = {}
        self._load_cache()

    def _load_cache(self):
        if self._cache_file.exists():
            with open(self._cache_file, "rb") as f:
                self._mem_cache = pickle.load(f)

    def save_cache(self):
        with open(self._cache_file, "wb") as f:
            pickle.dump(self._mem_cache, f)

    @staticmethod
    def _text_key(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def embed(self, text: str) -> np.ndarray:
        key = self._text_key(text)
        if key not in self._mem_cache:
            vec = self._model.encode(text, normalize_embeddings=True)
            self._mem_cache[key] = vec.astype(np.float32)
        return self._mem_cache[key]

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        to_compute = []
        to_compute_idx = []
        results = [None] * len(texts)

        for i, t in enumerate(texts):
            key = self._text_key(t)
            if key in self._mem_cache:
                results[i] = self._mem_cache[key]
            else:
                to_compute.append(t)
                to_compute_idx.append(i)

        if to_compute:
            vecs = self._model.encode(to_compute, normalize_embeddings=True, batch_size=256)
            for j, idx in enumerate(to_compute_idx):
                vec = vecs[j].astype(np.float32)
                self._mem_cache[self._text_key(to_compute[j])] = vec
                results[idx] = vec

        return np.array(results, dtype=np.float32)

    def fused_embedding(
        self, embeddings: list[np.ndarray], weights: list[float] | None = None
    ) -> np.ndarray:
        embs = np.array(embeddings, dtype=np.float32)
        if weights is not None:
            w = np.array(weights, dtype=np.float32).reshape(-1, 1)
            mean_vec = (embs * w).sum(axis=0) / (w.sum() + 1e-12)
        else:
            mean_vec = embs.mean(axis=0)
        norm = np.linalg.norm(mean_vec)
        if norm > 0:
            mean_vec /= norm
        return mean_vec.astype(np.float32)
