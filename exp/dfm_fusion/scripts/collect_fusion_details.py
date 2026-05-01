# Replay memory ingestion + fusion to capture per-event details (cluster sizes,
# fused-item token lengths) that were not logged in the original experiment runs.
# DFM-Fusion: full replay (CPU only). LLM-Fusion: clustering (CPU) + 1-conv LLM sample.

import json
import logging
import os
import sys
from datetime import timedelta
from pathlib import Path

import numpy as np
import tiktoken
import yaml
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from dfm_fusion.data.locomo_loader import load_locomo
from dfm_fusion.memory.embeddings import EmbeddingManager
from dfm_fusion.memory.fusion_deterministic import DeterministicFusionOperator
from dfm_fusion.memory.fusion_llm import LLMFusionOperator
from dfm_fusion.memory.memory_store import MemoryStore

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger(__name__)

ENC = tiktoken.encoding_for_model("gpt-4o-mini")

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _setup_env():
    load_dotenv(PROJECT_ROOT / ".env")
    os.environ.setdefault("OPENAI_API_KEY", os.environ.get("LEMMA_MAAS_API_KEY", ""))
    base = os.environ.get("LEMMA_MAAS_BASE_URL", "")
    if base and "OPENAI_BASE_URL" not in os.environ:
        os.environ["OPENAI_BASE_URL"] = f"http://{base}/v1"


def ingest_and_get_clusters(conv, config, emb_manager):
    store = MemoryStore(config)
    texts = [s.text for s in conv.snippets]
    embeddings = emb_manager.embed_batch(texts)

    current_session = -1
    session_embs = []
    all_clusters = []

    for i, snippet in enumerate(conv.snippets):
        if snippet.session_idx != current_session:
            if current_session > 0:
                if session_embs:
                    ctx = np.mean(session_embs, axis=0)
                    ctx /= (np.linalg.norm(ctx) + 1e-12)
                    store.update_context(ctx)
                store.apply_decay(snippet.timestamp)
                store.prune(snippet.timestamp, soft=True)
                clusters = store.get_fusion_candidates(snippet.timestamp)
                all_clusters.extend(clusters)
            current_session = snippet.session_idx
            session_embs = []

        store.add_memory(text=snippet.text, embedding=embeddings[i],
                         timestamp=snippet.timestamp, mid=snippet.dia_id)
        session_embs.append(embeddings[i])

    final_ts = conv.snippets[-1].timestamp
    store.apply_decay(final_ts)
    clusters = store.get_fusion_candidates(final_ts)
    all_clusters.extend(clusters)

    return store, all_clusters


def replay_dfm_fusion(conv, config, emb_manager):
    store = MemoryStore(config)
    fusion_cfg = config.get("fusion", {})
    pres_cfg = config.get("preservation", {})
    all_texts = [s.text for s in conv.snippets]

    fusion_op = DeterministicFusionOperator(
        embedding_manager=emb_manager,
        dedup_threshold=fusion_cfg.get("dedup_threshold", 0.85),
        mmr_lambda=fusion_cfg.get("mmr_lambda", 0.7),
        token_budget=fusion_cfg.get("token_budget", 768),
        coverage_threshold=pres_cfg.get("coverage_threshold", 0.85),
        top_k_tfidf=pres_cfg.get("salient_top_k", 20),
        epsilon=0.1,
        lambda_base=config["decay"]["lambda_base"],
        corpus_texts=all_texts,
    )

    texts = [s.text for s in conv.snippets]
    embeddings = emb_manager.embed_batch(texts)

    current_session = -1
    session_embs = []
    fusion_events = []

    for i, snippet in enumerate(conv.snippets):
        if snippet.session_idx != current_session:
            if current_session > 0:
                if session_embs:
                    ctx = np.mean(session_embs, axis=0)
                    ctx /= (np.linalg.norm(ctx) + 1e-12)
                    store.update_context(ctx)
                store.apply_decay(snippet.timestamp)
                store.prune(snippet.timestamp, soft=True)
                results = fusion_op.run_fusion(store, snippet.timestamp)
                for r in results:
                    fusion_events.append(r)
            current_session = snippet.session_idx
            session_embs = []

        store.add_memory(text=snippet.text, embedding=embeddings[i],
                         timestamp=snippet.timestamp, mid=snippet.dia_id)
        session_embs.append(embeddings[i])

    final_ts = conv.snippets[-1].timestamp
    store.apply_decay(final_ts)
    results = fusion_op.run_fusion(store, final_ts)
    for r in results:
        fusion_events.append(r)

    return fusion_events, store


def replay_llm_fusion_one_conv(conv, config, emb_manager):
    store = MemoryStore(config)
    fusion_op = LLMFusionOperator(
        embedding_manager=emb_manager,
        model=config.get("answer_model", "gpt-4o-mini"),
        preservation_threshold=0.7,
        lambda_base=config["decay"]["lambda_base"],
    )

    texts = [s.text for s in conv.snippets]
    embeddings = emb_manager.embed_batch(texts)

    current_session = -1
    session_embs = []
    fusion_events = []

    for i, snippet in enumerate(conv.snippets):
        if snippet.session_idx != current_session:
            if current_session > 0:
                if session_embs:
                    ctx = np.mean(session_embs, axis=0)
                    ctx /= (np.linalg.norm(ctx) + 1e-12)
                    store.update_context(ctx)
                store.apply_decay(snippet.timestamp)
                store.prune(snippet.timestamp, soft=True)
                results = fusion_op.run_fusion(store, snippet.timestamp)
                for r in results:
                    if r.get("accepted") and r.get("new_mid"):
                        mem = store.memories.get(r["new_mid"])
                        if mem:
                            r["fused_text"] = mem.text
                            r["fused_token_count"] = len(ENC.encode(mem.text))
                    fusion_events.append(r)
            current_session = snippet.session_idx
            session_embs = []

        store.add_memory(text=snippet.text, embedding=embeddings[i],
                         timestamp=snippet.timestamp, mid=snippet.dia_id)
        session_embs.append(embeddings[i])

    final_ts = conv.snippets[-1].timestamp
    store.apply_decay(final_ts)
    results = fusion_op.run_fusion(store, final_ts)
    for r in results:
        if r.get("accepted") and r.get("new_mid"):
            mem = store.memories.get(r["new_mid"])
            if mem:
                r["fused_text"] = mem.text
                r["fused_token_count"] = len(ENC.encode(mem.text))
        fusion_events.append(r)

    return fusion_events, store


def main():
    _setup_env()

    base_cfg_path = PROJECT_ROOT / "dfm_fusion" / "configs" / "base_config.yaml"
    dfm_cfg_path = PROJECT_ROOT / "dfm_fusion" / "configs" / "dfm_fusion_config.yaml"

    with open(base_cfg_path) as f:
        base_config = yaml.safe_load(f)
    with open(dfm_cfg_path) as f:
        dfm_config = yaml.safe_load(f)

    data_path = base_config.get("data", {}).get("locomo_path", "external/locomo/data/locomo10.json")
    conversations = load_locomo(data_path)
    emb_manager = EmbeddingManager(model_name=base_config.get("embedding_model", "all-MiniLM-L6-v2"))

    output = {
        "dfm_fusion": {
            "cluster_sizes": [],
            "fused_token_lengths": [],
            "acceptance_details": [],
            "per_conv": {},
        },
        "llm_fusion": {
            "cluster_sizes": [],
            "fused_token_lengths_sample": [],
            "per_conv_cluster_sizes": {},
            "sample_conv": "conv-26",
        },
    }

    log.info("=== DFM-Fusion: Full replay for all conversations ===")
    for conv in conversations:
        log.info(f"DFM replay: {conv.sample_id}")
        events, store = replay_dfm_fusion(conv, dfm_config, emb_manager)

        conv_cluster_sizes = []
        conv_fused_lengths = []
        conv_details = []

        for ev in events:
            cs = ev.get("cluster_size", 0)
            conv_cluster_sizes.append(cs)
            output["dfm_fusion"]["cluster_sizes"].append(cs)

            if ev.get("accepted") and ev.get("new_mid"):
                mem = store.memories.get(ev["new_mid"])
                if mem:
                    tl = len(ENC.encode(mem.text))
                    conv_fused_lengths.append(tl)
                    output["dfm_fusion"]["fused_token_lengths"].append(tl)

            conv_details.append({
                "cluster_size": cs,
                "accepted": ev.get("accepted", False),
                "preservation_passed": ev.get("preservation_passed"),
                "preservation_recall": ev.get("preservation_recall"),
                "num_sentences_in": ev.get("num_sentences_in"),
                "num_sentences_deduped": ev.get("num_sentences_deduped"),
                "num_sentences_selected": ev.get("num_sentences_selected"),
            })

        output["dfm_fusion"]["per_conv"][conv.sample_id] = {
            "cluster_sizes": conv_cluster_sizes,
            "fused_token_lengths": conv_fused_lengths,
            "num_events": len(events),
            "num_accepted": sum(1 for e in events if e.get("accepted")),
        }
        output["dfm_fusion"]["acceptance_details"].extend(conv_details)

    log.info("=== LLM-Fusion: Cluster-only replay for all conversations ===")
    for conv in conversations:
        log.info(f"LLM cluster replay: {conv.sample_id}")
        _, clusters = ingest_and_get_clusters(conv, base_config, emb_manager)
        sizes = [len(c) for c in clusters]
        output["llm_fusion"]["cluster_sizes"].extend(sizes)
        output["llm_fusion"]["per_conv_cluster_sizes"][conv.sample_id] = sizes

    log.info("=== LLM-Fusion: Full replay for conv-26 (with LLM calls) ===")
    sample_conv = [c for c in conversations if c.sample_id == "conv-26"][0]
    llm_events, llm_store = replay_llm_fusion_one_conv(sample_conv, base_config, emb_manager)
    for ev in llm_events:
        if ev.get("accepted") and ev.get("fused_token_count"):
            output["llm_fusion"]["fused_token_lengths_sample"].append(ev["fused_token_count"])

    output["llm_fusion"]["sample_fusion_events"] = []
    for ev in llm_events:
        output["llm_fusion"]["sample_fusion_events"].append({
            "cluster_size": ev.get("cluster_size", 0),
            "accepted": ev.get("accepted", False),
            "fused_token_count": ev.get("fused_token_count"),
            "preservation_score": ev.get("preservation_score"),
        })

    out_path = PROJECT_ROOT / "dfm_fusion" / "results" / "fusion_event_details.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)

    log.info(f"Saved fusion event details to {out_path}")
    log.info(f"DFM cluster sizes: n={len(output['dfm_fusion']['cluster_sizes'])}, "
             f"mean={np.mean(output['dfm_fusion']['cluster_sizes']):.1f}")
    log.info(f"DFM fused token lengths: n={len(output['dfm_fusion']['fused_token_lengths'])}, "
             f"mean={np.mean(output['dfm_fusion']['fused_token_lengths']):.1f}")
    log.info(f"LLM cluster sizes: n={len(output['llm_fusion']['cluster_sizes'])}, "
             f"mean={np.mean(output['llm_fusion']['cluster_sizes']):.1f}")
    log.info(f"LLM fused token lengths (conv-26 sample): n={len(output['llm_fusion']['fused_token_lengths_sample'])}")

    emb_manager.save_cache()


if __name__ == "__main__":
    main()
