"""End-to-end evaluation pipeline for memory fusion experiments on LoCoMo.
Runs memory ingestion, optional fusion, QA retrieval+answer generation, and scoring.
Supports conditions: llm_fusion, no_fusion, dfm_fusion.
"""

import argparse
import json
import logging
import os
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import yaml
from dotenv import load_dotenv
from openai import OpenAI

from dfm_fusion.data.locomo_loader import Conversation, load_locomo
from dfm_fusion.evaluation.f1_scorer import aggregate_scores, score_qa
from dfm_fusion.memory.embeddings import EmbeddingManager
from dfm_fusion.memory.fusion_deterministic import DeterministicFusionOperator
from dfm_fusion.memory.fusion_llm import LLMFusionOperator
from dfm_fusion.memory.memory_store import MemoryStore

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger(__name__)

ANSWER_PROMPT = """You are a helpful assistant with access to conversation memories. Answer the question based on the provided memory context. Extract relevant details from the context to form your answer. Be concise and specific — give short, direct answers (a few words or a short phrase when possible).

If the question asks about something that is clearly NOT mentioned or discussed anywhere in the context, respond with "Not mentioned".

Memory context:
{context}

Question: {question}

Answer:"""


def _setup_env():
    load_dotenv()
    os.environ.setdefault("OPENAI_API_KEY", os.environ.get("LEMMA_MAAS_API_KEY", ""))
    base = os.environ.get("LEMMA_MAAS_BASE_URL", "")
    if base and "OPENAI_BASE_URL" not in os.environ:
        os.environ["OPENAI_BASE_URL"] = f"http://{base}/v1"


def _get_answer_client(config: dict) -> OpenAI:
    return OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY", ""),
        base_url=os.environ.get("OPENAI_BASE_URL", None),
    )


def generate_answer(client: OpenAI, model: str, context: str, question: str, temperature: float = 0.0) -> str:
    prompt = ANSWER_PROMPT.format(context=context, question=question)
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=256,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            log.warning(f"Answer gen attempt {attempt+1} failed: {e}")
            if attempt < 2:
                time.sleep(2 ** attempt)
    return ""


def process_conversation(
    conv: Conversation,
    config: dict,
    condition: str,
    emb_manager: EmbeddingManager,
    answer_client: OpenAI,
) -> dict:
    store = MemoryStore(config)
    fusion_op = None
    if condition == "llm_fusion":
        fusion_op = LLMFusionOperator(
            embedding_manager=emb_manager,
            model=config.get("answer_model", "gpt-4o-mini"),
            preservation_threshold=0.7,
            lambda_base=config["decay"]["lambda_base"],
        )
    elif condition == "dfm_fusion":
        fusion_cfg = config.get("fusion", {})
        pres_cfg = config.get("preservation", {})
        all_texts = [s.text for s in conv.snippets]
        fusion_op = DeterministicFusionOperator(
            embedding_manager=emb_manager,
            dedup_threshold=fusion_cfg.get("dedup_threshold", 0.90),
            mmr_lambda=fusion_cfg.get("mmr_lambda", 0.7),
            token_budget=fusion_cfg.get("token_budget", 512),
            coverage_threshold=pres_cfg.get("coverage_threshold", 0.85),
            top_k_tfidf=pres_cfg.get("salient_top_k", 20),
            epsilon=0.1,
            lambda_base=config["decay"]["lambda_base"],
            corpus_texts=all_texts,
            skip_coverage_check=config.get("ablation_skip_coverage_check", False),
        )

    log.info(f"Ingesting {len(conv.snippets)} snippets for {conv.sample_id}")
    texts = [s.text for s in conv.snippets]
    embeddings = emb_manager.embed_batch(texts)

    current_session = -1
    session_embs = []
    for i, snippet in enumerate(conv.snippets):
        if snippet.session_idx != current_session:
            if current_session > 0:
                if session_embs:
                    ctx = np.mean(session_embs, axis=0)
                    ctx /= (np.linalg.norm(ctx) + 1e-12)
                    store.update_context(ctx)
                store.apply_decay(snippet.timestamp)
                store.prune(snippet.timestamp, soft=True)
                if fusion_op is not None:
                    fusion_results = fusion_op.run_fusion(store, snippet.timestamp)
                    n_acc = sum(1 for r in fusion_results if r.get("accepted"))
                    if fusion_results:
                        log.info(f"  Session {current_session} fusion: {len(fusion_results)} clusters, {n_acc} accepted")
            current_session = snippet.session_idx
            session_embs = []

        store.add_memory(
            text=snippet.text,
            embedding=embeddings[i],
            timestamp=snippet.timestamp,
            mid=snippet.dia_id,
        )
        session_embs.append(embeddings[i])

    if fusion_op is not None:
        final_ts = conv.snippets[-1].timestamp
        store.apply_decay(final_ts)
        fusion_results = fusion_op.run_fusion(store, final_ts)

    log.info(f"Memory store: {store.active_count()} active, layers={store.layer_counts()}")

    query_time = conv.snippets[-1].timestamp + timedelta(hours=1)
    store.apply_decay(query_time)

    qa_results = []
    model = config.get("answer_model", "gpt-4o-mini")
    total = len(conv.qa_pairs)

    for qi, qa in enumerate(conv.qa_pairs):
        q_emb = emb_manager.embed(qa.question)
        retrieved = store.retrieve(q_emb, query_time)
        context = "\n\n".join(r["text"] for r in retrieved)
        truncated_count = sum(1 for r in retrieved if r["truncated"])

        answer = generate_answer(answer_client, model, context, qa.question, config.get("answer_temperature", 0.0))
        f1 = score_qa(answer, qa.answer, qa.category)

        result = {
            "question": qa.question,
            "ground_truth": qa.answer,
            "prediction": answer,
            "category": qa.category,
            "category_name": qa.category_name,
            "f1": f1,
            "evidence": qa.evidence,
            "retrieved_mids": [r["mid"] for r in retrieved],
            "truncated_count": truncated_count,
            "num_retrieved": len(retrieved),
        }
        qa_results.append(result)

        if (qi + 1) % 20 == 0 or qi == total - 1:
            running_f1 = np.mean([r["f1"] for r in qa_results])
            log.info(f"  [{conv.sample_id}] {qi+1}/{total} QAs, running F1={running_f1:.4f}")

    conv_agg = aggregate_scores(qa_results)
    fusion_stats = fusion_op.stats() if fusion_op else {"total_llm_calls": 0}
    truncation_rate = np.mean([r["truncated_count"] / max(r["num_retrieved"], 1) for r in qa_results])

    return {
        "sample_id": conv.sample_id,
        "qa_results": qa_results,
        "aggregated": conv_agg,
        "fusion_stats": fusion_stats,
        "memory_stats": {
            "active_count": store.active_count(),
            "layers": store.layer_counts(),
        },
        "truncation_rate": float(truncation_rate),
        "retrieval_stats": store.retrieval_stats(),
    }


def run_experiment(config: dict, condition: str, run_id: int, output_dir: Path, conversations: list[Conversation] | None = None, conv_filter: str | None = None):
    _setup_env()
    emb_manager = EmbeddingManager(model_name=config.get("embedding_model", "all-MiniLM-L6-v2"))
    answer_client = _get_answer_client(config)

    data_path = config.get("data", {}).get("locomo_path", "external/locomo/data/locomo10.json")
    if conversations is None:
        conversations = load_locomo(data_path)

    if conv_filter:
        conversations = [c for c in conversations if c.sample_id == conv_filter]
        if not conversations:
            log.error(f"No conversation found with sample_id={conv_filter}")
            return

    run_dir = output_dir / f"run_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)

    all_results = []
    all_qa = []
    t_start = time.time()

    for conv in conversations:
        log.info(f"=== Processing {conv.sample_id} (run {run_id}, condition={condition}) ===")
        result = process_conversation(conv, config, condition, emb_manager, answer_client)
        all_results.append(result)
        all_qa.extend(result["qa_results"])

        with open(run_dir / f"{conv.sample_id}.json", "w") as f:
            json.dump(result, f, indent=2, default=str)

    elapsed = time.time() - t_start
    overall_agg = aggregate_scores(all_qa)

    total_fusion_calls = sum(r["fusion_stats"].get("total_llm_calls", 0) for r in all_results)
    total_accepted = sum(r["fusion_stats"].get("accepted", 0) for r in all_results)
    total_rejected = sum(r["fusion_stats"].get("rejected", 0) for r in all_results)
    mean_trunc_rate = np.mean([r["truncation_rate"] for r in all_results])

    abl_coverage_would_rejected = sum(r["fusion_stats"].get("would_have_rejected", 0) for r in all_results)
    abl_coverage_recalls = []
    for r in all_results:
        abl_coverage_recalls.extend(r["fusion_stats"].get("coverage_recall_values", []))
    abl_coverage_recall_mean = float(np.mean(abl_coverage_recalls)) if abl_coverage_recalls else 0.0

    abl_len_dom_count = sum(r["retrieval_stats"].get("length_dominance_count", 0) for r in all_results)
    abl_total_retrieves = sum(r["retrieval_stats"].get("total_retrieve_calls", 0) for r in all_results)
    abl_len_dom_frac = abl_len_dom_count / max(abl_total_retrieves, 1)

    summary = {
        "condition": condition,
        "run_id": run_id,
        "num_conversations": len(conversations),
        "total_qa": len(all_qa),
        "elapsed_seconds": elapsed,
        "overall": overall_agg,
        "fusion_stats": {
            "total_llm_calls": total_fusion_calls,
            "accepted": total_accepted,
            "rejected": total_rejected,
        },
        "mean_truncation_rate": float(mean_trunc_rate),
        "ablation_stats": {
            "coverage_would_have_rejected": abl_coverage_would_rejected,
            "coverage_recall_mean": abl_coverage_recall_mean,
            "length_dominance_count": abl_len_dom_count,
            "total_retrieve_calls": abl_total_retrieves,
            "length_dominance_fraction": float(abl_len_dom_frac),
        },
        "per_conversation": {r["sample_id"]: r["aggregated"] for r in all_results},
    }

    with open(run_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    emb_manager.save_cache()

    log.info(f"=== Run {run_id} complete: {condition} ===")
    log.info(f"Overall F1: {overall_agg['overall_f1']:.4f}")
    for k, v in overall_agg.items():
        if k.endswith("_f1"):
            log.info(f"  {k}: {v:.4f}")
    log.info(f"Elapsed: {elapsed:.1f}s, Fusion LLM calls: {total_fusion_calls}")

    return summary


def aggregate_runs(output_dir: Path, num_runs: int) -> dict:
    summaries = []
    for run_id in range(num_runs):
        path = output_dir / f"run_{run_id}" / "summary.json"
        if path.exists():
            summaries.append(json.loads(path.read_text()))

    if not summaries:
        return {}

    metric_keys = [k for k in summaries[0]["overall"] if k.endswith("_f1")]
    agg = {}
    for k in metric_keys:
        vals = [s["overall"][k] for s in summaries]
        agg[k] = {"mean": float(np.mean(vals)), "std": float(np.std(vals)), "values": vals}

    return agg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="dfm_fusion/configs/base_config.yaml")
    parser.add_argument("--condition", default="llm_fusion", choices=["llm_fusion", "no_fusion", "dfm_fusion"])
    parser.add_argument("--run-id", type=int, default=0)
    parser.add_argument("--num-runs", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--conv-filter", default=None)
    args = parser.parse_args()

    config = yaml.safe_load(open(args.config))

    output_dir = Path(args.output_dir) if args.output_dir else Path("dfm_fusion/results") / args.condition
    output_dir.mkdir(parents=True, exist_ok=True)

    num_runs = args.num_runs or config.get("evaluation", {}).get("num_runs", 3)

    if args.num_runs:
        conversations = load_locomo(config.get("data", {}).get("locomo_path", "external/locomo/data/locomo10.json"))
        for run_id in range(num_runs):
            run_experiment(config, args.condition, run_id, output_dir, conversations=conversations, conv_filter=args.conv_filter)
        agg = aggregate_runs(output_dir, num_runs)
        log.info(f"=== Aggregated over {num_runs} runs ===")
        for k, v in agg.items():
            log.info(f"  {k}: {v['mean']:.4f} +/- {v['std']:.4f}")
        with open(output_dir / "aggregated.json", "w") as f:
            json.dump(agg, f, indent=2)
    else:
        run_experiment(config, args.condition, args.run_id, output_dir, conv_filter=args.conv_filter)


if __name__ == "__main__":
    main()
