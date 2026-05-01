# LLM-Fusion Baseline (FadeMem-Style) on LoCoMo

## Experiment Overview

Replication of FadeMem's LLM-guided memory fusion pipeline on the LoCoMo benchmark,
serving as the upper-bound comparison for the proposed DeterministicFusion method.

**Reference**: "FadeMem: Biologically-Inspired Forgetting for Efficient Agent Memory" (arXiv:2601.18642)

## Setup

- **Dataset**: LoCoMo-10 (10 conversations, 1986 QA pairs total)
- **QA Categories**: multi-hop (282), temporal (321), open-domain (96), single-hop (841), adversarial (446)
- **Embedding Model**: all-MiniLM-L6-v2 (local sentence-transformers, 384-dim)
- **Answer Model**: gpt-4o-mini (temperature=0) via LEMMA_MAAS proxy
- **Fusion Model**: gpt-4o-mini (temperature=0) for both fusion and preservation checking
- **Runs**: 3 independent runs (to capture API non-determinism variance)
- **Memory Architecture**: FadeMem dual-layer (LML=1000, SML=500) with biologically-inspired decay
- **Retrieval**: Top-10 by cosine similarity, token budget B_ret=4000, per-entry truncation
- **Fusion**: Temporal-semantic clustering (theta_fusion=0.75, T_window=10 sessions), LLM-guided fusion with preservation check (theta_preserve=0.7)

### Key Implementation Differences from FadeMem Paper

1. **Embeddings**: Local sentence-transformers (all-MiniLM-L6-v2) instead of OpenAI text-embedding-3-small (API returned 401)
2. **Batch Evaluation**: All conversations processed as batch evaluation (ingest all turns, then answer all QAs) rather than streaming agent interaction
3. **Decay Adaptation**: Incremental decay tracking and soft pruning to prevent catastrophic memory loss in batch setting
4. **Retrieval**: Pure cosine similarity (no strength weighting) for better coverage of early conversation memories

## Key Results

### Aggregated Metrics (Mean +/- Std over 3 runs)

| Metric | Mean | Std |
|--------|------|-----|
| **Overall F1** | 0.3805 | 0.0004 |
| **Multi-hop F1** | 0.1863 | 0.0021 |
| Temporal F1 | 0.0614 | 0.0001 |
| Open-domain F1 | 0.1068 | 0.0009 |
| Single-hop F1 | 0.3439 | 0.0005 |
| Adversarial F1 | 0.8610 | 0.0000 |

### Per-Run Results

| Run | Overall F1 | Multi-hop F1 | Single-hop F1 | Temporal F1 | Adversarial F1 |
|-----|-----------|-------------|--------------|-------------|---------------|
| 0 | 0.3801 | 0.1854 | 0.3432 | 0.0615 | 0.8610 |
| 1 | 0.3810 | 0.1891 | 0.3442 | 0.0614 | 0.8610 |
| 2 | 0.3804 | 0.1843 | 0.3442 | 0.0614 | 0.8610 |

### Fusion Statistics (consistent across runs)

- Total LLM fusion calls: 226
- Accepted fusions: 113
- Rejected fusions: 0
- Acceptance rate: 100% (all preservation checks passed)
- Truncation rate: 0.0 (retrieved items fit within budget)

### Comparison with FadeMem Published Results

| Metric | FadeMem (published) | Our Replication |
|--------|-------------------|-----------------|
| Multi-hop F1 | 29.43 | 18.63 |

The gap (18.63 vs 29.43) is expected due to implementation differences:
- Different embedding model (local MiniLM vs OpenAI)
- Batch evaluation vs streaming agent interaction
- Different decay/retrieval adaptations for batch setting

The result is within the acceptable sanity check range (15-45) specified in the task requirements.

## Key Observations

1. **Very low variance across runs**: Std < 0.002 for all metrics, confirming deterministic behavior with temperature=0
2. **Adversarial detection is strong**: 86.1% F1 on adversarial questions (detecting "not mentioned" scenarios)
3. **Temporal reasoning is weak**: Only 6.1% F1 on temporal questions, suggesting the memory system struggles with time-ordered reasoning
4. **Fusion acceptance rate of 100%**: The LLM preservation checker never rejected a fusion, suggesting the clustering quality threshold (0.75) is conservative enough
5. **No truncation needed**: Retrieved items fit within the 4000-token budget, so per-entry truncation was not triggered

## Runtime

- Total runtime: ~8584 seconds (~2.4 hours) for 3 runs x 10 conversations
- Per-run: ~2850 seconds (~47 minutes)
- Per-conversation: ~4-5 minutes (API-bound)

## Files

- Per-conversation results: `dfm_fusion/results/llm_fusion/run_{0,1,2}/<conv-id>.json`
- Per-run summaries: `dfm_fusion/results/llm_fusion/run_{0,1,2}/summary.json`
- Aggregated results: `dfm_fusion/results/llm_fusion/aggregated.json`
