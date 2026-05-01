# DFM-Fusion Ablation Study

## Experiment Overview

Ablation study measuring the contribution of two key DFM-Fusion components:
1. **Coverage Check (preservation gate)**: Deterministic salient-token coverage recall check that rejects fusions with recall < theta_cov (0.85)
2. **Per-Entry Truncation**: Retrieval-time budget control that allocates floor(B_ret/k) tokens per item instead of post-concatenation truncation

Two DFM-Fusion variants were tested against the full DFM-Fusion and No-Fusion baseline.

## Setup

- **Dataset**: LoCoMo-10 (10 conversations, 1986 QA pairs, 282 multi-hop questions)
- **Runs per variant**: 3
- **Answer model**: gpt-4o-mini (temperature=0)
- **Embedding**: all-MiniLM-L6-v2 (local)
- **Config**: Identical to full DFM-Fusion (theta_dup=0.85, lambda_MMR=0.7, B_fuse=768, theta_cov=0.85, K_tfidf=20, top_k=10, B_ret=4000)

## Key Results

| Variant | Multi-hop F1 (mean +/- std) | Delta from full DFM | p-value |
|---------|----------------------------|---------------------|---------|
| Full DFM-Fusion | 0.1872 +/- 0.0034 | -- | -- |
| w/o Coverage Check | 0.1851 +/- 0.0003 | -0.0021 | 0.122 |
| w/o Per-Entry Truncation | 0.1850 +/- 0.0008 | -0.0023 | 0.193 |
| No-Fusion baseline | 0.1710 +/- 0.0009 | -0.0162 | 0.031* |

*significant at p<0.05

## Key Observations

### Coverage Check Ablation
- Neither variant's removal causes a statistically significant drop in multi-hop F1 (p > 0.05)
- 9 fusions (across 3 runs) would have been rejected by the coverage gate, but their downstream QA impact is minimal
- Average coverage recall across all fusions: 0.961 (well above threshold of 0.85)
- The coverage check functions as a safety net: it rarely triggers under the current config, but protects against destructive merges where salient tokens would be lost

### Per-Entry Truncation Ablation
- 0% of queries had the first retrieved item consuming >50% of the retrieval budget (B_ret=4000)
- This indicates memory items are relatively short compared to the per-item budget (4000/10 = 400 tokens per item)
- Per-entry truncation has no practical effect under current settings because items are rarely longer than 400 tokens
- The component would become critical with smaller B_ret or larger fused items

### Overall
- Both components contribute small directional improvements (+0.0021 and +0.0023) but are not individually statistically significant
- Their combined absence would likely have a larger effect
- Full DFM-Fusion significantly outperforms No-Fusion baseline (p=0.031), confirming the fusion mechanism itself is the primary driver of improvement
- The coverage check and per-entry truncation are "safety nets" that prevent edge-case degradation rather than systematically improving average performance
