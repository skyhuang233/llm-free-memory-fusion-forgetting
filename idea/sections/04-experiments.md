## Experiments

### Experimental Setup

**Task framing.** We implement a FadeMem-style memory store for each LoCoMo conversation, then answer each benchmark QA query using a fixed answer model with retrieved memory appended.

**Main conditions (≤3):**
1. **LLM-Fusion (FadeMem-style)**: Temporal-semantic clustering + LLM fusion + LLM preservation check (as described in FadeMem §2.4).
2. **DeterministicFusion (DFM-Fusion, ours)**: Same clustering and forgetting, but deterministic quote-preserving fusion + deterministic coverage check.
3. **No-Fusion**: Same clustering and forgetting, but never merge memories (this matches FadeMem’s “w/o Fusion” ablation concept).

All other components are held constant: embeddings, decay, pruning, conflict handling, retriever, answer model, decoding settings, and retrieval-time token budget.

**Baseline Ladder (REQUIRED):**
- **Prompting / long-context baseline**: Fixed-16K FIFO context (published in FadeMem Table 3).
- **Strongest existing method**: FadeMem with LLM fusion (published; and also re-run in our harness).
- **Ablation baseline**: w/o Fusion (published; and re-run in our harness).

**Base Models:**

| Model | Size | Download Link | Notes |
|-------|------|---------------|-------|
| gpt-4o-mini | API | (available via platform) | Used for answer generation and for LLM-fusion baseline’s fusion/preservation prompts (temp=0) |

**Training Data (if applicable):**

No training data needed – inference only.

**Resource Estimate**:
- **Compute budget**: 0 GPU-hours (API-based inference) + optional ≤50 GPU-hours if running embeddings locally on a single A100.
- **API usage** (order-of-magnitude):
  - Answer generation: ~1,540 LoCoMo questions → ~1,540 calls.
  - LLM-Fusion baseline: additional calls for fusion + preservation checks, roughly proportional to #fusion events (expected O(1k–5k) calls on LoCoMo10 depending on thresholds).
  - DeterministicFusion: no additional LLM calls beyond answer generation.
- **Wall-clock**: With modest parallelism, expected hours-scale for LLM-Fusion; DeterministicFusion reduces this substantially.

### Benchmarks and Metrics

| Benchmark | Description | Metrics | Split | Download Link | Evaluation Script |
|-----------|-------------|---------|-------|---------------|-------------------|
| LoCoMo | Long-horizon conversational memory benchmark with multi-hop, temporal, open-domain, and single-hop QA; ground-truth answers are drawn from the conversation | **Multi-hop F1** (primary), overall F1 (secondary), retrieval recall@k (diagnostic) | test | https://github.com/snap-research/locomo | Official repo + deterministic F1 computation described in LoCoMo §4.1 |

### Main Results

**Published reference numbers (FadeMem Table 3; LoCoMo multi-hop F1 protocol):**

| Method | Base Model | Benchmark | LoCoMo F1 ↑ | FCR ↑ (Factual Consistency Rate) | SRR ↑ (Storage Reduction Rate) | Source | Notes |
|--------|------------|-----------|-------------|-------|-------|--------|-------|
| Fixed-16K | GPT-4o-mini | LoCoMo | 5.17 | 78.9% | 0.00 | [FadeMem §3.4](<./references/FadeMem-Biologically-Inspired-Forgetting-for-Efficient-Agent-Memory/sections/3.4 Cross-Dataset Evaluation.md>) | Published |
| LangChain | GPT-4o-mini | LoCoMo | 25.75 | 81.2% | 0.00 | [FadeMem §3.4](<./references/FadeMem-Biologically-Inspired-Forgetting-for-Efficient-Agent-Memory/sections/3.4 Cross-Dataset Evaluation.md>) | Published |
| Mem0 | GPT-4o-mini | LoCoMo | 28.37 | 83.6% | 0.00 | [FadeMem §3.4](<./references/FadeMem-Biologically-Inspired-Forgetting-for-Efficient-Agent-Memory/sections/3.4 Cross-Dataset Evaluation.md>) | Published |
| FadeMem (LLM-Fusion) | GPT-4o-mini | LoCoMo | 29.43 | 85.9% | 0.45 | [FadeMem §3.4](<./references/FadeMem-Biologically-Inspired-Forgetting-for-Efficient-Agent-Memory/sections/3.4 Cross-Dataset Evaluation.md>) | Published |
| FadeMem w/o Fusion | GPT-4o-mini | LoCoMo | 13.63 | N/A | N/A | [FadeMem §3.5](<./references/FadeMem-Biologically-Inspired-Forgetting-for-Efficient-Agent-Memory/sections/3.5 Ablation Study.md>) | Published ablation |

**Verification table (to be filled by experiments in this proposal; all methods run in the same harness):**

| Method | Base Model | Benchmark | Multi-hop F1 (mean±std) | Source | Notes |
|--------|------------|-----------|--------------------------|--------|-------|
| LLM-Fusion (FadeMem-style) | gpt-4o-mini | LoCoMo | **TBD** | - | Run 3 seeds / 3 runs; temp=0 |
| DeterministicFusion (ours) | gpt-4o-mini | LoCoMo | **TBD** | - | Run 3 seeds / 3 runs; temp=0 |
| No-Fusion | gpt-4o-mini | LoCoMo | **TBD** | - | Run 3 seeds / 3 runs; temp=0 |

### Ablation Studies

| Variant | What’s changed | Expected finding |
|---------|----------------|------------------|
| DeterministicFusion w/o coverage check | Always accept deterministic fusion | If performance drops, coverage check prevents destructive merges |
| DeterministicFusion (per-entry truncation off) | Use post-concat truncation only | If performance drops, per-entry truncation is an important control against length confounds |

### Experimental Rigor

**Variance & Reproducibility:**
- Primary runs use temperature=0 for all LLM calls; non-determinism is still possible across API calls.
- Report mean±std over 3 runs (treated as “seeds”) for each main condition.

**Confounders and controls:**
- **Token budget confound**: Fused items may be longer and more often truncated. Control with per-entry truncation and log truncation rates.
- **Retrieval confound from embeddings**: Different fusion texts could change embeddings. Control by using mean-of-constituents embeddings for all fused items.
- **Evaluation protocol mismatch**: Some papers report LoCoMo with LLM-judge accuracy. We use deterministic F1 per LoCoMo §4.1.

**Sanity checks:**
- Reproduce FadeMem’s large fusion gap directionally (LLM-Fusion > No-Fusion) in our harness; if not, the harness is invalid.
- Random retrieval baseline (retrieve random memories) should perform near the Fixed-16K baseline.

---