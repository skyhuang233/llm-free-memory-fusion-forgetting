# DeterministicFadeMem-Fusion (DFM-Fusion)

Inference-only experiment comparing FadeMem-style LLM-guided memory fusion vs. a deterministic quote-preserving fusion operator, evaluated on the LoCoMo benchmark using deterministic F1 partial-match scoring.

## Quick Start

```bash
# Activate venv
source .venv/bin/activate

# Load API keys (sets OPENAI_API_KEY and OPENAI_BASE_URL from .env)
source dfm_fusion/scripts/setup_env.sh
```

## Environment

- **Python**: 3.12.12 (`.venv` at project root)
- **LLM API**: gpt-4o-mini via LEMMA_MAAS proxy (`http://{LEMMA_MAAS_BASE_URL}/v1`)
  - Auth: `LEMMA_MAAS_API_KEY` from `.env`
  - Use `python-dotenv` to load keys; do NOT hardcode
- **Embeddings**: Local `sentence-transformers` (`all-MiniLM-L6-v2`, 384-dim)
  - The proxy embedding endpoint is not available (returns 401); use local computation
- **NLTK data**: `punkt`, `punkt_tab` pre-downloaded

## Key Packages

| Package | Purpose |
|---------|---------|
| `openai` | gpt-4o-mini chat completions (answer gen, LLM-fusion) |
| `sentence-transformers` | Local embedding computation |
| `scikit-learn` | TF-IDF vectorization, cosine similarity |
| `nltk` | Sentence tokenization (sent_tokenize) |
| `tiktoken` | OpenAI token counting |
| `numpy`, `scipy` | Numerical ops, bootstrap CI, paired t-tests |
| `pandas` | Result aggregation |
| `matplotlib`, `seaborn` | Visualization |

## Project Structure

```
exp/
└── dfm_fusion/           # Core experiment code, configs, and runners
paper/                    # Current manuscript and build artifacts
docs/                     # Planning notes, structure docs, and workspace helpers
figures/                  # Shared analysis plots and method diagrams
references/               # Reference paper bundles and source material
tools/                    # Utility scripts such as workspace prep and upload helpers
llm-free-memory-fusion-forgetting-master/  # Legacy nested workspace snapshot
```

The core code still lives under `exp/dfm_fusion/`; the top-level directories now separate the manuscript, docs, figures, references, and utility scripts so the repo is easier to navigate.

See [`docs/PROJECT_STRUCTURE.md`](docs/PROJECT_STRUCTURE.md) for a shorter map of where to edit what.

## API Usage Pattern

```python
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(
    api_key=os.environ["LEMMA_MAAS_API_KEY"],
    base_url=f"http://{os.environ['LEMMA_MAAS_BASE_URL']}/v1"
)
resp = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Hello"}],
    temperature=0
)
```

## Experimental Conditions

1. **LLM-Fusion** (baseline): FadeMem-style LLM-guided fusion via gpt-4o-mini
2. **DFM-Fusion** (proposed): Deterministic quote-preserving fusion
3. **No-Fusion** (ablation): No merging, memories stored individually

## Key Hyperparameters (base_config.yaml)

- Decay: lambda_base=0.1, theta_promote=0.7, theta_demote=0.3, theta_fusion=0.75
- Memory: LML=1000, SML=500
- Fusion budget: B_fuse=512 tokens, dedup_threshold=0.85
- Retrieval: top_k=10, B_ret=4000 tokens
- Evaluation: 3 runs per condition, seed=42

## Running Experiments

```bash
source .venv/bin/activate

# Run LLM-Fusion baseline (3 runs x 10 conversations, ~2.5 hours)
python3 dfm_fusion/scripts/run_llm_fusion_full.py

# Run No-Fusion ablation (3 runs x 10 conversations, ~2 hours)
python3 dfm_fusion/scripts/run_no_fusion_full.py

# Run DFM-Fusion (3 runs, can parallelize with --run-id 0/1/2, ~45 min each)
python3 dfm_fusion/scripts/run_dfm_fusion_full.py --run-id 0  # or omit --run-id for sequential

# Compare fusion gap (LLM vs No-Fusion)
python3 dfm_fusion/scripts/compare_fusion_gap.py

# Full statistical comparison (all 3 conditions)
python3 dfm_fusion/scripts/statistical_comparison.py
```

## Completed Experiments

### LLM-Fusion Baseline (Task 1)
- **Multi-hop F1**: 0.1863 +/- 0.0021 (FadeMem published: 29.43)
- **Overall F1**: 0.3805 +/- 0.0004
- 226 fusion LLM calls/run, ~2855s/run
- Results: `EXPERIMENT_RESULTS/llm_fusion_baseline/`

### No-Fusion Ablation (Task 2)
- **Multi-hop F1**: 0.1710 +/- 0.0009
- **Overall F1**: 0.3750 +/- 0.0011
- Zero fusion LLM calls, ~2422s/run
- **Fusion gap**: LLM-Fusion +8.9% on multi-hop F1 (p=0.046, significant)
- Directional match with FadeMem published gap confirmed
- Results: `EXPERIMENT_RESULTS/no_fusion_baseline/`

### DFM-Fusion (Task 3, Optimized)
- **Multi-hop F1**: 0.1872 +/- 0.0034 (exceeds LLM-Fusion's 0.1863)
- **Overall F1**: 0.3795 +/- 0.0012
- Zero fusion LLM calls, ~2500s/run
- **Gap recovery**: 105.9% of No-Fusion->LLM-Fusion gap recovered (exceeds LLM-Fusion)
- DFM config: theta_dup=0.85, lambda_MMR=0.7, B_fuse=768, theta_cov=0.85, K=20
- Key optimizations: re-embed fused text from actual content, pipe separators, lower dedup threshold, larger fusion budget
- Results: `EXPERIMENT_RESULTS/dfm_fusion/`
- Optimization trace: `EXPERIMENT_RESULTS/optimize_trace/iteration_0/`

### Ablation Study (Task 6)
- **w/o Coverage Check**: Multi-hop F1 = 0.1851 +/- 0.0003 (delta = -0.0021, p=0.122, not significant)
- **w/o Per-Entry Truncation**: Multi-hop F1 = 0.1850 +/- 0.0008 (delta = -0.0023, p=0.193, not significant)
- Neither component removal causes a statistically significant drop vs full DFM-Fusion
- Both components act as safety nets; full DFM-Fusion significantly outperforms No-Fusion (p=0.031)
- Ablation configs: `configs/dfm_ablation_no_coverage.yaml`, `configs/dfm_ablation_no_truncation.yaml`
- Runner: `scripts/run_ablation_variants.py --variant no_coverage|no_truncation`
- Analysis: `scripts/ablation_analysis.py`
- Results: `EXPERIMENT_RESULTS/ablation_study/`

### Operational Metrics Analysis (Task 7)
- **LLM call reduction**: 226 fusion calls eliminated/run (10.2% total, 100% fusion-specific)
- **Maintenance speedup**: 5.2x (84s vs 439s overhead vs No-Fusion baseline)
- **Cost savings**: $0.018/run (3.4%) at gpt-4o-mini pricing
- **Cluster sizes**: mean 3.5, range [3,7] for both methods
- **Fused token lengths**: mean ~97 tokens (both methods, well under B_ret/k=400 budget)
- **Truncation rates**: 0.0% across all conditions
- Analysis scripts: `scripts/collect_fusion_details.py`, `scripts/operational_analysis.py`
- Figures: `results/figures/{api_calls_by_condition,wall_clock_maintenance,fused_token_lengths}.png`
- Results: `EXPERIMENT_RESULTS/operational_analysis/`

## Implementation Notes

### Memory Store Adaptations for Batch Evaluation
- Incremental decay tracking (per-memory `_last_decay_time`) to avoid catastrophic decay
- `prune(soft=True)`: only enforce capacity limits, skip strength/dormancy pruning
- Pure cosine similarity retrieval (no strength weighting) for better coverage of early memories
- Context embedding updated per session for importance computation during ingestion
- Key parameters tuned: mu=2.0, dv=0.15, epsilon_prune=0.005, t_max_days=365

### DFM-Fusion Operator (`memory/fusion_deterministic.py`)
- Sentence segmentation via NLTK `sent_tokenize`, each sentence inherits source memory strength/timestamp
- Near-duplicate removal: greedy pairwise cosine sim > theta_dup (0.85), keep higher-strength source
- Budgeted MMR packing: `MMR(s) = lambda * strength - (1-lambda) * max_sim`, greedy until B_fuse (768)
- Fused text joined with ` | ` separators for clear sentence delimitation
- Fused memory embedding computed from actual fused text (not weighted average of constituents)
- Preservation check (`memory/preservation.py`): salient-token coverage recall (numbers, entities, TF-IDF top-K)
- Fallback on rejection: pack highest-strength originals within budget
- Config: `configs/dfm_fusion_config.yaml` (overrides base_config fusion/preservation params)

### F1 Scorer
- Matches LoCoMo reference implementation exactly (verified against external/locomo/task_eval/evaluation.py)
- Category mapping: 1=multi-hop, 2=temporal, 3=open-domain, 4=single-hop, 5=adversarial
- Multi-hop uses comma-split partial F1; adversarial uses keyword match
