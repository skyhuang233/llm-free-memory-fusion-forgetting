## Proposed Approach

### Overview

We propose **DeterministicFadeMem-Fusion (DFM-Fusion)**: keep FadeMem’s forgetting/decay, retrieval, and answer generation unchanged, but replace only the fusion operator with a fully deterministic algorithm that:

1. **Identifies fusion candidates** using the same temporal-semantic clustering rule as FadeMem.
2. **Fuses by extractive packing**: selects a subset of verbatim sentences/spans from the cluster under a strict token budget, removing near-duplicates.
3. **Validates information preservation without an LLM**: uses a deterministic coverage test on salient tokens (numbers, capitalized entities, rare tokens) to reject unsafe fusions.

The goal is not to beat FadeMem’s accuracy, but to test whether LLM-based fusion is *necessary* for the gains FadeMem attributes to fusion.

### Method Details

**(A) Memory representation and forgetting (kept identical to FadeMem).** Each memory item is represented as \(m_i(t)=(c_i, s_i, v_i(t), \tau_i, f_i)\) where \(c_i\) is an embedding, \(s_i\) is text, \(v_i(t)\in[0,1]\) is strength, \(\tau_i\) is timestamp, and \(f_i\) is access frequency ([FadeMem](./references/FadeMem-Biologically-Inspired-Forgetting-for-Efficient-Agent-Memory/meta/meta_info.txt), §2.1–2.2). Dual-layer assignment and decay follow FadeMem’s equations and hyperparameters (\(\lambda_{base}=0.1\), \(\theta_{fusion}=0.75\), etc.).

**(B) Fusion candidate identification (kept identical to FadeMem).** For each candidate memory \(m_k\), form a cluster:
\[
C_k = \{m_i : \text{sim}(c_i,c_k) > \theta_{fusion} \wedge |\tau_i-\tau_k| < T_{window}\}.
\]
This matches FadeMem’s temporal-semantic clustering ([FadeMem](./references/FadeMem-Biologically-Inspired-Forgetting-for-Efficient-Agent-Memory/meta/meta_info.txt), §2.4).

**(C) Deterministic quote-preserving fusion operator (ours).** For a cluster \(C\) that exceeds a minimum size (same trigger as FadeMem), we produce fused text \(s_{fused}\) as follows:

1. **Sentence segmentation**: split each \(s_i\) into sentences/spans (rule-based splitter; no model).
2. **Deduplicate near-duplicates**: greedily remove sentences whose embedding similarity to any kept sentence exceeds \(\theta_{dup}\) (e.g., 0.90), keeping the sentence from the memory with larger \(v_i(t)\) (or more recent \(\tau_i\) as tie-break).
3. **Budgeted packing via deterministic MMR**: select remaining sentences with a deterministic greedy Maximum Marginal Relevance objective that prefers (a) higher-strength sources and (b) diversity among sentences, until reaching a per-fused-item token budget \(B_{fuse}\).
4. **Emit fused memory** as newline-separated verbatim sentences, ordered by timestamp of their source memory.

**(D) Deterministic preservation check (replaces FadeMem’s LLM verification).** FadeMem accepts/rejects a fusion using an LLM “information preservation” check with threshold \(\theta_{preserve}\) (§2.4). We replace this with a deterministic check:

- Extract a set of **salient tokens** from the cluster: all numbers, all capitalized wordpieces, and top-\(K\) TF-IDF tokens (where **TF-IDF = term frequency–inverse document frequency**, a standard heuristic for identifying rare but informative words in the conversation).
- Compute **coverage recall**: fraction of salient tokens appearing in \(s_{fused}\).
- If recall < \(\theta_{cov}\), reject fusion and fall back to a safe alternative: concatenate the top-\(n\) highest-strength original memories in \(C\) (truncated to \(B_{fuse}\)).

**(E) Embeddings for fused items (control to reduce retrieval confounds).** To avoid confounding retrieval changes from different fused texts, we store fused embeddings as the L2-normalized **mean of constituent embeddings** (optionally weighted by \(v_i(t)\)). This is applied to both the LLM-fusion baseline and DFM-Fusion.

**(F) Retrieval-time token budget control (critical).** “Same memory budget” is defined as the **total tokens appended to the answer model per query**, \(B_{ret}\). To avoid the confound that fused items are longer and get truncated more, we:

- retrieve top-\(k\) items,
- truncate **each retrieved item** to \(\lfloor B_{ret}/k\rfloor\) tokens before concatenation,
- log the fraction of items truncated for each method.

### Key Innovations

- **A deterministic, quote-preserving fusion operator** explicitly targeted to LoCoMo’s verbatim-answer evaluation protocol.
- **A non-LLM preservation criterion** (salient-token coverage) that makes “fusion safety” auditable and reproducible.
- **A retrieval-budget-controlled evaluation** that isolates fusion quality from token-length confounds.

---