# Deterministic (LLM-Free) Memory Fusion for FadeMem-Style Long-Horizon Conversational Agents

## Scope and Constraints

- **Paper Type**: Short paper
- **Target Venues**: NeurIPS, ICML, ICLR, ACL, EMNLP (or similar top AI conferences)

## Introduction

### Context and Motivation

Long-horizon conversational agents (e.g., personal assistants that interact with users over days or months) cannot rely on a single fixed context window. A common engineering pattern is to maintain an external memory store: (i) write new interaction snippets into memory, (ii) retrieve a small set of relevant memories for each user query, and (iii) answer using the retrieved memory as additional context.

Recent systems report large gains on long-term conversational memory benchmarks, but many rely on *additional* LLM calls to maintain the memory store itself (e.g., to merge redundant memories, resolve conflicts, or verify information preservation). This creates three deployment pain points in 2026: (1) **cost/latency** due to extra LLM calls beyond answer generation, (2) **non-determinism** and brittle behavior due to hidden prompt+model changes, and (3) **audit difficulty** because the memory store becomes the output of an LLM summarizer rather than a transparent transformation of the original conversation.

FadeMem is a representative recent example: it adds biologically-inspired forgetting (dual-layer decay) and achieves strong LoCoMo QA performance, but its biggest ablation gain comes from an **LLM-guided memory fusion** module that merges temporally/semantically related memories and uses an LLM “preservation check” to accept/reject fusions.

### The Problem

**Problem:** Are LLM calls actually necessary for the *fusion operator* in long-horizon conversational memory, or is FadeMem’s fusion gain primarily coming from a simpler effect—reducing retrieval noise by deduplicating and packing verbatim evidence into a fixed retrieval-time token budget?

This question is not cosmetic: if LLM-based fusion is unnecessary, practitioners can implement a deterministic, auditable, and cheaper memory-maintenance pipeline while preserving benchmark quality.

The LoCoMo benchmark is a good testbed because its QA task was designed to allow **deterministic automated scoring**: ground-truth answers are taken from the conversation “as much as possible”, and evaluation uses a normalized **F1 partial-match** metric rather than an LLM judge ([LoCoMo](./references/Evaluating-Very-Long-Term-Conversational-Memory-of-LLM-Agents/meta/meta_info.txt), §4.1).

Key prior work illustrating the current landscape:
- **[FadeMem](./references/FadeMem-Biologically-Inspired-Forgetting-for-Efficient-Agent-Memory/meta/meta_info.txt)** (LLM-guided fusion + conflict resolution): LoCoMo F1=29.43, but w/o fusion drops to 13.63 (−53.7%) (Table 3; Ablation §3.5).
- **[FluxMem / “Choosing How to Remember”](./references/Choosing-How-to-Remember-Adaptive-Memory-Structures-for-LLM-Agents/meta/meta_info.txt)** learns memory *structure* selection and replaces fixed similarity thresholds with a probabilistic (Beta mixture) merge criterion, but still uses LLM-heavy memory formation.
- **[EverMemOS](./references/EverMemOS-A-Self-Organizing-Memory-Operating-System-for-Structured-Long-Horizon-Reasoning/meta/meta_info.txt)** achieves strong LoCoMo judge accuracy via structured MemCells and MemScenes, but similarly relies on many LLM-mediated stages.

### Key Insight and Hypothesis

**Key insight:** On LoCoMo, the metric rewards reproducing *verbatim* conversational facts. This suggests that the primary value of “fusion” may be **information packing and redundancy removal** (so that retrieval returns fewer, denser, less noisy memories), not paraphrastic rewriting or high-level abstraction. If so, a deterministic fusion operator that (i) preserves verbatim spans and (ii) enforces an explicit token budget should recover most of FadeMem’s fusion gain.

**Hypothesis:** Replacing FadeMem’s LLM-guided fusion + LLM preservation check with a **deterministic, quote-preserving, budgeted fusion** (extractive deduplication + MMR-style sentence selection, where **MMR = maximal marginal relevance**—a standard greedy objective that trades off relevance vs redundancy + deterministic coverage check) will yield LoCoMo multi-hop F1 that is statistically indistinguishable from the LLM-fusion baseline under the same retrieval-time token budget.

Why this could be wrong: LLM fusion might be doing more than deduplication (e.g., rewriting memories into query-aligned phrasing, resolving coreference/temporal expressions, or preserving causal links that extractive rules miss). If this is true, deterministic fusion will regress toward the “w/o fusion” ablation regime.

---