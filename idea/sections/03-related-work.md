## Related Work

### Field Overview

Research on long-horizon agent memory for LLMs spans several axes: (i) *what is stored* (raw turns, extracted facts, events, graphs), (ii) *how memory is maintained* (append-only, update/delete, decay/forgetting, consolidation/fusion), and (iii) *how memory is retrieved* (dense, sparse, hybrid, graph traversal, query planning). A recurring empirical theme is that naively increasing context length does not solve long-horizon memory due to attention degradation and noise in long contexts (e.g., “lost in the middle”).

A second theme is that many recent “memory systems” are in fact pipelines that invoke a strong LLM multiple times: to extract memory, rewrite or consolidate it, validate consistency, and plan retrieval. This can improve benchmark scores, but makes the memory store itself opaque and expensive.

Our proposal focuses narrowly on one module that appears important in FadeMem: **fusion**. We ask whether the benefit attributed to LLM fusion can be reproduced by a deterministic operator that preserves verbatim evidence under a fixed token budget.

### Related Papers

- **[FadeMem](./references/FadeMem-Biologically-Inspired-Forgetting-for-Efficient-Agent-Memory/meta/meta_info.txt)**: Introduces dual-layer decay and uses LLM-guided conflict resolution and fusion; provides a strong ablation showing fusion is critical on LoCoMo.
- **[LoCoMo](./references/Evaluating-Very-Long-Term-Conversational-Memory-of-LLM-Agents/meta/meta_info.txt)**: Defines long-term conversational memory tasks and a deterministic F1 evaluation protocol for QA with answers drawn verbatim from conversations.
- **[Mem0](./references/Mem0-Building-Production-Ready-AI-Agents-with-Scalable-Long-Term-Memory/meta/meta_info.txt)**: Uses LLM function-calling to add/update/delete memory facts; reports LoCoMo results under both F1 and LLM-judge protocols.
- **[EverMemOS](./references/EverMemOS-A-Self-Organizing-Memory-Operating-System-for-Structured-Long-Horizon-Reasoning/meta/meta_info.txt)**: Introduces MemCell/MemScene lifecycle and necessity-sufficiency retrieval; strong LoCoMo judge accuracy but LLM-heavy memory construction.
- **[Choosing How to Remember / FluxMem](./references/Choosing-How-to-Remember-Adaptive-Memory-Structures-for-LLM-Agents/meta/meta_info.txt)**: Learns to select among linear/graph/hierarchical memory structures; replaces fixed fusion thresholds with a Beta-mixture posterior criterion.
- **[SimpleMem](./references/SimpleMem-Efficient-Lifelong-Memory-for-LLM-Agents/meta/meta_info.txt)**: Uses storage-time semantic compression and recursive consolidation; reports strong LoCoMo F1 with large token savings.
- **[TiMem](./references/TiMem-Temporal-Hierarchical-Memory-Consolidation-for-Long-Horizon-Conversational-Agents/meta/meta_info.txt)**: Builds a temporal memory tree with instruction-guided consolidation and query planning/gating for retrieval depth.
- **[MemWeaver](./references/MemWeaver-Weaving-Hybrid-Memories-for-Traceable-Long-Horizon-Agentic-Reasoning/meta/meta_info.txt)**: Combines temporal KGs, experience abstractions, and passage evidence; shows strong LoCoMo gains and emphasizes traceability.
- **[EMem](./references/A-Simple-Yet-Strong-Baseline-for-Long-Term-Conversational-Memory-of-LLM-Agents/meta/meta_info.txt)**: Proposes event-centric EDUs and graph retrieval; highlights that lossy compression can hurt long-term QA.
- **[Zep / Graphiti](./references/Zep-A-Temporal-Knowledge-Graph-Architecture-for-Agent-Memory/meta/meta_info.txt)**: Temporal knowledge graph memory with edge invalidation; strong on LongMemEval and temporal reasoning.
- **[Memory OS of AI Agent](https://arxiv.org/abs/2506.06326)**: OS-inspired memory management framing for LLM agents.
- **[A-Mem](https://arxiv.org/abs/2502.12110)**: Agentic atomic memory that evolves via LLM-driven operations.
- **[MemoryBank](https://arxiv.org/abs/2305.10250)**: Early long-term personalized memory bank for dialogue agents.
- **[Generative Agents](https://arxiv.org/abs/2304.03442)**: Demonstrates reflection and memory summaries for simulated agents; motivates consolidation but uses LLM generation.
- **[HippoRAG](https://arxiv.org/abs/2405.14831)**: Neuro-inspired graph retrieval using personalized PageRank over entity–passage graphs.
- **[GraphRAG](https://arxiv.org/abs/2404.16130)**: Uses community detection and hierarchical summaries over entity graphs for retrieval-augmented QA.
- **[Lost in the Middle](https://arxiv.org/abs/2307.03172)**: Shows long-context models underutilize middle context, motivating selective retrieval/compression.
- **[RAG Survey](https://arxiv.org/abs/2312.10997)**: Surveys retrieval-augmented generation methods and retrieval/re-ranking practices.
- **[SelfCheckGPT](https://arxiv.org/abs/2303.08896)**: An LLM hallucination detection method used in some memory systems for factual-consistency checking.
- **[Memory-R1](https://arxiv.org/abs/2502.04301)**: Uses reinforcement learning to manage memory operations for agents.
- **[ENGRAM](https://arxiv.org/abs/2409.15796)**: Lightweight memory orchestration for conversational agents (memory selection/organization).
- **[SGMem](https://arxiv.org/abs/2406.15939)**: Sentence graph memory that connects sentences across sessions for retrieval.

### Taxonomy

| Family / cluster | Core idea | Representative papers | Benchmarks / evaluation | Known limitations |
|---|---|---|---|---|
| Forgetting + fusion in external memory | Maintain a memory store with decay and consolidate redundant items | FadeMem | LoCoMo (F1), MSC, synthetic long-term | Fusion/conflict often uses LLM calls; fusion text is opaque |
| LLM-mediated memory CRUD | Extract facts and update memory via LLM function calls | Mem0, A-Mem | LoCoMo (F1 + judge), DMR | Expensive; non-deterministic updates |
| Hierarchical/temporal consolidation | Build multi-level summaries/personas over time | TiMem, EverMemOS, SimpleMem | LoCoMo (often judge + F1), LongMemEval | Consolidation often uses LLM generation and thresholds |
| Structured (graph) memory | Store entities/relations/events and traverse for retrieval | Zep, MemWeaver, EMem, HippoRAG | LongMemEval, LoCoMo | Requires extraction/normalization; often LLM-heavy |
| Long-context-only baselines | Rely on longer context windows without external memory | Fixed-16K / full-context | LoCoMo | Suffers from noise + attention degradation |

### Closest Prior Work

**FadeMem** ([meta](./references/FadeMem-Biologically-Inspired-Forgetting-for-Efficient-Agent-Memory/meta/meta_info.txt)). FadeMem introduces a dual-layer decay memory with LLM-guided conflict resolution and LLM-guided fusion, plus an LLM information-preservation check. It reports that removing fusion drops LoCoMo F1 from 29.43 to 13.63 (Ablation §3.5), making fusion its largest contributor. **Our difference** is to keep FadeMem’s architecture but replace the fusion operator and preservation check with fully deterministic, quote-preserving procedures.

**SimpleMem** ([meta](./references/SimpleMem-Efficient-Lifelong-Memory-for-LLM-Agents/meta/meta_info.txt)). SimpleMem performs storage-time semantic compression (coreference/time normalization) and recursive consolidation, achieving strong LoCoMo F1 with large token savings. **Our difference** is narrower: we do not redesign indexing/retrieval; we test whether the *fusion step alone* needs an LLM, using a deterministic operator and an auditable preservation metric.

**FluxMem / Choosing How to Remember** ([meta](./references/Choosing-How-to-Remember-Adaptive-Memory-Structures-for-LLM-Agents/meta/meta_info.txt)). FluxMem replaces fixed similarity thresholds for merge decisions with a probabilistic Beta-mixture criterion and adapts memory structure, but still uses LLM-based memory formation and does not propose an LLM-free fusion operator for memory content. **Our difference** is to remove LLM generation from fusion itself.

**TiMem** ([meta](./references/TiMem-Temporal-Hierarchical-Memory-Consolidation-for-Long-Horizon-Conversational-Agents/meta/meta_info.txt)). TiMem’s gains come from hierarchical consolidation and query planning/gating, implemented via LLM prompts. **Our difference** is to avoid adding new LLM middleware and instead test a deterministic fusion hypothesis inside an existing decay-based memory.

**EMem** ([meta](./references/A-Simple-Yet-Strong-Baseline-for-Long-Term-Conversational-Memory-of-LLM-Agents/meta/meta_info.txt)). EMem argues against lossy compression and instead stores enriched event-level EDUs, relying on retrieval-time LLM filtering. **Our difference** is not to change the stored representation, but to ask whether redundancy-removal fusion can be done deterministically without losing the verbatim facts LoCoMo requires.

**Novelty Kill Search Summary:** Searched for combinations of “deterministic fusion + LoCoMo”, “LLM-free memory fusion LoCoMo”, “quote-preserving memory fusion”, and “extractive memory consolidation for conversational agents” (plus OpenReview queries for “memory fusion conversational agent”). No prior work was found that (i) targets FadeMem-style fusion specifically and (ii) replaces the fusion operator and fusion acceptance check with a fully deterministic, quote-preserving algorithm evaluated under LoCoMo’s deterministic F1 protocol (as of 2026-02-20). Full query log is in `notes.md`.

### Comparison Table

| Related work | What it does | Key limitation | What we change | Why ours should win |
|---|---|---|---|---|
| FadeMem | Decay-based dual-layer memory + LLM conflict + LLM fusion | Fusion is expensive/opaque; LLM preservation check is non-auditable | Replace fusion + preservation check with deterministic extractive fusion + coverage check | If fusion gains are mostly noise reduction, LLM rewriting is unnecessary |
| SimpleMem | Storage-time compression + recursive consolidation | Consolidation uses LLM generation; more pipeline complexity | Keep retrieval/indexing fixed; only replace fusion operator | More isolated test of whether LLM fusion is needed |
| FluxMem | Adaptive memory structures + probabilistic merge gating | Still uses LLM memory formation; merge affects structure not content operator | Deterministic content fusion in FadeMem-style clusters | Directly targets the expensive content rewrite step |
| TiMem | Temporal memory tree + LLM recall planning/gating | Extra LLM middleware; hard to make fully deterministic | No new LLM planners; deterministic fusion only | Lower latency and simpler audit surface |
| EMem | Event-level EDU storage + LLM filtering/graph retrieval | Requires LLM extraction/filtering; different representation | Keep representation but make fusion deterministic | Quote-preservation matches LoCoMo evaluation needs |

---