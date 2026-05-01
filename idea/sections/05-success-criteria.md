## Success Criteria

**Hypothesis** (directional): DeterministicFusion will close most of the No-Fusion→LLM-Fusion gap, yielding LoCoMo multi-hop F1 close to LLM-Fusion under the same retrieval budget.

**Decision Rule** (concrete):
- **Continue/Proceed** if DeterministicFusion is **not statistically worse** than LLM-Fusion on per-question multi-hop F1 (paired bootstrap 95% CI includes 0 difference, or paired t-test p≥0.05), and both beat No-Fusion by a statistically distinguishable margin.
- **Pivot** if DeterministicFusion is worse than LLM-Fusion but still significantly better than No-Fusion; then iterate on the deterministic operator (e.g., add rule-based coreference/time normalization from SimpleMem) while keeping the same evaluation harness.
- **Refute** if DeterministicFusion is statistically significantly worse than LLM-Fusion and close to No-Fusion (i.e., fails to recover the fusion gain), implying that LLM rewriting/normalization is necessary for this benchmark.

---