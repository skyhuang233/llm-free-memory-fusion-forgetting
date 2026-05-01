# Statistical comparison of DFM-Fusion vs LLM-Fusion and No-Fusion baselines.
# Paired bootstrap 95% CI, paired t-test, and gap recovery metric on multi-hop F1.
# Saves results to dfm_fusion/results/statistical_tests.json.

import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


def load_per_question_scores(results_dir: Path, num_runs: int = 3) -> list[dict]:
    all_runs = []
    for run_id in range(num_runs):
        run_dir = results_dir / f"run_{run_id}"
        qa_scores = {}
        for conv_file in sorted(run_dir.glob("conv-*.json")):
            data = json.loads(conv_file.read_text())
            for qa in data["qa_results"]:
                key = (data["sample_id"], qa["question"])
                qa_scores[key] = {
                    "f1": qa["f1"],
                    "category": qa["category"],
                    "category_name": qa["category_name"],
                }
        all_runs.append(qa_scores)
    return all_runs


def paired_bootstrap_ci(scores_a, scores_b, n_bootstrap=10000, alpha=0.05, seed=42):
    rng = np.random.RandomState(seed)
    n = len(scores_a)
    diffs = np.array(scores_a) - np.array(scores_b)
    boot_means = []
    for _ in range(n_bootstrap):
        idx = rng.randint(0, n, size=n)
        boot_means.append(np.mean(diffs[idx]))
    boot_means = sorted(boot_means)
    lo = boot_means[int(n_bootstrap * alpha / 2)]
    hi = boot_means[int(n_bootstrap * (1 - alpha / 2))]
    return float(np.mean(diffs)), float(lo), float(hi)


def get_category_scores(runs_a, runs_b, category_id):
    all_keys = set()
    for r in runs_a:
        all_keys.update(r.keys())
    for r in runs_b:
        all_keys.update(r.keys())

    a_scores = []
    b_scores = []
    for key in sorted(all_keys):
        a_vals = [r[key]["f1"] for r in runs_a if key in r and r[key]["category"] == category_id]
        b_vals = [r[key]["f1"] for r in runs_b if key in r and r[key]["category"] == category_id]
        if a_vals and b_vals:
            a_scores.append(np.mean(a_vals))
            b_scores.append(np.mean(b_vals))
    return np.array(a_scores), np.array(b_scores)


def compare_pair(name_a, name_b, runs_a, runs_b, category_id=1):
    a_scores, b_scores = get_category_scores(runs_a, runs_b, category_id)
    if len(a_scores) == 0:
        return None

    mean_a = float(np.mean(a_scores))
    mean_b = float(np.mean(b_scores))
    mean_diff, ci_lo, ci_hi = paired_bootstrap_ci(a_scores.tolist(), b_scores.tolist())
    t_stat, p_val = stats.ttest_rel(a_scores, b_scores)

    return {
        "comparison": f"{name_a} vs {name_b}",
        "n_questions": len(a_scores),
        f"{name_a}_mean_f1": mean_a,
        f"{name_b}_mean_f1": mean_b,
        "mean_difference": mean_diff,
        "bootstrap_95ci": [ci_lo, ci_hi],
        "paired_t_test": {
            "t_statistic": float(t_stat),
            "p_value": float(p_val),
            "significant_at_005": bool(p_val < 0.05),
        },
        "ci_includes_zero": bool(ci_lo <= 0 <= ci_hi),
    }


def main():
    dfm_dir = Path("dfm_fusion/results/dfm_fusion")
    llm_dir = Path("dfm_fusion/results/llm_fusion")
    nofus_dir = Path("dfm_fusion/results/no_fusion")

    for d, name in [(dfm_dir, "dfm_fusion"), (llm_dir, "llm_fusion"), (nofus_dir, "no_fusion")]:
        if not d.exists():
            print(f"ERROR: {name} results directory not found: {d}")
            sys.exit(1)

    num_runs = 3
    dfm_runs = load_per_question_scores(dfm_dir, num_runs)
    llm_runs = load_per_question_scores(llm_dir, num_runs)
    nofus_runs = load_per_question_scores(nofus_dir, num_runs)

    print("=" * 70)
    print("STATISTICAL COMPARISON: DFM-Fusion vs Baselines")
    print("=" * 70)

    dfm_agg = json.loads((dfm_dir / "aggregated.json").read_text())
    llm_agg = json.loads((llm_dir / "aggregated.json").read_text())
    nofus_agg = json.loads((nofus_dir / "aggregated.json").read_text())

    print("\n--- Aggregated Multi-hop F1 ---")
    print(f"  LLM-Fusion:  {llm_agg['multi-hop_f1']['mean']:.4f} +/- {llm_agg['multi-hop_f1']['std']:.4f}")
    print(f"  DFM-Fusion:  {dfm_agg['multi-hop_f1']['mean']:.4f} +/- {dfm_agg['multi-hop_f1']['std']:.4f}")
    print(f"  No-Fusion:   {nofus_agg['multi-hop_f1']['mean']:.4f} +/- {nofus_agg['multi-hop_f1']['std']:.4f}")

    results = {}

    print("\n--- DFM-Fusion vs LLM-Fusion (multi-hop) ---")
    cmp_dfm_llm = compare_pair("dfm_fusion", "llm_fusion", dfm_runs, llm_runs, category_id=1)
    if cmp_dfm_llm:
        print(f"  DFM mean: {cmp_dfm_llm['dfm_fusion_mean_f1']:.4f}")
        print(f"  LLM mean: {cmp_dfm_llm['llm_fusion_mean_f1']:.4f}")
        print(f"  Diff: {cmp_dfm_llm['mean_difference']:+.4f}")
        print(f"  Bootstrap 95% CI: [{cmp_dfm_llm['bootstrap_95ci'][0]:+.4f}, {cmp_dfm_llm['bootstrap_95ci'][1]:+.4f}]")
        print(f"  Paired t-test: t={cmp_dfm_llm['paired_t_test']['t_statistic']:.3f}, p={cmp_dfm_llm['paired_t_test']['p_value']:.4f}")
        print(f"  CI includes 0: {cmp_dfm_llm['ci_includes_zero']}")
        results["dfm_vs_llm_multihop"] = cmp_dfm_llm

    print("\n--- DFM-Fusion vs No-Fusion (multi-hop) ---")
    cmp_dfm_nofus = compare_pair("dfm_fusion", "no_fusion", dfm_runs, nofus_runs, category_id=1)
    if cmp_dfm_nofus:
        print(f"  DFM mean: {cmp_dfm_nofus['dfm_fusion_mean_f1']:.4f}")
        print(f"  NoFus mean: {cmp_dfm_nofus['no_fusion_mean_f1']:.4f}")
        print(f"  Diff: {cmp_dfm_nofus['mean_difference']:+.4f}")
        print(f"  Bootstrap 95% CI: [{cmp_dfm_nofus['bootstrap_95ci'][0]:+.4f}, {cmp_dfm_nofus['bootstrap_95ci'][1]:+.4f}]")
        print(f"  Paired t-test: t={cmp_dfm_nofus['paired_t_test']['t_statistic']:.3f}, p={cmp_dfm_nofus['paired_t_test']['p_value']:.4f}")
        print(f"  Significant at 0.05: {cmp_dfm_nofus['paired_t_test']['significant_at_005']}")
        results["dfm_vs_nofusion_multihop"] = cmp_dfm_nofus

    f1_llm = llm_agg["multi-hop_f1"]["mean"]
    f1_nofus = nofus_agg["multi-hop_f1"]["mean"]
    f1_dfm = dfm_agg["multi-hop_f1"]["mean"]

    gap = f1_llm - f1_nofus
    if gap > 0:
        recovery = (f1_dfm - f1_nofus) / gap * 100
    else:
        recovery = float("nan")

    print(f"\n--- Gap Recovery Metric ---")
    print(f"  LLM-Fusion multi-hop F1:  {f1_llm:.4f}")
    print(f"  No-Fusion multi-hop F1:   {f1_nofus:.4f}")
    print(f"  DFM-Fusion multi-hop F1:  {f1_dfm:.4f}")
    print(f"  NoFusion->LLMFusion gap:  {gap:+.4f}")
    print(f"  DFM gap recovery:         {recovery:.1f}%")

    results["gap_recovery"] = {
        "llm_fusion_multihop_f1": f1_llm,
        "no_fusion_multihop_f1": f1_nofus,
        "dfm_fusion_multihop_f1": f1_dfm,
        "nofusion_to_llmfusion_gap": gap,
        "dfm_gap_recovery_pct": recovery,
    }

    cat_map = {1: "multi-hop", 2: "temporal", 3: "open-domain", 4: "single-hop", 5: "adversarial"}
    per_category = {}
    print("\n--- Per-Category Comparison ---")
    print(f"{'Category':<15} {'LLM-Fusion':>12} {'DFM-Fusion':>12} {'No-Fusion':>12} {'DFM-LLM':>10} {'DFM-NoFus':>10}")
    print("-" * 71)
    for cat_id, cat_name in sorted(cat_map.items()):
        key = f"{cat_name}_f1"
        if key in llm_agg and key in dfm_agg and key in nofus_agg:
            lm = llm_agg[key]["mean"]
            dm = dfm_agg[key]["mean"]
            nm = nofus_agg[key]["mean"]
            print(f"{cat_name:<15} {lm:>12.4f} {dm:>12.4f} {nm:>12.4f} {dm-lm:>+10.4f} {dm-nm:>+10.4f}")
            per_category[cat_name] = {"llm_fusion": lm, "dfm_fusion": dm, "no_fusion": nm}

    results["per_category_f1"] = per_category

    overall_key = "overall_f1"
    if overall_key in llm_agg and overall_key in dfm_agg and overall_key in nofus_agg:
        results["overall_f1"] = {
            "llm_fusion": llm_agg[overall_key]["mean"],
            "dfm_fusion": dfm_agg[overall_key]["mean"],
            "no_fusion": nofus_agg[overall_key]["mean"],
        }

    out_path = Path("dfm_fusion/results/statistical_tests.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
