"""Compare No-Fusion vs LLM-Fusion results and validate the fusion gap.
Loads per-question predictions from both conditions, computes paired
bootstrap CIs and paired t-test on multi-hop F1, and prints summary.
"""

import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


def load_per_question_scores(results_dir: Path, num_runs: int = 3) -> dict:
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
                    "prediction": qa["prediction"],
                    "ground_truth": qa["ground_truth"],
                }
        all_runs.append(qa_scores)
    return all_runs


def paired_bootstrap_ci(scores_a, scores_b, n_bootstrap=10000, alpha=0.05, seed=42):
    rng = np.random.RandomState(seed)
    n = len(scores_a)
    diffs = np.array(scores_a) - np.array(scores_b)
    boot_diffs = []
    for _ in range(n_bootstrap):
        idx = rng.randint(0, n, size=n)
        boot_diffs.append(np.mean(diffs[idx]))
    boot_diffs = sorted(boot_diffs)
    lo = boot_diffs[int(n_bootstrap * alpha / 2)]
    hi = boot_diffs[int(n_bootstrap * (1 - alpha / 2))]
    return float(np.mean(diffs)), float(lo), float(hi)


def main():
    llm_dir = Path("dfm_fusion/results/llm_fusion")
    nofus_dir = Path("dfm_fusion/results/no_fusion")

    if not llm_dir.exists() or not nofus_dir.exists():
        print("ERROR: Both llm_fusion and no_fusion results directories must exist.")
        sys.exit(1)

    llm_agg_path = llm_dir / "aggregated.json"
    nofus_agg_path = nofus_dir / "aggregated.json"

    print("=" * 70)
    print("FUSION GAP ANALYSIS: LLM-Fusion vs No-Fusion")
    print("=" * 70)

    if llm_agg_path.exists() and nofus_agg_path.exists():
        llm_agg = json.loads(llm_agg_path.read_text())
        nofus_agg = json.loads(nofus_agg_path.read_text())

        print("\n--- Aggregated Metrics (mean +/- std over 3 runs) ---")
        print(f"{'Metric':<25} {'LLM-Fusion':>15} {'No-Fusion':>15} {'Delta':>12} {'Rel %':>10}")
        print("-" * 77)

        for key in sorted(set(list(llm_agg.keys()) + list(nofus_agg.keys()))):
            if key in llm_agg and key in nofus_agg:
                lm = llm_agg[key]["mean"]
                ls = llm_agg[key]["std"]
                nm = nofus_agg[key]["mean"]
                ns = nofus_agg[key]["std"]
                delta = lm - nm
                rel = (delta / nm * 100) if nm > 0 else float("inf")
                print(f"{key:<25} {lm:>10.4f}+/-{ls:<4.4f} {nm:>10.4f}+/-{ns:<4.4f} {delta:>+10.4f} {rel:>+9.1f}%")

    num_runs = 3
    llm_runs = load_per_question_scores(llm_dir, num_runs)
    nofus_runs = load_per_question_scores(nofus_dir, num_runs)

    print("\n--- Paired Per-Question Analysis (averaged across runs) ---")

    all_keys = set()
    for r in llm_runs:
        all_keys.update(r.keys())
    for r in nofus_runs:
        all_keys.update(r.keys())
    common_keys = sorted(all_keys)

    cat_map = {1: "multi-hop", 2: "temporal", 3: "open-domain", 4: "single-hop", 5: "adversarial"}

    for cat_id, cat_name in sorted(cat_map.items()):
        llm_scores = []
        nofus_scores = []
        for key in common_keys:
            llm_vals = [r[key]["f1"] for r in llm_runs if key in r and r[key]["category"] == cat_id]
            nofus_vals = [r[key]["f1"] for r in nofus_runs if key in r and r[key]["category"] == cat_id]
            if llm_vals and nofus_vals:
                llm_scores.append(np.mean(llm_vals))
                nofus_scores.append(np.mean(nofus_vals))

        if not llm_scores:
            continue

        llm_arr = np.array(llm_scores)
        nofus_arr = np.array(nofus_scores)

        llm_mean = np.mean(llm_arr)
        nofus_mean = np.mean(nofus_arr)
        n_q = len(llm_arr)

        t_stat, p_val = stats.ttest_rel(llm_arr, nofus_arr)

        mean_diff, ci_lo, ci_hi = paired_bootstrap_ci(llm_scores, nofus_scores)

        print(f"\n  {cat_name} (n={n_q} questions)")
        print(f"    LLM-Fusion mean F1: {llm_mean:.4f}")
        print(f"    No-Fusion  mean F1: {nofus_mean:.4f}")
        print(f"    Delta:              {llm_mean - nofus_mean:+.4f}")
        print(f"    Paired t-test:      t={t_stat:.3f}, p={p_val:.4f} {'***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'ns'}")
        print(f"    Bootstrap 95% CI:   [{ci_lo:+.4f}, {ci_hi:+.4f}]")

    print("\n--- FadeMem Reference Comparison ---")
    print(f"  FadeMem published:  Fusion=29.43, No-Fusion=13.63, Gap=-53.7%")
    if llm_agg_path.exists() and nofus_agg_path.exists():
        llm_mh = llm_agg.get("multi-hop_f1", {}).get("mean", 0)
        nofus_mh = nofus_agg.get("multi-hop_f1", {}).get("mean", 0)
        if nofus_mh > 0:
            our_gap_pct = (llm_mh - nofus_mh) / nofus_mh * 100
        else:
            our_gap_pct = float("inf")
        print(f"  Our harness:        Fusion={llm_mh*100:.2f}, No-Fusion={nofus_mh*100:.2f}, Gap={our_gap_pct:+.1f}%")
        if llm_mh > nofus_mh:
            print("  PASS: LLM-Fusion > No-Fusion (directional match)")
        else:
            print("  WARNING: Gap is negligible or reversed -- potential harness bug!")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
