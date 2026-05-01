# Ablation analysis: compare DFM-Fusion variants (w/o coverage check, w/o per-entry truncation)
# against full DFM-Fusion and No-Fusion baseline using per-question paired t-tests.

import json
import logging
import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger(__name__)

RESULTS_DIRS = {
    "full_dfm": Path("dfm_fusion/results/dfm_fusion_optimized"),
    "no_coverage": Path("dfm_fusion/results/dfm_ablation_no_coverage"),
    "no_truncation": Path("dfm_fusion/results/dfm_ablation_no_truncation"),
    "no_fusion": Path("dfm_fusion/results/no_fusion"),
}

NUM_RUNS = 3
MULTI_HOP_CATEGORY = 1


def load_per_question_f1(results_dir: Path, num_runs: int = 3) -> dict:
    """Load per-question F1 scores across runs. Returns {question: [f1_run0, f1_run1, ...]}"""
    question_f1s: dict[str, list[float]] = {}
    for run_id in range(num_runs):
        run_dir = results_dir / f"run_{run_id}"
        if not run_dir.exists():
            log.warning(f"Run dir not found: {run_dir}")
            continue
        for conv_file in sorted(run_dir.glob("conv-*.json")):
            with open(conv_file) as f:
                data = json.load(f)
            for qa in data["qa_results"]:
                key = f"{data['sample_id']}::{qa['question']}"
                if key not in question_f1s:
                    question_f1s[key] = []
                question_f1s[key].append(qa["f1"])
    return question_f1s


def load_per_question_multihop_f1(results_dir: Path, num_runs: int = 3) -> dict:
    """Load per-question F1 for multi-hop questions only."""
    question_f1s: dict[str, list[float]] = {}
    for run_id in range(num_runs):
        run_dir = results_dir / f"run_{run_id}"
        if not run_dir.exists():
            continue
        for conv_file in sorted(run_dir.glob("conv-*.json")):
            with open(conv_file) as f:
                data = json.load(f)
            for qa in data["qa_results"]:
                if qa["category"] != MULTI_HOP_CATEGORY:
                    continue
                key = f"{data['sample_id']}::{qa['question']}"
                if key not in question_f1s:
                    question_f1s[key] = []
                question_f1s[key].append(qa["f1"])
    return question_f1s


def load_aggregated(results_dir: Path) -> dict:
    agg_path = results_dir / "aggregated.json"
    if agg_path.exists():
        return json.load(open(agg_path))
    return {}


def load_ablation_stats(results_dir: Path, num_runs: int = 3) -> dict:
    """Aggregate ablation stats across runs."""
    stats_all = []
    for run_id in range(num_runs):
        summary_path = results_dir / f"run_{run_id}" / "summary.json"
        if summary_path.exists():
            with open(summary_path) as f:
                summary = json.load(f)
            stats_all.append(summary.get("ablation_stats", {}))
    return stats_all


def paired_ttest(condition_a: dict, condition_b: dict) -> dict:
    """Paired t-test on per-question mean F1 (averaged across runs per question)."""
    common_keys = sorted(set(condition_a.keys()) & set(condition_b.keys()))
    if not common_keys:
        return {"error": "no common questions"}

    a_means = [np.mean(condition_a[k]) for k in common_keys]
    b_means = [np.mean(condition_b[k]) for k in common_keys]

    t_stat, p_value = stats.ttest_rel(a_means, b_means)
    diff = np.mean(a_means) - np.mean(b_means)

    return {
        "n_questions": len(common_keys),
        "mean_a": float(np.mean(a_means)),
        "mean_b": float(np.mean(b_means)),
        "mean_difference": float(diff),
        "t_statistic": float(t_stat),
        "p_value": float(p_value),
        "significant_at_0.05": bool(p_value < 0.05),
    }


def main():
    log.info("Loading per-question multi-hop F1 scores for all conditions...")

    mh_f1s = {}
    for name, rdir in RESULTS_DIRS.items():
        mh_f1s[name] = load_per_question_multihop_f1(rdir)
        log.info(f"  {name}: {len(mh_f1s[name])} multi-hop questions loaded")

    log.info("Loading aggregated metrics...")
    aggs = {}
    for name, rdir in RESULTS_DIRS.items():
        aggs[name] = load_aggregated(rdir)
        if aggs[name]:
            mh = aggs[name].get("multi_hop_f1", aggs[name].get("multi-hop_f1", {}))
            log.info(f"  {name}: multi-hop F1 = {mh.get('mean', 'N/A')}")

    log.info("Running paired t-tests (multi-hop F1)...")

    ttest_no_cov = paired_ttest(mh_f1s["full_dfm"], mh_f1s["no_coverage"])
    log.info(f"  full_dfm vs no_coverage: diff={ttest_no_cov['mean_difference']:.4f}, p={ttest_no_cov['p_value']:.4f}")

    ttest_no_trunc = paired_ttest(mh_f1s["full_dfm"], mh_f1s["no_truncation"])
    log.info(f"  full_dfm vs no_truncation: diff={ttest_no_trunc['mean_difference']:.4f}, p={ttest_no_trunc['p_value']:.4f}")

    ttest_no_fusion = paired_ttest(mh_f1s["full_dfm"], mh_f1s["no_fusion"])
    log.info(f"  full_dfm vs no_fusion: diff={ttest_no_fusion['mean_difference']:.4f}, p={ttest_no_fusion['p_value']:.4f}")

    log.info("Loading ablation-specific stats...")
    no_cov_stats = load_ablation_stats(RESULTS_DIRS["no_coverage"])
    no_trunc_stats = load_ablation_stats(RESULTS_DIRS["no_truncation"])

    total_would_rejected = sum(s.get("coverage_would_have_rejected", 0) for s in no_cov_stats)
    cov_recalls = [s.get("coverage_recall_mean", 0) for s in no_cov_stats if s.get("coverage_recall_mean", 0) > 0]
    avg_cov_recall = float(np.mean(cov_recalls)) if cov_recalls else 0.0

    len_dom_fracs = [s.get("length_dominance_fraction", 0) for s in no_trunc_stats]
    avg_len_dom_frac = float(np.mean(len_dom_fracs)) if len_dom_fracs else 0.0
    total_len_dom = sum(s.get("length_dominance_count", 0) for s in no_trunc_stats)
    total_retrieves = sum(s.get("total_retrieve_calls", 0) for s in no_trunc_stats)

    def get_mh_metric(agg, key_options=("multi_hop_f1", "multi-hop_f1")):
        for k in key_options:
            if k in agg:
                return agg[k]
        return {"mean": "N/A", "std": "N/A"}

    full_dfm_mh = get_mh_metric(aggs["full_dfm"])
    no_cov_mh = get_mh_metric(aggs["no_coverage"])
    no_trunc_mh = get_mh_metric(aggs["no_truncation"])
    no_fusion_mh = get_mh_metric(aggs["no_fusion"])

    def safe_delta(a, b):
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            return float(a - b)
        return "N/A"

    analysis = {
        "comparison_table": {
            "full_dfm_fusion": {
                "multi_hop_f1_mean": full_dfm_mh.get("mean"),
                "multi_hop_f1_std": full_dfm_mh.get("std"),
                "delta_from_full_dfm": 0.0,
            },
            "wo_coverage_check": {
                "multi_hop_f1_mean": no_cov_mh.get("mean"),
                "multi_hop_f1_std": no_cov_mh.get("std"),
                "delta_from_full_dfm": safe_delta(no_cov_mh.get("mean"), full_dfm_mh.get("mean")),
            },
            "wo_per_entry_truncation": {
                "multi_hop_f1_mean": no_trunc_mh.get("mean"),
                "multi_hop_f1_std": no_trunc_mh.get("std"),
                "delta_from_full_dfm": safe_delta(no_trunc_mh.get("mean"), full_dfm_mh.get("mean")),
            },
            "no_fusion_baseline": {
                "multi_hop_f1_mean": no_fusion_mh.get("mean"),
                "multi_hop_f1_std": no_fusion_mh.get("std"),
                "delta_from_full_dfm": safe_delta(no_fusion_mh.get("mean"), full_dfm_mh.get("mean")),
            },
        },
        "paired_ttests": {
            "full_dfm_vs_wo_coverage": ttest_no_cov,
            "full_dfm_vs_wo_truncation": ttest_no_trunc,
            "full_dfm_vs_no_fusion": ttest_no_fusion,
        },
        "ablation_diagnostics": {
            "wo_coverage_check": {
                "total_fusions_would_have_rejected": total_would_rejected,
                "average_coverage_recall": avg_cov_recall,
                "interpretation": (
                    "Coverage check never rejected fusions (all recall >= theta_cov=0.85). "
                    "The coverage gate is a safety net that did not trigger under current config."
                ) if total_would_rejected == 0 else (
                    f"{total_would_rejected} fusions would have been rejected by coverage check. "
                    "Removing the gate allows destructive merges that lose salient information."
                ),
            },
            "wo_per_entry_truncation": {
                "total_length_dominance_queries": total_len_dom,
                "total_retrieve_calls": total_retrieves,
                "length_dominance_fraction": avg_len_dom_frac,
                "interpretation": (
                    f"{avg_len_dom_frac:.1%} of queries had first item consuming >50% of budget. "
                    "Without per-entry truncation, longer items dominate the retrieval context."
                ),
            },
        },
        "overall_interpretation": "",
    }

    interp_parts = []
    if ttest_no_cov.get("significant_at_0.05"):
        interp_parts.append(
            "Removing the coverage check causes a statistically significant drop in multi-hop F1 "
            f"(p={ttest_no_cov['p_value']:.4f}), indicating the coverage gate prevents destructive merges "
            f"({total_would_rejected} fusions would have been rejected)."
        )
    else:
        if total_would_rejected > 0:
            interp_parts.append(
                "Removing the coverage check does NOT cause a significant drop in multi-hop F1 "
                f"(p={ttest_no_cov['p_value']:.4f}). Although {total_would_rejected} fusions would have been "
                f"rejected by the coverage gate, their impact on downstream QA is small. "
                "The coverage check acts as a safety net with minimal overhead."
            )
        else:
            interp_parts.append(
                "Removing the coverage check does NOT cause a significant drop in multi-hop F1 "
                f"(p={ttest_no_cov['p_value']:.4f}). The coverage check never rejected any fusions with "
                "theta_cov=0.85, acting as a safety net that did not need to trigger."
            )

    if ttest_no_trunc.get("significant_at_0.05"):
        interp_parts.append(
            "Removing per-entry truncation causes a statistically significant drop in multi-hop F1 "
            f"(p={ttest_no_trunc['p_value']:.4f}), confirming that per-entry budget control prevents "
            "length-biased retrieval where fused/long items dominate the context."
        )
    else:
        interp_parts.append(
            "Removing per-entry truncation does NOT cause a significant drop in multi-hop F1 "
            f"(p={ttest_no_trunc['p_value']:.4f}). Per-entry truncation may not be critical "
            "when retrieved items are relatively short or uniform in length."
        )

    analysis["overall_interpretation"] = " ".join(interp_parts)

    out_path = Path("dfm_fusion/results/ablation_analysis.json")
    with open(out_path, "w") as f:
        json.dump(analysis, f, indent=2)
    log.info(f"Analysis saved to {out_path}")

    log.info("\n===== ABLATION COMPARISON TABLE =====")
    log.info(f"{'Variant':<30} {'Multi-hop F1':>20} {'Delta':>12}")
    log.info("-" * 65)
    for name, entry in analysis["comparison_table"].items():
        m = entry["multi_hop_f1_mean"]
        s = entry["multi_hop_f1_std"]
        d = entry["delta_from_full_dfm"]
        m_str = f"{m:.4f} +/- {s:.4f}" if isinstance(m, float) else str(m)
        d_str = f"{d:+.4f}" if isinstance(d, float) else str(d)
        log.info(f"  {name:<28} {m_str:>20} {d_str:>12}")


if __name__ == "__main__":
    main()
