"""Paired bootstrap CI and paired t-test for optimized DFM-Fusion vs LLM-Fusion and No-Fusion.
Reads per-question F1 scores from per-conversation JSON files for each condition x run,
averages across 3 runs per question, then computes paired tests on multi-hop questions."""

import json
import os
import numpy as np
from scipy import stats
from collections import defaultdict

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE, "results")
CONDITIONS = {
    "dfm_fusion_optimized": os.path.join(RESULTS_DIR, "dfm_fusion_optimized"),
    "llm_fusion": os.path.join(RESULTS_DIR, "llm_fusion"),
    "no_fusion": os.path.join(RESULTS_DIR, "no_fusion"),
}
NUM_RUNS = 3
CONV_IDS = ["conv-26", "conv-30", "conv-41", "conv-42", "conv-43",
            "conv-44", "conv-47", "conv-48", "conv-49", "conv-50"]
N_BOOTSTRAP = 10000
SEED = 42


def load_per_question_scores(condition_dir):
    all_runs = []
    for run_id in range(NUM_RUNS):
        run_dir = os.path.join(condition_dir, f"run_{run_id}")
        questions = []
        for conv_id in CONV_IDS:
            fpath = os.path.join(run_dir, f"{conv_id}.json")
            with open(fpath) as f:
                data = json.load(f)
            for qa in data["qa_results"]:
                questions.append({
                    "key": f"{conv_id}_{qa['question'][:60]}",
                    "category": qa["category_name"],
                    "f1": qa["f1"],
                })
        all_runs.append(questions)
    return all_runs


def average_across_runs(all_runs):
    by_key = defaultdict(lambda: {"f1_sum": 0.0, "count": 0, "category": None})
    for run in all_runs:
        for q in run:
            by_key[q["key"]]["f1_sum"] += q["f1"]
            by_key[q["key"]]["count"] += 1
            by_key[q["key"]]["category"] = q["category"]
    result = {}
    for key, v in by_key.items():
        result[key] = {"f1": v["f1_sum"] / v["count"], "category": v["category"]}
    return result


def paired_bootstrap_ci(diffs, n_boot=N_BOOTSTRAP, alpha=0.05, rng=None):
    if rng is None:
        rng = np.random.RandomState(SEED)
    n = len(diffs)
    boot_means = []
    for _ in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        boot_means.append(np.mean(diffs[idx]))
    boot_means = np.array(boot_means)
    lo = np.percentile(boot_means, 100 * alpha / 2)
    hi = np.percentile(boot_means, 100 * (1 - alpha / 2))
    return float(lo), float(hi)


def run_tests():
    scores = {}
    for cond_name, cond_dir in CONDITIONS.items():
        all_runs = load_per_question_scores(cond_dir)
        scores[cond_name] = average_across_runs(all_runs)

    dfm = scores["dfm_fusion_optimized"]
    llm = scores["llm_fusion"]
    nof = scores["no_fusion"]

    common_keys = set(dfm.keys()) & set(llm.keys()) & set(nof.keys())
    multihop_keys = sorted([k for k in common_keys if dfm[k]["category"] == "multi-hop"])

    dfm_mh = np.array([dfm[k]["f1"] for k in multihop_keys])
    llm_mh = np.array([llm[k]["f1"] for k in multihop_keys])
    nof_mh = np.array([nof[k]["f1"] for k in multihop_keys])

    rng = np.random.RandomState(SEED)

    diff_dfm_llm = dfm_mh - llm_mh
    t_stat_dl, p_val_dl = stats.ttest_rel(dfm_mh, llm_mh)
    ci_lo_dl, ci_hi_dl = paired_bootstrap_ci(diff_dfm_llm, rng=rng)

    diff_dfm_nof = dfm_mh - nof_mh
    t_stat_dn, p_val_dn = stats.ttest_rel(dfm_mh, nof_mh)
    ci_lo_dn, ci_hi_dn = paired_bootstrap_ci(diff_dfm_nof, rng=rng)

    diff_llm_nof = llm_mh - nof_mh
    t_stat_ln, p_val_ln = stats.ttest_rel(llm_mh, nof_mh)
    ci_lo_ln, ci_hi_ln = paired_bootstrap_ci(diff_llm_nof, rng=rng)

    dfm_mean = float(np.mean(dfm_mh))
    llm_mean = float(np.mean(llm_mh))
    nof_mean = float(np.mean(nof_mh))
    gap_recovery = (dfm_mean - nof_mean) / (llm_mean - nof_mean) * 100 if (llm_mean - nof_mean) != 0 else float('inf')

    categories = sorted(set(dfm[k]["category"] for k in common_keys))
    per_category = {}
    for cat in categories:
        cat_keys = [k for k in common_keys if dfm[k]["category"] == cat]
        per_category[cat] = {
            "n_questions": len(cat_keys),
            "dfm_fusion_optimized": float(np.mean([dfm[k]["f1"] for k in cat_keys])),
            "llm_fusion": float(np.mean([llm[k]["f1"] for k in cat_keys])),
            "no_fusion": float(np.mean([nof[k]["f1"] for k in cat_keys])),
        }

    results = {
        "note": "Statistical tests on OPTIMIZED DFM-Fusion (iteration 0 fixes applied)",
        "n_multihop_questions": len(multihop_keys),
        "dfm_vs_llm_multihop": {
            "dfm_mean_f1": dfm_mean,
            "llm_mean_f1": llm_mean,
            "mean_difference": float(np.mean(diff_dfm_llm)),
            "bootstrap_95ci": [ci_lo_dl, ci_hi_dl],
            "ci_includes_zero": ci_lo_dl <= 0 <= ci_hi_dl,
            "paired_t_test": {
                "t_statistic": float(t_stat_dl),
                "p_value": float(p_val_dl),
                "significant_at_005": p_val_dl < 0.05,
            },
        },
        "dfm_vs_nofusion_multihop": {
            "dfm_mean_f1": dfm_mean,
            "nofusion_mean_f1": nof_mean,
            "mean_difference": float(np.mean(diff_dfm_nof)),
            "bootstrap_95ci": [ci_lo_dn, ci_hi_dn],
            "ci_includes_zero": ci_lo_dn <= 0 <= ci_hi_dn,
            "paired_t_test": {
                "t_statistic": float(t_stat_dn),
                "p_value": float(p_val_dn),
                "significant_at_005": p_val_dn < 0.05,
            },
        },
        "llm_vs_nofusion_multihop": {
            "llm_mean_f1": llm_mean,
            "nofusion_mean_f1": nof_mean,
            "mean_difference": float(np.mean(diff_llm_nof)),
            "bootstrap_95ci": [ci_lo_ln, ci_hi_ln],
            "ci_includes_zero": ci_lo_ln <= 0 <= ci_hi_ln,
            "paired_t_test": {
                "t_statistic": float(t_stat_ln),
                "p_value": float(p_val_ln),
                "significant_at_005": p_val_ln < 0.05,
            },
        },
        "gap_recovery": {
            "dfm_multihop_f1": dfm_mean,
            "llm_multihop_f1": llm_mean,
            "nofusion_multihop_f1": nof_mean,
            "gap_recovery_pct": gap_recovery,
        },
        "per_category_f1": per_category,
    }

    def convert(obj):
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert(v) for v in obj]
        return obj

    results = convert(results)
    out_path = os.path.join(RESULTS_DIR, "statistical_tests_optimized.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved to {out_path}")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    run_tests()
