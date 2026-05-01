# Runner for DFM-Fusion ablation variants.
# --variant no_coverage: skip coverage check (always accept fusions)
# --variant no_truncation: skip per-entry truncation (post-concat only)

import json
import logging
import os
import sys
import time
from pathlib import Path

import yaml
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from dfm_fusion.data.locomo_loader import load_locomo
from dfm_fusion.evaluation.eval_pipeline import aggregate_runs, run_experiment

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger(__name__)

VARIANT_CONFIGS = {
    "no_coverage": "dfm_fusion/configs/dfm_ablation_no_coverage.yaml",
    "no_truncation": "dfm_fusion/configs/dfm_ablation_no_truncation.yaml",
}

VARIANT_DIRS = {
    "no_coverage": "dfm_fusion/results/dfm_ablation_no_coverage",
    "no_truncation": "dfm_fusion/results/dfm_ablation_no_truncation",
}


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", type=str, required=True, choices=["no_coverage", "no_truncation"])
    parser.add_argument("--run-id", type=int, default=None)
    parser.add_argument("--conv-filter", type=str, default=None)
    args = parser.parse_args()

    load_dotenv()
    os.environ["WANDB_MODE"] = "offline"

    import wandb

    config_path = Path(VARIANT_CONFIGS[args.variant])
    config = yaml.safe_load(config_path.read_text())
    output_dir = Path(VARIANT_DIRS[args.variant])
    output_dir.mkdir(parents=True, exist_ok=True)

    num_runs = config.get("evaluation", {}).get("num_runs", 3)
    data_path = config.get("data", {}).get("locomo_path", "external/locomo/data/locomo10.json")
    conversations = load_locomo(data_path)

    if args.conv_filter:
        conversations = [c for c in conversations if c.sample_id == args.conv_filter]

    if args.run_id is not None:
        runs = [args.run_id]
    else:
        runs = list(range(num_runs))

    wandb.init(
        project=os.environ.get("WANDB_PROJECT", "llm-free-memory-fusion-forgetting"),
        name=f"dfm_ablation_{args.variant}_runs_{'_'.join(str(r) for r in runs)}",
        config={
            "condition": f"dfm_ablation_{args.variant}",
            "variant": args.variant,
            "num_runs": len(runs),
            "num_conversations": len(conversations),
            **config,
        },
    )

    log.info(f"Starting DFM-Fusion ablation variant={args.variant}: runs={runs} x {len(conversations)} conversations")
    log.info(f"Config: {config_path}")
    t_total = time.time()

    all_summaries = []
    for run_id in runs:
        log.info(f"===== RUN {run_id}/{num_runs} =====")
        t_run = time.time()
        summary = run_experiment(
            config=config,
            condition="dfm_fusion",
            run_id=run_id,
            output_dir=output_dir,
            conversations=conversations,
        )
        run_elapsed = time.time() - t_run
        all_summaries.append(summary)

        log.info(f"Run {run_id} complete in {run_elapsed:.0f}s.")
        for k, v in summary["overall"].items():
            if k.endswith("_f1"):
                log.info(f"  {k}: {v:.4f}")
                wandb.log({f"run_{run_id}/{k}": v})
        wandb.log({f"run_{run_id}/elapsed_s": run_elapsed})

        abl_stats = summary.get("ablation_stats", {})
        if args.variant == "no_coverage":
            wandb.log({
                f"run_{run_id}/coverage_would_have_rejected": abl_stats.get("coverage_would_have_rejected", 0),
                f"run_{run_id}/coverage_recall_mean": abl_stats.get("coverage_recall_mean", 0),
            })
            log.info(f"  Ablation: would_have_rejected={abl_stats.get('coverage_would_have_rejected', 0)}, "
                      f"coverage_recall_mean={abl_stats.get('coverage_recall_mean', 0):.4f}")
        elif args.variant == "no_truncation":
            wandb.log({
                f"run_{run_id}/length_dominance_fraction": abl_stats.get("length_dominance_fraction", 0),
                f"run_{run_id}/length_dominance_count": abl_stats.get("length_dominance_count", 0),
            })
            log.info(f"  Ablation: length_dominance_frac={abl_stats.get('length_dominance_fraction', 0):.4f}, "
                      f"count={abl_stats.get('length_dominance_count', 0)}/{abl_stats.get('total_retrieve_calls', 0)}")

    if args.run_id is None and len(runs) > 1:
        agg = aggregate_runs(output_dir, num_runs)
        with open(output_dir / "aggregated.json", "w") as f:
            json.dump(agg, f, indent=2)

        total_elapsed = time.time() - t_total
        log.info(f"===== ALL RUNS COMPLETE in {total_elapsed:.0f}s =====")
        for k, v in agg.items():
            log.info(f"  {k}: {v['mean']:.4f} +/- {v['std']:.4f}")
            wandb.log({f"agg/{k}_mean": v["mean"], f"agg/{k}_std": v["std"]})
        wandb.log({"total_elapsed_s": total_elapsed})

    wandb.finish()


if __name__ == "__main__":
    main()
