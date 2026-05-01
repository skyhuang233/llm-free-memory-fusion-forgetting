# Optimized DFM-Fusion experiment runner with wandb logging.
# Changes from original: re-embed fused text, increased top_k/budget,
# lower dedup threshold, larger fusion budget, improved text formatting.
# Saves results to dfm_fusion/results/dfm_fusion_optimized/.

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


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", type=int, default=None)
    parser.add_argument("--conv-filter", type=str, default=None)
    args = parser.parse_args()

    load_dotenv()
    os.environ["WANDB_MODE"] = "offline"

    import wandb

    config_path = Path("dfm_fusion/configs/dfm_fusion_config.yaml")
    config = yaml.safe_load(config_path.read_text())
    output_dir = Path("dfm_fusion/results/dfm_fusion_optimized")
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
        name=f"dfm_fusion_optimized_runs_{'_'.join(str(r) for r in runs)}",
        config={
            "condition": "dfm_fusion_optimized",
            "num_runs": len(runs),
            "num_conversations": len(conversations),
            **config,
        },
    )

    log.info(f"Starting optimized DFM-Fusion: runs={runs} x {len(conversations)} conversations")
    log.info(f"Config overrides: top_k={config.get('retrieval_top_k')}, budget={config.get('retrieval_token_budget')}")
    log.info(f"  fusion: {config.get('fusion')}")
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
