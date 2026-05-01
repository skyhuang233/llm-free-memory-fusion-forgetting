# Run the full DFM-Fusion experiment: 3 runs x 10 conversations.
# Loads dfm_fusion_config.yaml (not base_config). Zero extra LLM calls beyond answer gen.
# Saves results to dfm_fusion/results/dfm_fusion/.

import json
import logging
import sys
import time
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from dfm_fusion.data.locomo_loader import load_locomo
from dfm_fusion.evaluation.eval_pipeline import aggregate_runs, run_experiment

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger(__name__)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", type=int, default=None,
                        help="Run a single run_id (for parallel execution). If omitted, run all.")
    args = parser.parse_args()

    config_path = Path("dfm_fusion/configs/dfm_fusion_config.yaml")
    config = yaml.safe_load(config_path.read_text())
    output_dir = Path("dfm_fusion/results/dfm_fusion")
    output_dir.mkdir(parents=True, exist_ok=True)

    num_runs = config.get("evaluation", {}).get("num_runs", 3)
    data_path = config.get("data", {}).get("locomo_path", "external/locomo/data/locomo10.json")
    conversations = load_locomo(data_path)

    if args.run_id is not None:
        runs = [args.run_id]
    else:
        runs = list(range(num_runs))

    log.info(f"Starting DFM-Fusion: runs={runs} x {len(conversations)} conversations")
    t_total = time.time()

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
        log.info(f"Run {run_id} complete in {run_elapsed:.0f}s. Overall F1: {summary['overall']['overall_f1']:.4f}")
        mh_key = "multi-hop_f1"
        if mh_key in summary["overall"]:
            log.info(f"  Multi-hop F1: {summary['overall'][mh_key]:.4f}")

    if args.run_id is None:
        agg = aggregate_runs(output_dir, num_runs)
        with open(output_dir / "aggregated.json", "w") as f:
            json.dump(agg, f, indent=2)

        total_elapsed = time.time() - t_total
        log.info(f"===== ALL RUNS COMPLETE in {total_elapsed:.0f}s =====")
        for k, v in agg.items():
            log.info(f"  {k}: {v['mean']:.4f} +/- {v['std']:.4f}")


if __name__ == "__main__":
    main()
