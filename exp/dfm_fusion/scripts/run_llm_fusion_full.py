"""Run the full LLM-Fusion baseline: 3 runs x 10 conversations.
Saves results to dfm_fusion/results/llm_fusion/.
"""

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
    config_path = Path("dfm_fusion/configs/base_config.yaml")
    config = yaml.safe_load(config_path.read_text())
    output_dir = Path("dfm_fusion/results/llm_fusion")
    output_dir.mkdir(parents=True, exist_ok=True)

    num_runs = config.get("evaluation", {}).get("num_runs", 3)
    data_path = config.get("data", {}).get("locomo_path", "external/locomo/data/locomo10.json")
    conversations = load_locomo(data_path)

    log.info(f"Starting LLM-Fusion baseline: {num_runs} runs x {len(conversations)} conversations")
    t_total = time.time()

    for run_id in range(num_runs):
        log.info(f"===== RUN {run_id}/{num_runs} =====")
        t_run = time.time()
        summary = run_experiment(
            config=config,
            condition="llm_fusion",
            run_id=run_id,
            output_dir=output_dir,
            conversations=conversations,
        )
        run_elapsed = time.time() - t_run
        log.info(f"Run {run_id} complete in {run_elapsed:.0f}s. Overall F1: {summary['overall']['overall_f1']:.4f}")
        mh_key = "multi-hop_f1"
        if mh_key in summary["overall"]:
            log.info(f"  Multi-hop F1: {summary['overall'][mh_key]:.4f}")

    agg = aggregate_runs(output_dir, num_runs)
    with open(output_dir / "aggregated.json", "w") as f:
        json.dump(agg, f, indent=2)

    total_elapsed = time.time() - t_total
    log.info(f"===== ALL RUNS COMPLETE in {total_elapsed:.0f}s =====")
    for k, v in agg.items():
        log.info(f"  {k}: {v['mean']:.4f} +/- {v['std']:.4f}")


if __name__ == "__main__":
    main()
