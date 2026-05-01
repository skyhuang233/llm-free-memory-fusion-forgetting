#!/bin/bash
# Run LLM-Fusion baseline (FadeMem-style) on LoCoMo
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/setup_env.sh"
python -m dfm_fusion.evaluation.eval_pipeline --config dfm_fusion/configs/llm_fusion.yaml "$@"
