#!/bin/bash
# Load environment variables from .env and activate venv
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

source "$PROJECT_ROOT/.venv/bin/activate"

if [ -f "$PROJECT_ROOT/.env" ]; then
    set -a
    source "$PROJECT_ROOT/.env"
    set +a
fi

export OPENAI_API_KEY="${LEMMA_MAAS_API_KEY}"
export OPENAI_BASE_URL="http://${LEMMA_MAAS_BASE_URL}/v1"
