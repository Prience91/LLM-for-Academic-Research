#!/usr/bin/env bash
set -euo pipefail
ROOT="${GIIS_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
export GIIS_ROOT="$ROOT"
OUT="$ROOT/artifacts/v17_turbo/heldout_authoring"
cd "$ROOT"
if [[ ! -s "$OUT/review_packets_oss_single_source.jsonl" ]]; then
  PYTHONPATH=src "$PYTHON_BIN" -u giis_pipeline.py --config config/settings_v17_turbo.toml author-heldout --output artifacts/v17_turbo/heldout_authoring
fi
exec "$ROOT/scripts/run_v17_turbo_heldout_review.sh"
