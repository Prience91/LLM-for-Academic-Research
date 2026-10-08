#!/usr/bin/env bash
set -euo pipefail
GIIS_ROOT="${GIIS_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export GIIS_ROOT
ROOT="$GIIS_ROOT"
exec "$ROOT/scripts/run_v17_turbo_heldout_pipeline_v2.sh" "${1:?Supply a fresh held-out run namespace}" "${2:?Supply the completed main run directory}"
