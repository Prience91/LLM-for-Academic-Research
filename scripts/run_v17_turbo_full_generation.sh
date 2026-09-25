#!/usr/bin/env bash
set -euo pipefail
ROOT="${GIIS_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
TEACHER_ENV_FILE="${GIIS_TEACHER_ENV_FILE:-$HOME/.config/giis/teacher.env}"
set -a
source "$TEACHER_ENV_FILE"
set +a
cd "$ROOT"
exec env PYTHONPATH=src "$PYTHON_BIN" -u -m giis_prod.v17_turbo_canary run --out artifacts/v17_turbo/sft_generation_1000
