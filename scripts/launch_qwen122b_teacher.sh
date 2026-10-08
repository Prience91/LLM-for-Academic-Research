#!/usr/bin/env bash
set -euo pipefail
GIIS_ROOT="${GIIS_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export GIIS_ROOT
export PATH="${GIIS_RUNTIME_BIN:-$(dirname "$(command -v python3)")}:$PATH"
set -a
if [[ -f "${GIIS_TEACHER_ENV:-$HOME/.config/giis/mini_teacher.env}" ]]; then source "${GIIS_TEACHER_ENV:-$HOME/.config/giis/mini_teacher.env}"; fi
set +a
cd "$GIIS_ROOT"
exec env PYTHONPATH=src ${GIIS_PYTHON:-python3} giis_pipeline.py --config config/settings_v17_turbo.toml teacher-serve
