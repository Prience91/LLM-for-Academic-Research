#!/usr/bin/env bash
set -euo pipefail
GIIS_ROOT="${GIIS_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export GIIS_ROOT
set -a
if [[ -f "${GIIS_TEACHER_ENV:-$HOME/.config/giis/mini_teacher.env}" ]]; then source "${GIIS_TEACHER_ENV:-$HOME/.config/giis/mini_teacher.env}"; fi
set +a
ROOT="$GIIS_ROOT"
RUN_REL="${1:?Supply a fresh held-out run namespace}"
OUT="$ROOT/$RUN_REL"
RUNTIME_LOG_DIR="$ROOT/artifacts/v17_turbo/runtime_logs/$(basename "$RUN_REL")"
QWEN_SESSION="${QWEN_SESSION:-giis-mini-qwen-teacher}"
MAIN_RUN="${2:?Supply the OSS-complete main run directory}"
mkdir -p "$RUNTIME_LOG_DIR"
cd "$ROOT"
PYTHONPATH=src python3 -m giis_prod.v17_turbo_pre_gold pre-gold --main "$MAIN_RUN" >/dev/null
python3 - "$OUT" <<'PY'
import json,pathlib,sys
out=pathlib.Path(sys.argv[1])
if (out/'status.json').exists():
    status=json.loads((out/'status.json').read_text())
    if status.get('review_routing_revision')!='split_review_v2':
        raise SystemExit('Historical held-out namespace cannot resume under new routing')
PY
if curl -fsS --max-time 3 http://127.0.0.1:8600/health >/dev/null 2>&1; then
  echo "OSS port 8600 must be free before held-out Qwen authoring" >&2
  exit 70
fi
if ! curl -fsS --max-time 3 http://127.0.0.1:8800/health >/dev/null 2>&1; then
  if ss -ltn 2>/dev/null | grep -q ':8800 '; then
    echo "Generation port 8800 is occupied by an unhealthy or unverified process" >&2
    exit 71
  fi
  available=$(awk '/MemAvailable:/ {print $2}' /proc/meminfo)
  if (( available < 104857600 )); then
    echo "Qwen preflight requires at least 100 GiB available memory" >&2
    exit 72
  fi
  tmux kill-session -t "$QWEN_SESSION" 2>/dev/null || true
  tmux new-session -d -s "$QWEN_SESSION" "'$ROOT/scripts/launch_qwen122b_teacher.sh' >> '$RUNTIME_LOG_DIR/qwen_server.log' 2>&1"
fi
healthy=0
for _ in $(seq 1 120); do
  if curl -fsS --max-time 3 http://127.0.0.1:8800/health >/dev/null 2>&1; then healthy=1; break; fi
  if ! tmux has-session -t "$QWEN_SESSION" 2>/dev/null; then break; fi
  sleep 10
done
if (( healthy != 1 )); then echo "Qwen-122B did not become healthy" >&2; exit 73; fi
PYTHONPATH=src python3 - <<'PY'
from giis_prod.config import Settings
from giis_prod.teacher_ab import running_teacher_profile
s=Settings.load(__import__('os').path.join(__import__('os').environ['GIIS_ROOT'], 'config/settings_v17_turbo.toml'))
p=running_teacher_profile(s)
actual={k:int(p[k]) for k in ('max_model_len','max_num_seqs')}
if actual != {'max_model_len':16384,'max_num_seqs':2}:
    raise SystemExit(f'Wrong held-out Qwen profile: {actual}')
PY
if [[ ! -s "$OUT/review_packets.jsonl" ]]; then
  PYTHONPATH=src ${GIIS_PYTHON:-python3} -u giis_pipeline.py --config config/settings_v17_turbo.toml author-heldout --output "$RUN_REL"
fi
exec "$ROOT/scripts/run_v17_turbo_heldout_handoff.sh" "$RUN_REL"
