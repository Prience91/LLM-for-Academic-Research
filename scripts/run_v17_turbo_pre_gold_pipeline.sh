#!/usr/bin/env bash
# One normal-path launch: main generation -> OSS -> held-out authoring -> both frozen gold queues.
set -euo pipefail
GIIS_ROOT="${GIIS_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export GIIS_ROOT
set -a
if [[ -f "${GIIS_TEACHER_ENV:-$HOME/.config/giis/mini_teacher.env}" ]]; then source "${GIIS_TEACHER_ENV:-$HOME/.config/giis/mini_teacher.env}"; fi
set +a
ROOT="$GIIS_ROOT"
MAIN_REL="${1:?Supply a fresh planned main run namespace}"
HELD_REL="${2:?Supply a fresh held-out run namespace}"
QWEN_SESSION="${QWEN_SESSION:-giis-mini-qwen-teacher}"
export PATH="${GIIS_RUNTIME_BIN:-$(dirname "$(command -v python3)")}:$PATH"
export PYTHONPATH="$ROOT/src"
cd "$ROOT"
python3 - "$MAIN_REL" "$HELD_REL" <<'PY'
import pathlib,sys
root=pathlib.Path(__import__('os').path.join(__import__('os').environ['GIIS_ROOT'], '')).resolve()
base=(root/'artifacts/v17_turbo').resolve()
if pathlib.Path(sys.argv[1]).is_absolute() or pathlib.Path(sys.argv[2]).is_absolute():
    raise SystemExit('Supply relative run namespaces below artifacts/v17_turbo')
main=(root/sys.argv[1]).resolve();held=(root/sys.argv[2]).resolve()
if not main.is_relative_to(base) or not held.is_relative_to(base) or main==held:
    raise SystemExit('Both distinct run namespaces must be within artifacts/v17_turbo')
if not (main/'plan.json').is_file(): raise SystemExit('Main frozen plan is missing')
if held.exists() and any(held.iterdir()) and not (held/'status.json').is_file():
    raise SystemExit('Held-out namespace is not fresh or resumable')
PY
if curl -fsS --max-time 3 http://127.0.0.1:8600/health >/dev/null 2>&1; then
  echo "OSS is still active; Qwen and OSS must remain mutually exclusive" >&2
  exit 70
fi
if ! curl -fsS --max-time 3 http://127.0.0.1:8800/health >/dev/null 2>&1; then
  if ss -ltn 2>/dev/null | grep -q ':8800 '; then
    echo "Qwen port 8800 has an unhealthy or foreign process" >&2
    exit 71
  fi
  available=$(awk '/MemAvailable:/ {print $2}' /proc/meminfo)
  if (( available < 104857600 )); then
    echo "Qwen preflight requires at least 100 GiB available memory" >&2
    exit 72
  fi
  tmux new-session -d -s "$QWEN_SESSION" "bash '$ROOT/scripts/launch_qwen122b_teacher.sh' >> '$ROOT/$MAIN_REL/qwen_pre_gold_server.log' 2>&1"
fi
healthy=0
for _ in $(seq 1 120); do
  if curl -fsS --max-time 3 http://127.0.0.1:8800/health >/dev/null 2>&1; then healthy=1; break; fi
  if ! tmux has-session -t "$QWEN_SESSION" 2>/dev/null; then break; fi
  sleep 10
done
if (( healthy != 1 )); then echo "Qwen-122B did not become healthy" >&2; exit 73; fi
python3 - <<'PY'
from giis_prod.config import Settings
from giis_prod.teacher_ab import running_teacher_profile
s=Settings.load(__import__('os').path.join(__import__('os').environ['GIIS_ROOT'], 'config/settings_v17_turbo.toml'))
p=running_teacher_profile(s)
actual={k:int(p[k]) for k in ('max_model_len','max_num_seqs')}
if actual!={'max_model_len':16384,'max_num_seqs':2}:
    raise SystemExit(f'Wrong Qwen profile: {actual}')
PY
python3 -u -m giis_prod.v17_turbo_canary run --out "$MAIN_REL"
GENERATION_SESSION="giis-v17-direct-${BASHPID}" bash "$ROOT/scripts/run_v17_turbo_main_handoff.sh" "$MAIN_REL"
python3 -m giis_prod.v17_turbo_pre_gold pre-gold --main "$ROOT/$MAIN_REL" >/dev/null
python3 -u giis_pipeline.py --config config/settings_v17_turbo.toml heldout-pipeline \
  --output "$HELD_REL" --main-run "$MAIN_REL"
python3 -m giis_prod.v17_turbo_pre_gold checkpoint --main "$ROOT/$MAIN_REL" --heldout "$ROOT/$HELD_REL"
