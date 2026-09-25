#!/usr/bin/env bash
set -euo pipefail
ROOT="${GIIS_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export GIIS_ROOT="$ROOT"
RUN_REL="${1:-artifacts/v17_turbo/heldout_authoring_retry_100}"
OUT="$ROOT/$RUN_REL"
QWEN_SESSION="${QWEN_SESSION:-giis-mini-qwen-teacher}"
MAIN_RUN="$ROOT/artifacts/v17_turbo/sft_generation_1000"
cd "$ROOT"
python3 - "$MAIN_RUN" <<'PY'
import json,pathlib,sys
run=pathlib.Path(sys.argv[1])
plan=json.loads((run/'plan.json').read_text())
status=json.loads((run/'orchestration_status.json').read_text())
manifest=json.loads((run/'main_semantic_review_manifest.json').read_text())
if status.get('stage')!='gold_review_complete':
    raise SystemExit('Main V17-Turbo workflow has not completed permission-authorized gold review')
if manifest.get('status')!='complete' or manifest.get('plan_digest')!=plan.get('plan_digest'):
    raise SystemExit('Main semantic-review completion manifest is absent, incomplete, or foreign')
if manifest.get('gold_review_authorized') is not True:
    raise SystemExit('Main gold review authorization is not bound')
if manifest.get('unreviewed_multi_or_creative_candidates') != 0:
    raise SystemExit('Main multi-source or creative candidates remain unreviewed')
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
  mkdir -p "$OUT/logs"
  tmux new-session -d -s "$QWEN_SESSION" "'$ROOT/scripts/launch_qwen122b_teacher.sh' >> '$OUT/logs/qwen_server.log' 2>&1"
fi
healthy=0
for _ in $(seq 1 120); do
  if curl -fsS --max-time 3 http://127.0.0.1:8800/health >/dev/null 2>&1; then healthy=1; break; fi
  if ! tmux has-session -t "$QWEN_SESSION" 2>/dev/null; then break; fi
  sleep 10
done
if (( healthy != 1 )); then echo "Qwen-122B did not become healthy" >&2; exit 73; fi
PYTHONPATH=src python3 - <<'PY'
import os
from giis_prod.config import Settings
from giis_prod.teacher_ab import running_teacher_profile
s=Settings.load(os.path.join(os.environ['GIIS_ROOT'], 'config/settings_v17_turbo.toml'))
p=running_teacher_profile(s)
actual={k:int(p[k]) for k in ('max_model_len','max_num_seqs')}
if actual != {'max_model_len':16384,'max_num_seqs':2}:
    raise SystemExit(f'Wrong held-out Qwen profile: {actual}')
PY
if [[ ! -s "$OUT/review_packets_oss_single_source.jsonl" || ! -s "$OUT/review_packets_codex_gold.jsonl" ]]; then
  PYTHONPATH=src "${PYTHON_BIN:-python3}" -u giis_pipeline.py --config config/settings_v17_turbo.toml author-heldout --output "$RUN_REL"
fi
exec "$ROOT/scripts/run_v17_turbo_heldout_handoff.sh" "$RUN_REL"
