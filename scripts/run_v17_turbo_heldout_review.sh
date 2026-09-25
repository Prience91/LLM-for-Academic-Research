#!/usr/bin/env bash
set -euo pipefail
ROOT="${GIIS_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
export GIIS_ROOT="$ROOT"
AUTHORING="$ROOT/artifacts/v17_turbo/heldout_authoring"
STATUS="$AUTHORING/status.json"
PACKETS="$AUTHORING/review_packets_oss_single_source.jsonl"
GOLD="$AUTHORING/review_packets_codex_gold.jsonl"
OSS_SESSION="giis-mini-oss-reviewer"
QWEN_SESSION="giis-mini-teacher"
if [[ ! -s "$PACKETS" || ! -s "$GOLD" ]]; then
  echo "Held-out pre-review packets are missing" >&2
  exit 80
fi
python3 - "$STATUS" "$PACKETS" "$GOLD" <<'PY'
import json,sys
from pathlib import Path
status=Path(sys.argv[1]); single=Path(sys.argv[2]); gold=Path(sys.argv[3])
payload=json.loads(status.read_text())
payload.update(status='handoff_to_oss',oss_packets=sum(1 for _ in single.open()),codex_gold_packets=sum(1 for _ in gold.open()))
status.write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n')
PY
if tmux has-session -t "$QWEN_SESSION" 2>/dev/null; then
  tmux kill-session -t "$QWEN_SESSION"
fi
for _ in $(seq 1 60); do
  if ! ss -ltn 2>/dev/null | grep -q ':8800 '; then break; fi
  sleep 1
done
if ss -ltn 2>/dev/null | grep -q ':8800 '; then
  echo "Qwen port 8800 did not close" >&2
  exit 81
fi
if ! curl -fsS --max-time 3 http://127.0.0.1:8600/health >/dev/null 2>&1; then
  for _ in $(seq 1 60); do
    AVAILABLE_KB="$(awk '/MemAvailable:/ {print $2}' /proc/meminfo)"
    if (( AVAILABLE_KB >= 104857600 )); then break; fi
    sleep 1
  done
  AVAILABLE_KB="$(awk '/MemAvailable:/ {print $2}' /proc/meminfo)"
  if (( AVAILABLE_KB < 104857600 )); then
    echo "Memory did not recover above 100 GiB" >&2
    exit 82
  fi
  if tmux has-session -t "$OSS_SESSION" 2>/dev/null; then tmux kill-session -t "$OSS_SESSION"; fi
  if ss -ltn 2>/dev/null | grep -q ':8600 '; then
    echo "OSS port 8600 is occupied by an unverified process" >&2
    exit 83
  fi
  tmux new-session -d -s "$OSS_SESSION" "bash '$ROOT/scripts/launch_oss120b_reviewer.sh'"
fi
for _ in $(seq 1 900); do
  if curl -fsS --max-time 3 http://127.0.0.1:8600/health >/dev/null 2>&1; then break; fi
  if ! tmux has-session -t "$OSS_SESSION" 2>/dev/null; then
    echo "OSS server exited before becoming healthy" >&2
    exit 84
  fi
  sleep 1
done
if ! curl -fsS --max-time 3 http://127.0.0.1:8600/health >/dev/null; then
  echo "OSS health timeout" >&2
  exit 85
fi
cd "$ROOT"
PYTHONPATH=src "$PYTHON_BIN" -u -m giis_prod.heldout_oss_review
python3 - "$STATUS" <<'PY'
import json,sys
from pathlib import Path
status=Path(sys.argv[1]); payload=json.loads(status.read_text())
payload['status']='awaiting_codex_gold_review'
payload['oss_review_complete']=True
status.write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n')
PY
