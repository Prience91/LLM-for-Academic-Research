#!/usr/bin/env bash
set -euo pipefail
ROOT="${GIIS_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
RUN_REL="${1:-artifacts/v17_turbo/sft_generation_1000}"
RUN="$ROOT/$RUN_REL"
STATUS="$RUN/orchestration_status.json"
GENERATION_SESSION="${GENERATION_SESSION:-giis-mini-sft-generation}"
QWEN_SESSION="${QWEN_SESSION:-giis-mini-qwen-teacher}"
OSS_SESSION="${OSS_SESSION:-giis-mini-oss-reviewer}"
export GIIS_ROOT="$ROOT"
export PYTHONPATH="$ROOT/src"

write_status() {
  python3 - "$STATUS" "$1" "$2" <<'PY'
import datetime,json,pathlib,sys
p=pathlib.Path(sys.argv[1]); p.parent.mkdir(parents=True,exist_ok=True)
v={'stage':sys.argv[2],'detail':sys.argv[3],'updated_at':datetime.datetime.now(datetime.timezone.utc).isoformat()}
t=pathlib.Path(str(p)+'.tmp'); t.write_text(json.dumps(v,indent=2)+'\n'); t.replace(p)
PY
}

cd "$ROOT"
write_status waiting_for_generation "Waiting for the plan-bound Qwen generation and deterministic qualification"
while true; do
  while tmux has-session -t "$GENERATION_SESSION" 2>/dev/null; do sleep 60; done
  if python3 -m giis_prod.v17_turbo_review_packets --run "$RUN" > "$RUN/oss_packet_freeze.log" 2>&1; then
    break
  fi
  write_status generation_resume_required "Generation is incomplete; checkpoints are preserved and handoff is waiting for the generation session to resume"
  while ! tmux has-session -t "$GENERATION_SESSION" 2>/dev/null; do sleep 60; done
  write_status waiting_for_generation "Checkpointed Qwen generation resumed; waiting for deterministic qualification"
done
write_status packets_frozen "Blinded single-source census packets are frozen and plan-bound"

tmux kill-session -t "$QWEN_SESSION" 2>/dev/null || true
for _ in $(seq 1 60); do
  if ! ss -ltn 2>/dev/null | grep -q ':8800 '; then break; fi
  sleep 5
done
if ss -ltn 2>/dev/null | grep -q ':8800 '; then
  write_status qwen_shutdown_failed "Generation port 8800 remained active"
  exit 80
fi
for _ in $(seq 1 60); do
  available=$(awk '/MemAvailable:/ {print $2}' /proc/meminfo)
  (( available >= 104857600 )) && break
  sleep 5
done
available=$(awk '/MemAvailable:/ {print $2}' /proc/meminfo)
if (( available < 104857600 )); then
  write_status oss_preflight_failed "System memory did not recover to 100 GiB"
  exit 81
fi
if ss -ltn 2>/dev/null | grep -q ':8600 '; then
  write_status port_collision "Reserved OSS port 8600 is occupied"
  exit 82
fi
tmux new-session -d -s "$OSS_SESSION" "bash '$ROOT/scripts/launch_oss120b_reviewer.sh' >> '$RUN/oss120b_single_source_semantic_census/oss_server.log' 2>&1"
write_status loading_oss "GPT-OSS-120B is loading on reserved review port 8600"
healthy=0
for _ in $(seq 1 120); do
  if curl -fsS --max-time 3 http://127.0.0.1:8600/health >/dev/null 2>&1; then healthy=1; break; fi
  if ! tmux has-session -t "$OSS_SESSION" 2>/dev/null; then break; fi
  sleep 10
done
if (( healthy != 1 )); then
  write_status oss_server_failed "GPT-OSS-120B did not become healthy"
  exit 83
fi
write_status reviewing_single_source "GPT-OSS-120B dual-axis single-source census is active"
python3 -u -m giis_prod.v17_turbo_oss_census --run "$RUN" > "$RUN/oss120b_single_source_semantic_census/census.log" 2>&1
write_status awaiting_gold_review "OSS census complete; multi-source, table, and comparison candidates await explicit gold-review authorization"
tmux kill-session -t "$OSS_SESSION" 2>/dev/null || true
