#!/usr/bin/env bash
set -euo pipefail
ROOT="${GIIS_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
RUN_REL="${1:-artifacts/v17_turbo/heldout_authoring_retry_100}"
OUT="$ROOT/$RUN_REL"
STATUS="$OUT/status.json"
PACKETS="$OUT/review_packets_oss_single_source.jsonl"
GOLD="$OUT/review_packets_codex_gold.jsonl"
BINDING="$ROOT/config/v17_turbo_heldout_handoff_binding.json"
QWEN_SESSION="${QWEN_SESSION:-giis-mini-qwen-teacher}"
OSS_SESSION="${OSS_SESSION:-giis-mini-oss-reviewer}"
export GIIS_ROOT="$ROOT"
export PYTHONPATH="$ROOT/src"

write_status() {
  python3 - "$STATUS" "$1" "$2" <<'PY'
import datetime,json,pathlib,sys
p=pathlib.Path(sys.argv[1]); value=json.loads(p.read_text())
value.update(status=sys.argv[2],status_detail=sys.argv[3],updated_at=datetime.datetime.now(datetime.timezone.utc).isoformat())
t=pathlib.Path(str(p)+'.tmp'); t.write_text(json.dumps(value,indent=2,sort_keys=True)+'\n'); t.replace(p)
PY
}

cd "$ROOT"
python3 - "$STATUS" "$PACKETS" "$GOLD" "$BINDING" <<'PY'
import hashlib,json,os,pathlib,sys,tomllib
status_path,single_path,gold_path,binding_path=map(pathlib.Path,sys.argv[1:])
status=json.loads(status_path.read_text()); binding=json.loads(binding_path.read_text())
root=pathlib.Path(os.environ['GIIS_ROOT'])
for rel,digest in binding['bindings'].items():
    if hashlib.sha256((root/rel).read_bytes()).hexdigest()!=digest: raise SystemExit('Held-out handoff source binding mismatch: '+rel)
with (root/'config/settings_v17_turbo.toml').open('rb') as handle: config=tomllib.load(handle)
policy=config['v17_turbo_heldout_benchmark']
single=[json.loads(x) for x in single_path.read_text().splitlines() if x.strip()]
gold=[json.loads(x) for x in gold_path.read_text().splitlines() if x.strip()]
expected_single=int(policy['single_source_cases'])
expected_gold=int(policy['multi_source_cases'])+int(policy['table_cases'])+int(policy['comparison_cases'])
if status.get('total')!=100 or status.get('completed')!=100 or status.get('failed')!=0: raise SystemExit('Canonical held-out authoring is incomplete')
if len(single)!=expected_single or len(gold)!=expected_gold: raise SystemExit('Canonical held-out packet allocation mismatch')
ids=[x['review_id'] for x in single+gold]
if len(ids)!=len(set(ids)): raise SystemExit('Duplicate held-out review IDs')
if not status.get('selection_digest') or not status.get('allocation_digest'): raise SystemExit('Held-out identity digest is missing')
manifest={'schema':'giis_v17_turbo_heldout_handoff_preflight_v1','selection_digest':status['selection_digest'],'allocation_digest':status['allocation_digest'],'oss_packets':len(single),'gold_packets':len(gold),'oss_packet_sha256':hashlib.sha256(single_path.read_bytes()).hexdigest(),'gold_packet_sha256':hashlib.sha256(gold_path.read_bytes()).hexdigest(),'gold_review_authorized':False}
p=status_path.parent/'handoff_preflight.json'; t=pathlib.Path(str(p)+'.tmp'); t.write_text(json.dumps(manifest,indent=2)+'\n'); t.replace(p)
PY
write_status handoff_to_oss "Canonical held-out packets are frozen; beginning guarded Qwen-to-OSS transition"

tmux kill-session -t "$QWEN_SESSION" 2>/dev/null || true
for _ in $(seq 1 60); do
  if ! ss -ltn 2>/dev/null | grep -q ':8800 '; then break; fi
  sleep 5
done
if ss -ltn 2>/dev/null | grep -q ':8800 '; then write_status qwen_shutdown_failed "Port 8800 remained active"; exit 81; fi
for _ in $(seq 1 60); do
  available=$(awk '/MemAvailable:/ {print $2}' /proc/meminfo)
  (( available >= 104857600 )) && break
  sleep 5
done
available=$(awk '/MemAvailable:/ {print $2}' /proc/meminfo)
if (( available < 104857600 )); then write_status oss_preflight_failed "System memory did not recover to 100 GiB"; exit 82; fi
if ss -ltn 2>/dev/null | grep -q ':8600 '; then write_status port_collision "Reserved OSS port 8600 is occupied"; exit 83; fi
tmux new-session -d -s "$OSS_SESSION" "bash '$ROOT/scripts/launch_oss120b_reviewer.sh' >> '$OUT/oss120b_single_source_review/oss_server.log' 2>&1"
write_status loading_oss "GPT-OSS-120B is loading on reserved review port 8600"
healthy=0
for _ in $(seq 1 120); do
  if curl -fsS --max-time 3 http://127.0.0.1:8600/health >/dev/null 2>&1; then healthy=1; break; fi
  if ! tmux has-session -t "$OSS_SESSION" 2>/dev/null; then break; fi
  sleep 10
done
if (( healthy != 1 )); then write_status oss_server_failed "GPT-OSS-120B did not become healthy"; exit 84; fi
write_status reviewing_single_source "GPT-OSS-120B held-out dual-axis census is active"
python3 -u -m giis_prod.v17_turbo_heldout_oss_census --run "$OUT" > "$OUT/oss120b_single_source_review/review.log" 2>&1
errors=$(python3 -c "import json; print(json.load(open('$OUT/oss120b_single_source_review/summary.json'))['unresolved_errors'])")
tmux kill-session -t "$OSS_SESSION" 2>/dev/null || true
if (( errors > 0 )); then
  write_status oss_review_errors_fail_closed "OSS census completed with unresolved fail-closed errors; gold review remains unauthorized"
else
  write_status awaiting_codex_gold_review_authorization "OSS census complete; explicit operator authorization is required before Codex gold review"
fi
