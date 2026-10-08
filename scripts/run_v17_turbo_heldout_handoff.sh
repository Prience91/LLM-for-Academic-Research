#!/usr/bin/env bash
set -euo pipefail
GIIS_ROOT="${GIIS_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export GIIS_ROOT
ROOT="$GIIS_ROOT"
RUN_REL="${1:?Supply a fresh held-out run namespace}"
OUT="$ROOT/$RUN_REL"
QWEN_SESSION="${QWEN_SESSION:-giis-mini-qwen-teacher}"
export PATH="${GIIS_RUNTIME_BIN:-$(dirname "$(command -v python3)")}:$PATH"
export PYTHONPATH="$ROOT/src"
cd "$ROOT"
python3 - "$ROOT" <<'PYBIND'
import hashlib,json,pathlib,sys
root=pathlib.Path(sys.argv[1])
binding=json.loads((root/'config/v17_turbo_heldout_handoff_binding.json').read_text())
if binding.get('routing_revision')!='split_review_v2': raise SystemExit('Wrong held-out routing binding')
for rel,expected in binding['bindings'].items():
    if hashlib.sha256((root/rel).read_bytes()).hexdigest()!=expected:
        raise SystemExit('Held-out execution binding drift: '+rel)
PYBIND
python3 -m giis_prod.v17_turbo_semantic_finalize prepare-heldout --run "$OUT"
tmux kill-session -t "$QWEN_SESSION" 2>/dev/null || true
for _ in $(seq 1 60); do
  if ! ss -ltn 2>/dev/null | grep -q ':8800 '; then break; fi
  sleep 5
done
if ss -ltn 2>/dev/null | grep -q ':8800 '; then
  echo "Qwen generation port did not close" >&2; exit 81
fi
python3 - "$OUT/status.json" <<'PYSTATUS'
import json,pathlib,sys
p=pathlib.Path(sys.argv[1]);s=json.loads(p.read_text())
s.update(status='awaiting_codex_gold_review_authorization',gold_review_authorized=False,
         gold_packets=s['completed'],oss_review_required=False)
t=p.with_suffix('.json.tmp');t.write_text(json.dumps(s,indent=2)+'\n');t.replace(p)
PYSTATUS
