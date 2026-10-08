#!/usr/bin/env bash
set -euo pipefail
GIIS_ROOT="${GIIS_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export GIIS_ROOT
MODE=continuous
if (( $# > 0 )); then MODE="$1"; fi
if [[ "$MODE" != "continuous" && "$MODE" != "once" ]]; then
  echo "usage: $0 [continuous|once]" >&2
  exit 2
fi
ROOT="${GIIS_FULL_RAG_ROOT:-$GIIS_ROOT/Full_RAG}"
export GIIS_FULL_RAG_ROOT="$ROOT"
PIPELINE="$GIIS_ROOT"
PYTHON=${GIIS_PYTHON:-python3}
export PYTHONPATH="$PIPELINE/src"
export CUDA_VISIBLE_DEVICES="0"
set -a
if [[ -f "${GIIS_OPENALEX_ENV:-$HOME/.config/giis/openalex.env}" ]]; then source "${GIIS_OPENALEX_ENV:-$HOME/.config/giis/openalex.env}"; fi
set +a
umask 0002
cd "$PIPELINE"

deferred_count() {
  "$PYTHON" -c 'import json; from pathlib import Path; p=Path(__import__("os").environ["GIIS_FULL_RAG_ROOT"] + "/Parsed_Corpus/reports/ingestion_report.json"); print(int(json.loads(p.read_text()).get("metadata_deferred_count", 0)))'
}

index_if_changed() {
  local chunk_version active_version
  chunk_version=$("$PYTHON" -c 'import hashlib; from pathlib import Path; p=Path(__import__("os").environ["GIIS_FULL_RAG_ROOT"] + "/Parsed_Corpus/chunks.jsonl"); print(hashlib.file_digest(p.open("rb"), "sha256").hexdigest())')
  active_version=$("$PYTHON" -c 'import json; from pathlib import Path; p=Path(__import__("os").environ["GIIS_FULL_RAG_ROOT"] + "/Vector_Index/active_index.json"); print(json.loads(p.read_text()).get("corpus_version", "") if p.exists() else "")')
  if [[ "$chunk_version" != "$active_version" ]]; then
    echo "$(date -u +%FT%TZ) Full_RAG CUDA embedding and CPU lexical indexing starting"
    "$PYTHON" production_ingestion.py index --root "$ROOT" --embedding-device cuda
  else
    echo "$(date -u +%FT%TZ) Full_RAG index unchanged; current content version retained"
  fi
}

run_cycle() {
  "$PYTHON" -m giis_prod.metadata_adjudication promote --settings "$PIPELINE/config/settings_full_rag.toml"
  echo "$(date -u +%FT%TZ) Full_RAG parsing/resume starting; deferred papers have priority"
  "$PYTHON" production_ingestion.py parse --root "$ROOT" --settings "$PIPELINE/config/settings_full_rag.toml"
  "$PYTHON" production_ingestion.py materialize --root "$ROOT"
  index_if_changed
  echo "$(date -u +%FT%TZ) Full_RAG cycle complete; $(deferred_count) OpenAlex-deferred papers remain"
}

next_wait_seconds() {
  "$PYTHON" - "$ROOT" <<'PY'
import json
import sys
import time
from pathlib import Path
root = Path(sys.argv[1])
state_path = root / "Parsed_Corpus/metadata/cache/openalex/rate_limit_state.json"
state = json.loads(state_path.read_text()) if state_path.exists() else {}
now = time.time()
blocked_until = float(state.get("blocked_until_epoch") or 0)
reason = state.get("reason")
if reason == "daily_budget_floor" and blocked_until > now:
    delay = blocked_until - now + 2
elif reason == "provider_429" and blocked_until > now:
    delay = max(3600, blocked_until - now + 2)
else:
    delay = 3600
print(int(max(60, min(delay, 86400))))
PY
}

echo "$(date -u +%FT%TZ) Full_RAG inventory starting"
"$PYTHON" production_ingestion.py inventory --root "$ROOT"
run_cycle
if [[ "$MODE" == "once" ]]; then
  exit 0
fi
while (( $(deferred_count) > 0 )); do
  wait_seconds=$(next_wait_seconds)
  echo "$(date -u +%FT%TZ) Full_RAG deferred queue waiting $wait_seconds seconds for renewed OpenAlex allowance; current index remains active"
  sleep "$wait_seconds"
  run_cycle
done
echo "$(date -u +%FT%TZ) Full_RAG deferred queue cleared"
