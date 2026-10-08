#!/usr/bin/env bash
set -uo pipefail
ROOT="$GIIS_ROOT"
PYTHON="${GIIS_PYTHON:-python3}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
LOG_DIR="$ROOT/artifacts/v17_turbo/logs"
LOG_FILE="$LOG_DIR/corpus_1000_${STAMP}.log"
STATUS_FILE="$LOG_DIR/corpus_1000_status.json"
mkdir -p "$LOG_DIR"
printf '{"status":"running","started_at_utc":"%s","pid":%s,"log":"%s"}\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$$" "$LOG_FILE" > "$STATUS_FILE"
cd "$ROOT"
set +e
"$PYTHON" -u giis_pipeline.py corpus --expected-documents 1000 2>&1 | tee "$LOG_FILE"
code=${PIPESTATUS[0]}
set -e
if [ "$code" -eq 0 ]; then state="complete"; else state="failed"; fi
printf '{"status":"%s","finished_at_utc":"%s","exit_code":%s,"log":"%s"}\n' "$state" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$code" "$LOG_FILE" > "$STATUS_FILE"
exit "$code"
