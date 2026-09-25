#!/usr/bin/env bash
set -euo pipefail
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
KEY_FILE="${GIIS_OSS_REVIEWER_KEY_FILE:-/tmp/giis_gptoss_reviewer_api_key}"
export VLLM_API_KEY="$(cat "$KEY_FILE")"
MODEL="${GIIS_OSS_MODEL:-openai/gpt-oss-120b}"
VLLM_BIN="${VLLM_BIN:-vllm}"
AVAILABLE_KB="$(awk '/MemAvailable:/ {print $2}' /proc/meminfo)"
MIN_AVAILABLE_KB=104857600
if (( AVAILABLE_KB < MIN_AVAILABLE_KB )); then
  echo "Preflight refused: MemAvailable ${AVAILABLE_KB} KiB is below 100 GiB" >&2
  exit 70
fi
exec "$VLLM_BIN" serve "$MODEL" \
  --served-model-name gpt-oss-120b-reviewer \
  --host 127.0.0.1 \
  --port 8600 \
  --api-key "$VLLM_API_KEY" \
  --dtype auto \
  --max-model-len 16384 \
  --max-num-seqs 1 \
  --gpu-memory-utilization 0.82 \
  --kv-cache-dtype fp8 \
  --enable-prefix-caching \
  --enable-chunked-prefill \
  --enforce-eager \
  --reasoning-parser openai_gptoss \
  --no-enable-log-requests \
  --generation-config vllm
