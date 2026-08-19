#!/bin/bash
# =============================================================================
# serve_sglang.sh — load ANY HF model on an SGLang server (generic smoke test).
#
# This is the bare-minimum "does SGLang come up on this model + these GPUs" script.
# It is NOT tied to OPD/teachers — use it to sanity-check a checkpoint or a node
# before wiring it into a training run. (For the OPD routing flow, use
# serve_teacher.sh, which adds teacher-specific labels/health messaging.)
#
# It launches the server in the background, waits until /health_generate passes,
# prints the endpoint, and stays in the foreground so Ctrl-C tears it down.
# Once it prints "is UP", run test_inference.sh in another terminal.
#
# ---- usage --------------------------------------------------------------------
#   MODEL_PATH=/root/models/Qwen3-8B GPUS=0 PORT=30000 bash serve_sglang.sh
#
# ---- knobs (env vars) ---------------------------------------------------------
#   MODEL_PATH   (required) HF checkpoint dir (or hub id) to serve.
#   GPUS         GPU id(s), comma-separated. Default "0". Must equal TP*DP.
#   PORT         HTTP port. Default 30000.
#   TP           Tensor-parallel size. Default = number of ids in GPUS.
#   HOST         Bind address. Default 0.0.0.0.
#   MEM_FRACTION Static KV/cache fraction. Default 0.85.
# =============================================================================
set -euo pipefail

MODEL_PATH="${MODEL_PATH:?set MODEL_PATH to the HF checkpoint dir or hub id}"
GPUS="${GPUS:-0}"
PORT="${PORT:-30000}"
HOST="${HOST:-0.0.0.0}"
MEM_FRACTION="${MEM_FRACTION:-0.85}"

# Default TP = number of GPU ids given (e.g. GPUS="0,1" -> TP=2).
if [ -z "${TP:-}" ]; then
  TP="$(awk -F, '{print NF}' <<<"$GPUS")"
fi

LOG_FILE="/tmp/sglang_${PORT}.log"
HEALTH_URL="http://127.0.0.1:${PORT}/health_generate"

echo "[serve_sglang] model=${MODEL_PATH}"
echo "[serve_sglang] GPUS=${GPUS} TP=${TP} port=${PORT} log=${LOG_FILE}"

# Launch in the background so we can health-check, then hand control back via `wait`.
CUDA_VISIBLE_DEVICES="${GPUS}" python3 -m sglang.launch_server \
    --model-path "${MODEL_PATH}" \
    --host "${HOST}" \
    --port "${PORT}" \
    --tp "${TP}" \
    --chunked-prefill-size 4096 \
    --mem-fraction-static "${MEM_FRACTION}" \
    > "${LOG_FILE}" 2>&1 &
SERVER_PID=$!

# Kill the server whenever this script exits (Ctrl-C, error, or normal end).
cleanup() {
  echo "[serve_sglang] stopping server (pid ${SERVER_PID})"
  kill "${SERVER_PID}" 2>/dev/null || true
  wait "${SERVER_PID}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

echo "[serve_sglang] waiting for server to become healthy..."
until curl -sf "${HEALTH_URL}" > /dev/null; do
  # If the server died during startup, surface the log and bail.
  if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
    echo "[serve_sglang] ERROR: server exited during startup. Last log lines:"
    tail -n 30 "${LOG_FILE}"
    exit 1
  fi
  sleep 5
done

echo "=============================================================="
echo "[serve_sglang] server is UP:  http://127.0.0.1:${PORT}"
echo "  generate:      http://127.0.0.1:${PORT}/generate"
echo "  openai chat:   http://127.0.0.1:${PORT}/v1/chat/completions"
echo "  now run:       PORT=${PORT} bash test_inference.sh"
curl -s "http://127.0.0.1:${PORT}/get_model_info" || true
echo ""
echo "  (leave this running; Ctrl-C to stop)"
echo "=============================================================="

# Stay in the foreground so the endpoint lives as long as this terminal does.
wait "${SERVER_PID}"
