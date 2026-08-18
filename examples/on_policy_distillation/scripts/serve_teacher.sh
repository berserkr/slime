#!/bin/bash
# =============================================================================
# serve_teacher.sh — stand up ONE SGLang teacher endpoint.
#
# Run this once per teacher, in its own terminal / tmux pane / nohup. It pins the
# server to a set of GPUs, waits until it is healthy, prints the endpoint URL, and
# then stays in the foreground so Ctrl-C (or the EXIT trap) tears it down cleanly.
#
# The training scripts (train_opd_1teacher.sh / train_opd_2teachers.sh) do NOT start
# any server — they just curl the endpoint(s) you bring up here and then launch slime.
#
# ---- usage --------------------------------------------------------------------
#   # teacher A on GPU 3, port 13141
#   TEACHER_NAME=math MODEL_PATH=/root/models/teacher-math \
#     GPUS=3 PORT=13141 bash serve_teacher.sh
#
#   # teacher B on GPU 2, port 13142   (second terminal, same node)
#   TEACHER_NAME=code MODEL_PATH=/root/models/teacher-code \
#     GPUS=2 PORT=13142 bash serve_teacher.sh
#
# ---- knobs (env vars) ---------------------------------------------------------
#   MODEL_PATH   (required) HF checkpoint dir for this teacher.
#   GPUS         GPU id(s) for THIS server, comma-separated. Default "3".
#                Must equal TP*DP; for one GB200 with TP=1 just use a single id.
#   PORT         HTTP port. Default 13141.
#   TP           Tensor-parallel size. Default = number of ids in GPUS.
#   HOST         Bind address. Default 0.0.0.0 (reachable by the trainer on-node).
#   MEM_FRACTION Static KV/cache fraction. Default 0.85 (GB200 has plenty of HBM).
#   TEACHER_NAME Label used only for the log filename / messages. Default "teacher".
#
# NOTE: every teacher MUST share the student's tokenizer/vocab — teachers score the
#       student's exact token ids. Different weights are fine; different vocab is not.
# =============================================================================
set -euo pipefail

MODEL_PATH="${MODEL_PATH:?set MODEL_PATH to the teacher checkpoint dir}"
GPUS="${GPUS:-3}"
PORT="${PORT:-13141}"
HOST="${HOST:-0.0.0.0}"
MEM_FRACTION="${MEM_FRACTION:-0.85}"
TEACHER_NAME="${TEACHER_NAME:-teacher}"

# Default TP = number of GPU ids given (e.g. GPUS="2,3" -> TP=2).
if [ -z "${TP:-}" ]; then
  TP="$(awk -F, '{print NF}' <<<"$GPUS")"
fi

LOG_FILE="/tmp/sglang_${TEACHER_NAME}_${PORT}.log"
HEALTH_URL="http://127.0.0.1:${PORT}/health_generate"

echo "[serve_teacher] name=${TEACHER_NAME} model=${MODEL_PATH}"
echo "[serve_teacher] GPUS=${GPUS} TP=${TP} port=${PORT} log=${LOG_FILE}"

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
  echo "[serve_teacher] stopping ${TEACHER_NAME} (pid ${SERVER_PID})"
  kill "${SERVER_PID}" 2>/dev/null || true
  wait "${SERVER_PID}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

echo "[serve_teacher] waiting for ${TEACHER_NAME} to become healthy..."
until curl -sf "${HEALTH_URL}" > /dev/null; do
  # If the server died during startup, surface the log and bail.
  if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
    echo "[serve_teacher] ERROR: server exited during startup. Last log lines:"
    tail -n 30 "${LOG_FILE}"
    exit 1
  fi
  sleep 5
done

echo "=============================================================="
echo "[serve_teacher] '${TEACHER_NAME}' is UP:  http://127.0.0.1:${PORT}/generate"
echo "  use this in the trainer as   name=http://127.0.0.1:${PORT}/generate"
curl -s "http://127.0.0.1:${PORT}/get_model_info" || true
echo ""
echo "  (leave this running; Ctrl-C to stop)"
echo "=============================================================="

# Stay in the foreground so the endpoint lives as long as this terminal does.
wait "${SERVER_PID}"
