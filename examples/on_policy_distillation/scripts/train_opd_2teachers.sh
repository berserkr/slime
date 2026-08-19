#!/bin/bash
# =============================================================================
# train_opd_2teachers.sh — MULTI-teacher OPD training that USES two endpoints that
# are already running (each started by its own serve_teacher.sh).
#
# This is the routing path: every trajectory is sent to the teacher named by its
# own metadata.teacher tag (see MOPD_GETTING_STARTED.md, Step 1). This script starts
# NO servers — it waits for both URLs, then launches slime with --opd-teacher-urls.
#
# ---- GPU layout: 1 node, 4x GB200 -------------------------------------------
#     GPU 2   -> teacher_a   (owned by serve_teacher.sh #1)
#     GPU 3   -> teacher_b   (owned by serve_teacher.sh #2)
#     GPU 0,1 -> slime training + student rollout (colocated)
#
# ---- run order ---------------------------------------------------------------
#   # terminal 1:
#   TEACHER_NAME=math MODEL_PATH=/root/models/teacher-math \
#     GPUS=2 PORT=13141 bash .../serve_teacher.sh
#   # terminal 2:
#   TEACHER_NAME=code MODEL_PATH=/root/models/teacher-code \
#     GPUS=3 PORT=13142 bash .../serve_teacher.sh
#   # terminal 3 (once both print "is UP"):
#   bash examples/on_policy_distillation/scripts/train_opd_2teachers.sh
#
# ---- knobs (env vars) ---------------------------------------------------------
#   TEACHER_A_URL  Default http://127.0.0.1:13141/generate
#   TEACHER_B_URL  Default http://127.0.0.1:13142/generate
#   TEACHER_A_NAME Default "math"   } these NAMES are the vocabulary your data's
#   TEACHER_B_NAME Default "code"   } metadata.teacher values must match exactly.
#   TRAIN_GPUS     Default "0,1"
#
# ---- data requirement (important) --------------------------------------------
#   Every prompt row must carry metadata.teacher set to one of the teacher NAMES,
#   e.g.  {"prompt": "...", "metadata": {"teacher": "math"}}
#   An untagged row (or an unknown name) raises ValueError at rollout time.
#   A ready-made sample lives at examples/on_policy_distillation/data/prompts_tagged.jsonl
#   (tagged math/code) — the PROMPT_DATA default below points at it.
# =============================================================================
set -euo pipefail

TEACHER_A_URL="${TEACHER_A_URL:-http://127.0.0.1:13141/generate}"
TEACHER_B_URL="${TEACHER_B_URL:-http://127.0.0.1:13142/generate}"
TEACHER_A_NAME="${TEACHER_A_NAME:-math}"
TEACHER_B_NAME="${TEACHER_B_NAME:-code}"
TRAIN_GPUS="${TRAIN_GPUS:-0,1}"
NUM_TRAIN_GPUS="$(awk -F, '{print NF}' <<<"$TRAIN_GPUS")"

# ----- EDIT ME: model + data ------------------------------------------------- #
SLIME_DIR="${SLIME_DIR:-/root/slime}"
MODEL_SCRIPT="${MODEL_SCRIPT:-${SLIME_DIR}/scripts/models/qwen3-8B.sh}"  # defines MODEL_ARGS
STUDENT_HF="${STUDENT_HF:-/root/models/student}"
STUDENT_REF="${STUDENT_REF:-/root/models/student_torch_dist}"
STUDENT_SAVE="${STUDENT_SAVE:-/root/models/student_slime}"
# Default points at the bundled math/code sample; swap for your own tagged jsonl.
PROMPT_DATA="${PROMPT_DATA:-${SLIME_DIR}/examples/on_policy_distillation/data/prompts_tagged.jsonl}"
# ----------------------------------------------------------------------------- #

export PYTHONUNBUFFERED=1

# 1) Wait for BOTH (already-running) teacher endpoints.
wait_healthy() {
  local url="$1" name="$2"
  local health="${url%/generate}/health_generate"
  echo "[train] waiting for ${name} at ${health} ..."
  until curl -sf "${health}" > /dev/null; do
    echo "[train] ${name} not ready yet; is its serve_teacher.sh running? retrying..."
    sleep 5
  done
  echo "[train] ${name} healthy: ${url}"
}
wait_healthy "${TEACHER_A_URL}" "${TEACHER_A_NAME}"
wait_healthy "${TEACHER_B_URL}" "${TEACHER_B_NAME}"

# 2) Restrict this process (and Ray) to the training GPUs only.
export CUDA_VISIBLE_DEVICES="${TRAIN_GPUS}"

source "${MODEL_SCRIPT}"   # -> MODEL_ARGS

CKPT_ARGS=(
   --hf-checkpoint "${STUDENT_HF}"
   --ref-load "${STUDENT_REF}"
   --load "${STUDENT_SAVE}"
   --save "${STUDENT_SAVE}"
   --save-interval 50
)

ROLLOUT_ARGS=(
   --prompt-data "${PROMPT_DATA}"
   --input-key prompt
   --metadata-key metadata
   --apply-chat-template
   --rollout-shuffle
   --num-rollout 100
   --rollout-batch-size 16
   --n-samples-per-prompt 4
   --rollout-max-response-len 8192
   --rollout-temperature 1.0
   --global-batch-size 64
   --balance-data
)

# Multi-teacher: the routing map. LEFT-hand names must equal your data's
# metadata.teacher values; --opd-routing-key names the field to read (default "teacher").
RM_ARGS=(
   --custom-rm-path slime.rollout.on_policy_distillation.reward_func
   --custom-reward-post-process-path slime.rollout.on_policy_distillation.post_process_rewards
   --opd-teacher-urls "${TEACHER_A_NAME}=${TEACHER_A_URL},${TEACHER_B_NAME}=${TEACHER_B_URL}"
   --opd-routing-key teacher
)

GRPO_ARGS=(
   --advantage-estimator grpo
   --use-opd
   --opd-type sglang
   --opd-kl-coef 1.0
   --use-kl-loss
   --kl-loss-coef 0.00
   --kl-loss-type low_var_kl
   --entropy-coef 0.00
)

PERF_ARGS=(
   --tensor-model-parallel-size 1
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1
   --use-dynamic-batch-size
   --max-tokens-per-gpu 16384
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 1
   --sglang-mem-fraction-static 0.6
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
   --colocate
)

# 3) Start Ray on the training GPUs only, then submit.
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
ray start --head --node-ip-address "${MASTER_ADDR}" \
   --num-gpus "${NUM_TRAIN_GPUS}" --disable-usage-stats \
   --dashboard-host=0.0.0.0 --dashboard-port=8265

cleanup() {
  # Tear down only Ray. Do NOT `pkill python` — the two teacher SGLang servers are python
  # processes on this same node and must survive so you can launch another trainer.
  echo "[train] cleaning up ray"
  ray stop --force || true
  pkill -9 ray || true
}
trap cleanup EXIT

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json='{
     "env_vars": {
        "PYTHONPATH": "/root/Megatron-LM/",
        "CUDA_DEVICE_MAX_CONNECTIONS": "1"
     }
   }' \
   -- python3 train.py \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node "${NUM_TRAIN_GPUS}" \
   "${MODEL_ARGS[@]}" \
   "${CKPT_ARGS[@]}" \
   "${ROLLOUT_ARGS[@]}" \
   "${OPTIMIZER_ARGS[@]}" \
   "${GRPO_ARGS[@]}" \
   "${PERF_ARGS[@]}" \
   "${SGLANG_ARGS[@]}" \
   "${MISC_ARGS[@]}" \
   "${RM_ARGS[@]}"
