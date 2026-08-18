#!/bin/bash
# =============================================================================
# train_opd_1teacher.sh — single-teacher OPD training that USES an endpoint that
# is already running (started by serve_teacher.sh in another terminal).
#
# This script starts NO server. It waits for the teacher URL to be healthy, then
# launches slime with --rm-url pointing at it.
#
# ---- GPU layout: 1 node, 4x GB200 -------------------------------------------
#     GPU 3        -> teacher   (owned by serve_teacher.sh)
#     GPU 0,1,2    -> slime training + student rollout (colocated)
#
# ---- run order ---------------------------------------------------------------
#   # terminal 1:
#   MODEL_PATH=/root/models/teacher GPUS=3 PORT=13141 \
#     bash examples/on_policy_distillation/scripts/serve_teacher.sh
#
#   # terminal 2 (once teacher prints "is UP"):
#   bash examples/on_policy_distillation/scripts/train_opd_1teacher.sh
#
# ---- knobs (env vars) ---------------------------------------------------------
#   TEACHER_URL   Default http://127.0.0.1:13141/generate  (match serve_teacher PORT)
#   TRAIN_GPUS    Default "0,1,2"   (GPUs for training; keep teacher's GPU out)
#   Edit the EDIT-ME block below for your model script, checkpoints, and data.
# =============================================================================
set -euo pipefail

TEACHER_URL="${TEACHER_URL:-http://127.0.0.1:13141/generate}"
TRAIN_GPUS="${TRAIN_GPUS:-0,1,2}"
NUM_TRAIN_GPUS="$(awk -F, '{print NF}' <<<"$TRAIN_GPUS")"

# ----- EDIT ME: model + data ------------------------------------------------- #
SLIME_DIR="${SLIME_DIR:-/root/slime}"
MODEL_SCRIPT="${MODEL_SCRIPT:-${SLIME_DIR}/scripts/models/qwen3-8B.sh}"  # defines MODEL_ARGS
STUDENT_HF="${STUDENT_HF:-/root/models/student}"                        # student HF checkpoint
STUDENT_REF="${STUDENT_REF:-/root/models/student_torch_dist}"           # student, mcore/torch_dist
STUDENT_SAVE="${STUDENT_SAVE:-/root/models/student_slime}"
PROMPT_DATA="${PROMPT_DATA:-/root/datasets/prompts.jsonl}"              # rows need input-key below
# ----------------------------------------------------------------------------- #

export PYTHONUNBUFFERED=1

# 1) Wait for the (already-running) teacher endpoint.
HEALTH_URL="${TEACHER_URL%/generate}/health_generate"
echo "[train] waiting for teacher at ${HEALTH_URL} ..."
until curl -sf "${HEALTH_URL}" > /dev/null; do
  echo "[train] teacher not ready yet; is serve_teacher.sh running? retrying..."
  sleep 5
done
echo "[train] teacher is healthy: ${TEACHER_URL}"

# 2) Restrict this process (and Ray) to the training GPUs only; the teacher owns its own.
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

# Single teacher: classic --rm-url. (No routing; every trajectory hits this one server.)
RM_ARGS=(
   --custom-rm-path slime.rollout.on_policy_distillation.reward_func
   --custom-reward-post-process-path slime.rollout.on_policy_distillation.post_process_rewards
   --rm-url "${TEACHER_URL}"
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
  # Tear down only Ray. Do NOT `pkill python` — the teacher SGLang server is a python
  # process on this same node and must survive so you can launch another trainer.
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
