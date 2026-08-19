#!/bin/bash
# =============================================================================
# run-tau-bench-opd.sh — train the tau-bench AGENTIC example WITH on-policy
# distillation (OPD). The student runs multi-turn tau-bench episodes; a teacher
# SGLang endpoint scores the student's tokens; we distill toward it.
#
# Teacher scoring is done by a ROLLOUT SAMPLE HOOK (tau_bench_opd.opd_teacher_hook)
# in BOTH modes; the modes differ only in the generate function:
#   MODE=B  (default) task reward + OPD KL     — RL on tau-bench success PLUS
#                                                 distillation toward the teacher.
#                                                 Uses stock generate_with_tau.generate.
#   MODE=A            pure distillation         — env reward ignored; the ONLY signal
#                                                 is the teacher KL. Uses
#                                                 tau_bench_opd.generate_pure_distill.
#
# This script starts NO teacher. Bring one up first (own terminal), e.g.:
#     MODEL_PATH=/root/Qwen3-32B GPUS=3 PORT=30000 \
#       bash examples/on_policy_distillation/scripts/serve_sglang.sh
# then smoke-test it:
#     PORT=30000 bash examples/on_policy_distillation/scripts/test_inference.sh
# The teacher MUST share the student's tokenizer/vocab (it scores the student's
# exact token ids). A stronger model in the same family is the right choice.
#
# See TAU_BENCH_OPD.md for the full explanation of how the pieces fit together.
#
# ---- GPU layout suggestion: 1 node, 4x GB200 --------------------------------
#     GPU 3    -> teacher (serve_sglang.sh)
#     GPU 0,1  -> tau-bench student training + rollout (colocated, TP=2)
#     (GPU 2 spare / raise NUM_GPUS or teacher TP as you like)
#
# ---- knobs (env vars) --------------------------------------------------------
#   MODE          "B" (task reward + KL, default) or "A" (pure distillation)
#   TEACHER_URL   Default http://127.0.0.1:30000/generate
#   OPD_KL_COEF   OPD KL penalty coefficient. Default 1.0
#   NUM_GPUS      Training GPUs (colocated). Default 2 (matches TP=2 below)
#   TAU_BENCH_DIR Path to slime's examples/tau-bench (for generate_with_tau)
#   TAU_DATA_DIR  Path holding retail_{train,dev}_tasks.jsonl. Default /root/tau-bench
# =============================================================================
set -euo pipefail

pkill -9 sglang || true   # only stray student-side sglang; the teacher is a SEPARATE
sleep 2                    # process you launched elsewhere and is not killed here.
ray stop --force || true
pkill -9 ray || true
sleep 2

export PYTHONUNBUFFERED=1

MODE="${MODE:-B}"
TEACHER_URL="${TEACHER_URL:-http://127.0.0.1:30000/generate}"
OPD_KL_COEF="${OPD_KL_COEF:-1.0}"
NUM_GPUS="${NUM_GPUS:-2}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
SLIME_DIR="$(cd -- "${SCRIPT_DIR}/../.." &>/dev/null && pwd)"
TAU_BENCH_DIR="${TAU_BENCH_DIR:-${SLIME_DIR}/examples/tau-bench}"
TAU_DATA_DIR="${TAU_DATA_DIR:-/root/tau-bench}"

# 1) Wait for the (already-running) teacher endpoint.
HEALTH="${TEACHER_URL%/generate}/health_generate"
echo "[run] MODE=${MODE}; waiting for teacher at ${HEALTH} ..."
until curl -sf "${HEALTH}" > /dev/null; do
  echo "[run] teacher not ready; is serve_sglang.sh running on the teacher GPU? retrying..."
  sleep 5
done
echo "[run] teacher healthy: ${TEACHER_URL}"

source "${SLIME_DIR}/scripts/models/qwen3-4B-Instruct-2507.sh"   # -> MODEL_ARGS

CKPT_ARGS=(
   --hf-checkpoint /root/Qwen3-4B-Instruct-2507/
   --ref-load /root/Qwen3-4B-Instruct-2507_torch_dist/
   --load /root/Qwen3-4B-Instruct-2507_slime/
   --save /root/Qwen3-4B-Instruct-2507_slime/
   --save-interval 20
)

ROLLOUT_ARGS=(
   --prompt-data "${TAU_DATA_DIR}/retail_train_tasks.jsonl"
   --input-key index
   --rollout-shuffle
   --num-rollout 500
   --rollout-batch-size 32
   --n-samples-per-prompt 8
   --rollout-max-response-len 1024
   --rollout-temperature 1
   --global-batch-size 256
   --balance-data
)
# In MODE B the task reward varies across a group, so the nonzero-std filter is
# useful (drops all-equal groups). In MODE A the task reward is a constant 0.0, so
# every group has zero std and the filter would drop EVERYTHING — omit it there.
if [ "${MODE}" = "B" ]; then
   ROLLOUT_ARGS+=( --dynamic-sampling-filter-path slime.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std )
fi

EVAL_ARGS=(
   --eval-interval 5
   --eval-prompt-data retail-dev "${TAU_DATA_DIR}/retail_dev_tasks.jsonl"
   --n-samples-per-eval-prompt 1
   --eval-max-response-len 1024
   --eval-top-k 1
)

PERF_ARGS=(
   --tensor-model-parallel-size 2
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1
   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1
   --use-dynamic-batch-size
   --max-tokens-per-gpu 9216
)

# GRPO + OPD. The OPD KL is additive on top of the advantage estimator.
# --rm-url is where the teacher is scored (read by opd_teacher_hook via
# _resolve_teacher_url); swap it for --opd-teacher-urls "name=url,..." to route.
GRPO_ARGS=(
   --advantage-estimator grpo
   --use-opd
   --opd-type sglang
   --opd-kl-coef "${OPD_KL_COEF}"
   --rm-url "${TEACHER_URL}"
   --use-kl-loss
   --kl-loss-coef 0.00
   --kl-loss-type low_var_kl
   --entropy-coef 0.00
   --eps-clip 0.2
   --eps-clip-high 0.28
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
)

WANDB_ARGS=(
   # --use-wandb
   # --wandb-project slime-tau-bench-opd
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 1
   --sglang-mem-fraction-static 0.7
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
)

# OPD teacher scoring is done by a ROLLOUT SAMPLE HOOK (opd_teacher_hook), which sets
# sample.teacher_log_probs and leaves the env reward alone. This is NOT a --custom-rm-path
# (tau-bench already fills sample.reward, so the RM step never fires) — see TAU_BENCH_OPD.md.
# Mode B: real task reward + KL (stock tau-bench generate). Mode A: pure distillation
# (generate_pure_distill zeros the task reward so only the KL trains).
RM_ARGS=( --rollout-sample-hook-path tau_bench_opd.opd_teacher_hook )
if [ "${MODE}" = "B" ]; then
   CUSTOM_ARGS=( --custom-generate-function-path generate_with_tau.generate )
elif [ "${MODE}" = "A" ]; then
   CUSTOM_ARGS=( --custom-generate-function-path tau_bench_opd.generate_pure_distill )
else
   echo "[run] ERROR: MODE must be 'A' or 'B' (got '${MODE}')"; exit 1
fi

export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
ray start --head --node-ip-address "${MASTER_ADDR}" --num-gpus "${NUM_GPUS}" \
   --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265

cleanup() {
  # Tear down only this run's Ray/student sglang. The teacher is a separate process
  # you started with serve_sglang.sh — leave it running for the next attempt.
  echo "[run] cleaning up ray"
  ray stop --force || true
  pkill -9 ray || true
}
trap cleanup EXIT

# tau-bench's generate_with_tau AND this dir's tau_bench_opd must both be importable.
RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"/root/Megatron-LM/:${TAU_BENCH_DIR}:${SCRIPT_DIR}\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\"
  }
}"

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 train.py \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node "${NUM_GPUS}" \
   --rollout-num-gpus "${NUM_GPUS}" \
   --colocate \
   ${MODEL_ARGS[@]} \
   "${CKPT_ARGS[@]}" \
   "${ROLLOUT_ARGS[@]}" \
   "${OPTIMIZER_ARGS[@]}" \
   "${GRPO_ARGS[@]}" \
   "${WANDB_ARGS[@]}" \
   "${PERF_ARGS[@]}" \
   "${EVAL_ARGS[@]}" \
   "${SGLANG_ARGS[@]}" \
   "${MISC_ARGS[@]}" \
   "${RM_ARGS[@]}" \
   "${CUSTOM_ARGS[@]}"
