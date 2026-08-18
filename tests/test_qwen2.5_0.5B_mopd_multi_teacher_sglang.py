"""Multi-teacher OPD (MOPD) routing smoke test — sglang backend, 2 teachers.

This is the multi-teacher analogue of ``test_qwen2.5_0.5B_opd_sglang.py``. It exercises
the routing patch end-to-end:

  * two sglang teacher servers are launched (here they serve the *same* Qwen2.5-0.5B
    checkpoint, so this stays a self-distillation test and ``opd_reverse_kl`` should sit
    near 0 — but the two endpoints are genuinely distinct processes/ports/GPUs);
  * a small prompt dataset is generated from gsm8k where every row carries
    ``metadata.teacher`` set to ``"teacher_a"`` or ``"teacher_b"`` (alternating);
  * training is launched with ``--opd-teacher-urls "teacher_a=<url_a>,teacher_b=<url_b>"``
    (instead of the single ``--rm-url``), so ``reward_func`` routes each trajectory to the
    server named by its own ``metadata.teacher`` tag.

Using the same model for both teachers is deliberate: it lets the routing plumbing be
tested on any 8-GPU box without needing two different real teachers, while keeping the
self-distillation correctness signal (KL ~ 0). Swap either ``--model-path`` below for a
different checkpoint (same tokenizer!) to turn this into a true multi-teacher run.
"""

import json
import os
import subprocess
import time
import urllib.request

import slime.utils.external_utils.command_utils as U

MODEL_NAME = "Qwen2.5-0.5B-Instruct"
MODEL_TYPE = "qwen2.5-0.5B"
NUM_GPUS = 8
NUM_TRAIN_GPUS = 4

TEACHER_HOST = "127.0.0.1"
# One port (and one GPU) per teacher. Both serve MODEL_NAME here.
TEACHERS = {
    "teacher_a": 13141,
    "teacher_b": 13142,
}

# Where the tagged prompt/eval datasets get written by prepare().
TRAIN_DATA = "/root/datasets/gsm8k_mopd/train_tagged.jsonl"
EVAL_DATA = "/root/datasets/gsm8k_mopd/test_tagged.jsonl"


def _build_tagged_dataset(src_parquet: str, dst_jsonl: str, teacher_names, limit: int):
    """Read a gsm8k parquet and write a jsonl where each row gains metadata.teacher.

    Teachers are assigned round-robin so both endpoints receive traffic. Every row that
    can reach a teacher server (train prompts and eval prompts alike) is tagged, so the
    router never sees an untagged sample.
    """
    import pyarrow.parquet as pq

    os.makedirs(os.path.dirname(dst_jsonl), exist_ok=True)
    names = list(teacher_names)
    written = 0
    with open(dst_jsonl, "w", encoding="utf-8") as out:
        pf = pq.ParquetFile(src_parquet)
        for batch in pf.iter_batches():
            for row in batch.to_pylist():
                if written >= limit:
                    break
                row = dict(row)
                # gsm8k rows expose `messages` (conversation) + `label`; preserve them and
                # attach the routing tag. metadata is merged so any existing keys survive.
                metadata = dict(row.get("metadata") or {})
                metadata["teacher"] = names[written % len(names)]
                row["metadata"] = metadata
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
                written += 1
            if written >= limit:
                break
    print(f"Wrote {written} tagged rows -> {dst_jsonl}")


def prepare():
    U.exec_command("mkdir -p /root/models /root/datasets")
    U.exec_command(f"hf download Qwen/{MODEL_NAME} --local-dir /root/models/{MODEL_NAME}")
    U.hf_download_dataset("zhuzilin/gsm8k")
    # Small tagged slices are enough for a smoke test; keep them tiny so it stays fast.
    _build_tagged_dataset("/root/datasets/gsm8k/train.parquet", TRAIN_DATA, TEACHERS, limit=64)
    _build_tagged_dataset("/root/datasets/gsm8k/test.parquet", EVAL_DATA, TEACHERS, limit=16)


def _get_gpu_split():
    """Split GPUs: first NUM_TRAIN_GPUS for training, one GPU per teacher after that."""
    all_gpus = os.environ.get("CUDA_VISIBLE_DEVICES", ",".join(str(i) for i in range(NUM_GPUS))).split(",")
    needed = NUM_TRAIN_GPUS + len(TEACHERS)
    assert len(all_gpus) >= needed, f"Expected at least {needed} GPUs, got {len(all_gpus)}"
    train_gpus = all_gpus[:NUM_TRAIN_GPUS]
    teacher_gpus = all_gpus[NUM_TRAIN_GPUS : NUM_TRAIN_GPUS + len(TEACHERS)]
    return train_gpus, teacher_gpus


def _launch_teacher_server(name: str, port: int, teacher_gpu: str):
    """Launch one sglang teacher server on a specific GPU/port. Returns the process."""
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = teacher_gpu

    log_path = f"/tmp/sglang_{name}.log"
    log_file = open(log_path, "w")
    process = subprocess.Popen(
        [
            "python3",
            "-m",
            "sglang.launch_server",
            "--model-path",
            f"/root/models/{MODEL_NAME}",
            "--host",
            "0.0.0.0",
            "--port",
            str(port),
            "--tp",
            "1",
            "--mem-fraction-static",
            "0.6",
        ],
        env=env,
        stdout=log_file,
        stderr=subprocess.STDOUT,
    )

    print(f"Starting teacher '{name}' on GPU {teacher_gpu} port {port} (pid={process.pid}), log: {log_path}")

    # Wait for server to be ready (up to 10 minutes).
    for _ in range(120):
        if process.poll() is not None:
            raise RuntimeError(f"Teacher '{name}' exited with code {process.returncode}. Check {log_path}")
        try:
            req = urllib.request.urlopen(f"http://{TEACHER_HOST}:{port}/health_generate", timeout=2)
            if req.status == 200:
                print(f"Teacher '{name}' is ready on GPU {teacher_gpu} port {port}")
                return process
        except Exception:
            pass
        time.sleep(5)

    process.kill()
    raise RuntimeError(f"Teacher '{name}' failed to start within timeout. Check {log_path}")


def execute():
    train_gpus, teacher_gpus = _get_gpu_split()
    teacher_processes = []

    # Restrict CUDA_VISIBLE_DEVICES to training GPUs before Ray starts.
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(train_gpus)

    def launch_teachers():
        for (name, port), gpu in zip(TEACHERS.items(), teacher_gpus):
            teacher_processes.append(_launch_teacher_server(name, port, gpu))

    try:
        ckpt_args = f"--hf-checkpoint /root/models/{MODEL_NAME}/ " f"--ref-load /root/models/{MODEL_NAME}/ "

        rollout_args = (
            f"--prompt-data {TRAIN_DATA} "
            "--input-key messages "
            "--label-key label "
            "--metadata-key metadata "
            "--apply-chat-template "
            "--rollout-shuffle "
            "--rm-type math "
            "--num-rollout 2 "
            "--rollout-batch-size 4 "
            "--n-samples-per-prompt 4 "
            "--rollout-max-response-len 1024 "
            "--rollout-temperature 0.8 "
            "--global-batch-size 16 "
        )

        eval_args = (
            f"--eval-prompt-data gsm8k {EVAL_DATA} "
            "--n-samples-per-eval-prompt 1 "
            "--eval-max-response-len 1024 "
            "--eval-top-k 1 "
        )

        perf_args = (
            "--tensor-model-parallel-size 1 "
            "--sequence-parallel "
            "--pipeline-model-parallel-size 1 "
            "--context-parallel-size 1 "
            "--expert-model-parallel-size 1 "
            "--expert-tensor-parallel-size 1 "
            "--use-dynamic-batch-size "
            "--max-tokens-per-gpu 9216 "
        )

        # MOPD routing: two named teachers instead of a single --rm-url. Each trajectory is
        # routed to the server named by its own metadata.teacher tag.
        teacher_urls = ",".join(f"{name}=http://{TEACHER_HOST}:{port}/generate" for name, port in TEACHERS.items())
        rm_args = (
            "--custom-rm-path slime.rollout.on_policy_distillation.reward_func "
            "--custom-reward-post-process-path slime.rollout.on_policy_distillation.post_process_rewards "
            f"--opd-teacher-urls {teacher_urls} "
            "--opd-routing-key teacher "
        )

        grpo_args = (
            "--advantage-estimator grpo "
            # OPD with sglang teachers (self-distillation for CI test)
            "--use-opd "
            "--opd-type sglang "
            "--opd-kl-coef 1.0 "
            "--use-kl-loss "
            "--kl-loss-coef 0.00 "
            "--kl-loss-type low_var_kl "
            "--entropy-coef 0.00 "
            "--eps-clip 0.2 "
            "--eps-clip-high 0.28 "
        )

        optimizer_args = (
            "--optimizer adam "
            "--lr 1e-6 "
            "--lr-decay-style constant "
            "--weight-decay 0.1 "
            "--adam-beta1 0.9 "
            "--adam-beta2 0.98 "
        )

        sglang_args = (
            "--rollout-num-gpus-per-engine 1 "
            "--sglang-mem-fraction-static 0.7 "
            "--sglang-cuda-graph-max-bs 16 "
            "--sglang-enable-metrics "
        )

        ci_args = "--ci-test "

        misc_args = (
            "--attention-dropout 0.0 "
            "--hidden-dropout 0.0 "
            "--accumulate-allreduce-grads-in-fp32 "
            "--attention-softmax-in-fp32 "
            "--attention-backend flash "
            "--actor-num-nodes 1 "
            f"--actor-num-gpus-per-node {NUM_TRAIN_GPUS} "
            "--colocate "
        )

        train_args = (
            f"{ckpt_args} "
            f"{rollout_args} "
            f"{optimizer_args} "
            f"{grpo_args} "
            f"{U.get_default_wandb_args(__file__)} "
            f"{perf_args} "
            f"{eval_args} "
            f"{sglang_args} "
            f"{ci_args} "
            f"{misc_args} "
            f"{rm_args} "
        )

        U.execute_train(
            train_args=train_args,
            num_gpus_per_node=NUM_TRAIN_GPUS,
            megatron_model_type=MODEL_TYPE,
            before_ray_job_submit=launch_teachers,
        )
    finally:
        for process in teacher_processes:
            process.kill()
            process.wait()
        U.exec_command("pkill -9 sglang; true")


if __name__ == "__main__":
    prepare()
    for proxy_var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(proxy_var, None)
    execute()
