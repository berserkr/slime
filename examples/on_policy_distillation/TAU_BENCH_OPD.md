# Training tau-bench (agentic) with On-Policy Distillation

This walkthrough explains how to combine slime's **tau-bench agentic example** with
**on-policy distillation (OPD)** — so a student learns a multi-turn, tool-using policy
while being distilled toward a stronger teacher. It documents the two supported modes,
the one non-obvious mechanic (the *reward slot collision*), and how to run and verify it.

Files that implement this:
- [`run-tau-bench-opd.sh`](run-tau-bench-opd.sh) — the runnable training script (both modes).
- [`tau_bench_opd.py`](tau_bench_opd.py) — glue: a teacher-scoring **sample hook** + a Mode-A generate wrapper.
- Teacher server: [`scripts/serve_sglang.sh`](scripts/serve_sglang.sh) + [`scripts/test_inference.sh`](scripts/test_inference.sh).

---

## 1. The mental model

OPD adds a per-token KL penalty to the advantages. Look at what actually consumes the teacher
signal — `slime/backends/megatron_utils/loss.py::apply_opd_kl_to_advantages`:

```python
reverse_kl   = student_log_probs[i] - teacher_log_probs[i]
advantages[i] = adv - args.opd_kl_coef * reverse_kl
```

So the **only** thing OPD needs per sample is `sample.teacher_log_probs`. The task reward →
advantage path is entirely separate and is where GRPO's group normalization happens.

Key facts:
- **The agent loop is your code.** slime has no `AgentLoop` base class; the loop lives inside
  the generate function. tau-bench already implements a full multi-turn tool-calling loop in
  `examples/tau-bench/generate_with_tau.py` and — importantly — sets `loss_mask` so tool
  outputs / observations are excluded from the loss. We reuse it as-is.
- **OPD is orthogonal to the advantage estimator.** The KL is *additive* on top of GRPO; it
  does not replace it. `adv` above is the (group-normalized) task advantage.
- **The teacher is an external SGLang server.** It only scores tokens (`max_new_tokens=0,
  return_logprob=True`); it never generates. It **must share the student's tokenizer/vocab**,
  because it scores the student's exact token ids.

---

## 2. The one thing that bites: how the teacher gets queried

The base OPD example (plain prompt dataset) wires the teacher through `--custom-rm-path`
(`reward_func` → POSTs tokens → returns logprob JSON) and a `post_process_rewards` that
extracts `teacher_log_probs` and returns a task reward of `0.0`. That works there **only
because the default rollout leaves `sample.reward = None`.**

In `slime/rollout/sglang_rollout.py`, the RM step is guarded:

```python
# Some custom generate paths may have already filled the reward.
if sample.reward is None:
    sample.reward = await async_rm(args, sample)   # <-- only here does reward_func run
```

tau-bench's `generate` **fills `sample.reward`** with the env success reward. So on the agentic
path that guard is false, `reward_func` **never runs**, the teacher is never queried, and a
stock post-process would then crash trying to read logprobs out of a float. Two dead ends:

- Reusing `--custom-rm-path` → teacher never queried (reward already set).
- Using a **custom** `--custom-reward-post-process-path` to inject the task reward → it
  **bypasses GRPO's built-in group normalization** (that normalization only runs when no
  custom post-process is configured — see `slime/ray/rollout.py::_post_process_rewards`).

The clean fix avoids both.

---

## 3. The fix: a rollout sample hook

Register a **rollout sample hook** with `--rollout-sample-hook-path`. Hooks run *before* reward
computation (`slime/rollout/sample_hooks.py`), so the hook can score the student's tokens with
the teacher and set `sample.teacher_log_probs`, while leaving `sample.reward` (the env reward)
completely alone. Then:

- the env reward flows through slime's **normal, group-normalized** reward path → GRPO advantages;
- `teacher_log_probs` is already on the sample → the OPD KL rides on top;
- **no** `--custom-rm-path`, **no** `--custom-reward-post-process-path`.

> **Zero slime-core changes.** This entire agentic setup modifies **no** files under `slime/`.
> It composes existing extension points — `--rollout-sample-hook-path`,
> `--custom-generate-function-path`, and the upstream OPD flags (`--use-opd --opd-type sglang
> --rm-url`). The base OPD pathway (`apply_opd_kl_to_advantages`, `on_policy_distillation.py`)
> is upstream slime; multi-teacher routing added only two small core files (see
> [`MT_PATCH.md`](MT_PATCH.md)). The tau-bench glue lives entirely in this example directory.

[`tau_bench_opd.py`](tau_bench_opd.py) implements it, with **no edits to the upstream tau-bench
example**:

```python
async def opd_teacher_hook(args, sample, *, rollout_id=None, evaluation=False):
    if evaluation:                       # eval reports task success; don't query the teacher
        return sample
    payload = {"input_ids": sample.tokens,
               "sampling_params": {"temperature": 0, "max_new_tokens": 0, "skip_special_tokens": False},
               "return_logprob": True, "logprob_start_len": 0}
    async with aiohttp.ClientSession() as s:
        async with s.post(_resolve_teacher_url(args, sample), json=payload) as r:
            r.raise_for_status(); data = await r.json()
    lp = torch.tensor([x[0] for x in data["meta_info"]["input_token_logprobs"][1:]], dtype=torch.float32)
    sample.teacher_log_probs = lp[-sample.response_length:]   # trim to response span
    return sample
```

It reuses OPD's `_resolve_teacher_url`, so **multi-teacher routing works unchanged**: swap
`--rm-url` for `--opd-teacher-urls "math=...,retail=..."` and tag each task's `metadata.teacher`.

### The two modes

Both modes use `opd_teacher_hook`; they differ only in the generate function:

| | generate | task reward | dynamic-sampling filter |
|---|---|---|---|
| **Mode B** (default) — task reward + KL | `generate_with_tau.generate` (stock) | env success reward, GRPO-normalized | keep `check_reward_nonzero_std` |
| **Mode A** — pure distillation | `tau_bench_opd.generate_pure_distill` (zeros the reward) | constant `0.0` → no task advantage | **drop** it (all-zero groups would be filtered) |

`generate_pure_distill` just runs tau-bench's loop and sets `reward = 0.0` in training (eval
keeps the real reward so eval still reports task success). With a constant reward the task
advantage collapses to 0 and the KL is the sole training signal.

---

## 4. Multi-turn masking (why it's already correct)

In a multi-turn trajectory the token sequence interleaves the model's actions with injected
tool outputs / observations. The OPD KL must apply **only to model-generated tokens**, never to
observations the student didn't choose — otherwise you distill the teacher against text the
student never produced.

tau-bench already handles this: `res_to_sample` sets `sample.loss_mask` (assistant tokens = 1,
observation tokens = 0) and `sample.response_length = len(loss_mask)`. `opd_teacher_hook`
trims teacher logprobs to the last `response_length` tokens (the response span), and the
`loss_mask` then restricts the actual gradient/KL to assistant tokens within that span. So the
alignment is: teacher logprobs cover the response span; `loss_mask` selects which of those
count. Nothing extra to do — but **verify it** (Section 7) before trusting the run.

---

## 5. GPU layout (1 node, 4× GB200)

```
  GPU 3     -> teacher SGLang server        (serve_sglang.sh)
  GPU 0,1   -> tau-bench student, colocated  (train + rollout, TP=2)
  GPU 2     -> spare (raise NUM_GPUS, or give the teacher TP=2 on GPUs 2,3)
```

The teacher runs as its own process with its own `CUDA_VISIBLE_DEVICES`; the trainer's Ray head
is started with `--num-gpus ${NUM_GPUS}` on the training GPUs only.

---

## 6. Running it

**Step 1 — bring up the teacher** (own terminal; must share the student's tokenizer):
```bash
MODEL_PATH=/root/Qwen3-32B GPUS=3 PORT=30000 \
  bash examples/on_policy_distillation/scripts/serve_sglang.sh
```

**Step 2 — smoke-test it** (own terminal; optional but recommended):
```bash
PORT=30000 bash examples/on_policy_distillation/scripts/test_inference.sh
```

**Step 3 — train.** Default is Mode B (task reward + KL):
```bash
# task reward + distillation (default)
TEACHER_URL=http://127.0.0.1:30000/generate OPD_KL_COEF=1.0 \
  bash examples/on_policy_distillation/run-tau-bench-opd.sh

# pure distillation (env reward ignored)
MODE=A TEACHER_URL=http://127.0.0.1:30000/generate \
  bash examples/on_policy_distillation/run-tau-bench-opd.sh
```

Knobs: `MODE` (`B`/`A`), `TEACHER_URL`, `OPD_KL_COEF`, `NUM_GPUS`, `TAU_BENCH_DIR`,
`TAU_DATA_DIR`. Data + checkpoint paths follow the base tau-bench example
(`retail_train_tasks.jsonl` etc.; see `examples/tau-bench/README.md`).

---

## 7. Verifying the run

**Offline first (no GPU):** the glue in `opd_teacher_hook` — sglang JSON → trimmed
`teacher_log_probs`, eval short-circuit, and multi-teacher routing — has a self-contained
unit test that stands up a real local aiohttp teacher on an ephemeral port (no mocking):

```bash
PYTHONPATH=examples/on_policy_distillation \
  python -m pytest examples/on_policy_distillation/tests/test_opd_teacher_hook.py -v
# (or, with no pytest installed:)
PYTHONPATH=examples/on_policy_distillation \
  python examples/on_policy_distillation/tests/test_opd_teacher_hook.py
```

Then, at training time, check these in order — a masking or alignment bug is silent and
quietly corrupts the signal.

1. **Teacher reachable** — the script blocks on `/health_generate`; if it loops forever, the
   server didn't come up (check `/tmp/sglang_30000.log`).
2. **Masking sanity** (do this once, early): confirm that observation tokens have
   `loss_mask == 0` and only assistant-generated tokens have `1`. If tool outputs leak in with
   `1`, fix tau-bench's mask before trusting any OPD numbers.
3. **KL is finite and moving** — watch `opd_reverse_kl` (or the OPD KL metric) in the logs. It
   should be finite from step 1 and trend **down** as the student approaches the teacher. `NaN`/
   `inf` almost always means a token-id / length misalignment between student and teacher.
4. **Mode B reward is non-trivial** — the logged task reward should vary across groups (that's
   what the `check_reward_nonzero_std` filter needs). If it's a flat `0.0` in Mode B, the hook
   or reward wiring is off — confirm you're on stock `generate_with_tau.generate` (not
   `generate_pure_distill`) and that `--rollout-sample-hook-path` is set.
5. **teacher_log_probs present** — if training raises `OPD ... requires teacher_log_probs, but
   it is missing`, the hook didn't run (check `--rollout-sample-hook-path
   tau_bench_opd.opd_teacher_hook` and PYTHONPATH).

---

## 8. Common failures

| Symptom | Cause | Fix |
|---|---|---|
| Whole batch filtered / no updates in Mode A | constant `0.0` reward + `check_reward_nonzero_std` | the script already drops the filter in Mode A; don't re-add it |
| `requires teacher_log_probs, but it is missing` | hook not registered / not imported | set `--rollout-sample-hook-path tau_bench_opd.opd_teacher_hook`; check PYTHONPATH |
| Task reward always `0.0` in Mode B | accidentally on `generate_pure_distill` | Mode B must use stock `generate_with_tau.generate` |
| `ModuleNotFoundError: generate_with_tau` / `tau_bench_opd` | PYTHONPATH missing a dir | the run script adds both `TAU_BENCH_DIR` and this example dir; check they're correct |
| OPD KL is `NaN`/`inf` | student/teacher token ids or lengths don't align | ensure teacher shares the student's tokenizer; confirm `response_length == len(loss_mask)` |
| Teacher OOM / slow | teacher `mem-fraction` or concurrency | lower `MEM_FRACTION` on serve, or add `--sglang-server-concurrency` on the trainer |

---

## 9. How this generalizes to any gym

Nothing here is tau-bench-specific except the import. For your own environment:

1. Write a `generate(args, sample, sampling_params) -> Sample` that runs your env loop
   (`reset → act via SGLang → env.step → reward`) and sets `tokens`, **`loss_mask`
   (assistant-only)**, `reward`, `metadata`.
2. Reuse `opd_teacher_hook` **as-is** — it's completely env-agnostic (it only reads
   `sample.tokens` / `sample.response_length`). Just point `--custom-generate-function-path`
   at your generate and `--rollout-sample-hook-path` at `tau_bench_opd.opd_teacher_hook`.
3. Reuse the run script; change `--custom-generate-function-path`, `TAU_BENCH_DIR`, and PYTHONPATH.

The masking discipline in step 1 is the load-bearing part — get it right first, then turn on
the teacher.
