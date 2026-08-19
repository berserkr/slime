# OPD testing — baby steps

A ladder from "is SGLang even alive" to "multi-teacher agentic OPD is training." Each rung is
independently verifiable, cheap, and builds on the one below. **Do them in order** — when
something breaks, the lowest failing rung tells you exactly where. Don't skip ahead to a
training run; most failures are two or three rungs down and much faster to catch there.

Assumptions: 1 node, ≥1 GPU; slime installed at `/root/slime`; commands run from
`/root/slime`. Adjust `MODEL_PATH` / checkpoint paths to yours.

---

## Rung 0 — an endpoint comes up and answers

Goal: prove SGLang serves your model and responds. No slime, no training.

```bash
# terminal 1 — bring a server up on one GPU:
MODEL_PATH=/root/Qwen3-8B GPUS=0 PORT=30000 \
  bash examples/on_policy_distillation/scripts/serve_sglang.sh
# wait for "server is UP"

# terminal 2 — smoke-test it:
PORT=30000 bash examples/on_policy_distillation/scripts/test_inference.sh
```

**Pass when:** `test_inference.sh` prints "all requests succeeded" (health, model info,
`/generate`, and `/v1/chat/completions` all return). If it hangs on health, read
`/tmp/sglang_30000.log`.

---

## Rung 1 — the teacher can *score* tokens (return_logprob)

Goal: OPD doesn't generate from the teacher, it asks for logprobs of given tokens. Prove that
works (this is exactly what `opd_teacher_hook` / OPD `reward_func` do).

```bash
curl -s http://127.0.0.1:30000/generate -H 'Content-Type: application/json' -d '{
  "input_ids": [9906, 1917, 0],
  "sampling_params": {"temperature": 0, "max_new_tokens": 0, "skip_special_tokens": false},
  "return_logprob": true, "logprob_start_len": 0
}' | python3 -m json.tool | grep -A3 input_token_logprobs | head
```

**Pass when:** the response contains `meta_info.input_token_logprobs` (a list of `[logprob,
token_id, ...]`). If this is empty or errors, no OPD variant will work — stop and fix the server
first.

> Tokenizer check: the teacher **must** share the student's vocab. If you're unsure, tokenize
> the same string on both and compare ids before going further.

---

## Rung 2 — base single-teacher OPD on a plain dataset (no agent, no routing)

Goal: get the *simplest* real OPD training loop running end-to-end. This is the upstream
example — self-distillation (student == teacher) is fine and expected to drive the KL toward 0.

```bash
# Follow examples/on_policy_distillation/README.md "Using SGLang Teacher":
#   - convert the student to torch_dist
#   - point run-qwen3-8B-opd.sh at your teacher endpoint (--rm-url) + data
bash examples/on_policy_distillation/run-qwen3-8B-opd.sh
```

**Pass when:** training starts, and `opd_reverse_kl` shows up in the logs and is **finite**.
This rung validates your slime install, checkpoints, and the OPD KL path — everything above
agentic/routing. If this doesn't run, nothing more complex will.

---

## Rung 3 — multi-teacher routing, same model on two ports

Goal: prove the routing plumbing (`--opd-teacher-urls` + `metadata.teacher`) before involving
different teachers. Use the **same** checkpoint on two ports — cheap, and any difference in
behavior is then purely routing, not model.

```bash
# two endpoints, same model, different ports:
MODEL_PATH=/root/Qwen3-8B GPUS=2 PORT=13141 TEACHER_NAME=math \
  bash examples/on_policy_distillation/scripts/serve_teacher.sh
MODEL_PATH=/root/Qwen3-8B GPUS=3 PORT=13142 TEACHER_NAME=code \
  bash examples/on_policy_distillation/scripts/serve_teacher.sh

# tagged sample data already ships with the repo:
jq -r '.metadata.teacher' examples/on_policy_distillation/data/prompts_tagged.jsonl | sort -u
#   code
#   math   <- these must equal the LEFT-hand names in --opd-teacher-urls

# train with the routing map (defaults point at the sample data):
bash examples/on_policy_distillation/scripts/train_opd_2teachers.sh
```

**Pass when:** training runs without a `metadata[...] ... not in --opd-teacher-urls` error. To
*prove* routing actually happened, temporarily point the two ports at different-sized models, or
add a log line in `_resolve_teacher_url`. See [MOPD_GETTING_STARTED.md](MOPD_GETTING_STARTED.md).

**First, test the failure too** (fast, no training): launch with data whose tag is misspelled
(`"mathz"`) and confirm you get the `ValueError` at rollout time. Knowing the guard fires is as
important as the happy path.

---

## Rung 4 — the agent loop alone (tau-bench, NO OPD)

Goal: get the agentic rollout working *before* adding distillation. Two independent things;
debug them separately.

```bash
# stand up tau-bench per its own README (env/mock, data files), then:
bash examples/tau-bench/run_qwen3_4B.sh
```

**Pass when:** episodes roll out and the logged task reward **varies** across groups (some
successes, some failures). If tau-bench itself doesn't run, fix that first — OPD sits on top of
it and can't paper over a broken env.

**Then eyeball the mask** (the silent-failure risk): dump one sample and confirm
`loss_mask == 0` on tool/observation tokens, `1` only on assistant-generated tokens. Set
`--save-debug-rollout-data` and inspect, or add a one-off print in `res_to_sample`. Do this
once — a wrong mask corrupts OPD invisibly.

---

## Rung 5 — tau-bench + OPD, pure distillation (Mode A)

Goal: add the teacher to the working agent loop, with the simplest reward story (constant 0.0,
so only the KL trains). Fewest moving parts on the OPD side.

```bash
MODEL_PATH=/root/Qwen3-32B GPUS=3 PORT=30000 \
  bash examples/on_policy_distillation/scripts/serve_sglang.sh      # teacher

MODE=A TEACHER_URL=http://127.0.0.1:30000/generate \
  bash examples/on_policy_distillation/run-tau-bench-opd.sh
```

**Pass when:** training runs and `opd_reverse_kl` is finite and trends **down**. If you hit
`requires teacher_log_probs, but it is missing`, the hook isn't registered — check
`--rollout-sample-hook-path` and PYTHONPATH. See [TAU_BENCH_OPD.md](TAU_BENCH_OPD.md) §7–8.

---

## Rung 6 — tau-bench + OPD, task reward + KL (Mode B, the goal)

Goal: the real thing — GRPO on task success *plus* the teacher KL.

```bash
TEACHER_URL=http://127.0.0.1:30000/generate OPD_KL_COEF=1.0 \
  bash examples/on_policy_distillation/run-tau-bench-opd.sh          # MODE defaults to B
```

**Pass when:** both signals are alive — the logged **task reward varies across groups** (GRPO is
learning task success) **and** `opd_reverse_kl` is finite and trending down (distillation is
working). If the reward is a flat 0.0, you're accidentally on `generate_pure_distill`; if the KL
error appears, the hook didn't run.

---

## Rung 7 — agentic + multi-teacher routing (everything on)

Only after Rungs 3 and 6 both pass. Swap the single `--rm-url` in `run-tau-bench-opd.sh` for a
routing map and tag each task:

```bash
# edit GRPO_ARGS: replace  --rm-url ...  with
#   --opd-teacher-urls "retail=http://127.0.0.1:13141/generate,general=http://127.0.0.1:13142/generate"
#   --opd-routing-key teacher
# and make sure each task's metadata.teacher is "retail" or "general".
```

`opd_teacher_hook` already routes via `_resolve_teacher_url` — no code change needed.

---

## Debugging ladder (when a rung fails)

| Rung | If it fails, suspect |
|---|---|
| 0 | model path, GPU free, sglang install (`/tmp/sglang_*.log`) |
| 1 | wrong endpoint route, model doesn't return logprobs, tokenizer mismatch |
| 2 | slime install, checkpoint conversion, `--rm-url` wrong, OPD args |
| 3 | tag vocabulary ≠ `--opd-teacher-urls` names, `--metadata-key`/`--opd-routing-key` |
| 4 | tau-bench env/data, **loss_mask**, reward wiring in the example |
| 5–6 | `--rollout-sample-hook-path` not set, PYTHONPATH, teacher/student vocab mismatch |
| 7 | routing tags on the agentic data; re-check Rung 3 in isolation |

**Golden rule:** never debug two layers at once. If Rung 6 misbehaves, confirm Rung 4 (agent) and
Rung 2 (OPD) still pass on their own — the bug is almost always in the layer you just added.
