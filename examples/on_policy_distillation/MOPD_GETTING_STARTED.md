# Multi-Teacher On-Policy Distillation (MOPD) on slime — Getting Started

This guide gets you from *"I have a warmed-up RLVR MoE student and a few specialist
teachers"* to *"I'm running multi-teacher on-policy distillation"* on slime.

slime ships **single-teacher** OPD upstream (see
[`docs/en/advanced/on-policy-distillation.md`](../../docs/en/advanced/on-policy-distillation.md)).
MOPD adds **per-trajectory routing**: each prompt is tagged with a teacher name, and
at rollout time the reward function fetches that teacher's log-probs. One teacher scores
each trajectory — this is the specialist-routing model (à la Nemotron), *not* per-token
averaging.

Because slime's OPD penalty is already applied **per sample**
(`apply_opd_kl_to_advantages` in `slime/backends/megatron_utils/loss.py`), routing is
almost free: the loss/advantage path needs **zero changes**. All we add is a way to pick
a teacher URL per sample.

---

## The mental model

```
                 prompt row (jsonl)
                 { "input": ...,  "metadata": { "teacher": "math" } }
                          │
                          ▼
   ┌──────────── slime rollout (student MoE generates) ────────────┐
   │  student samples response tokens a_t ~ π_student              │
   └──────────────────────────┬────────────────────────────────────┘
                               │ reward_func routes by metadata.teacher
              ┌────────────────┼─────────────────┐
              ▼                ▼                  ▼
        SGLang teacher    SGLang teacher     SGLang teacher
          "math"             "code"            "general"
        (scores a_t)       (scores a_t)       (scores a_t)
              └────────────────┼─────────────────┘
                               │ teacher_log_probs (per sample)
                               ▼
        Â_t = A_t − λ_opd · ( logπ_student(a_t) − logπ_teacher(a_t) )
                               │  (sampled reverse-KL, per token, per sample)
                               ▼
                     Megatron policy update on the student
```

- **One teacher per trajectory.** Routing key = the dataset/domain the prompt came from.
- **A_t** is your base advantage (GRPO / PPO / REINFORCE++ / …). For *pure* distillation
  set task reward to 0 and let the OPD term be the whole signal.
- **All teachers must share the student's tokenizer/vocabulary** — they score the
  student's exact token IDs. For a Qwen3 MoE student, use Qwen3-family teachers.

---

## Prerequisites

| Thing | What you need |
|---|---|
| **Student** | Your warmed-up **RLVR MoE** checkpoint, in Megatron `torch_dist` format. |
| **Teachers** | N specialist MoE checkpoints, **same tokenizer/vocab** as the student. HF format is fine — SGLang serves them directly. |
| **Data** | One or more jsonl prompt files, each row tagged with a teacher name (see below). |
| **GPUs** | Student training job (Megatron) + one SGLang server per teacher (separate nodes). |

Convert the **student** to Megatron format (teachers stay in HF, served by SGLang):

```bash
cd /root/slime
source scripts/models/qwen3-30B-A3B.sh          # your MoE's model-args script
PYTHONPATH=/root/Megatron-LM python tools/convert_hf_to_torch_dist.py \
    ${MODEL_ARGS[@]} \
    --hf-checkpoint /root/student-rlvr-moe \
    --save /root/student-rlvr-moe_torch_dist
```

---

## Step 1 — Structure your datasets (the important part)

MOPD routing is driven entirely by a **`metadata.teacher`** field on each prompt.
slime's training loader (`slime/utils/data.py`) copies each jsonl row's `metadata`
object straight onto `sample.metadata`, so this works with **no code change**.

### Row format

Each line of your `--prompt-data` jsonl:

```json
{"input": "Solve: integral of x^2 dx", "label": "x^3/3 + C", "metadata": {"teacher": "math"}}
{"input": "Write a function to reverse a linked list", "label": "...", "metadata": {"teacher": "code"}}
{"input": "Explain photosynthesis to a 10 year old", "label": null, "metadata": {"teacher": "general"}}
```

- `input` — the prompt (key set by `--input-key`, default `input`).
- `label` — optional; only needed if you also compute a **task reward** (verifiable RLVR).
  For pure distillation, `label` can be omitted/null.
- `metadata.teacher` — **the routing key.** Its value must match a teacher name you
  register at launch (Step 3). This is the only field MOPD adds.

### One file or many?

Both work — pick whichever is easier to curate:

- **Single mixed file:** every row carries its own `metadata.teacher`. Simplest.
- **Per-domain files:** keep `math.jsonl`, `code.jsonl`, … each internally consistent,
  and concatenate. (Every row still needs the `metadata.teacher` field — slime does not
  infer it from the filename.)

### Domain balance = teacher load balance

The fraction of a batch tagged `"math"` is exactly the query load hitting the `math`
teacher server. If one domain dominates, that teacher becomes the bottleneck — either
rebalance the data or give the hot teacher more SGLang replicas (Step 2).

> **Custom routing key.** If you'd rather route on `label` or another field, set
> `--opd-routing-key <field>` (Step 4). Default is `teacher`, read from `sample.metadata`.

---

## Step 2 — Serve the teachers (one SGLang server each)

Launch one SGLang server per specialist. Each must return input-token log-probs (the
OPD reward function requests `return_logprob=True`, so a stock server is fine). Size
`--tp`/`--ep` to the teacher MoE and keep each on its own node group.

```bash
# math teacher  (node A)
python -m sglang.launch_server \
    --model-path /root/teacher-math-moe \
    --tp 8 --ep 8 \
    --host 0.0.0.0 --port 8001

# code teacher  (node B)
python -m sglang.launch_server \
    --model-path /root/teacher-code-moe \
    --tp 8 --ep 8 \
    --host 0.0.0.0 --port 8002

# general teacher  (node C)
python -m sglang.launch_server \
    --model-path /root/teacher-general-moe \
    --tp 8 --ep 8 \
    --host 0.0.0.0 --port 8003
```

Note each server's `IP:PORT` — you'll map teacher names to these `/generate` endpoints
in Step 4.

---

## Step 3 — Implementing the routing patch

Upstream slime resolves a **single** teacher endpoint (`args.rm_url`). Multi-teacher
routing needs exactly two edits: register two new args, then choose the URL per sample.
Nothing in the loss/advantage path changes — each sample still ends up carrying one
teacher's log-probs, and the OPD penalty is already applied per sample.

This section is a full implementation reference: exact file, function, insertion point,
and before/after for each edit.

### 3a. Register the args — `slime/utils/arguments.py`

Find `add_on_policy_distillation_arguments` (around line 1117). The last argument it
registers is `--opd-teacher-ckpt-step`, immediately followed by `return parser`
(around line 1155–1158):

```python
            parser.add_argument(
                "--opd-teacher-ckpt-step", type=int, default=None, help="The checkpoint step for OPD teacher model."
            )
            return parser          # <-- insert the two new args ABOVE this line
```

Insert the two MOPD args just before that `return parser`:

```python
            parser.add_argument(
                "--opd-teacher-urls",
                type=str,
                default=None,
                help=(
                    "MOPD multi-teacher routing (opd-type=sglang only). Comma-separated "
                    "name=url pairs, e.g. "
                    "'math=http://h1:8001/generate,code=http://h2:8002/generate'. "
                    "When set, the teacher is chosen per sample via --opd-routing-key "
                    "instead of the single --rm-url."
                ),
            )
            parser.add_argument(
                "--opd-routing-key",
                type=str,
                default="teacher",
                help=(
                    "Which sample.metadata key holds the teacher name for MOPD routing. "
                    "Default 'teacher' (reads sample.metadata['teacher'])."
                ),
            )
            return parser
```

### 3b. Parse + validate the map — `slime/utils/arguments.py`

The OPD validation block is near line 1780 (`# Validate on-policy distillation (OPD) arguments`).
The `sglang` branch currently only guards `--opd-teacher-load` (around line 1801):

```python
        elif args.opd_type == "sglang":
            if args.opd_teacher_load is not None:
                raise ValueError(
                    "--opd-teacher-load should not be set when --opd-type=sglang. "
                    "In sglang mode, teacher log-probs are obtained from external server during rollout."
                )
```

Extend that `sglang` branch to build `args.opd_teacher_url_map`. Parse defensively
(strip whitespace, reject malformed/empty pairs) so a typo fails fast at startup rather
than mid-rollout:

```python
        elif args.opd_type == "sglang":
            if args.opd_teacher_load is not None:
                raise ValueError(
                    "--opd-teacher-load should not be set when --opd-type=sglang. "
                    "In sglang mode, teacher log-probs are obtained from external server during rollout."
                )

            # MOPD: build the teacher name -> url map (None => single-teacher via --rm-url)
            args.opd_teacher_url_map = None
            if args.opd_teacher_urls:
                url_map = {}
                for pair in args.opd_teacher_urls.split(","):
                    pair = pair.strip()
                    if not pair:
                        continue
                    if "=" not in pair:
                        raise ValueError(
                            f"--opd-teacher-urls entry {pair!r} is malformed; "
                            "expected 'name=url' pairs separated by commas."
                        )
                    name, url = pair.split("=", 1)
                    name, url = name.strip(), url.strip()
                    if not name or not url:
                        raise ValueError(
                            f"--opd-teacher-urls entry {pair!r} has an empty name or url."
                        )
                    if name in url_map:
                        raise ValueError(f"--opd-teacher-urls has a duplicate teacher name {name!r}.")
                    url_map[name] = url
                if not url_map:
                    raise ValueError("--opd-teacher-urls was set but parsed to an empty map.")
                args.opd_teacher_url_map = url_map
            elif args.rm_url is None:
                raise ValueError(
                    "opd-type=sglang requires either --rm-url (single teacher) or "
                    "--opd-teacher-urls (multi-teacher routing)."
                )
```

There is one more spot to keep consistent: the `else` branch (`use_opd` disabled)
already rejects a stray `--opd-teacher-load`. You don't need to guard `--opd-teacher-urls`
there — it simply has no effect when `--use-opd` is off — but you may add a symmetric
warning if you want misconfig to be loud.

> **Why set `args.opd_teacher_url_map = None` explicitly?** So the rollout code can do a
> plain `getattr(args, "opd_teacher_url_map", None)` and treat "attribute missing"
> (megatron mode, or older configs) and "no map" identically — single-teacher behavior.

### 3c. Route per sample — `slime/rollout/on_policy_distillation.py`

This is the only behavioral change. The current `reward_func` posts to the single
`args.rm_url` (line 27). Add a resolver and use it. **Before:**

```python
async def reward_func(args, sample, **kwargs):
    payload = {
        "input_ids": sample.tokens,
        "sampling_params": { "temperature": 0, "max_new_tokens": 0, "skip_special_tokens": False },
        "return_logprob": True,
        "logprob_start_len": 0,
    }
    if sample.multimodal_inputs and sample.multimodal_inputs.get("images"):
        image_data = sample.multimodal_inputs["images"]
        payload["image_data"] = [encode_image_for_rollout_engine(image) for image in image_data]

    session_kwargs = {}
    async with aiohttp.ClientSession(**session_kwargs) as session:
        async with session.post(args.rm_url, json=payload) as resp:
            resp.raise_for_status()
            return await resp.json()
```

**After** — add the helper above `reward_func`, and swap the one `session.post` target:

```python
def _resolve_teacher_url(args, sample):
    """Pick the teacher endpoint for this sample.

    Single-teacher (no map): use args.rm_url, unchanged upstream behavior.
    MOPD routing: read the teacher name from sample.metadata[args.opd_routing_key]
    and look it up in args.opd_teacher_url_map.
    """
    url_map = getattr(args, "opd_teacher_url_map", None)
    if not url_map:
        return args.rm_url
    routing_key = getattr(args, "opd_routing_key", "teacher")
    metadata = sample.metadata or {}
    name = metadata.get(routing_key)
    if name is None:
        raise ValueError(
            f"MOPD routing: sample is missing metadata[{routing_key!r}]. "
            f"Every prompt row must carry a teacher tag; known teachers: {list(url_map)}."
        )
    if name not in url_map:
        raise ValueError(
            f"MOPD routing: teacher {name!r} not in --opd-teacher-urls {list(url_map)}. "
            f"Fix the prompt's metadata.{routing_key} or the launch map."
        )
    return url_map[name]


async def reward_func(args, sample, **kwargs):
    payload = {
        "input_ids": sample.tokens,
        "sampling_params": {"temperature": 0, "max_new_tokens": 0, "skip_special_tokens": False},
        "return_logprob": True,
        "logprob_start_len": 0,
    }
    if sample.multimodal_inputs and sample.multimodal_inputs.get("images"):
        image_data = sample.multimodal_inputs["images"]
        payload["image_data"] = [encode_image_for_rollout_engine(image) for image in image_data]

    async with aiohttp.ClientSession() as session:
        async with session.post(_resolve_teacher_url(args, sample), json=payload) as resp:
            resp.raise_for_status()
            return await resp.json()
```

### 3d. Why nothing else changes

- **`post_process_rewards`** (same file) reads `reward["meta_info"]["input_token_logprobs"]`
  and writes `sample.teacher_log_probs` — it operates on whatever server answered, so
  routing is transparent to it. No change.
- **`slime/backends/megatron_utils/loss.py::apply_opd_kl_to_advantages`** (line 663)
  iterates samples and computes `reverse_kl = student_log_probs[i] - teacher_log_probs[i]`
  **per sample**. Since each sample already carries its own routed teacher's log-probs,
  the multi-teacher case is already handled. No change.
- **The advantage estimator** is untouched — OPD rides on top of it exactly as in the
  single-teacher case.

That per-sample structure is precisely why routing is a ~40-line patch and not a rewrite.

### 3e. Verify the patch before a full run

1. **Arg parsing / map build** — dry-check the parser without launching training:
   ```bash
   python -c "
   import sys; sys.argv = ['x','--use-opd','--opd-type','sglang',
       '--opd-teacher-urls','math=http://h1:8001/generate,code=http://h2:8002/generate']
   from slime.utils.arguments import parse_args   # adjust to slime's entrypoint
   a = parse_args(); print(a.opd_teacher_url_map, a.opd_routing_key)
   "
   ```
   Expect the dict `{'math': ..., 'code': ...}` and `teacher`. Feed it a malformed
   `--opd-teacher-urls 'mathhttp://...'` and confirm it raises.
2. **Routing resolver unit check** — no servers needed:
   ```python
   from types import SimpleNamespace
   from slime.rollout.on_policy_distillation import _resolve_teacher_url
   args = SimpleNamespace(opd_teacher_url_map={"math": "URL_M"}, opd_routing_key="teacher", rm_url=None)
   ok  = SimpleNamespace(metadata={"teacher": "math"})
   bad = SimpleNamespace(metadata={"teacher": "physics"})
   assert _resolve_teacher_url(args, ok) == "URL_M"
   try: _resolve_teacher_url(args, bad); assert False
   except ValueError: pass
   # back-compat: no map -> falls back to rm_url
   assert _resolve_teacher_url(SimpleNamespace(opd_teacher_url_map=None, rm_url="URL_S"), ok) == "URL_S"
   ```
3. **End-to-end self-distill** — point every teacher name at one server serving a copy of
   the student and watch `opd_reverse_kl` (logged from `loss.py`) sit near 0. This is the
   existing single-teacher test path (`tests/test_qwen2.5_0.5B_opd_sglang.py`) with a
   two-name map that resolves to the same URL — a minimal routing regression test.

### 3f. Optional: extend the CI test

`tests/test_qwen2.5_0.5B_opd_sglang.py` covers single-teacher sglang OPD. To lock in
routing, add a case that (a) passes `--opd-teacher-urls "a=<url>,b=<url>"` pointing both
names at the one test server, and (b) uses a tiny prompt file where rows are tagged
`metadata.teacher` `"a"` or `"b"`. Assert the run completes and `opd_reverse_kl` is
finite — proving the routing path parses, resolves, and feeds the loss identically to the
single-teacher case.

---

## Step 4 — Launch MOPD

Start from the single-teacher example
([`run-qwen3-8B-opd.sh`](run-qwen3-8B-opd.sh)) and swap the single `--rm-url` for the
teacher map. The OPD-specific flags:

```bash
# --- MOPD (multi-teacher, routed) ---
--use-opd
--opd-type sglang
--opd-kl-coef 1.0
--opd-routing-key teacher
--opd-teacher-urls "math=http://10.0.0.1:8001/generate,code=http://10.0.0.2:8002/generate,general=http://10.0.0.3:8003/generate"
--custom-rm-path slime.rollout.on_policy_distillation.reward_func
--custom-reward-post-process-path slime.rollout.on_policy_distillation.post_process_rewards

# --- base RL (advantage estimator OPD rides on top of) ---
--advantage-estimator grpo          # or gspo / ppo / reinforce_plus_plus
--kl-coef 0.0                        # OPD term is the learning signal for pure distill

# --- student model + data ---
--load /root/student-rlvr-moe_torch_dist
--ref-load /root/student-rlvr-moe_torch_dist
--prompt-data /root/mopd_prompts.jsonl
--input-key input
--label-key label
--metadata-key metadata
--apply-chat-template
```

> For pure distillation, keep task reward at 0 (the stock `post_process_rewards` already
> returns `0.0` rewards) and set `--kl-coef 0.0` so the only signal is the OPD reverse-KL.
> To blend in verifiable rewards, add them in a custom post-process and keep `--opd-kl-coef`
> to weight distillation against task reward.

---

## Step 5 — MoE-specific notes

- **Tokenizer/vocab must match.** Every teacher scores the student's exact token IDs.
  Mixing tokenizer families (e.g., a Llama teacher for a Qwen student) will silently
  produce garbage log-probs — keep all teachers in the student's model family.
- **EP / TP on teachers is independent** of the student's parallelism. Size each SGLang
  teacher to its own MoE; they're decoupled processes.
- **Routing replay** (`--use-rollout-routing-replay`) affects the *student's* MoE
  expert consistency between rollout and training — it's orthogonal to MOPD and you can
  enable it as usual. slime already fences the teacher forward path for routing replay in
  `actor.py` (the `megatron` teacher mode), but in `sglang` mode teachers are external so
  this doesn't apply to them.
- **Hot teacher = more replicas.** Because routing sends whole domains to one server,
  give over-represented domains more SGLang replicas behind the same name (round-robin at
  the URL, or run several and list them — a natural extension of the map).

---

## Sanity checklist before a real run

1. **Self-distill smoke test:** point *all* teacher names at one server serving a copy of
   the student. The OPD reverse-KL should hover near 0 and training should be stable.
   Watch `opd_reverse_kl` in the logs (logged from `loss.py`).
2. **Routing coverage:** every distinct `metadata.teacher` value in your data must appear
   in `--opd-teacher-urls`, or `reward_func` raises. Grep your jsonl for the set of teacher
   names and diff against the map.
3. **Teacher gap:** confirm each specialist actually beats the student on its domain —
   distilling from a teacher no better than the student wastes compute.
4. **Reverse-KL scale:** if `opd_reverse_kl` is huge at step 0, the student's rollouts are
   out-of-distribution for the teacher — that's what the RLVR/warmup student is for; a
   larger gap may need a lower `--opd-kl-coef` to start.

---

## What this does and doesn't cover

- ✅ Per-**trajectory** routing to one specialist teacher (the scalable, evidence-backed mode).
- ✅ Any base advantage estimator; pure-distill or reward-blended.
- ❌ Per-**token** teacher switching (much harder; not needed for specialist routing).
- ❌ Teacher **ensembling/averaging** across teachers on the same token (N× cost, weaker evidence).
- ❌ Cross-tokenizer teachers (fundamental to the sampled-token objective).

For the single-teacher baseline and the in-process `megatron` teacher mode, see
[`docs/en/advanced/on-policy-distillation.md`](../../docs/en/advanced/on-policy-distillation.md).
