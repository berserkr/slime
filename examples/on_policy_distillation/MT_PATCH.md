# MT_PATCH.md — Multi-Teacher OPD Routing Patch Summary

Post-patch reference for the multi-teacher on-policy distillation (MOPD) routing feature.
Branch: `multi-teacher`. **Two slime-core source files changed (`+90 / −1`)**, plus new
tests and example docs/scripts. The loss/advantage path is **not** touched — routing is
transparent to it because the OPD penalty is already per-sample.

> **Scope note.** The base OPD pathway (`--use-opd`, `apply_opd_kl_to_advantages`,
> `slime/rollout/on_policy_distillation.py`) is **upstream slime** (PRs #1538, #1610) — this
> branch did not create it. This branch adds only *multi-teacher routing* on top (the two
> files below). The separate **agentic tau-bench OPD** example (`run-tau-bench-opd.sh`,
> `tau_bench_opd.py`) modifies **zero** core files — it composes with existing extension
> points (`--rollout-sample-hook-path`, `--custom-generate-function-path`, `--rm-url`).

For the full workflow (datasets, teacher serving, launch), see
[`MOPD_GETTING_STARTED.md`](MOPD_GETTING_STARTED.md).

---

## slime footprint & provenance (how much of slime is ours)

It helps to separate three layers. Only the middle one changes slime core, and only by
two files.

| Layer | Where it lives | slime-core change | Origin |
|---|---|---|---|
| **1. Base OPD engine** — `--use-opd`, `apply_opd_kl_to_advantages`, sglang/megatron teacher modes, `on_policy_distillation.py` (`reward_func`, `post_process_rewards`) | `slime/` core | — (inherited, unchanged) | **Upstream slime**: PR #1538 (megatron OPD + KL-on-advantages + args) and PR #1610 (moved OPD to `slime/rollout`, CI test, docs) |
| **2. Multi-teacher routing (MOPD)** — per-sample teacher selection | `slime/rollout/on_policy_distillation.py`, `slime/utils/arguments.py` | **2 files, `+90 / −1`** | This branch (`multi-teacher`) |
| **3. Agentic tau-bench OPD** — multi-turn rollout distilled toward a teacher | `examples/on_policy_distillation/` only | **none** | This branch (example-only) |

**Why layer 1 is untouched.** The KL penalty is computed per token from
`student_log_probs − teacher_log_probs` inside `apply_opd_kl_to_advantages`. Nothing in that
function cares *which* teacher produced the logprobs. So both new layers had to change only
*how `teacher_log_probs` gets onto each sample*, never the math that consumes it.

**Why layer 2 is only 2 files (`+90 / −1`).** Routing is a single decision — "for this
sample, which endpoint do I POST to?" — so the entire behavioral change is one line in
`reward_func` (`args.rm_url` → `_resolve_teacher_url(args, sample)`); everything else is the
new helper, two argparse flags, a defensive parser, and one validation guard. All of it is
gated on `args.opd_teacher_url_map` being non-empty, so single-teacher (`--rm-url`) and
megatron (`--opd-teacher-load`) runs are byte-for-byte the old behavior. Full detail below.

**Why layer 3 needs zero core.** The agentic example gets `teacher_log_probs` onto samples
through an *existing* extension point — a rollout sample hook (`--rollout-sample-hook-path`)
that runs before reward computation — and lets the env task reward flow through slime's stock
group-normalized reward path. It reuses layer 2's `_resolve_teacher_url`, so multi-teacher
routing works there for free. The glue (`tau_bench_opd.py`, `run-tau-bench-opd.sh`) lives
entirely in this example directory; see [`TAU_BENCH_OPD.md`](TAU_BENCH_OPD.md).

### How the axes combine (MOPD supports both rollout types)

Teacher count and rollout type are **independent**. MOPD (multiple teachers) is orthogonal to
whether the student generates a one-shot completion or a multi-turn agent episode — the same
`_resolve_teacher_url` serves all four cells:

| | **single teacher** (`--rm-url`) | **multiple teachers / MOPD** (`--opd-teacher-urls`) |
|---|---|---|
| **regular data** | base OPD example (`run-qwen3-8B-opd.sh`) | MOPD as shipped (`MOPD_GETTING_STARTED.md`) |
| **agentic rollout** | tau-bench OPD, single `--rm-url` | tau-bench OPD + `--opd-teacher-urls` (Rung 7) |

The only thing that differs by **row** is *how* `teacher_log_probs` is attached: `reward_func`
when the reward slot is free (regular data), a sample hook when the agent already filled
`sample.reward` (agentic). The **column** — which teacher scores each sample — is the same code
either way. Constant across all cells: every teacher must share the student's tokenizer/vocab.

---

## Summary of what changed

| File | Function / block | Change |
|---|---|---|
| `slime/rollout/on_policy_distillation.py` | `_resolve_teacher_url` (new) | Picks the teacher endpoint per sample from `metadata[routing_key]`. |
| `slime/rollout/on_policy_distillation.py` | `reward_func` | POSTs to `_resolve_teacher_url(args, sample)` instead of the single `args.rm_url`. |
| `slime/utils/arguments.py` | `get_slime_extra_args_provider` → `add_on_policy_distillation_arguments` | Registers `--opd-teacher-urls` and `--opd-routing-key`. |
| `slime/utils/arguments.py` | `_parse_teacher_url_map` (new, module-level) | Pure parser for `--opd-teacher-urls` → dict; raises on malformed input. |
| `slime/utils/arguments.py` | `slime_validate_args` (sglang branch) | Calls `_parse_teacher_url_map`, sets `args.opd_teacher_url_map`; enforces `--rm-url` XOR map. |
| `tests/test_opd_multi_teacher_routing.py` | new | Fast unit tests for both helpers (no GPU/servers). |
| `tests/test_qwen2.5_0.5B_mopd_multi_teacher_sglang.py` | new | End-to-end 2-teacher routing smoke test (sglang, self-distill, `opd_reverse_kl ≈ 0`). |

**Not changed (intentionally):** `post_process_rewards` (server-agnostic — reads whatever
server answered), and `slime/backends/megatron_utils/loss.py::apply_opd_kl_to_advantages`
(already iterates per sample). Each sample carries exactly one routed teacher's
`teacher_log_probs`, so the multi-teacher case falls out for free.

---

## File 1 — `slime/rollout/on_policy_distillation.py`  (+29 / −1)

### `_resolve_teacher_url(args, sample)` — NEW (inserted above `reward_func`, ~line 8)

Chooses the teacher server for a single sample.

- **Single-teacher (back-compat):** if `args.opd_teacher_url_map` is unset/`None`
  (megatron mode, or no `--opd-teacher-urls`), returns `args.rm_url` — identical to
  upstream behavior.
- **Multi-teacher:** reads the teacher name from
  `sample.metadata[args.opd_routing_key]` (default key `"teacher"`) and looks it up in
  `args.opd_teacher_url_map`.
- **Fail-fast `ValueError`** with an actionable message on:
  - missing tag — `sample.metadata` has no routing key (points at the prompt row);
  - unknown teacher — the tag isn't in the launch map (points at data or map).

Uses `getattr(..., None)` / `getattr(..., "teacher")` so a config without the new
attributes (e.g., megatron mode) behaves exactly as single-teacher.

### `reward_func(args, sample, **kwargs)` — MODIFIED (1 line, ~line 54)

Only the POST target changed:

```diff
- async with session.post(args.rm_url, json=payload) as resp:
+ async with session.post(_resolve_teacher_url(args, sample), json=payload) as resp:
```

Payload, multimodal handling, and the response contract are unchanged, so
`post_process_rewards` still parses `meta_info.input_token_logprobs` as before.

---

## File 2 — `slime/utils/arguments.py`  (+62)

### `add_on_policy_distillation_arguments` — 2 args added (~line 1158)

Inserted just before `return parser`, after `--opd-teacher-ckpt-step`:

| Flag | Type / default | Purpose |
|---|---|---|
| `--opd-teacher-urls` | `str` / `None` | Comma-separated `name=url` pairs, e.g. `math=http://h1:8001/generate,code=http://h2:8002/generate`. Enables routing (sglang mode). |
| `--opd-routing-key` | `str` / `"teacher"` | Which `sample.metadata` key holds the teacher name. |

### `_parse_teacher_url_map(opd_teacher_urls)` — NEW, module-level (just above `slime_validate_args`)

Pure, unit-testable parser. Returns `None` when unset (single-teacher mode via
`--rm-url`); otherwise a `name -> url` dict. Parses **defensively** — strips whitespace,
skips blanks, splits on the **first** `=` (so URLs with query strings survive), and raises
`ValueError` on: malformed entry (no `=`), empty name or url, duplicate teacher name, or an
all-empty map. Extracted as a standalone function so it can be tested without running full
argument validation.

### OPD validation, `sglang` branch — wire in the parser + guard (~line 1840)

Extends the existing `elif args.opd_type == "sglang":` branch (in the
`# Validate on-policy distillation (OPD) arguments` block):

```python
args.opd_teacher_url_map = _parse_teacher_url_map(args.opd_teacher_urls)
if args.opd_teacher_url_map is None and args.rm_url is None:
    raise ValueError("opd-type=sglang requires either --rm-url ... or --opd-teacher-urls ...")
```

- Always sets `args.opd_teacher_url_map` (possibly `None`) so the rollout side can rely on
  the attribute existing in sglang mode.
- If **no** map is given, requires `--rm-url` — preserves single-teacher mode and turns a
  "no teacher configured" mistake into a startup error rather than a rollout-time failure.

Net rule: in sglang mode you must provide **either** `--rm-url` (single teacher) **or**
`--opd-teacher-urls` (multi-teacher routing).

## Tests — two new files (NEW)

### `tests/test_opd_multi_teacher_routing.py` — fast unit tests (no GPU/servers/downloads)

Follows the repo's `test_*_validation.py` convention (`SimpleNamespace` args,
`pytest.raises(match=...)`):

- `_parse_teacher_url_map`: unset→`None`, valid pairs w/ whitespace, URL containing `=`,
  and the four raise paths (malformed / empty name-or-url / duplicate / empty map).
- `_resolve_teacher_url`: routes by tag, honors a custom `--opd-routing-key`, raises on
  missing tag (incl. `metadata=None`) and unknown teacher, and falls back to `rm_url` both
  when the map is `None` and when the attribute is absent (megatron mode).

### `tests/test_qwen2.5_0.5B_mopd_multi_teacher_sglang.py` — end-to-end smoke test (needs GPUs)

The multi-teacher analogue of the upstream `test_qwen2.5_0.5B_opd_sglang.py`: launches **two**
distinct sglang teacher servers (both serving the same Qwen2.5-0.5B, so it stays a
self-distillation test with `opd_reverse_kl ≈ 0`), builds a gsm8k dataset whose rows alternate
`metadata.teacher = "teacher_a"/"teacher_b"`, and trains with
`--opd-teacher-urls "teacher_a=<url_a>,teacher_b=<url_b>"` — exercising the routing patch through
a real run. Point either server at a different checkpoint (same tokenizer!) for a true
multi-teacher run.

> The **agentic** hook (`opd_teacher_hook`) has its own offline test —
> `examples/on_policy_distillation/tests/test_opd_teacher_hook.py` — covering the sglang-JSON →
> `teacher_log_probs` trim, the eval short-circuit, and routing, against a real local aiohttp
> teacher (no GPU). See [`TAU_BENCH_OPD.md`](TAU_BENCH_OPD.md) §7.

---

## New argument surface

```bash
# single teacher (unchanged)
--use-opd --opd-type sglang --rm-url http://teacher:8000/generate

# multi-teacher routing (new)
--use-opd --opd-type sglang \
  --opd-teacher-urls "math=http://h1:8001/generate,code=http://h2:8002/generate" \
  --opd-routing-key teacher      # optional; default "teacher"
```

Resolved at parse time into `args.opd_teacher_url_map: dict[str, str] | None`.

---

## Verification performed

Both touched functions are pure Python; validated by exact-copy inline tests
(deps `aiohttp`/`torch` aren't installed in this checkout, so the module isn't imported):

- `ast.parse` clean on both files.
- **Resolver:** correct routing (`math`→URL_M, `code`→URL_C); raises on missing tag,
  `metadata=None`, and unknown teacher; falls back to `rm_url` when no map. ✅
- **Map parse:** accepts valid pairs (with surrounding whitespace); raises on malformed,
  empty-name, empty-url, duplicate, and empty-map. ✅

Not yet run: full `parse_args` end-to-end (needs a valid model config) and an
end-to-end training smoke test. Recommended next check — the self-distill smoke test in
`MOPD_GETTING_STARTED.md` §3e/§Sanity checklist (point every teacher name at one server
serving a copy of the student; watch `opd_reverse_kl ≈ 0`).

---

## Rollback

```bash
git checkout -- slime/rollout/on_policy_distillation.py slime/utils/arguments.py
```

Both changes are additive and gated on `args.opd_teacher_url_map` being non-empty, so
existing single-teacher (`--rm-url`) and megatron (`--opd-teacher-load`) runs are
unaffected.
