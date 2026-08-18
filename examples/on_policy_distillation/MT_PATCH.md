# MT_PATCH.md — Multi-Teacher OPD Routing Patch Summary

Post-patch reference for the multi-teacher on-policy distillation (MOPD) routing feature.
Branch: `multi-teacher`. Two source files changed (**+90 / −1**) plus one new unit test.
The loss/advantage path is **not** touched — routing is transparent to it because the
OPD penalty is already per-sample.

For the full workflow (datasets, teacher serving, launch), see
[`MOPD_GETTING_STARTED.md`](MOPD_GETTING_STARTED.md).

---

## Summary of what changed

| File | Function / block | Change |
|---|---|---|
| `slime/rollout/on_policy_distillation.py` | `_resolve_teacher_url` (new) | Picks the teacher endpoint per sample from `metadata[routing_key]`. |
| `slime/rollout/on_policy_distillation.py` | `reward_func` | POSTs to `_resolve_teacher_url(args, sample)` instead of the single `args.rm_url`. |
| `slime/utils/arguments.py` | `get_slime_extra_args_provider` → `add_on_policy_distillation_arguments` | Registers `--opd-teacher-urls` and `--opd-routing-key`. |
| `slime/utils/arguments.py` | `_parse_teacher_url_map` (new, module-level) | Pure parser for `--opd-teacher-urls` → dict; raises on malformed input. |
| `slime/utils/arguments.py` | `slime_validate_args` (sglang branch) | Calls `_parse_teacher_url_map`, sets `args.opd_teacher_url_map`; enforces `--rm-url` XOR map. |
| `tests/test_opd_multi_teacher_routing.py` | new | Unit tests for both helpers (no GPU/servers). |

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

## File 3 — `tests/test_opd_multi_teacher_routing.py`  (NEW)

Fast pytest unit tests (no GPU, servers, or downloads), following the repo's
`test_*_validation.py` convention (`SimpleNamespace` args, `pytest.raises(match=...)`):

- `_parse_teacher_url_map`: unset→`None`, valid pairs w/ whitespace, URL containing `=`,
  and the four raise paths (malformed / empty name-or-url / duplicate / empty map).
- `_resolve_teacher_url`: routes by tag, honors a custom `--opd-routing-key`, raises on
  missing tag (incl. `metadata=None`) and unknown teacher, and falls back to `rm_url` both
  when the map is `None` and when the attribute is absent (megatron mode).

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
