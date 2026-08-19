# MOPD data preparation

Multi-teacher OPD reuses slime's **normal** prompt format and adds exactly **one** field:
`metadata.teacher`. If you can already run slime RL on a dataset, you make it MOPD-ready by
tagging each row with the teacher that should score it.

## Sample files here

| File | Shape | Use with |
|---|---|---|
| [`prompts_tagged.jsonl`](prompts_tagged.jsonl) | string prompt | `--input-key prompt --label-key label` |
| [`prompts_tagged_messages.jsonl`](prompts_tagged_messages.jsonl) | chat messages | `--input-key messages --label-key label` |

Both carry `--metadata-key metadata` (the default) and a `metadata.teacher` tag on every row.

## The two supported row shapes

slime's loader (`slime/utils/data.py::_build_messages`) reads the field named by
`--input-key` and accepts either shape:

**A. String prompt** — like the `dapo-math-17k` dataset the base OPD example uses
(`--input-key prompt`). With `--apply-chat-template`, slime wraps it as a single user turn.

```json
{"prompt": "What is the derivative of x^3 + 2x?", "label": "3x^2 + 2", "metadata": {"teacher": "math"}}
```

**B. Chat messages** — like the `gsm8k` dataset (`--input-key messages`). A list of
`{"role", "content"}` dicts, used as-is (multi-turn / system prompts supported).

```json
{"messages": [{"role": "user", "content": "Solve: integral of x^2 dx"}], "label": "x^3/3 + C", "metadata": {"teacher": "math"}}
```

## Field-by-field

| Field | Controlled by | Required? | Notes |
|---|---|---|---|
| prompt text | `--input-key` (default `input`) | yes | string **or** a `messages` list — see above |
| `label` | `--label-key` | only for a task reward | omit / `null` for pure distillation (OPD signal comes from the teacher KL, not the label) |
| `metadata` | `--metadata-key` (default `metadata`) | yes for routing | a JSON object; MOPD reads `metadata[<routing-key>]` |
| `metadata.teacher` | `--opd-routing-key` (default `teacher`) | yes for routing | **value** must exactly match a **name** in `--opd-teacher-urls` (case-sensitive) |

> The single-teacher path (`--rm-url`) ignores `metadata.teacher` — one server scores
> everything, so tagging is optional there. It is **required** for `--opd-teacher-urls`.

## The one rule that bites people

The `metadata.teacher` **value** is matched against the **left-hand names** of
`--opd-teacher-urls`. If your launch map is:

```
--opd-teacher-urls "math=http://h1:8001/generate,code=http://h2:8002/generate"
```

then every row's `metadata.teacher` must be `"math"` or `"code"`. Any other value (or a
missing tag) raises `ValueError` at rollout time naming the offending tag. Grep your data
for the tag vocabulary before launching:

```bash
jq -r '.metadata.teacher' prompts_tagged.jsonl | sort -u
# math
# code
```

## Adding tags to an existing dataset

If you already have an untagged dataset (e.g. a downloaded parquet with no `metadata`
column), attach tags programmatically. The integration test
`tests/test_qwen2.5_0.5B_mopd_multi_teacher_sglang.py::_build_tagged_dataset` is a worked
example: it reads a gsm8k parquet and writes a jsonl where each row gains
`metadata.teacher`. The core is just:

```python
row = dict(row)
metadata = dict(row.get("metadata") or {})
metadata["teacher"] = pick_teacher(row)   # by source file, a classifier, round-robin, ...
row["metadata"] = metadata
```

## Domain balance = teacher load

The fraction of a batch tagged `"math"` is exactly the query load hitting the `math`
teacher server. If one domain dominates, that teacher becomes the rollout bottleneck —
rebalance the data or give the hot teacher more SGLang replicas.

See [`../MOPD_GETTING_STARTED.md`](../MOPD_GETTING_STARTED.md) for the end-to-end flow.
