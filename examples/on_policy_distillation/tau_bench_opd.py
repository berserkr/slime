"""
tau_bench_opd.py — glue to train slime's AGENTIC tau-bench example WITH on-policy
distillation (OPD), the correct way.

Design (why a sample hook, not a custom reward function)
--------------------------------------------------------
The OPD KL penalty (slime/backends/megatron_utils/loss.py::apply_opd_kl_to_advantages)
only needs one thing per sample: ``sample.teacher_log_probs``. It subtracts
``opd_kl_coef * (student_log_probs - teacher_log_probs)`` from the advantages. The
*task reward → advantage* path is entirely separate and benefits from GRPO's built-in
group normalization.

The base OPD example wires the teacher through ``--custom-rm-path`` (the reward slot),
which is fine when the rollout leaves ``sample.reward = None``. But tau-bench's
``generate`` fills ``sample.reward`` with the env success reward, and slime only runs
the RM step when ``sample.reward is None`` (sglang_rollout.py) — so that path would
NEVER query the teacher, and a custom post-process would then choke on a float. A
custom post-process would also *bypass* GRPO group normalization of the task reward.

So instead we use a **rollout sample hook** (``--rollout-sample-hook-path``): it runs
BEFORE reward computation, scores the student's tokens with the teacher, and stores
``teacher_log_probs`` on the sample — leaving ``sample.reward`` (the env reward) alone.
Result: GRPO normalizes the real task reward as usual, and the OPD KL rides on top.

Two entry points:
  * ``opd_teacher_hook``       — the hook; used in BOTH modes.
  * ``generate_pure_distill``  — Mode A only: zeros the task reward so the KL is the
                                 sole training signal (pure distillation).

Both this module and tau-bench's ``generate_with_tau`` must be importable — the run
script puts the tau-bench example dir and this dir on PYTHONPATH.
"""

import aiohttp
import torch

# Reuse OPD's single/multi-teacher endpoint resolver unchanged.
from slime.rollout.on_policy_distillation import _resolve_teacher_url

# NOTE: tau-bench's agent loop (`generate_with_tau`) is imported *lazily* inside
# `generate_pure_distill` (the only function that uses it). The hook below does not,
# so importing this module to exercise the hook never drags in the tau-bench example.


async def opd_teacher_hook(args, sample, *, rollout_id=None, evaluation=False):
    """Score the student's tokens with the teacher and stash ``teacher_log_probs``.

    Registered via ``--rollout-sample-hook-path``. Runs before reward computation, so
    it does not touch ``sample.reward`` — the env reward flows through slime's normal
    (group-normalized) reward path and OPD subtracts this KL from the advantages.

    Skipped during eval: teacher logprobs are only needed to train, and eval must keep
    the real env reward for its task-success metric.
    """
    if evaluation:
        return sample

    payload = {
        "input_ids": sample.tokens,
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": 0,
            "skip_special_tokens": False,
        },
        "return_logprob": True,
        "logprob_start_len": 0,
    }

    async with aiohttp.ClientSession() as session:
        async with session.post(_resolve_teacher_url(args, sample), json=payload) as resp:
            resp.raise_for_status()
            data = await resp.json()

    # sglang returns per-input-token logprobs; [0] of each entry is the logprob. The
    # first token has none, hence [1:]; then trim to the response span (same as the
    # stock OPD post-process). loss.py aligns these with student_log_probs elementwise
    # and the sample's loss_mask restricts the KL to assistant-generated tokens.
    teacher_log_probs = torch.tensor(
        [item[0] for item in data["meta_info"]["input_token_logprobs"][1:]],
        dtype=torch.float32,
    )
    sample.teacher_log_probs = teacher_log_probs[-sample.response_length :]
    return sample


async def generate_pure_distill(args, sample, sampling_params, evaluation: bool = False):
    """Mode A: run tau-bench's loop but ZERO the training task reward.

    With a constant 0.0 reward the GRPO advantage from the task collapses to 0, so the
    ONLY training signal is the OPD KL (pure distillation on tau-bench trajectories).
    Eval keeps the real env reward so eval metrics still report task success.

    Pair this with ``opd_teacher_hook`` (which supplies teacher_log_probs). Also drop
    the nonzero-std dynamic-sampling filter — every group would otherwise be filtered.
    """
    import generate_with_tau  # lazy: resolved via PYTHONPATH; see run-tau-bench-opd.sh

    sample = await generate_with_tau.generate(args, sample, sampling_params)
    if not evaluation:
        targets = sample if isinstance(sample, list) else [sample]
        for one in targets:
            one.reward = 0.0
    return sample
