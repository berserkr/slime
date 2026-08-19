"""Unit tests for ``tau_bench_opd.opd_teacher_hook``.

These lock in the glue contract between an SGLang teacher's ``return_logprob``
response and the ``teacher_log_probs`` tensor the OPD KL consumes
(``slime/backends/megatron_utils/loss.py::apply_opd_kl_to_advantages``). A change to
the sglang response shape, or to the trimming math, fails here instead of silently
producing a misaligned KL at training time.

No mocking of aiohttp internals: each test stands up a *real* local aiohttp server on
an ephemeral port that returns canned logprob JSON, and lets the hook make a genuine
HTTP round-trip against it. Real ``Sample`` and real ``_resolve_teacher_url`` are used.
Needs only slime + aiohttp + torch (+ pytest if you run it under pytest); it does NOT
import the tau-bench example.

Run:
    cd <slime root>
    PYTHONPATH=examples/on_policy_distillation \
      python -m pytest examples/on_policy_distillation/tests/test_opd_teacher_hook.py -v

or standalone (no pytest needed):
    PYTHONPATH=examples/on_policy_distillation \
      python examples/on_policy_distillation/tests/test_opd_teacher_hook.py
"""

import asyncio
from types import SimpleNamespace

import torch
from aiohttp import web

from slime.utils.types import Sample

import tau_bench_opd


# A canned sglang /generate (return_logprob) reply. Each input_token_logprobs entry is
# [logprob, token_id, ...]; the FIRST token has no predecessor so its logprob is null
# and the hook must drop it via [1:]. Five input tokens -> five entries.
CANNED_LOGPROBS = [
    [None, 10],
    [-0.1, 11],
    [-0.2, 12],
    [-0.3, 13],
    [-0.4, 14],
]
CANNED_RESPONSE = {"meta_info": {"input_token_logprobs": CANNED_LOGPROBS}}

TOKENS = [10, 11, 12, 13, 14]
RESPONSE_LENGTH = 3
# [1:] drops [None,10]; item[0] of the rest -> [-0.1,-0.2,-0.3,-0.4]; [-3:] -> last 3.
EXPECTED = [-0.2, -0.3, -0.4]


class _TeacherServer:
    """A minimal real aiohttp server that records requests and replies with canned JSON.

    Register one or more named routes; ``hits`` records which route names were called
    (in order) and ``requests`` records the JSON bodies received.
    """

    def __init__(self):
        self.app = web.Application()
        self.hits = []
        self.requests = []
        self._runner = None

    def add_route(self, name, path):
        async def handler(request):
            self.hits.append(name)
            self.requests.append(await request.json())
            return web.json_response(CANNED_RESPONSE)

        self.app.router.add_post(path, handler)
        return self

    async def start(self):
        self._runner = web.AppRunner(self.app)
        await self._runner.setup()
        site = web.TCPSite(self._runner, "127.0.0.1", 0)
        await site.start()
        host, port = self._runner.addresses[0][:2]
        self.base_url = f"http://{host}:{port}"
        return self.base_url

    async def stop(self):
        if self._runner is not None:
            await self._runner.cleanup()


def _make_sample(metadata=None):
    return Sample(tokens=list(TOKENS), response_length=RESPONSE_LENGTH, metadata=metadata or {})


# --------------------------------------------------------------------------------------
# 1) Trimming/parse contract: JSON in -> correctly shaped & valued tensor out.
# --------------------------------------------------------------------------------------
def test_hook_parses_and_trims_logprobs():
    async def _run():
        server = _TeacherServer().add_route("single", "/generate")
        base = await server.start()
        try:
            args = SimpleNamespace(
                rm_url=f"{base}/generate", opd_teacher_url_map=None, opd_routing_key="teacher"
            )
            sample = _make_sample()
            out = await tau_bench_opd.opd_teacher_hook(args, sample, evaluation=False)

            # tensor contract
            assert out.teacher_log_probs is not None
            assert out.teacher_log_probs.dtype == torch.float32
            assert list(out.teacher_log_probs.shape) == [RESPONSE_LENGTH]
            assert torch.allclose(
                out.teacher_log_probs, torch.tensor(EXPECTED, dtype=torch.float32), atol=1e-6
            ), out.teacher_log_probs

            # the teacher was asked to SCORE (not generate) the student's exact tokens
            assert len(server.requests) == 1
            req = server.requests[0]
            assert req["input_ids"] == TOKENS
            assert req["return_logprob"] is True
            assert req["sampling_params"]["max_new_tokens"] == 0
        finally:
            await server.stop()

    asyncio.run(_run())


# --------------------------------------------------------------------------------------
# 2) Eval short-circuits: no tensor set, no network call.
# --------------------------------------------------------------------------------------
def test_hook_skips_on_eval():
    async def _run():
        server = _TeacherServer().add_route("single", "/generate")
        base = await server.start()
        try:
            args = SimpleNamespace(
                rm_url=f"{base}/generate", opd_teacher_url_map=None, opd_routing_key="teacher"
            )
            sample = _make_sample()
            out = await tau_bench_opd.opd_teacher_hook(args, sample, evaluation=True)

            assert out is sample
            assert out.teacher_log_probs is None
            assert server.hits == []  # teacher never contacted during eval
        finally:
            await server.stop()

    asyncio.run(_run())


# --------------------------------------------------------------------------------------
# 3) Multi-teacher routing: the sample's metadata.teacher picks the endpoint.
# --------------------------------------------------------------------------------------
def test_hook_routes_by_metadata():
    async def _run():
        server = _TeacherServer()
        server.add_route("retail", "/retail/generate").add_route("general", "/general/generate")
        base = await server.start()
        try:
            args = SimpleNamespace(
                rm_url=None,
                opd_teacher_url_map={
                    "retail": f"{base}/retail/generate",
                    "general": f"{base}/general/generate",
                },
                opd_routing_key="teacher",
            )
            sample = _make_sample(metadata={"teacher": "retail"})
            await tau_bench_opd.opd_teacher_hook(args, sample, evaluation=False)

            assert server.hits == ["retail"]  # routed to the retail teacher, not general
        finally:
            await server.stop()

    asyncio.run(_run())


# --------------------------------------------------------------------------------------
# 4) Routing guards fire (no network needed): bad tag and missing tag both raise.
# --------------------------------------------------------------------------------------
def test_routing_rejects_unknown_and_missing_tag():
    async def _run():
        args = SimpleNamespace(
            rm_url=None,
            opd_teacher_url_map={"retail": "http://127.0.0.1:1/retail/generate"},
            opd_routing_key="teacher",
        )
        # tag not in the map
        try:
            await tau_bench_opd.opd_teacher_hook(args, _make_sample(metadata={"teacher": "bogus"}), evaluation=False)
            raise AssertionError("expected ValueError for unknown teacher tag")
        except ValueError:
            pass
        # tag missing entirely
        try:
            await tau_bench_opd.opd_teacher_hook(args, _make_sample(metadata={}), evaluation=False)
            raise AssertionError("expected ValueError for missing teacher tag")
        except ValueError:
            pass

    asyncio.run(_run())


if __name__ == "__main__":
    # Standalone runner (no pytest): run every test_* function and report.
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failures = 0
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
        except Exception as e:  # noqa: BLE001 - surface any failure
            failures += 1
            print(f"FAIL {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    raise SystemExit(1 if failures else 0)
