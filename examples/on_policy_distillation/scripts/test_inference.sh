#!/bin/bash
# =============================================================================
# test_inference.sh — fire a couple of requests at a running SGLang server.
#
# Point it at an endpoint stood up by serve_sglang.sh (or serve_teacher.sh) and it
# will: (1) check health, (2) print model info, (3) hit the native /generate route,
# (4) hit the OpenAI-compatible /v1/chat/completions route. If any call fails the
# script exits non-zero, so it doubles as a smoke test in CI.
#
# ---- usage --------------------------------------------------------------------
#   PORT=30000 bash test_inference.sh
#   HOST=127.0.0.1 PORT=13141 PROMPT="What is 2+2?" bash test_inference.sh
#
# ---- knobs (env vars) ---------------------------------------------------------
#   HOST         Server host. Default 127.0.0.1.
#   PORT         Server port. Default 30000.
#   PROMPT       Prompt text to send. Default a short math question.
#   MAX_TOKENS   Max new tokens. Default 64.
# =============================================================================
set -euo pipefail

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-30000}"
PROMPT="${PROMPT:-What is the capital of France? Answer in one word.}"
MAX_TOKENS="${MAX_TOKENS:-64}"
BASE="http://${HOST}:${PORT}"

# Pretty-print JSON if jq is present, otherwise pass through raw.
pp() { if command -v jq > /dev/null 2>&1; then jq .; else cat; fi; }

echo "[test] target ${BASE}"
echo "[test] prompt: ${PROMPT}"
echo

# 1) Health -------------------------------------------------------------------
echo "== 1. health_generate =="
if ! curl -sf "${BASE}/health_generate" > /dev/null; then
  echo "[test] ERROR: ${BASE}/health_generate did not pass. Is the server up?"
  exit 1
fi
echo "ok"
echo

# 2) Model info ---------------------------------------------------------------
echo "== 2. get_model_info =="
curl -sf "${BASE}/get_model_info" | pp
echo

# 3) Native /generate ---------------------------------------------------------
# SGLang's native route: text in, text out, sampling params under "sampling_params".
echo "== 3. /generate (native) =="
curl -sf "${BASE}/generate" \
  -H 'Content-Type: application/json' \
  -d "$(cat <<JSON
{
  "text": "${PROMPT}",
  "sampling_params": {"temperature": 0.0, "max_new_tokens": ${MAX_TOKENS}}
}
JSON
)" | pp
echo

# 4) OpenAI-compatible chat completions ---------------------------------------
# Uses the chat template baked into the model's tokenizer_config.
echo "== 4. /v1/chat/completions (OpenAI-compatible) =="
curl -sf "${BASE}/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d "$(cat <<JSON
{
  "model": "default",
  "messages": [{"role": "user", "content": "${PROMPT}"}],
  "temperature": 0.0,
  "max_tokens": ${MAX_TOKENS}
}
JSON
)" | pp
echo

echo "[test] all requests succeeded."
