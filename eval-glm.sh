#!/usr/bin/env bash
set -euo pipefail

source venv/bin/activate
cd -- harbor

export GLM_BASE_URL=
export GLM_API_KEY=
export GLM_MODEL=GLM-5.3-Flash
export DDB_SEARCH_API_KEY=

# CUA uses Chat Completions with image inputs and function tools.
export DDB_CUA_BASE_URL=${DDB_CUA_BASE_URL:-$GLM_BASE_URL}
export DDB_CUA_AGENT_PROTOCOL=${DDB_CUA_AGENT_PROTOCOL:-function_calling}
export DDB_CUA_API_KEY=${DDB_CUA_API_KEY:-$GLM_API_KEY}
export DDB_CUA_MODEL=${DDB_CUA_MODEL:-$GLM_MODEL}

# The judge uses the same GLM Chat Completions endpoint.
export DDB_JUDGE_BASE_URL=${DDB_JUDGE_BASE_URL:-$GLM_BASE_URL}
export DDB_JUDGE_PROVIDER=${DDB_JUDGE_PROVIDER:-openai}
export DDB_JUDGE_API_KEY=${DDB_JUDGE_API_KEY:-$GLM_API_KEY}
export DDB_JUDGE_MODEL=${DDB_JUDGE_MODEL:-$GLM_MODEL}
export DDB_JUDGE_MAX_TOKENS=${DDB_JUDGE_MAX_TOKENS:-32000}

# Claude Code appends /v1/messages, so its base URL must be the server root.
GLM_AGENT_BASE_URL=${GLM_BASE_URL%/}
GLM_AGENT_BASE_URL=${GLM_AGENT_BASE_URL%/v1}
exec uv run --active --project adapters/ddb-benchmark harbor run \
    --yes \
    -c adapters/ddb-benchmark/run_ddb-benchmark.yaml \
    --path datasets/ddb-benchmark \
    --n-tasks 8 \
    --n-concurrent 8 \
    -a claude-code \
    -m "anthropic/${GLM_MODEL}" \
    --base-url "$GLM_AGENT_BASE_URL" \
    --api-key-name GLM_API_KEY \
    --api-format anthropic_messages \
    --ak disable_web_search=true \
    --agent-timeout-multiplier 3 \
    "$@"
