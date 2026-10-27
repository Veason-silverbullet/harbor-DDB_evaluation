#!/usr/bin/env bash
set -euo pipefail

source venv/bin/activate
cd -- harbor

export CLAUDE_BASE_URL=
export CLAUDE_API_KEY=
export CLAUDE_MODEL=claude-opus-5
export DDB_SEARCH_API_KEY=

# CUA uses Anthropic Messages with image inputs and function tools.
export DDB_CUA_BASE_URL=${DDB_CUA_BASE_URL:-$CLAUDE_BASE_URL}
export DDB_CUA_AGENT_PROTOCOL=${DDB_CUA_AGENT_PROTOCOL:-function_calling_anthropic}
export DDB_CUA_API_KEY=${DDB_CUA_API_KEY:-$CLAUDE_API_KEY}
export DDB_CUA_MODEL=${DDB_CUA_MODEL:-$CLAUDE_MODEL}

# The judge uses the gateway's Chat Completions route with the same Claude model.
export DDB_JUDGE_BASE_URL=${DDB_JUDGE_BASE_URL:-$CLAUDE_BASE_URL}
export DDB_JUDGE_PROVIDER=${DDB_JUDGE_PROVIDER:-openai}
export DDB_JUDGE_API_KEY=${DDB_JUDGE_API_KEY:-$CLAUDE_API_KEY}
export DDB_JUDGE_MODEL=${DDB_JUDGE_MODEL:-$CLAUDE_MODEL}
export DDB_JUDGE_MAX_TOKENS=${DDB_JUDGE_MAX_TOKENS:-32000}

# Claude Code appends /v1/messages, so its base URL must be the server root.
CLAUDE_AGENT_BASE_URL=${CLAUDE_BASE_URL%/}
CLAUDE_AGENT_BASE_URL=${CLAUDE_AGENT_BASE_URL%/v1}
exec uv run --active --project adapters/ddb-benchmark harbor run \
    --yes \
    -c adapters/ddb-benchmark/run_ddb-benchmark.yaml \
    --path datasets/ddb-benchmark \
    --n-tasks 8 \
    --n-concurrent 8 \
    -a claude-code \
    -m "anthropic/${CLAUDE_MODEL}" \
    --base-url "$CLAUDE_AGENT_BASE_URL" \
    --api-key-name CLAUDE_API_KEY \
    --api-format anthropic_messages \
    --ak disable_web_search=true \
    --agent-timeout-multiplier 3 \
    "$@"
