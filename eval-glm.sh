#!/usr/bin/env bash
set -euo pipefail

# Select exported task datasets; this does not run checkpoint generation.
# Usage: bash eval-glm.sh [--ddb-mode bundled|long|lite] [harbor run options]
DDB_MODE=${DDB_MODE:-bundled}
if [[ ${1:-} == --ddb-mode ]]; then
    DDB_MODE=${2:-}
    if [[ $# -lt 2 ]]; then
        echo "--ddb-mode requires bundled, long, or lite" >&2
        exit 2
    fi
    shift 2
fi
case "$DDB_MODE" in
    bundled) DDB_TASK_PATH=datasets/ddb-benchmark ;;
    long|lite) DDB_TASK_PATH="datasets/ddb-benchmark-${DDB_MODE}" ;;
    *) echo "Invalid DDB mode '$DDB_MODE'; choose bundled, long, or lite" >&2; exit 2 ;;
esac

source venv/bin/activate
cd -- harbor

if [[ $DDB_MODE != bundled && ! -d $DDB_TASK_PATH ]]; then
    echo "Missing $DDB_TASK_PATH. Import the $DDB_MODE artifacts and generate Harbor tasks there first; see adapters/ddb-benchmark/README.md." >&2
    exit 2
fi

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
    --path "$DDB_TASK_PATH" \
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
