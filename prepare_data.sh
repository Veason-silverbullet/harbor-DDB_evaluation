#!/usr/bin/env bash
set -euo pipefail

source venv/bin/activate
cd -- harbor

# Regenerate from the bundled snapshot; set DDB_OVERWRITE=0 to preserve existing tasks.
prepare=(ddb-benchmark --output-dir datasets/ddb-benchmark)
if [[ -n ${DDB_SOURCE_DIR:-} ]]; then prepare+=(--source-dir "$DDB_SOURCE_DIR"); fi
if [[ -n ${DDB_DATA:-} ]]; then prepare+=(--data "$DDB_DATA"); fi
if [[ -n ${DDB_FIXTURES_DIR:-} ]]; then prepare+=(--fixtures-dir "$DDB_FIXTURES_DIR"); fi
if [[ ${DDB_OVERWRITE:-1} == 1 ]]; then prepare+=(--overwrite); fi

exec uv run --active --project adapters/ddb-benchmark "${prepare[@]}" "$@"
