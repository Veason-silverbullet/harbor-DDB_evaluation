#!/usr/bin/env bash
set -euo pipefail
mkdir -p /logs/verifier
# Infrastructure failures deliberately leave no reward, so Harbor records an error.
# evaluate.py writes the upstream score to /logs/verifier/reward.txt.
exec /opt/ddb-venv/bin/python /tests/evaluate.py
