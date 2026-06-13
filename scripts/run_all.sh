#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python -m experiments.run_benchmark --preset full
python -m experiments.run_v3_evidence
