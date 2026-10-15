#!/usr/bin/env bash
# Last detector: Desklib on all essays, then on edited essays.
set -uo pipefail
cd "$(dirname "$0")/.."
export HF_HUB_DISABLE_XET=1 PYTHONUNBUFFERED=1
uv run python scripts/run_detectors.py --only desklib_2025
uv run python scripts/run_detectors.py --texts data/processed/rewrites.parquet --cache results/scores_rewrites --only desklib_2025
echo PIPELINE DONE
