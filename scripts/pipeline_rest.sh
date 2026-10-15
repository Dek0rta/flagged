#!/usr/bin/env bash
# Runs after the Qwen zero-shot pass: LLM rewrites first, then the slow detectors, then scoring of rewrites.
set -uo pipefail
cd "$(dirname "$0")/.."
export HF_HUB_DISABLE_XET=1 PYTHONUNBUFFERED=1
uv run python scripts/llm_rewrite.py grammar --n 200
uv run python scripts/llm_rewrite.py native --n 200
uv run python analysis/rewrites.py build
FAST="openai_2019 hc3_2023 radar_2023 superannotate_2024 fakespot_2025 logppl binoculars fastdetect_qwen"
uv run python scripts/run_detectors.py --texts data/processed/rewrites.parquet --cache results/scores_rewrites --only $FAST
uv run python scripts/run_detectors.py --only fastdetect_neo desklib_2025
uv run python scripts/run_detectors.py --texts data/processed/rewrites.parquet --cache results/scores_rewrites --only fastdetect_neo desklib_2025
echo PIPELINE DONE
