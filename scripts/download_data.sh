#!/usr/bin/env bash
# Fetch the public corpora into data/raw/ (none of them are redistributed in this repo).
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data/raw/{liang,persuade,ellipse,daigt}

# Liang et al. 2023 (MIT): TOEFL + college-essay sets
curl -sL -o data/raw/liang/all_data.json \
  https://huggingface.co/datasets/WxWx/ChatGPT-Detector-Bias/resolve/main/all_data.json

# ELLIPSE (CC BY-NC-SA 4.0), Crossley et al. 2023
curl -sL -o data/raw/ellipse/ellipse_train.csv \
  https://raw.githubusercontent.com/scrosseye/ELLIPSE-Corpus/main/ELLIPSE_Final_github_train.csv

# PERSUADE 2.0 (CC BY-NC-SA 4.0), Crossley et al. 2024; test zip password is public in their README
uvx gdown -q 13phHyDzIsb0MHyJr6q-B-qIa9P2tM135 -O data/raw/persuade/
uvx gdown -q 1K1SIJiG-2zWgMlTzxQeYOcLwOsFaVel1 -O data/raw/persuade/
uv run --with pyzipper python - <<'PY'
import pyzipper
with pyzipper.AESZipFile("data/raw/persuade/persuade_corpus_2.0_test.zip") as z:
    z.setpassword(b"persuade_test"); z.extractall("data/raw/persuade")
PY

# DAIGT v2 (LLM-written essays on the PERSUADE prompts, Kaggle 2023), HF mirror
curl -sL -o data/raw/daigt/daigt_v2.parquet \
  https://huggingface.co/datasets/Yunij/kaggle-comp-daigt/resolve/main/data/train-00000-of-00001.parquet
echo done
