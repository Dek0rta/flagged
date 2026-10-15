"""Make the LLM-touched texts for the two follow-up experiments, locally with MLX.

  grammar   Fix only spelling/grammar/punctuation -- what a student does with Grammarly.
  native    Liang et al.'s prompt: "Enhance the word choices to sound more like that
            of a native speaker."
  generate  Fresh essays from 2024-25 open models on the 15 PERSUADE prompts, so the
            detectors are also tested on AI text newer than the 2023 DAIGT set.

Usage: python scripts/llm_rewrite.py grammar|native|generate [--model ...] [--n ...]
"""

import argparse
import re
from pathlib import Path

import pandas as pd
from mlx_lm import batch_generate, load
from mlx_lm.sample_utils import make_sampler

from flagged.data import OUT as TEXTS, RAW

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "processed" / "rewrites"
SEED = 0
BATCH = 8

PROMPTS = {
    "grammar": (
        "Correct the spelling, grammar and punctuation mistakes in the essay below. "
        "Do not change word choice, sentence structure, tone or meaning, and do not add "
        "or remove content. Return only the corrected essay, with no comments.\n\n{text}"
    ),
    "native": (
        "Enhance the word choices to sound more like that of a native speaker. "
        "Return only the revised essay, with no comments.\n\n{text}"
    ),
    "generate": (
        "You are a student in grade {grade}. Write an essay for this assignment. "
        "Return only the essay.\n\nAssignment: {assignment}"
    ),
}


def rewrite_sample(n):
    """n ELL + n native PERSUADE essays (held-out half, same prompts) + all TOEFL essays."""
    df = pd.read_parquet(TEXTS)
    ell = df[(df.group == "ell")].sample(n, random_state=SEED)
    native = df[(df.group == "native") & (df.split == "eval")]
    native = native[native.prompt.isin(ell.prompt.unique())].sample(n, random_state=SEED)
    toefl = df[df.group == "toefl"]
    return pd.concat([ell, native, toefl])[["id", "text", "group"]]


def generate_sample(n):
    raw = pd.read_csv(RAW / "persuade" / "persuade_corpus_2.0_test.csv",
                      usecols=["prompt_name", "assignment"], low_memory=False).drop_duplicates("prompt_name")
    rows = []
    for i in range(n):
        r = raw.iloc[i % len(raw)]
        rows.append({"id": f"gen{i}", "prompt": r.prompt_name, "grade": 6 + i % 7,
                     "assignment": " ".join(str(r.assignment).split())})
    return pd.DataFrame(rows)


def strip_preamble(text):
    # Small models sometimes open with "Here is the corrected essay:"; drop that line.
    return re.sub(r"^(here('s| is)[^\n]*:|sure[^\n]*:)\s*", "", text.strip(), flags=re.I).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=PROMPTS)
    ap.add_argument("--model", default="mlx-community/Qwen2.5-7B-Instruct-4bit")
    ap.add_argument("--n", type=int, default=300)
    args = ap.parse_args()

    if args.mode == "generate":
        items = generate_sample(args.n)
        users = [PROMPTS["generate"].format(grade=r.grade, assignment=r.assignment) for r in items.itertuples()]
        sampler = make_sampler(temp=0.8, top_p=0.95)
    else:
        items = rewrite_sample(args.n)
        users = [PROMPTS[args.mode].format(text=t) for t in items.text]
        sampler = make_sampler(temp=0.0)

    model, tok = load(args.model)
    prompts = [tok.apply_chat_template([{"role": "user", "content": u}], add_generation_prompt=True)
               for u in users]
    outs = []
    for start in range(0, len(prompts), BATCH):
        res = batch_generate(model, tok, prompts[start:start + BATCH], max_tokens=1024, sampler=sampler)
        outs += [strip_preamble(t) for t in res.texts]
        print(f"{args.mode}: {len(outs)}/{len(prompts)}", flush=True)

    items = items.assign(new_text=outs, mode=args.mode, model=args.model)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tag = args.model.split("/")[-1]
    items.to_parquet(OUT_DIR / f"{args.mode}__{tag}.parquet", index=False)


if __name__ == "__main__":
    main()
