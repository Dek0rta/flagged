"""Build one table of human and AI essays for the detector audit.

Groups:
  native   PERSUADE 2.0 essays by US students not flagged as English learners
  ell      PERSUADE 2.0 essays by US students flagged as English learners (ELL)
  ellipse  ELLIPSE essays by ELL students, with 1-5 proficiency scores
  toefl    Liang et al. (2023) real TOEFL essays by non-native writers
  college  Liang et al. (2023) real US college-admission essays
  ai       DAIGT v2 essays written by LLMs on the same 15 PERSUADE prompts
"""

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed" / "texts.parquet"

MIN_WORDS = 100
NATIVE_SAMPLE = 6000
AI_PER_SOURCE = 200
SEED = 0

PERSUADE_COLS = [
    "essay_id_comp", "full_text", "holistic_essay_score", "prompt_name", "task",
    "grade_level", "ell_status", "race_ethnicity", "gender",
    "economically_disadvantaged", "student_disability_status",
]


def clean(text: str) -> str:
    text = str(text).replace("\r\n", "\n").replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    return text.strip()


def key(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()[:300]


def n_words(text: str) -> int:
    return len(text.split())


def load_persuade() -> pd.DataFrame:
    parts = []
    for split in ("train", "test"):
        df = pd.read_csv(RAW / "persuade" / f"persuade_corpus_2.0_{split}.csv",
                         usecols=PERSUADE_COLS, low_memory=False)
        parts.append(df.drop_duplicates("essay_id_comp"))
    df = pd.concat(parts).drop_duplicates("essay_id_comp")
    df = df[df.ell_status.isin(["Yes", "No"])]
    return pd.DataFrame({
        "id": "persuade_" + df.essay_id_comp.astype(str),
        "text": df.full_text.map(clean),
        "group": np.where(df.ell_status == "Yes", "ell", "native"),
        "is_ai": False,
        "source": "persuade",
        "prompt": df.prompt_name,
        "task": df.task,
        "grade": df.grade_level,
        "holistic": df.holistic_essay_score,
        "race": df.race_ethnicity,
        "gender": df.gender,
        "econ_disadv": df.economically_disadvantaged.eq("Economically disadvantaged")
                       .where(df.economically_disadvantaged.notna()),
        "disability": df.student_disability_status.eq("Identified as having disability")
                      .where(df.student_disability_status.notna()),
    })


def load_ellipse() -> pd.DataFrame:
    df = pd.read_csv(RAW / "ellipse" / "ellipse_train.csv")
    out = pd.DataFrame({
        "id": "ellipse_" + df.text_id_kaggle.astype(str),
        "text": df.full_text.map(clean),
        "group": "ellipse",
        "is_ai": False,
        "source": "ellipse",
        "prompt": df.prompt,
        "task": df.task,
        "grade": df.grade,
        "race": df.race_ethnicity,
        "gender": df.gender,
        "econ_disadv": df.SES.eq("Economically disadvantaged").where(df.SES.notna()),
    })
    for col in ["Overall", "Cohesion", "Syntax", "Vocabulary", "Phraseology", "Grammar", "Conventions"]:
        out["ell_" + col.lower()] = df[col].values
    return out


def load_liang() -> pd.DataFrame:
    rows = json.load(open(RAW / "liang" / "all_data.json"))
    names = {"Real TOEFL": "toefl", "Real College Essays": "college"}
    keep = [r for r in rows if r["name"] in names]
    return pd.DataFrame({
        "id": [f"liang_{names[r['name']]}_{i}" for i, r in enumerate(keep)],
        "text": [clean(r["text"]) for r in keep],
        "group": [names[r["name"]] for r in keep],
        "is_ai": False,
        "source": "liang2023",
    })


def load_daigt() -> pd.DataFrame:
    df = pd.read_parquet(RAW / "daigt" / "daigt_v2.parquet", columns=["text", "label", "prompt_name", "source"])
    df = df[(df.label == 1) & (df.source != "train_essays")]
    df = df.sample(frac=1, random_state=SEED).groupby("source").head(AI_PER_SOURCE)
    return pd.DataFrame({
        "id": [f"daigt_{i}" for i in df.index],
        "text": df.text.map(clean).values,
        "group": "ai",
        "is_ai": True,
        "source": "daigt",
        "ai_model": df.source.values,
        "prompt": df.prompt_name.values,
    })


def build() -> pd.DataFrame:
    persuade = load_persuade()
    native = persuade[persuade.group == "native"]
    native = native.sample(min(len(native), NATIVE_SAMPLE), random_state=SEED)
    ell = persuade[persuade.group == "ell"]

    ellipse = load_ellipse()
    # ELLIPSE reuses some PERSUADE essays; keep each text once, in PERSUADE.
    seen = set(persuade.text.map(key))
    ellipse["in_persuade"] = ellipse.text.map(key).isin(seen)

    df = pd.concat([native, ell, ellipse, load_liang(), load_daigt()], ignore_index=True)
    df["n_words"] = df.text.map(n_words)
    # Liang et al. kept all 91 TOEFL essays (median ~100 words); so do we, to replicate them.
    keep = (df.n_words >= MIN_WORDS) | (df.source == "liang2023")
    df = df[keep].drop_duplicates("id").reset_index(drop=True)

    # Half of the native essays set detector thresholds; the other half is held out.
    rng = np.random.default_rng(SEED)
    df["split"] = "eval"
    native_idx = df.index[df.group == "native"]
    df.loc[rng.choice(native_idx, len(native_idx) // 2, replace=False), "split"] = "calib"
    return df


def main() -> None:
    df = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT, index=False)
    print(df.groupby(["group", "split"]).agg(n=("id", "size"), words=("n_words", "median")).to_string())
    print(df[df.is_ai].ai_model.value_counts().to_string())


if __name__ == "__main__":
    main()
