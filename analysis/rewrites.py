"""Experiments 2 and 3: grammar-fixed / word-enhanced human essays, and newer AI essays.

  python analysis/rewrites.py build     -> data/processed/rewrites.parquet (to score with run_detectors.py)
  python analysis/rewrites.py analyze   -> results/rewrites.csv, results/new_ai.csv
"""

import difflib
import re
import sys
from pathlib import Path

import pandas as pd

from flagged.metrics import flag_rate
from analyze import ROOT, available, cuts_for, load

SRC = ROOT / "data" / "processed" / "rewrites"
TABLE = ROOT / "data" / "processed" / "rewrites.parquet"
SCORES = ROOT / "results" / "scores_rewrites"


def word_similarity(a, b):
    """Share of words kept in order (difflib ratio on word lists); 1.0 = untouched."""
    return difflib.SequenceMatcher(None, a.split(), b.split(), autojunk=False).ratio()


def truncated(n=1000, seed=0):
    """Native and ELL essays cut to TOEFL-like length, to separate length from writer group.

    Each essay keeps its first k words, k drawn from the TOEFL length distribution, then is
    cut back to the last full sentence inside those k words (if there is one).
    """
    main = load()
    rng = pd.Series(main.loc[main.group == "toefl", "n_words"].values)
    parts = []
    for grp in ("native", "ell"):
        g = main[(main.group == grp) & (main.split != "calib")].sample(n, random_state=seed)
        ks = rng.sample(n, replace=True, random_state=seed).values
        texts = []
        for t, k in zip(g.text, ks):
            words = t.split()[:k]
            cut = " ".join(words)
            end = max(cut.rfind(". "), cut.rfind("? "), cut.rfind("! "))
            texts.append(cut[:end + 1] if end > len(cut) // 2 else cut)
        parts.append(pd.DataFrame({"id": "truncate__" + g.id.values, "text": texts, "is_ai": False,
                                   "mode": "truncate", "model": None, "group": grp, "prompt": g.prompt.values,
                                   "orig_id": g.id.values, "similarity": None}))
    return pd.concat(parts, ignore_index=True)


def spelled():
    """Non-LLM control: fix misspelled lowercase words with a dictionary spellchecker only.

    Uses the same essays as the LLM grammar experiment. Words with capitals (names, sentence
    starts) and words of one or two letters are left alone, so the edit is conservative.
    """
    from spellchecker import SpellChecker

    grammar = sorted(SRC.glob("grammar__*.parquet"))
    if not grammar:
        return None
    src = pd.read_parquet(grammar[0])[["id", "text", "group"]]
    sp = SpellChecker()
    word = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")

    def fix(text):
        def repl(m):
            w = m.group(0)
            if len(w) < 3 or not w.islower() or w in sp:
                return w
            c = sp.correction(w)
            return c if c else w
        return word.sub(repl, text)

    fixed = [fix(t) for t in src.text]
    return pd.DataFrame({"id": "spell__" + src.id.values, "text": fixed, "is_ai": False, "mode": "spell",
                         "model": "pyspellchecker", "group": src.group.values, "prompt": None,
                         "orig_id": src.id.values,
                         "similarity": [word_similarity(a, b) for a, b in zip(src.text, fixed)]})


def build():
    parts = [truncated(), spelled()]
    for f in sorted(SRC.glob("*.parquet")):
        r = pd.read_parquet(f)
        tag = f.stem
        if r["mode"].iloc[0] == "generate":
            parts.append(pd.DataFrame({"id": tag + "__" + r.id, "text": r.new_text, "is_ai": True,
                                       "mode": "generate", "model": r.model, "group": "ai_new",
                                       "prompt": r.prompt, "orig_id": None, "similarity": None}))
        else:
            parts.append(pd.DataFrame({"id": tag + "__" + r.id, "text": r.new_text, "is_ai": False,
                                       "mode": r["mode"], "model": r.model, "group": r.group,
                                       "prompt": None, "orig_id": r.id,
                                       "similarity": [word_similarity(a, b) for a, b in zip(r.text, r.new_text)]}))
    df = pd.concat([p for p in parts if p is not None], ignore_index=True)
    df["n_words"] = df.text.str.split().str.len()
    df.to_parquet(TABLE, index=False)
    print(df.groupby(["mode", "group"]).agg(n=("id", "size"), sim=("similarity", "median")).to_string())


def analyze():
    main = load()
    rw = pd.read_parquet(TABLE)
    for f in SCORES.glob("*.parquet"):
        rw = rw.merge(pd.read_parquet(f).rename(columns={"score": f.stem}), on="id", how="left")

    rows, new_ai = [], []
    for name in available(main):
        if name not in rw:
            continue
        for policy, cut in cuts_for(main, name).items():
            for (mode, grp), g in rw[~rw.is_ai].groupby(["mode", "group"]):
                orig = main.set_index("id").loc[g.orig_id, name]
                before, after = flag_rate(orig, cut), flag_rate(g[name], cut)
                rows.append({"detector": name, "policy": policy, "mode": mode, "group": grp, "n": len(g),
                             "similarity": g.similarity.median(),
                             "rate_before": before["rate"], "rate_after": after["rate"],
                             "after_lo": after["lo"], "after_hi": after["hi"],
                             # essays flagged after the edit that were not flagged before
                             "newly_flagged": float(((g[name].values > cut) & (orig.values <= cut)).mean())})
            for model, g in rw[rw.is_ai].groupby("model"):
                new_ai.append({"detector": name, "policy": policy, "model": model, **flag_rate(g[name], cut)})
    pd.DataFrame(rows).to_csv(ROOT / "results" / "rewrites.csv", index=False)

    # Flag rate after the LLM grammar fix, by how much of the essay it changed.
    g = rw[(rw["mode"] == "grammar") & rw.group.isin(["ell", "native"])].copy()
    g["changed"] = pd.cut(1 - g.similarity, [0, .05, .10, .20, 1], labels=["<5%", "5-10%", "10-20%", ">20%"],
                          include_lowest=True)
    by_edit = []
    for name in available(main):
        if name in g:
            cut = cuts_for(main, name)["fpr5"]
            for lvl, x in g.groupby("changed", observed=True):
                by_edit.append({"detector": name, "words_changed": lvl, **flag_rate(x[name], cut)})
    pd.DataFrame(by_edit).to_csv(ROOT / "results" / "grammar_by_edit.csv", index=False)
    pd.DataFrame(new_ai).to_csv(ROOT / "results" / "new_ai.csv", index=False)
    with pd.option_context("display.width", 200, "display.precision", 3):
        print(pd.DataFrame(rows).query("policy == 'fpr5'").to_string(index=False))
        if new_ai:
            print(pd.DataFrame(new_ai).query("policy == 'fpr5'")[["detector", "model", "rate"]].to_string(index=False))


if __name__ == "__main__":
    {"build": build, "analyze": analyze}[sys.argv[1]]()
