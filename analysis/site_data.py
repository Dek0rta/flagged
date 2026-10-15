"""Export the numbers the site shows into site/data.json."""

import json
from pathlib import Path

import pandas as pd

from figures import GROUP, LABEL

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"


def clean(v):
    return None if pd.isna(v) else round(float(v), 4)


SPECIMEN_ID = "persuade_6204A1726C16"   # ELL, grade 12, "Distance learning"; chosen because it flips
SPECIMEN_DETECTORS = ["desklib_2025", "fakespot_2025", "openai_2019"]


def excerpt(text, n_paragraphs=2):
    """Opening paragraphs; a paragraph break inside a sentence (a corpus artifact) is joined."""
    paras, cur = [], ""
    for p in [p.strip() for p in text.split("\n\n") if p.strip()]:
        cur = f"{cur} {p}".strip() if cur and not cur.endswith((".", "!", "?")) else p
        if cur.endswith((".", "!", "?")):
            paras.append(cur)
            cur = ""
        if len(paras) == n_paragraphs:
            break
    return paras


def diff_tokens(a, b):
    """Word-level diff of b against a: list of [op, text] with op in eq/del/ins."""
    import difflib
    wa, wb = a.split(), b.split()
    ops = []
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, wa, wb, autojunk=False).get_opcodes():
        if tag == "equal":
            ops.append(["eq", " ".join(wa[i1:i2])])
        else:
            if i2 > i1:
                ops.append(["del", " ".join(wa[i1:i2])])
            if j2 > j1:
                ops.append(["ins", " ".join(wb[j1:j2])])
    return ops


def specimen(m):
    """A real ELL essay whose verdict flips after an LLM grammar fix but not after a spellcheck."""
    texts = pd.read_parquet(ROOT / "data" / "processed" / "texts.parquet").set_index("id")
    rw = pd.read_parquet(ROOT / "data" / "processed" / "rewrites.parquet")
    row = texts.loc[SPECIMEN_ID]
    versions = {"original": row.text}
    ids = {"original": SPECIMEN_ID}
    for mode in ("spell", "grammar"):
        r = rw[(rw.orig_id == SPECIMEN_ID) & (rw["mode"] == mode)].iloc[0]
        versions[mode], ids[mode] = r.text, r.id
    paras = {k: excerpt(v) for k, v in versions.items()}
    scores = {}
    for d in SPECIMEN_DETECTORS:
        main_s = pd.read_parquet(RES / "scores" / f"{d}.parquet").set_index("id").score
        rw_s = pd.read_parquet(RES / "scores_rewrites" / f"{d}.parquet").set_index("id").score
        cut = float(m[(m.detector == d) & (m.policy == "fpr5")].cut.iloc[0])
        scores[d] = {"label": LABEL[d], "cut": cut,
                     "original": float(main_s[ids["original"]]),
                     "spell": float(rw_s[ids["spell"]]), "grammar": float(rw_s[ids["grammar"]])}
    return {"grade": int(row.grade), "prompt": row.prompt, "words": int(row.n_words),
            "paragraphs": paras,
            "diff": {k: [diff_tokens(a, b) for a, b in zip(paras["original"], paras[k])] for k in ("spell", "grammar")},
            "scores": scores}


def main():
    m = pd.read_csv(RES / "main.csv")
    q = pd.read_csv(RES / "by_quality.csv")
    reg = pd.read_csv(RES / "regression.csv").query("controls == 'prompt+grade+length'").set_index("detector")
    dets = [d for d in LABEL if d in set(m.detector)]
    out = {
        "summary": json.loads((RES / "summary.json").read_text()),
        "groups": {g: {"label": l, "color": c} for g, (l, c) in GROUP.items()},
        "detectors": [],
    }
    for d in dets:
        rows = m[m.detector == d].set_index("policy")
        det = {"id": d, "label": LABEL[d],
               "kind": rows.kind.iloc[0], "year": int(rows.year.iloc[0]),
               "auroc": clean(rows.auroc_native_vs_ai.iloc[0]),
               "odds_ratio_ell": [clean(reg.loc[d, c]) for c in ("odds_ratio_ell", "or_lo", "or_hi")] if d in reg.index else None,
               "policies": {}}
        for policy, r in rows.iterrows():
            det["policies"][policy] = {
                "cut": clean(r.cut), "tpr_ai": clean(r.tpr_ai),
                **{g: [clean(r[f"fpr_{g}"]), clean(r[f"fpr_{g}_lo"]), clean(r[f"fpr_{g}_hi"]), int(r[f"n_{g}"])]
                   for g in GROUP},
            }
        qd = q[(q.detector == d) & (q.n >= 30)]
        det["by_score"] = {g: [[int(x.holistic), clean(x.rate)] for x in qd[qd.group == g].itertuples()]
                           for g in ("native", "ell")}
        out["detectors"].append(det)
    rw = RES / "rewrites.csv"
    if rw.exists():
        w = pd.read_csv(rw)
        out["edits"] = [{k: (clean(v) if isinstance(v, float) else v) for k, v in r.items()}
                        for r in w.to_dict("records")]
    out["specimen"] = specimen(m)
    (ROOT / "site").mkdir(exist_ok=True)
    (ROOT / "site" / "data.json").write_text(json.dumps(out, ensure_ascii=False))
    print("site/data.json:", len(out["detectors"]), "detectors")


if __name__ == "__main__":
    main()
