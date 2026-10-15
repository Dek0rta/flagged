"""Main audit: how often does each detector flag human essays, by writer group?

Writes results/*.csv. Thresholds:
  default  a supervised classifier's own cut (P(AI) > 0.5)
  fpr1     cut that flags 1% of held-out-half native PERSUADE essays (calibration half)
  fpr5     same at 5%
"""

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from flagged import detectors
from flagged.data import OUT as TEXTS
from flagged.metrics import (DEFAULT_CUT, auroc, bootstrap_ratio, cut_at_fpr, flag_rate,
                             stratified_auroc, two_prop_p)

ROOT = Path(__file__).resolve().parents[1]
SCORES = ROOT / "results" / "scores"
RES = ROOT / "results"

GROUPS = ["native", "ell", "ellipse", "toefl", "college"]
YEAR = {"openai_2019": 2019, "hc3_2023": 2023, "radar_2023": 2023, "superannotate_2024": 2024,
        "fakespot_2025": 2025, "desklib_2025": 2025, "logppl": 2023, "binoculars": 2024,
        "fastdetect_qwen": 2024, "fastdetect_neo": 2024}


def load(scores_dir=SCORES, texts=TEXTS):
    df = pd.read_parquet(texts)
    for name in detectors.ALL:
        f = scores_dir / f"{name}.parquet"
        if f.exists():
            df = df.merge(pd.read_parquet(f).rename(columns={"score": name}), on="id", how="left")
    return df


def available(df):
    return [n for n in detectors.ALL if n in df]


def cuts_for(df, name):
    calib = df.loc[(df.group == "native") & (df.split == "calib"), name].values
    cuts = {"fpr1": cut_at_fpr(calib, 0.01), "fpr5": cut_at_fpr(calib, 0.05)}
    if name in detectors.SUPERVISED:
        cuts["default"] = DEFAULT_CUT
    return cuts


def held_out(df):
    """Every row except the native calibration half."""
    return df[~((df.group == "native") & (df.split == "calib"))]


def main_table(df):
    rows = []
    ev = held_out(df)
    native = ev[ev.group == "native"]
    ai = ev[ev.is_ai]
    for name in available(df):
        base = {"detector": name, "year": YEAR[name],
                "kind": "supervised" if name in detectors.SUPERVISED else "zero-shot",
                "auroc_native_vs_ai": auroc(native[name], ai[name]),
                "auroc_ell_vs_ai": auroc(ev.loc[ev.group == "ell", name], ai[name]),
                # threshold-free bias: P(ELL essay looks more "AI" than a native essay on the same prompt)
                "auroc_native_vs_ell": stratified_auroc(native[name].values, ev.loc[ev.group == "ell", name].values,
                                                        native.prompt.values, ev.loc[ev.group == "ell", "prompt"].values)}
        for policy, cut in cuts_for(df, name).items():
            row = dict(base, policy=policy, cut=cut, tpr_ai=flag_rate(ai[name], cut)["rate"])
            for g in GROUPS:
                fr = flag_rate(ev.loc[ev.group == g, name], cut)
                row.update({f"fpr_{g}": fr["rate"], f"fpr_{g}_lo": fr["lo"], f"fpr_{g}_hi": fr["hi"],
                            f"n_{g}": fr["n"], f"k_{g}": fr["flagged"]})
            nf = native[name].values > cut
            ef = ev.loc[ev.group == "ell", name].values > cut
            row["ell_native_ratio"] = ef.mean() / nf.mean() if nf.mean() > 0 else np.inf
            row["ratio_lo"], row["ratio_hi"] = bootstrap_ratio(nf, ef)
            row["p_ell_vs_native"] = two_prop_p(ef.sum(), len(ef), nf.sum(), len(nf))
            rows.append(row)
    return pd.DataFrame(rows)


def per_ai_model(df):
    ev = held_out(df)
    rows = []
    for name in available(df):
        for policy, cut in cuts_for(df, name).items():
            for m, g in ev[ev.is_ai].groupby("ai_model"):
                rows.append({"detector": name, "policy": policy, "ai_model": m, "tpr": (g[name] > cut).mean(), "n": len(g)})
    return pd.DataFrame(rows)


def regression(df):
    """Logit(flagged) on ELL status, holding prompt, grade and length fixed; then also holistic score."""
    ev = held_out(df)
    p = ev[ev.group.isin(["native", "ell"])].copy()
    p["ell"] = (p.group == "ell").astype(int)
    p["log_words"] = np.log(p.n_words)
    p = p[p.grade.notna() & p.holistic.notna()]
    rows = []
    for name in available(df):
        cut = cuts_for(df, name)["fpr5"]
        p["flag"] = (p[name] > cut).astype(int)
        for label, formula in [("prompt+grade+length", "flag ~ ell + log_words + C(prompt) + grade"),
                               ("+ holistic score", "flag ~ ell + log_words + C(prompt) + grade + holistic")]:
            try:
                fit = smf.logit(formula, data=p).fit(disp=0, method="newton", maxiter=100)
                lo, hi = fit.conf_int().loc["ell"]
                rows.append({"detector": name, "controls": label, "odds_ratio_ell": np.exp(fit.params["ell"]),
                             "or_lo": np.exp(lo), "or_hi": np.exp(hi), "p": fit.pvalues["ell"], "n": int(fit.nobs)})
            except Exception as e:  # perfect separation etc.
                rows.append({"detector": name, "controls": label, "error": str(e)[:80]})
    return pd.DataFrame(rows)


def ellipse_dose(df):
    """Flag rate by ELLIPSE proficiency ratings (1 = beginner ... 5 = advanced), per rated skill."""
    e = df[df.group == "ellipse"]
    bins, labels = [0, 2.0, 2.5, 3.0, 3.5, 5.0], ["≤2.0", "2.5", "3.0", "3.5", "≥4.0"]
    rows = []
    for name in available(df):
        cut = cuts_for(df, name)["fpr5"]
        for skill in ["overall", "vocabulary", "grammar", "conventions", "syntax"]:
            level = pd.cut(e["ell_" + skill], bins, labels=labels)
            for lvl, g in e.groupby(level, observed=True):
                rows.append({"detector": name, "skill": skill, "level": lvl, **flag_rate(g[name], cut)})
            rows.append({"detector": name, "skill": skill, "level": "spearman",
                         "rate": e[[name, "ell_" + skill]].corr("spearman").iloc[0, 1]})
    return pd.DataFrame(rows)


def native_by_quality(df):
    """Does simple writing get flagged even from native writers? Flag rate by holistic score (1-6)."""
    ev = held_out(df)
    p = ev[ev.group.isin(["native", "ell"])]
    rows = []
    for name in available(df):
        cut = cuts_for(df, name)["fpr5"]
        for (grp, h), g in p.groupby(["group", "holistic"]):
            rows.append({"detector": name, "group": grp, "holistic": h, **flag_rate(g[name], cut)})
    return pd.DataFrame(rows)


def main():
    df = load()
    print("detectors:", available(df))
    out = {"main": main_table(df), "per_ai_model": per_ai_model(df), "regression": regression(df),
           "ellipse_dose": ellipse_dose(df), "by_quality": native_by_quality(df)}
    for k, v in out.items():
        v.to_csv(RES / f"{k}.csv", index=False)
    m = out["main"]
    cols = ["detector", "policy", "auroc_native_vs_ai", "auroc_native_vs_ell", "tpr_ai",
            "fpr_native", "fpr_ell", "ell_native_ratio", "fpr_ellipse", "fpr_toefl", "fpr_college"]
    with pd.option_context("display.width", 250, "display.max_columns", 30, "display.precision", 3):
        print(m[cols].to_string(index=False))
        print(out["regression"].to_string(index=False))


if __name__ == "__main__":
    main()
