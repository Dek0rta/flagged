"""Collect the numbers quoted in the paper text into results/summary.json."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from analyze import YEAR, load

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"


def r(x, d=4):
    return None if pd.isna(x) else round(float(x), d)


def main():
    df = load()
    m = pd.read_csv(RES / "main.csv")
    reg = pd.read_csv(RES / "regression.csv")
    q = pd.read_csv(RES / "by_quality.csv")
    s = {
        "n": {g: int((df.group == g).sum()) for g in ["native", "ell", "ellipse", "toefl", "college", "ai"]},
        "n_native_calib": int(((df.group == "native") & (df.split == "calib")).sum()),
        "n_ai_models": int(df.ai_model.nunique()),
        "n_prompts": int(df.loc[df.source == "persuade", "prompt"].nunique()),
        "median_words": {g: int(df.loc[df.group == g, "n_words"].median()) for g in
                         ["native", "ell", "ellipse", "toefl", "college", "ai"]},
        "detectors": [d for d in YEAR if d in set(m.detector)],
    }
    dflt = m[m.policy == "default"]
    s["toefl_default_mean"] = r(dflt.fpr_toefl.mean())
    s["toefl_default_range"] = [r(dflt.fpr_toefl.min()), r(dflt.fpr_toefl.max())]
    s["college_default_range"] = [r(dflt.fpr_college.min()), r(dflt.fpr_college.max())]
    f5 = m[m.policy == "fpr5"].set_index("detector")
    s["fpr5"] = {d: {k: r(f5.loc[d, k]) for k in ["fpr_native", "fpr_ell", "fpr_ellipse", "fpr_toefl", "fpr_college",
                                                  "ell_native_ratio", "tpr_ai", "auroc_native_vs_ai",
                                                  "auroc_native_vs_ell"]} for d in f5.index}
    s["n_ell_lower"] = int((f5.ell_native_ratio < 1).sum())
    s["auroc_ell_below_half"] = int((f5.auroc_native_vs_ell < 0.5).sum())
    rg = reg[reg.controls == "prompt+grade+length"].set_index("detector")
    s["or_ell"] = {d: [r(rg.loc[d, "odds_ratio_ell"], 3), r(rg.loc[d, "or_lo"], 3), r(rg.loc[d, "or_hi"], 3),
                       r(rg.loc[d, "p"], 6)] for d in rg.index}
    s["or_ell_sig_lower"] = int(((rg.or_hi < 1)).sum())
    s["or_ell_sig_higher"] = int(((rg.or_lo > 1)).sum())
    nat = q[q.group == "native"].pivot(index="detector", columns="holistic", values="rate")
    s["native_by_score"] = {d: {str(int(k)): r(v) for k, v in row.items()} for d, row in nat.iterrows()}
    ed = pd.read_csv(RES / "ellipse_dose.csv")
    rho = ed[(ed.level == "spearman") & (ed.skill == "overall")].rate
    s["ellipse_rho_min"], s["ellipse_rho_max"] = r(rho.min(), 2), r(rho.max(), 2)
    rho_by = ed[(ed.level == "spearman") & (ed.skill == "overall")].set_index("detector").rate
    s["ellipse_rho"] = {d: r(v, 2) for d, v in rho_by.items()}
    s["n_rho_pos"] = int((rho_by > 0).sum())
    pos = rho_by[rho_by > 0]
    s["rho_pos_min"], s["rho_pos_max"] = r(pos.min(), 2), r(pos.max(), 2)
    # Edits that raise flags for every detector: count detectors where the after-rate exceeds the before-rate.
    if rw_exists := (RES / "rewrites.csv").exists():
        w5 = pd.read_csv(RES / "rewrites.csv").query("policy == 'fpr5'")
        for mode in ("grammar", "native", "spell"):
            for grp in ("ell", "native"):
                x = w5[(w5["mode"] == mode) & (w5.group == grp)]
                s[f"n_up_{mode}_{grp}"] = int((x.rate_after > x.rate_before + 0.02).sum())
                s[f"ratio_{mode}_{grp}"] = [r((x.rate_after / x.rate_before.clip(lower=0.005)).min(), 1),
                                            r((x.rate_after / x.rate_before.clip(lower=0.005)).max(), 1)]
    rw = RES / "rewrites.csv"
    if rw.exists():
        w = pd.read_csv(rw).query("policy == 'fpr5'")
        s["edits"] = {f"{a}|{b}|{c}": {"before": r(x.rate_before), "after": r(x.rate_after),
                                       "newly": r(x.newly_flagged), "sim": r(x.similarity, 3), "n": int(x.n)}
                      for (a, b, c), x in w.set_index(["mode", "group", "detector"]).iterrows()}
    (RES / "summary.json").write_text(json.dumps(s, indent=1, ensure_ascii=False))
    print(json.dumps({k: v for k, v in s.items() if k in ("n", "toefl_default_mean", "toefl_default_range",
                                                         "college_default_range", "n_ell_lower",
                                                         "or_ell_sig_lower", "or_ell_sig_higher")}, indent=1))


if __name__ == "__main__":
    main()
