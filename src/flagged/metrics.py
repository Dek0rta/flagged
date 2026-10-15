"""Thresholds, flag rates and confidence intervals shared by the analysis scripts."""

import numpy as np
from scipy.stats import norm
from sklearn.metrics import roc_auc_score

DEFAULT_CUT = 0.5  # what a supervised classifier's own "AI" label means


def wilson(k, n, z=1.96):
    """Wilson 95% interval for a proportion k/n."""
    if n == 0:
        return np.nan, np.nan
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return centre - half, centre + half


def cut_at_fpr(human_scores, fpr):
    """Smallest threshold that flags at most `fpr` of the given human texts."""
    return float(np.quantile(human_scores, 1 - fpr, method="higher"))


def flag_rate(scores, cut):
    flagged = int((np.asarray(scores) > cut).sum())
    lo, hi = wilson(flagged, len(scores))
    return {"n": len(scores), "flagged": flagged, "rate": flagged / max(len(scores), 1), "lo": lo, "hi": hi}


def auroc(neg, pos):
    """P(random pos scores above random neg); 0.5 = no difference."""
    y = np.r_[np.zeros(len(neg)), np.ones(len(pos))]
    return roc_auc_score(y, np.r_[neg, pos])


def stratified_auroc(neg, pos, neg_strata, pos_strata):
    """AUROC computed within each stratum (e.g. prompt), averaged with weights n_neg*n_pos."""
    num = den = 0.0
    for s in np.intersect1d(np.unique(neg_strata), np.unique(pos_strata)):
        a, b = neg[neg_strata == s], pos[pos_strata == s]
        w = len(a) * len(b)
        num += w * auroc(a, b)
        den += w
    return num / den


def bootstrap_ratio(a_flags, b_flags, reps=2000, seed=0):
    """Bootstrap 95% CI for rate(b) / rate(a) from two boolean arrays."""
    rng = np.random.default_rng(seed)
    a, b = np.asarray(a_flags, float), np.asarray(b_flags, float)
    ra = rng.choice(a, (reps, len(a))).mean(1)
    rb = rng.choice(b, (reps, len(b))).mean(1)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = rb / ra
    r = r[np.isfinite(r)]
    return (np.quantile(r, 0.025), np.quantile(r, 0.975)) if len(r) else (np.nan, np.nan)


def two_prop_p(k1, n1, k2, n2):
    p = (k1 + k2) / (n1 + n2)
    se = np.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    return 2 * norm.sf(abs(k1 / n1 - k2 / n2) / se) if se > 0 else 1.0
