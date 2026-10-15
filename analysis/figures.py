"""Figures for the paper and site, from results/*.csv."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES, FIG = ROOT / "results", ROOT / "figures"

LABEL = {"openai_2019": "OpenAI RoBERTa (2019)", "hc3_2023": "HC3 RoBERTa (2023)",
         "radar_2023": "RADAR (2023)", "superannotate_2024": "SuperAnnotate (2024)",
         "fakespot_2025": "Fakespot / Mozilla (2025)", "desklib_2025": "Desklib (2025)",
         "logppl": "Perplexity (Qwen2.5-1.5B)", "binoculars": "Binoculars (2024)",
         "fastdetect_qwen": "Fast-DetectGPT, Qwen (2024)", "fastdetect_neo": "Fast-DetectGPT, Neo (2024)"}
# Same palette as the site (site/index.html).
GROUP = {"native": ("US students, native", "#2E5EAA"), "ell": ("US students, English learners", "#0E8A82"),
         "ellipse": ("English learners (ELLIPSE)", "#66B8B1"), "toefl": ("TOEFL essays (Liang)", "#C8102E"),
         "college": ("Published college essays (Liang)", "#7B4CC2")}

plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.edgecolor": "#5B616B", "axes.labelcolor": "#15181D",
                     "xtick.color": "#5B616B", "ytick.color": "#5B616B", "figure.dpi": 200})


def order(df):
    return [d for d in LABEL if d in set(df.detector)]


def fig_groups(policy, name):
    """Share of human essays flagged as AI, per detector and writer group."""
    m = pd.read_csv(RES / "main.csv").query("policy == @policy")
    dets = order(m)
    fig, ax = plt.subplots(figsize=(6.4, 0.42 * len(dets) + 0.9))
    for j, (g, (label, color)) in enumerate(GROUP.items()):
        for i, d in enumerate(dets):
            r = m[m.detector == d].iloc[0]
            y = i + (j - 2) * 0.13
            ax.plot([r[f"fpr_{g}_lo"] * 100, r[f"fpr_{g}_hi"] * 100], [y, y], color=color, lw=1, alpha=0.6)
            ax.plot(r[f"fpr_{g}"] * 100, y, "o", ms=4, color=color, label=label if i == 0 else None)
    ax.set_yticks(range(len(dets)), [LABEL[d] for d in dets])
    ax.invert_yaxis()
    ax.set_xlabel("Human essays flagged as AI (%)")
    ax.set_xlim(0, 100)
    ax.grid(axis="x", lw=0.3, alpha=0.5)
    ax.legend(loc="lower right", fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / name)
    fig.savefig(FIG / name.replace(".pdf", ".png"), dpi=200)
    plt.close(fig)


def fig_quality():
    """Flag rate by holistic essay score (1-6), native vs English learners, 5%-FPR threshold."""
    q = pd.read_csv(RES / "by_quality.csv")
    dets = [d for d in ["superannotate_2024", "fakespot_2025", "desklib_2025", "binoculars"] if d in set(q.detector)]
    fig, axes = plt.subplots(1, len(dets), figsize=(1.7 * len(dets) + 0.4, 2.2), sharey=True)
    for ax, d in zip(np.atleast_1d(axes), dets):
        for g in ["native", "ell"]:
            s = q[(q.detector == d) & (q.group == g) & (q.n >= 30)]
            ax.errorbar(s.holistic, s.rate * 100, yerr=[(s.rate - s.lo) * 100, (s.hi - s.rate) * 100],
                        fmt="o-", ms=3, lw=1, capsize=1.5, color=GROUP[g][1], label=GROUP[g][0])
        ax.set_title(LABEL[d], fontsize=8)
        ax.set_xlabel("Essay score (1–6)")
        ax.set_xticks(range(1, 7))
    np.atleast_1d(axes)[0].set_ylabel("Flagged as AI (%)")
    np.atleast_1d(axes)[0].legend(fontsize=6.5, frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIG / "quality.pdf")
    fig.savefig(FIG / "quality.png", dpi=200)
    plt.close(fig)


def fig_edits():
    """Flag rate before and after truncation / grammar fixing / word enhancement."""
    f = RES / "rewrites.csv"
    if not f.exists():
        return
    r = pd.read_csv(f).query("policy == 'fpr5'")
    modes = [m for m in ["spell", "grammar", "native"] if m in set(r["mode"])]
    titles = {"spell": "Dictionary spellcheck", "grammar": "LLM grammar fix",
              "native": "LLM 'sound native' rewrite"}
    dets = order(r)
    fig, axes = plt.subplots(1, len(modes), figsize=(2.3 * len(modes) + 1.2, 0.3 * len(dets) + 1.2), sharey=True)
    for ax, mode in zip(np.atleast_1d(axes), modes):
        for k, g in enumerate(["native", "ell", "toefl"]):
            s = r[(r["mode"] == mode) & (r.group == g)]
            for i, d in enumerate(dets):
                row = s[s.detector == d]
                if row.empty:
                    continue
                row = row.iloc[0]
                y = i + (k - 1) * 0.22
                ax.annotate("", xy=(row.rate_after * 100, y), xytext=(row.rate_before * 100, y),
                            arrowprops=dict(arrowstyle="->", color=GROUP[g][1], lw=1))
                ax.plot(row.rate_before * 100, y, "o", ms=2.5, color=GROUP[g][1],
                        label=GROUP[g][0] if i == 0 else None)
        ax.set_title(titles[mode], fontsize=8)
        ax.set_xlabel("Flagged as AI (%)")
        ax.set_xlim(0, 100)
        ax.grid(axis="x", lw=0.3, alpha=0.5)
    np.atleast_1d(axes)[0].set_yticks(range(len(dets)), [LABEL[d] for d in dets])
    np.atleast_1d(axes)[0].invert_yaxis()
    handles, labels = np.atleast_1d(axes)[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=7.5, frameon=False)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(FIG / "edits.pdf")
    fig.savefig(FIG / "edits.png", dpi=200)
    plt.close(fig)


def main():
    FIG.mkdir(exist_ok=True)
    fig_groups("default", "groups_default.pdf")
    fig_groups("fpr5", "groups_fpr5.pdf")
    fig_quality()
    fig_edits()
    print("figures:", sorted(p.name for p in FIG.glob("*.pdf")))


if __name__ == "__main__":
    main()
