"""Score every essay with every detector; cache one parquet per detector."""

import argparse
import time
from pathlib import Path

import pandas as pd

from flagged import detectors
from flagged.data import OUT as TEXTS

CACHE = Path(__file__).resolve().parents[1] / "results" / "scores"


def run(name, texts):
    if name in detectors.SUPERVISED:
        return {name: detectors.supervised(name).score(texts)}
    if name == "fastdetect_neo":
        return detectors.zero_shot_neo(texts)
    return detectors.zero_shot_qwen(texts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--limit", type=int, default=None, help="smoke test on a small balanced sample")
    ap.add_argument("--texts", type=Path, default=TEXTS, help="parquet with id, text (and is_ai)")
    ap.add_argument("--cache", type=Path, default=CACHE)
    args = ap.parse_args()

    df = pd.read_parquet(args.texts)
    if args.limit:
        df = df.groupby("is_ai").sample(args.limit // 2, random_state=0)
    cache = args.cache if not args.limit else args.cache / "smoke"
    cache.mkdir(parents=True, exist_ok=True)

    names = args.only or detectors.ALL

    def missing(name):
        """Rows of df this detector has not scored yet (scores are cached and appended)."""
        f = cache / f"{name}.parquet"
        return df if not f.exists() else df[~df.id.isin(pd.read_parquet(f).id)]

    # The three Qwen detectors come from one pass; run it once.
    jobs = []
    for n in names:
        job = "qwen" if n in ("logppl", "binoculars", "fastdetect_qwen") else n
        if job not in jobs and len(missing(n)):
            jobs.append(job)
    for job in jobs:
        members = ["logppl", "binoculars", "fastdetect_qwen"] if job == "qwen" else [job]
        todo = pd.concat([missing(m) for m in members]).drop_duplicates("id")
        t0 = time.time()
        scores = run(job, todo.text.tolist())
        for name, s in scores.items():
            f = cache / f"{name}.parquet"
            new = pd.DataFrame({"id": todo.id.values, "score": s})
            old = pd.read_parquet(f) if f.exists() else new.iloc[:0]
            pd.concat([old[~old.id.isin(new.id)], new]).to_parquet(f, index=False)
        print(f"{job}: {len(todo)} texts in {time.time() - t0:.0f}s", flush=True)
        if args.limit:
            from sklearn.metrics import roc_auc_score
            for name, s in scores.items():
                print(f"  {name} AUROC(AI vs human) = {roc_auc_score(todo.is_ai, s):.3f}")

if __name__ == "__main__":
    main()
