#!/usr/bin/env python
"""Phase 3: check the TEST, not the answer.

The 2026-09-13 retraction happened because every check asked whether the
result looked plausible and none asked whether the measurement was correct.
A 44.5% top-1 hit rate against a 19.4% base rate looks good, which is
exactly the condition under which nobody looks closer.

Three checks, each able to fail independently:

1. SHUFFLE. Retrain on a permuted label. Every real signal is destroyed, so
   top1 must fall to roughly the rate you get by naming somebody at random.
   If a shuffled model still scores well, the metric is measuring the shape
   of the data rather than the model, and the headline number means nothing.

2. IMPORTANCE. Read the top features. A genuine anytime-TD model should lean
   on inside-10 usage and the team's implied total. Anything that looks like
   a same-game outcome is leakage the allowlist failed to catch.

3. CALIBRATION. Split the predictions into deciles and compare predicted
   against observed. A model can rank well and still be wrong about how
   likely anything is, and the marker is going to print a number, not a rank.
"""
import pathlib
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from dataset import build          # noqa: E402
from evaluate import top1_rate     # noqa: E402

SEASON = 2025


def frames():
    df, feats = build()
    cats = [pd.get_dummies(df["position"], prefix="pos")]
    if "report_status" in df.columns:
        cats.append(pd.get_dummies(df["report_status"].fillna("None"), prefix="inj"))
    X = pd.concat([df[feats]] + cats, axis=1)
    return df, X, list(X.columns)


def fit(X, y, seed=0):
    m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06,
                                       max_leaf_nodes=31, l2_regularization=1.0,
                                       random_state=seed)
    m.fit(X, y)
    return m


def main():
    df, X, names = frames()
    tr, te = (df.season < SEASON).values, (df.season == SEASON).values
    test = df[te].copy()
    ytr, yte = df.loc[tr, "td"], df.loc[te, "td"].to_numpy()

    print("=== 1. SHUFFLE CONTROL ===")
    real = fit(X[tr], ytr)
    test["p"] = real.predict_proba(X[te])[:, 1]
    r_top1, n_games = top1_rate(test, "p")
    print(f"  real model     top1 {r_top1:.4f}   auc {roc_auc_score(yte, test.p):.4f}")
    for seed in (1, 2, 3):
        rng = np.random.default_rng(seed)
        shuf = fit(X[tr], pd.Series(rng.permutation(ytr.to_numpy()), index=ytr.index), seed)
        test["p_s"] = shuf.predict_proba(X[te])[:, 1]
        s_top1, _ = top1_rate(test, "p_s")
        print(f"  shuffled #{seed}    top1 {s_top1:.4f}   auc {roc_auc_score(yte, test.p_s):.4f}")
    print(f"  (a shuffled model should land near the name-anyone floor, over {n_games} games)")

    print("\n=== 2. WHAT THE MODEL LEANS ON ===")
    imp = permutation_importance(real, X[te], yte, n_repeats=5,
                                 random_state=0, scoring="roc_auc", n_jobs=-1)
    order = np.argsort(imp.importances_mean)[::-1][:12]
    for i in order:
        print(f"  {names[i]:<26} {imp.importances_mean[i]:+.4f}")

    print("\n=== 3. CALIBRATION (test season) ===")
    test["bin"] = pd.qcut(test.p, 10, labels=False, duplicates="drop")
    cal = test.groupby("bin").agg(pred=("p", "mean"), actual=("td", "mean"), n=("td", "size"))
    for b, r in cal.iterrows():
        gap = r.actual - r.pred
        flag = "  <-- off by >5pts" if abs(gap) > 0.05 else ""
        print(f"  decile {int(b)+1:>2}  predicted {r.pred:.3f}   actual {r.actual:.3f}   "
              f"n={int(r.n):>4}{flag}")
    print(f"  mean |gap| {np.abs(cal.actual - cal.pred).mean():.4f}")


if __name__ == "__main__":
    main()
