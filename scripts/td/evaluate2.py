#!/usr/bin/env python
"""Walk-forward evaluation of the serve-time feature set.

The earlier evaluation (scripts/td/evaluate.py) scored a model trained on
the EDA parquet, which cannot be reproduced at serving time. This one scores
the model that will actually run: same features, same function, same
universe. Its numbers are the ones that get to be quoted.

Two differences from the first pass, both of which should lower the score,
and both of which are more honest:

The universe is the ROSTER, not the box score. A rostered player who never
touched the ball is a real zero here. He was available and the book priced
him. That drops the base rate from 19.4% to 14.7% and makes top1 harder,
because there are more candidates per game.

The feature set is smaller: twelve columns plus position, against
forty-eight. Everything dropped was something the projection engine cannot
produce for a game that has not been played.
"""
import pathlib
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from nflprops.tdmodel import design_matrix  # noqa: E402

TRAIN = ROOT / "data" / "td_train.parquet"
TEST_SEASONS = [2021, 2022, 2023, 2024, 2025]


def top1_rate(frame, col):
    idx = frame.groupby("game_id")[col].idxmax()
    return frame.loc[idx, "td"].mean(), len(idx)


def fit(X, y, seed=0):
    m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06,
                                       max_leaf_nodes=31, l2_regularization=1.0,
                                       random_state=seed)
    m.fit(X, y)
    return m


def main():
    df = pd.read_parquet(TRAIN).dropna(subset=["game_id"])
    X = design_matrix(df)

    rows = []
    for season in TEST_SEASONS:
        tr, te = (df.season < season).values, (df.season == season).values
        test = df[te].copy()

        test["p_base"] = df.loc[tr, "td"].mean()
        pos_rate = df.loc[tr].groupby("position")["td"].mean()
        test["p_pos"] = test.position.map(pos_rate).fillna(test["p_base"])
        model = fit(X[tr], df.loc[tr, "td"])
        test["p_model"] = model.predict_proba(X[te])[:, 1]

        # Shuffle control, run per season rather than once at the end. The
        # floor is a property of the metric and the slate, not of the model,
        # so it has to be measured on the same games it is compared against.
        rng = np.random.default_rng(season)
        shuf = fit(X[tr], pd.Series(rng.permutation(df.loc[tr, "td"].to_numpy()),
                                    index=df.index[tr]), season)
        test["p_shuf"] = shuf.predict_proba(X[te])[:, 1]

        for name in ("base", "pos", "shuf", "model"):
            p, y = test[f"p_{name}"].to_numpy(), test["td"].to_numpy()
            t1, n_games = top1_rate(test, f"p_{name}")
            rows.append({"season": season, "model": name, "games": n_games,
                         "brier": brier_score_loss(y, p),
                         "logloss": log_loss(y, np.clip(p, 1e-6, 1 - 1e-6)),
                         "auc": roc_auc_score(y, p) if p.std() > 0 else np.nan,
                         "top1": t1})

        if season == TEST_SEASONS[-1]:
            test["bin"] = pd.qcut(test.p_model, 10, labels=False, duplicates="drop")
            cal = test.groupby("bin").agg(pred=("p_model", "mean"),
                                          actual=("td", "mean"), n=("td", "size"))
            print("calibration, final test season:")
            for b, r in cal.iterrows():
                print(f"  decile {int(b)+1:>2}  predicted {r.pred:.3f}  "
                      f"actual {r.actual:.3f}  n={int(r.n)}")
            print(f"  mean |gap| {np.abs(cal.actual - cal.pred).mean():.4f}\n")

    res = pd.DataFrame(rows)
    print("per season top1:")
    print(res.pivot(index="season", columns="model", values="top1").round(4).to_string())
    print("\nmean across test seasons:")
    agg = res.groupby("model")[["brier", "logloss", "auc", "top1"]].mean().round(4)
    print(agg.reindex(["base", "pos", "shuf", "model"]).to_string())
    m, s = agg.loc["model", "top1"], agg.loc["shuf", "top1"]
    print(f"\nThe number that matters: top1 {m:.4f} against a shuffled floor of "
          f"{s:.4f}. Quote the gap against shuf, never against base.")


if __name__ == "__main__":
    main()
