#!/usr/bin/env python
"""Phase 2: does a model beat the obvious baselines at picking the scorer?

Walk-forward by season. Train on every season strictly before the test
season, predict the test season, never the reverse. Seasons before 2021 are
training-only, since a model needs history to have any.

Four things are scored, and the fourth is the one that matters here. Sean
wants one marker per game naming the player most likely to score. That is a
RANKING claim about a single game, not a calibration claim about a
population, so a model can have a respectable Brier score and still be
useless for the marker. top1 measures the actual product: take the model's
highest-probability player in a game, and ask how often that player scored.

The controls exist because of 2026-09-17, when a full feature sweep found
that the biggest number in the table belonged to the control column. "Always
name the running back" is close to free, and a model that cannot beat it has
not earned a place on the page.
"""
import pathlib
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from dataset import build  # noqa: E402

TEST_SEASONS = [2021, 2022, 2023, 2024, 2025]


def top1_rate(frame, score_col):
    """Share of games whose highest-scored player actually scored.

    Ties are broken by taking the first row, which is arbitrary on purpose:
    a control that cannot separate its candidates should not be rescued by a
    tiebreak that quietly encodes knowledge.
    """
    idx = frame.groupby("game_id")[score_col].idxmax()
    return frame.loc[idx, "td"].mean(), len(idx)


def evaluate():
    df, feats = build()
    # Position is the single strongest thing anyone knows before kickoff, and
    # the controls below lean on it, so the model gets to see it too.
    # report_status comes along as dummies rather than as a string: it is on
    # the injury report, which is public before kickoff, and "Questionable"
    # is real information about whether a player will be on the field.
    cats = [pd.get_dummies(df["position"], prefix="pos")]
    if "report_status" in df.columns:
        cats.append(pd.get_dummies(df["report_status"].fillna("None"), prefix="inj"))
    X_all = pd.concat([df[feats]] + cats, axis=1)
    feat_names = list(X_all.columns)

    rows = []
    for season in TEST_SEASONS:
        tr = df.season < season
        te = df.season == season
        if tr.sum() == 0 or te.sum() == 0:
            continue

        test = df[te].copy()

        # Control 1: one number for everybody, the training base rate. It
        # cannot rank at all, so its top1 is whatever the arbitrary tiebreak
        # lands on -- that is the honest floor.
        base = df.loc[tr, "td"].mean()
        test["p_base"] = base

        # Control 2: the training base rate for that player's position.
        # "Always name the running back", costing one groupby.
        pos_rate = df.loc[tr].groupby("position")["td"].mean()
        test["p_pos"] = test.position.map(pos_rate).fillna(base)

        # The model.
        clf = HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.06, max_leaf_nodes=31,
            l2_regularization=1.0, random_state=0,
        )
        clf.fit(X_all[tr.values], df.loc[tr, "td"])
        test["p_model"] = clf.predict_proba(X_all[te.values])[:, 1]

        for name in ("base", "pos", "model"):
            p = test[f"p_{name}"].to_numpy()
            y = test["td"].to_numpy()
            t1, n_games = top1_rate(test, f"p_{name}")
            rows.append({
                "season": season, "model": name, "n": len(test), "games": n_games,
                "brier": brier_score_loss(y, p),
                "logloss": log_loss(y, np.clip(p, 1e-6, 1 - 1e-6)),
                "auc": roc_auc_score(y, p) if p.std() > 0 else np.nan,
                "top1": t1,
            })
    return pd.DataFrame(rows), feat_names


if __name__ == "__main__":
    res, feat_names = evaluate()
    pd.set_option("display.width", 130)
    print(f"{len(feat_names)} features (48 + position one-hots)\n")
    print("per season:")
    print(res.pivot(index="season", columns="model",
                    values=["brier", "auc", "top1"]).round(4).to_string())
    print("\nmean across the five test seasons:")
    agg = res.groupby("model")[["brier", "logloss", "auc", "top1"]].mean().round(4)
    print(agg.reindex(["base", "pos", "model"]).to_string())
    g = res[res.model == "model"].games.sum()
    print(f"\ntop1 is over {g} games.")
    print("The honest floor is NOT the name-anyone rate "
          f"({res[res.model=='base'].top1.mean():.4f}). A control with no ranking "
          "power still has to pick somebody, and idxmax does not pick at random -- "
          "it lands on whoever has extreme feature values, who tend to be the "
          "high-usage players who score. scripts/td/verify.py measures that floor "
          "directly by retraining on a shuffled label: it comes out near 0.25, "
          "close to the name-a-back control at "
          f"{res[res.model=='pos'].top1.mean():.4f}. Compare against those, not "
          "against name-anyone.")
