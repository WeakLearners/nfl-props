"""
Phase 4 debt #1 -- are the touchdown probabilities honest numbers?

WHAT WAS WRONG. td_probability.py ships. It beats all three baselines in all
four cells. But its closing table showed the stated chances drifting from
reality: it said 5% and reality delivered 8-11%; it said 50% and reality
delivered 37-46%. The ranking was good. The numbers themselves were not.

That gap matters here more than usual. A prop bet is priced off a stated
chance. "This player scores 30% of the time" is the product. If the real
answer is 22%, the ranking being right does not save the bet.

WHAT CALIBRATION IS. Picture a weather forecaster. Over a season he says "30%
chance of rain" two hundred times. If it rained on sixty of those days, he is
calibrated -- his numbers mean what they say. If it rained on ninety, he is a
useful forecaster with dishonest numbers. Calibration is the correction that
maps his 30% onto the 45% he actually meant, learned from his own history.

THE TRAP, AND IT IS THE WHOLE DESIGN. A correction fitted on the same rows it
is scored on will look perfect and mean nothing. It has memorised the answers.
So the corrector never sees a validation row before predicting it:
CalibratedClassifierCV holds out folds inside the TRAINING rows, fits the model
on the rest, and learns the correction from predictions on rows the model did
not train on. Only then is it applied to the validation week.

TWO CORRECTIONS ARE TRIED, because they fail differently.
  Platt (sigmoid)  -- fits one S-curve. Two parameters. Cannot overfit, but
                      cannot fix a kink either.
  Isotonic         -- fits any non-decreasing shape. Fixes more, and on thin
                      data it will happily memorise noise.

THE BAR. A correction that makes Brier worse is rejected. A correction that
improves honesty at no cost to Brier ships. The baselines are re-scored
alongside, because if calibration moves the model's Brier it moves the lift,
and the lift is what decision #25's bar is measured on.

Protocol, populations and splits are unchanged from td_probability.py. Nothing
is retrained differently; the same model, walk-forward the same way. Read-only
against features.db. Writes one JSON.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, brier_score_loss

from rung2_ridge import load, add_derived, build_design, TRAIN_SEASONS, VALIDATION_SEASONS

OUT = Path.home() / "Code/nfl-props/data/td_calibration.json"

# n_jobs capped deliberately: the Phase 4 opponent run holds ~3 cores and
# CalibratedClassifierCV triples the fit count. -1 here starves both.
PARAMS = dict(objective="binary", n_estimators=300, learning_rate=0.05,
              num_leaves=31, min_child_samples=50, subsample=0.8, subsample_freq=1,
              colsample_bytree=0.8, reg_lambda=1.0, n_jobs=3, verbose=-1, random_state=17)

STATS = [("receiving_tds", "y_receiving_tds", "qual_targets"),
         ("rushing_tds",   "y_rushing_tds",   "qual_carries")]

METHODS = ["raw", "platt", "iso"]


def calibration_table(y, p, bins=5):
    """When it said X%, how often did it happen? Returns (said, happened, n)."""
    d = pd.DataFrame({"y": y, "p": p})
    d["bucket"] = pd.qcut(d.p, bins, labels=False, duplicates="drop")
    g = d.groupby("bucket").agg(said=("p", "mean"), happened=("y", "mean"), n=("y", "size"))
    return g


def max_gap(c):
    """Worst distance between a stated chance and what actually happened."""
    return float((c.said - c.happened).abs().max())


def main():
    df = add_derived(load())
    X = build_design(df)[0].astype(float)
    in_tr = df.season.isin(TRAIN_SEASONS).to_numpy()

    g = df.groupby("player_id", sort=False)
    for _, col, _ in STATS:
        df[col + "_hit"] = (df[col] >= 1).astype(float)
        df[col + "_career_rate"] = g[col + "_hit"].transform(lambda s: s.shift(1).expanding().mean())

    results, table, calib = {}, [], []
    for season in VALIDATION_SEASONS:
        weeks = sorted(df.loc[df.season == season, "week"].unique())
        for stat, col, screen in STATS:
            hit = col + "_hit"
            out = []
            for i in range(1, len(weeks)):
                prior = (df.season.eq(season) & df.week.isin(weeks[:i])).to_numpy()
                tr = (in_tr | prior) & df[screen].to_numpy() & df[col].notna().to_numpy()
                te = (df.season.eq(season) & df.week.eq(weeks[i])).to_numpy() & df[screen].to_numpy() & df[col].notna().to_numpy()
                if te.sum() == 0 or tr.sum() < 200:
                    continue

                ytr = df.loc[tr, hit]

                # The uncorrected model, exactly as td_probability.py fits it.
                raw = lgb.LGBMClassifier(**PARAMS).fit(X[tr], ytr)

                # Each corrector learns from predictions on rows its own model
                # did not train on. No validation row is involved.
                platt = CalibratedClassifierCV(
                    lgb.LGBMClassifier(**PARAMS), method="sigmoid", cv=3
                ).fit(X[tr], ytr)
                iso = CalibratedClassifierCV(
                    lgb.LGBMClassifier(**PARAMS), method="isotonic", cv=3
                ).fit(X[tr], ytr)

                league = float(ytr.mean())
                by_pos = df.loc[tr].groupby("position_group")[hit].mean()
                out.append(pd.DataFrame({
                    "y": df.loc[te, hit].to_numpy(float),
                    "p_raw": raw.predict_proba(X[te])[:, 1],
                    "p_platt": platt.predict_proba(X[te])[:, 1],
                    "p_iso": iso.predict_proba(X[te])[:, 1],
                    "p_league": league,
                    "p_pos": df.loc[te, "position_group"].map(by_pos).fillna(league).to_numpy(float),
                    "p_career": df.loc[te, col + "_career_rate"].fillna(league).to_numpy(float),
                }))

            r = pd.concat(out, ignore_index=True)
            row = {"n": int(len(r)), "actual_rate": round(float(r.y.mean()), 4)}

            for who in METHODS + ["league", "pos", "career"]:
                p = r["p_" + who].clip(0, 1)
                entry = {"brier": round(float(brier_score_loss(r.y, p)), 5),
                         "auc": round(float(roc_auc_score(r.y, p)), 4) if p.nunique() > 1 else None}
                if who in METHODS:
                    c = calibration_table(r.y.to_numpy(), p.to_numpy())
                    entry["max_gap"] = round(max_gap(c), 4)
                    calib.append((season, stat, who, c))
                row[who] = entry

            # The bar is unchanged: best of the three baselines, per decision #25.
            best_base = min(("league", "pos", "career"), key=lambda k: row[k]["brier"])
            row["best_baseline"] = best_base
            for who in METHODS:
                row[who]["lift_pct"] = round(
                    (row[best_base]["brier"] - row[who]["brier"]) / row[best_base]["brier"] * 100, 2)

            results.setdefault(str(season), {})[stat] = row
            table.append((season, stat, row))

    OUT.write_text(json.dumps(dict(
        question="do the stated touchdown chances mean what they say",
        corrections=["platt (sigmoid, 2 params)", "isotonic (any non-decreasing shape)"],
        fitted_on="held-out folds inside the training rows only -- no validation row seen",
        bar="a correction that worsens Brier is rejected; lift re-scored per decision #25 (1% Brier)",
        results=results), indent=2))

    print(f"{'season':<7}{'stat':<15}{'method':<7}{'brier':>9}{'lift%':>8}{'maxGap':>8}{'AUC':>8}{'n':>7}")
    for season, stat, row in table:
        for who in METHODS:
            print(f"{season:<7}{stat:<15}{who:<7}{row[who]['brier']:>9.5f}"
                  f"{row[who]['lift_pct']:>+8.2f}{row[who]['max_gap']:>8.3f}"
                  f"{row[who]['auc']:>8.4f}{row['n']:>7}")
        print(f"{'':<7}{'':<15}{'(base)':<7}{row[row['best_baseline']]['brier']:>9.5f}"
              f"{'':>8}{'':>8}{'':>8}{'':>7}  <- {row['best_baseline']}")

    print("\n--- when it said X%, how often did it happen? ---")
    for season, stat, who, c in calib:
        s = "  ".join(f"{a*100:.0f}->{b*100:.0f}" for a, b in zip(c.said, c.happened))
        print(f"{season} {stat:<15} {who:<7} {s}")

    print(f"\nwritten -> {OUT}")


if __name__ == "__main__":
    main()
