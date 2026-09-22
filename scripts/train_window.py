"""
Does the model's dishonesty come from stale training data?

THE QUESTION. td_calibrate.py found the stated touchdown chances drifting from
reality, and the drift split by season rather than by stat. 2025 was badly off
(worst gap 0.137) and 2024 was nearly fine (0.047). 2025 sits two years from the
end of the training data; 2024 sits one. That is a drift story, and drift has a
cheaper fix than a correction layer: train on newer football.

THE OBVIOUS TEST IS THE WRONG ONE. "Put 2024 in training and see" costs a
validation season. The project has two, and the rule that both must agree is what
caught the 28% touchdown result that was really a loss to a constant. Trading
that rule for one answer is a bad trade.

SO: score 2025 twice. Same season, same rows, same screens, same walk-forward.
The ONLY difference is where training stops.

  window A   2019-2023   -- what ships today
  window B   2019-2024   -- one more season of football

If the 2025 calibration gap shrinks under window B, staleness was the cause and
the correction layer is treating a symptom. If it does not move, the drift story
is wrong and calibration has to be argued on its own merits.

2024 IS NOT CONSUMED. It moves into training for window B only, inside this
script, to answer this one question. The shipping split is untouched. Nothing
here writes to rung2_ridge.py's constants.

THE HOLDOUT IS NOT TOUCHED. 2026 stays shut, per decision #24.

Read-only against features.db. Writes one JSON.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, brier_score_loss

from rung2_ridge import load, add_derived, build_design

OUT = Path.home() / "Code/nfl-props/data/train_window.json"

TEST_SEASON = 2025
WINDOWS = {"A_through_2023": list(range(2019, 2024)),
           "B_through_2024": list(range(2019, 2025))}

# n_jobs held below the core count so a second run can share the machine.
CLS = dict(objective="binary", n_estimators=300, learning_rate=0.05,
           num_leaves=31, min_child_samples=50, subsample=0.8, subsample_freq=1,
           colsample_bytree=0.8, reg_lambda=1.0, n_jobs=4, verbose=-1, random_state=17)
REG = dict(objective="regression_l1", n_estimators=300, learning_rate=0.05,
           num_leaves=31, min_child_samples=50, subsample=0.8, subsample_freq=1,
           colsample_bytree=0.8, reg_lambda=1.0, n_jobs=4, verbose=-1, random_state=17)

TD_STATS = [("receiving_tds", "y_receiving_tds", "qual_targets"),
            ("rushing_tds",   "y_rushing_tds",   "qual_carries")]
MAE_STATS = [("targets", "y_targets", "qual_targets"),
             ("carries", "y_carries", "qual_carries"),
             ("receiving_yards", "y_receiving_yards", "qual_targets"),
             ("rushing_yards", "y_rushing_yards", "qual_carries")]


def gap(y, p, bins=5):
    """Worst distance between a stated chance and what actually happened."""
    d = pd.DataFrame({"y": y, "p": p})
    d["b"] = pd.qcut(d.p, bins, labels=False, duplicates="drop")
    g = d.groupby("b").agg(said=("p", "mean"), happened=("y", "mean"))
    return float((g.said - g.happened).abs().max()), g


def main():
    df = add_derived(load())
    X = build_design(df)[0].astype(float)

    g = df.groupby("player_id", sort=False)
    for _, col, _ in TD_STATS:
        df[col + "_hit"] = (df[col] >= 1).astype(float)
        df[col + "_career_rate"] = g[col + "_hit"].transform(lambda s: s.shift(1).expanding().mean())

    weeks = sorted(df.loc[df.season == TEST_SEASON, "week"].unique())
    results, curves = {}, []

    for wname, seasons in WINDOWS.items():
        in_tr = df.season.isin(seasons).to_numpy()
        res = {}

        for stat, col, screen in TD_STATS:
            hit = col + "_hit"
            out = []
            for i in range(1, len(weeks)):
                prior = (df.season.eq(TEST_SEASON) & df.week.isin(weeks[:i])).to_numpy()
                tr = (in_tr | prior) & df[screen].to_numpy() & df[col].notna().to_numpy()
                te = (df.season.eq(TEST_SEASON) & df.week.eq(weeks[i])).to_numpy() \
                     & df[screen].to_numpy() & df[col].notna().to_numpy()
                if te.sum() == 0 or tr.sum() < 200:
                    continue
                m = lgb.LGBMClassifier(**CLS).fit(X[tr], df.loc[tr, hit])
                league = float(df.loc[tr, hit].mean())
                by_pos = df.loc[tr].groupby("position_group")[hit].mean()
                out.append(pd.DataFrame({
                    "y": df.loc[te, hit].to_numpy(float),
                    "p": m.predict_proba(X[te])[:, 1],
                    "p_league": league,
                    "p_pos": df.loc[te, "position_group"].map(by_pos).fillna(league).to_numpy(float),
                    "p_career": df.loc[te, col + "_career_rate"].fillna(league).to_numpy(float),
                }))
            r = pd.concat(out, ignore_index=True)
            mg, curve = gap(r.y.to_numpy(), r.p.clip(0, 1).to_numpy())
            best = min(("p_league", "p_pos", "p_career"),
                       key=lambda c: brier_score_loss(r.y, r[c].clip(0, 1)))
            bb = float(brier_score_loss(r.y, r[best].clip(0, 1)))
            bm = float(brier_score_loss(r.y, r.p.clip(0, 1)))
            res[stat] = {"brier": round(bm, 5), "max_gap": round(mg, 4),
                         "auc": round(float(roc_auc_score(r.y, r.p)), 4),
                         "best_baseline": best.replace("p_", ""),
                         "lift_pct": round((bb - bm) / bb * 100, 2), "n": int(len(r))}
            curves.append((wname, stat, curve))
            print(f"  [{wname}] {stat}: brier {bm:.5f} gap {mg:.3f}", flush=True)

        for stat, col, screen in MAE_STATS:
            out = []
            for i in range(1, len(weeks)):
                prior = (df.season.eq(TEST_SEASON) & df.week.isin(weeks[:i])).to_numpy()
                tr = (in_tr | prior) & df[screen].to_numpy() & df[col].notna().to_numpy()
                te = (df.season.eq(TEST_SEASON) & df.week.eq(weeks[i])).to_numpy() \
                     & df[screen].to_numpy() & df[col].notna().to_numpy()
                if te.sum() == 0 or tr.sum() < 200:
                    continue
                m = lgb.LGBMRegressor(**REG).fit(X[tr], df.loc[tr, col].to_numpy(float))
                out.append(pd.DataFrame({"y": df.loc[te, col].to_numpy(float),
                                         "yhat": m.predict(X[te])}))
            r = pd.concat(out, ignore_index=True)
            res[stat] = {"mae": round(float((r.y - r.yhat).abs().mean()), 4), "n": int(len(r))}
            print(f"  [{wname}] {stat}: mae {res[stat]['mae']:.4f}", flush=True)

        results[wname] = res

    OUT.write_text(json.dumps(dict(
        question="does training one season later fix the 2025 calibration drift",
        test_season=TEST_SEASON, windows={k: [v[0], v[-1]] for k, v in WINDOWS.items()},
        note="2024 enters training for window B only; the shipping split is unchanged",
        results=results), indent=2))

    A, B = results["A_through_2023"], results["B_through_2024"]
    print(f"\n{'stat':<18}{'metric':<7}{'A 19-23':>10}{'B 19-24':>10}{'change':>10}")
    for stat, _, _ in TD_STATS:
        for k in ("brier", "max_gap"):
            d = B[stat][k] - A[stat][k]
            print(f"{stat:<18}{k:<7}{A[stat][k]:>10.5f}{B[stat][k]:>10.5f}{d:>+10.5f}"
                  f"   {'better' if d < 0 else 'worse'}")
    for stat, _, _ in MAE_STATS:
        d = B[stat]["mae"] - A[stat]["mae"]
        print(f"{stat:<18}{'mae':<7}{A[stat]['mae']:>10.4f}{B[stat]['mae']:>10.4f}{d:>+10.4f}"
              f"   {'better' if d < 0 else 'worse'}")

    print("\n--- when it said X%, how often did it happen? (2025) ---")
    for wname, stat, c in curves:
        s = "  ".join(f"{a*100:.0f}->{b*100:.0f}" for a, b in zip(c.said, c.happened))
        print(f"{wname:<16} {stat:<15} {s}")

    print(f"\nwritten -> {OUT}")


if __name__ == "__main__":
    main()
