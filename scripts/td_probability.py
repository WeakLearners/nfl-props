"""
Phase 3, final piece -- touchdowns as a probability. Decisions #21 and #22.

WHY THIS EXISTS. Rung 3 reported a 28% improvement on touchdowns and it was a
lie. 78% of qualifying games have zero touchdowns, so the answer that minimizes
average absolute error is "nobody scores". The model that learned to say so lost
to a literal constant: 0.2655 against always-zero's 0.2500.

MAE asked "how close is your number" when the only useful question was "how
likely is this". So the target changes shape. Not a count -- a chance.

TARGET (#21): did this player reach the end zone at all. Most non-zero
touchdown games are exactly one, so what this gives up is rare.

DECIDER (#22): Brier score -- squared error on probabilities. Say 30%, and it
either happened or it did not. Brier answers "are these numbers honest", which
is what makes a stated chance usable. AUC is reported beside it and answers
"did you rank the scorers above the non-scorers".

THE GUARD, WHICH IS THE WHOLE POINT. Every metric has a degenerate answer that
games it. For MAE on touchdowns it was zero. For Brier it is the league base
rate -- predict 0.22 for everyone and you score respectably while saying nothing
about anybody. So THREE baselines are scored explicitly, and the model has to
beat all of them:

  1. league base rate    -- one number for everybody. The degenerate answer.
  2. player career rate  -- his own history, leak-gated.
  3. position base rate  -- the rate for his position group.

If the model cannot beat a constant, that is the finding, and it gets printed
rather than buried.

PROTOCOL is unchanged from rungs 2 and 3: train 2019-2023 fixed, walk-forward
inside each validation season, per-stat screened populations, both seasons must
agree. Bar is decision #19's 3%, measured on Brier against the BEST of the three
baselines.

Read-only against features.db. Writes nothing to any database.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, brier_score_loss

from rung2_ridge import load, add_derived, build_design, TRAIN_SEASONS, VALIDATION_SEASONS

OUT = Path.home() / "Code/nfl-props/data/td_probability.json"

PARAMS = dict(objective="binary", n_estimators=300, learning_rate=0.05,
              num_leaves=31, min_child_samples=50, subsample=0.8, subsample_freq=1,
              colsample_bytree=0.8, reg_lambda=1.0, n_jobs=-1, verbose=-1, random_state=17)

STATS = [("receiving_tds", "y_receiving_tds", "qual_targets"),
         ("rushing_tds",   "y_rushing_tds",   "qual_carries")]


def main():
    df = add_derived(load())
    X = build_design(df)[0].astype(float)
    in_tr = df.season.isin(TRAIN_SEASONS).to_numpy()

    # Leak-gated per-player career TD rate: games strictly before this one.
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
                m = lgb.LGBMClassifier(**PARAMS).fit(X[tr], df.loc[tr, hit])
                # Baselines computed from TRAINING ROWS ONLY -- the same
                # information the model had at that moment, nothing more.
                league = float(df.loc[tr, hit].mean())
                by_pos = df.loc[tr].groupby("position_group")[hit].mean()
                out.append(pd.DataFrame({
                    "y": df.loc[te, hit].to_numpy(float),
                    "p_model": m.predict_proba(X[te])[:, 1],
                    "p_league": league,
                    "p_pos": df.loc[te, "position_group"].map(by_pos).fillna(league).to_numpy(float),
                    "p_career": df.loc[te, col + "_career_rate"].fillna(league).to_numpy(float),
                }))
            r = pd.concat(out, ignore_index=True)

            row = {"n": int(len(r)), "actual_rate": round(float(r.y.mean()), 4)}
            for who in ("model", "league", "pos", "career"):
                p = r["p_" + who].clip(0, 1)
                row[who] = {"brier": round(float(brier_score_loss(r.y, p)), 5),
                            "auc": round(float(roc_auc_score(r.y, p)), 4) if p.nunique() > 1 else None}
            best_base = min(("league", "pos", "career"), key=lambda k: row[k]["brier"])
            row["best_baseline"] = best_base
            row["lift_pct"] = round((row[best_base]["brier"] - row["model"]["brier"])
                                    / row[best_base]["brier"] * 100, 2)
            results.setdefault(str(season), {})[stat] = row
            table.append((season, stat, row))

            # Is the stated chance honest? Bucket the predictions and compare
            # predicted rate against what actually happened in each bucket.
            r["bucket"] = pd.qcut(r.p_model, 5, labels=False, duplicates="drop")
            c = r.groupby("bucket").agg(said=("p_model", "mean"), happened=("y", "mean"), n=("y", "size"))
            calib.append((season, stat, c))

    OUT.write_text(json.dumps(dict(
        target="P(player scores >= 1 TD)", decider="Brier score", reported="AUC",
        bar="beat the best of three baselines by >= 3% Brier, both seasons (decision #19)",
        results=results), indent=2))

    print(f"{'season':<8}{'stat':<16}{'rate':>7}{'modelBr':>9}{'leagueBr':>10}{'posBr':>9}{'careerBr':>10}"
          f"{'best':>8}{'lift%':>8}{'AUC':>7}{'n':>7}")
    for season, stat, row in table:
        print(f"{season:<8}{stat:<16}{row['actual_rate']:>7.3f}{row['model']['brier']:>9.5f}"
              f"{row['league']['brier']:>10.5f}{row['pos']['brier']:>9.5f}{row['career']['brier']:>10.5f}"
              f"{row['best_baseline']:>8}{row['lift_pct']:>+8.2f}{row['model']['auc']:>7.4f}{row['n']:>7}")

    print("\n--- calibration: when it said X%, how often did it happen? ---")
    for season, stat, c in calib:
        s = "  ".join(f"{a*100:.0f}%->{b*100:.0f}%" for a, b in zip(c.said, c.happened))
        print(f"{season} {stat:<16} {s}")

    print(f"\nwritten -> {OUT}")


if __name__ == "__main__":
    main()
