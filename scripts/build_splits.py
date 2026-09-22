"""
Phase 2B, Job 5 -- the split. Decision #10.

Writes two NEW tables to features.db, keyed to training_rows:

- `splits`: (player_id, game_id, season, week, season_type, split), split in
  {train, validate, holdout} assigned by SEASON alone -- 2019-2023 train,
  2024 AND 2025 validate, 2026 holdout. Never a random split: a random split
  can put week 10 of a player's season in train and week 9 in test, letting
  the model half-remember the answer.

  TWO validation seasons, Sean's call 2026-09-22 (decision #17). With one
  validation year, every rung's pass/fail bar is measured on that one year.
  Tune against it enough times and you have fit the season rather than the
  sport. A rung must now clear its bar on 2024 AND on 2025. The cost is one
  season of training data; the purchase is an answer to "did it only work
  in 2025?"

- `walk_forward_steps`: the walk-forward index, now carrying a `season`
  column because there are two validation seasons. Step i trains through
  week n and predicts week n+1, stepping through every week present in that
  season (REG 1-18 then POST 19-22, which do not overlap -- see
  build_training_table.py's docstring). n_predict_rows is how many
  training_rows fall in the predicted week, for sizing each step.

Read-only against training_rows; writes only these two new tables. Single
connection, single-threaded, timeout=60. Never touches nflprops.db.
"""

import sqlite3
from pathlib import Path

import pandas as pd

FEATURES_DB = Path.home() / "Code/nfl-props/data/features.db"


VALIDATION_SEASONS = (2024, 2025)


def assign_split(season):
    if 2019 <= season <= 2023:
        return "train"
    if season in VALIDATION_SEASONS:
        return "validate"
    if season == 2026:
        return "holdout"
    return None


def build_splits(tr):
    df = tr[["player_id", "game_id", "season", "week", "season_type"]].copy()
    df["split"] = df["season"].apply(assign_split)
    unassigned = df["split"].isna().sum()
    if unassigned:
        raise RuntimeError(f"{unassigned} row(s) have a season outside 2019-2026: cannot split")
    return df


def build_walk_forward(tr):
    steps = []
    for season in VALIDATION_SEASONS:
        weeks = sorted(tr.loc[tr["season"] == season, "week"].unique().tolist())
        for i in range(len(weeks) - 1):
            predict_week = weeks[i + 1]
            steps.append({
                "season": season,
                "step": i + 1,
                "train_through_week": weeks[i],
                "predict_week": predict_week,
                "n_predict_rows": int(((tr["season"] == season) & (tr["week"] == predict_week)).sum()),
            })
    return pd.DataFrame(steps)


def main():
    con = sqlite3.connect(FEATURES_DB, timeout=60)
    tr = pd.read_sql_query("SELECT player_id, game_id, season, week, season_type FROM training_rows", con)

    splits = build_splits(tr)
    splits.to_sql("splits", con, if_exists="replace", index=False)
    con.commit()

    wf = build_walk_forward(tr)
    wf.to_sql("walk_forward_steps", con, if_exists="replace", index=False)
    con.commit()

    print("=== Job 5: splits ===")
    counts = splits.groupby(["split", "season"]).size().reset_index(name="n_rows")
    print(counts.to_string(index=False))
    print(f"\ntotal rows: {len(splits)}")

    print("\n=== Job 5: walk_forward_steps ===")
    print(f"{len(wf)} steps across {VALIDATION_SEASONS}")
    print(wf.groupby("season").size().to_string())

    con.close()


if __name__ == "__main__":
    main()
