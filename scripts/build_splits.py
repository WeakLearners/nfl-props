"""
Phase 2B, Job 5 -- the split. Decision #10.

Writes two NEW tables to features.db, keyed to training_rows:

- `splits`: (player_id, game_id, season, week, season_type, split), split in
  {train, validate, holdout} assigned by SEASON alone -- 2019-2024 train,
  2025 validate, 2026 holdout. Never a random split: a random split can put
  week 10 of a player's season in train and week 9 in test, letting the
  model half-remember the answer.

- `walk_forward_steps`: the validation-season walk-forward index. Step i
  trains through week n and predicts week n+1, stepping through every week
  present in 2025 (REG 1-18 then POST 19-22, which do not overlap -- see
  build_training_table.py's docstring). n_predict_rows is how many
  training_rows fall in the predicted week, for sizing each step.

Read-only against training_rows; writes only these two new tables. Single
connection, single-threaded, timeout=60. Never touches nflprops.db.
"""

import sqlite3
from pathlib import Path

import pandas as pd

FEATURES_DB = Path.home() / "Code/nfl-props/data/features.db"


def assign_split(season):
    if 2019 <= season <= 2024:
        return "train"
    if season == 2025:
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
    weeks_2025 = sorted(tr.loc[tr["season"] == 2025, "week"].unique().tolist())
    steps = []
    for i in range(len(weeks_2025) - 1):
        train_through = weeks_2025[i]
        predict_week = weeks_2025[i + 1]
        n_predict_rows = int(((tr["season"] == 2025) & (tr["week"] == predict_week)).sum())
        steps.append({
            "step": i + 1,
            "train_through_week": train_through,
            "predict_week": predict_week,
            "n_predict_rows": n_predict_rows,
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
    print(f"{len(wf)} steps across 2025")
    print(wf.to_string(index=False))

    con.close()


if __name__ == "__main__":
    main()
