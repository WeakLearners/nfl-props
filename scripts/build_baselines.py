"""
Phase 2B, Job 4 then Job 3 — population screens, then the two baselines.

Job 4 (decision #6): writes a NEW table `population_flags` in features.db,
keyed to `training_rows` by (player_id, game_id). Two flags:
  - in_predict_pop: player averaging >=1 touch/game (targets+carries)
  - in_score_pop:   player averaging >=3 touches/game
Both averages use ONLY the player's STRICTLY PRIOR games (shift(1) before
expanding mean) -- a screen built from full-season totals is itself a leak,
per the task brief. Nothing is deleted from training_rows; every row still
trains. The flags just decide what counts toward reported error.

Job 3 (decision #8): for each outcome stat, score two naive baselines --
career-to-date average and last-4-game average, both already shift(1)-safe
-- on the validation season (2025), restricted to in_score_pop rows only
(standing rule #2: screens apply to the baseline identically). MAE decides
the winner per stat; RMSE is always reported beside it. Both layers of
target C get baselines (standing rule #4): usage (targets, carries) and
conversion (yards per target, yards per carry) -- not just the final
receiving/rushing yards totals, though those are scored too since they are
outcome stats in their own right.

Results are written to data/baselines.json before any model exists, per
decision #8 clause 4.

Reads training_rows only. Read-only would be ideal but this script also
writes population_flags, so it opens ONE writable connection to
features.db, single-threaded, timeout=60. Never touches nflprops.db.
"""

import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

FEATURES_DB = Path.home() / "Code/nfl-props/data/features.db"
BASELINES_JSON = Path.home() / "Code/nfl-props/data/baselines.json"


def shifted_expanding(g, col):
    return g[col].shift(1).expanding(min_periods=1).mean()


def shifted_roll4(g, col):
    return g[col].shift(1).rolling(4, min_periods=1).mean()


def mae(pred, actual):
    return float(np.mean(np.abs(pred - actual)))


def rmse(pred, actual):
    return float(np.sqrt(np.mean((pred - actual) ** 2)))


# ---------------------------------------------------------------------------
# Job 4 -- population screens
# ---------------------------------------------------------------------------

def build_population_flags(tr):
    df = tr.sort_values(["player_id", "season", "week"]).copy()
    df["touches"] = df["y_targets"] + df["y_carries"]
    df["avg_touches_before"] = df.groupby("player_id", group_keys=False).apply(
        lambda g: shifted_expanding(g, "touches")
    ).reset_index(drop=True)

    df["in_predict_pop"] = (df["avg_touches_before"] >= 1).astype(int)
    df["in_score_pop"] = (df["avg_touches_before"] >= 3).astype(int)
    # A row with no prior games (avg_touches_before is NaN) cannot yet be
    # judged to be averaging anything -- it stays out of both populations
    # until it has at least one prior game to average.
    no_history = df["avg_touches_before"].isna()
    df.loc[no_history, ["in_predict_pop", "in_score_pop"]] = 0

    return df[["player_id", "game_id", "avg_touches_before", "in_predict_pop", "in_score_pop"]]


# ---------------------------------------------------------------------------
# Job 3 -- the two baselines
# ---------------------------------------------------------------------------

def score_stat(df, actual_col, career_col, last4_col, pop):
    """df must already carry actual_col, career_col, last4_col, season, in_score_pop."""
    scored = df[
        (df["season"] == 2025)
        & (df["in_score_pop"] == 1)
        & df[actual_col].notna()
        & df[career_col].notna()
        & df[last4_col].notna()
    ]
    n = len(scored)
    if n == 0:
        return {"career": {"mae": None, "rmse": None}, "last4": {"mae": None, "rmse": None},
                "winner": None, "n": 0}

    actual = scored[actual_col].to_numpy()
    career_mae = mae(scored[career_col].to_numpy(), actual)
    career_rmse = rmse(scored[career_col].to_numpy(), actual)
    last4_mae = mae(scored[last4_col].to_numpy(), actual)
    last4_rmse = rmse(scored[last4_col].to_numpy(), actual)
    winner = "career" if career_mae <= last4_mae else "last4"

    return {
        "career": {"mae": round(career_mae, 4), "rmse": round(career_rmse, 4)},
        "last4": {"mae": round(last4_mae, 4), "rmse": round(last4_rmse, 4)},
        "winner": winner,
        "n": n,
    }


def build_usage_baselines(tr, pop):
    """career/last4 for raw usage counts: targets, carries. Built here (not
    reused from training_rows) because the model's existing rolling
    features are SHARES (team_target_share_*), and decision #8's known
    check was measured on raw counts, not shares."""
    df = tr.sort_values(["player_id", "season", "week"]).copy()
    for col, label in [("y_targets", "targets"), ("y_carries", "carries")]:
        df[f"{label}_career"] = df.groupby("player_id", group_keys=False).apply(
            lambda g, c=col: shifted_expanding(g, c)
        ).reset_index(drop=True)
        df[f"{label}_last4"] = df.groupby("player_id", group_keys=False).apply(
            lambda g, c=col: shifted_roll4(g, c)
        ).reset_index(drop=True)
    df = df.merge(pop[["player_id", "game_id", "in_score_pop"]], on=["player_id", "game_id"])

    out = {}
    out["targets"] = score_stat(df, "y_targets", "targets_career", "targets_last4", pop)
    out["carries"] = score_stat(df, "y_carries", "carries_career", "carries_last4", pop)
    return out


def build_final_stat_baselines(tr, pop):
    """career/last4 for the final outcome totals: receiving_yards,
    rushing_yards. This is the pair decision #8's known-result check is
    measured against."""
    df = tr.sort_values(["player_id", "season", "week"]).copy()
    for col, label in [("y_receiving_yards", "receiving_yards"), ("y_rushing_yards", "rushing_yards")]:
        df[f"{label}_career"] = df.groupby("player_id", group_keys=False).apply(
            lambda g, c=col: shifted_expanding(g, c)
        ).reset_index(drop=True)
        df[f"{label}_last4"] = df.groupby("player_id", group_keys=False).apply(
            lambda g, c=col: shifted_roll4(g, c)
        ).reset_index(drop=True)
    df = df.merge(pop[["player_id", "game_id", "in_score_pop"]], on=["player_id", "game_id"])

    out = {}
    out["receiving_yards"] = score_stat(
        df, "y_receiving_yards", "receiving_yards_career", "receiving_yards_last4", pop
    )
    out["rushing_yards"] = score_stat(
        df, "y_rushing_yards", "rushing_yards_career", "rushing_yards_last4", pop
    )
    return out


def build_conversion_baselines(tr, pop):
    """Layer 2 -- yards per target, yards per carry. The candidates are
    training_rows' EXISTING shift(1)-safe conversion columns
    (yards_per_target_career/_l4, yards_per_carry_career/_l4); the true
    value is this game's realized rate, undefined when the player had zero
    targets/carries that game."""
    df = tr.copy()
    df["realized_ypt"] = np.where(df["y_targets"] > 0, df["y_receiving_yards"] / df["y_targets"], np.nan)
    df["realized_ypc"] = np.where(df["y_carries"] > 0, df["y_rushing_yards"] / df["y_carries"], np.nan)
    df = df.merge(pop[["player_id", "game_id", "in_score_pop"]], on=["player_id", "game_id"])

    out = {}
    out["yards_per_target"] = score_stat(
        df, "realized_ypt", "yards_per_target_career", "yards_per_target_l4", pop
    )
    out["yards_per_carry"] = score_stat(
        df, "realized_ypc", "yards_per_carry_career", "yards_per_carry_l4", pop
    )
    return out


def main():
    con = sqlite3.connect(FEATURES_DB, timeout=60)
    tr = pd.read_sql_query(
        "SELECT player_id, game_id, season, week, y_targets, y_carries, y_receiving_yards, "
        "y_rushing_yards, yards_per_target_career, yards_per_target_l4, "
        "yards_per_carry_career, yards_per_carry_l4 FROM training_rows",
        con,
    )

    # --- Job 4 ---
    pop = build_population_flags(tr)
    pop.to_sql("population_flags", con, if_exists="replace", index=False)
    con.commit()

    n_predict_rows = int(pop["in_predict_pop"].sum())
    n_score_rows = int(pop["in_score_pop"].sum())
    n_predict_players = pop.loc[pop["in_predict_pop"] == 1, "player_id"].nunique()
    n_score_players = pop.loc[pop["in_score_pop"] == 1, "player_id"].nunique()

    print("=== Job 4: population_flags ===")
    print(f"in_predict_pop: {n_predict_rows} rows, {n_predict_players} distinct players")
    print(f"in_score_pop:   {n_score_rows} rows, {n_score_players} distinct players")

    # --- Job 3 ---
    results = {}
    results["usage"] = build_usage_baselines(tr, pop)
    results["final"] = build_final_stat_baselines(tr, pop)
    results["conversion"] = build_conversion_baselines(tr, pop)

    BASELINES_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(BASELINES_JSON, "w") as f:
        json.dump(results, f, indent=2)

    print("\n=== Job 3: baselines (validation season 2025, in_score_pop only) ===")
    for layer, stats in results.items():
        for stat, r in stats.items():
            print(f"[{layer}] {stat}: career MAE={r['career']['mae']} RMSE={r['career']['rmse']} | "
                  f"last4 MAE={r['last4']['mae']} RMSE={r['last4']['rmse']} | "
                  f"winner={r['winner']} n={r['n']}")

    print(f"\nWrote {BASELINES_JSON}")
    con.close()


if __name__ == "__main__":
    main()
