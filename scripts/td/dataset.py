#!/usr/bin/env python
"""Phase 1 of the anytime-TD model: the labelled dataset, and the leakage
guard that keeps it honest.

Built on data/eda_features.parquet, whose h_* columns are already decayed
and leakage-safe (see scripts/eda/build_features.py). What this file adds is
the label and, more importantly, an ALLOWLIST.

Why an allowlist rather than a blocklist. That parquet holds the target
row's own outcomes side by side with its history: rushing_tds and
receiving_tds are the label itself, and carries, targets, rz10_carries and
the rest are what the player did in the very game being predicted. Feeding
any of them back in produces a model that looks extraordinary and knows
nothing. A blocklist fails silently the first time a column is added
upstream; an allowlist fails loudly, which is the direction an error should
fail in a project that has already published one wrong number.

The 2026-09-13 retraction is the standing reason for the assertion at the
bottom of build(). Every check run before that retraction asked whether the
answer looked plausible. None asked whether the test was correct.
"""
import pathlib
import sys

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[2]
FEATURES_PARQUET = ROOT / "data" / "eda_features.parquet"

# Decayed player history. Every one of these is built strictly from games
# before the target row's (season, week).
HISTORY_PREFIX = "h_"

# Defense-vs-position, built as-of the target week by the same builder.
DVP_PREFIX = "dvp_"

# Known before kickoff: the betting market's own view, the venue, the
# weather, the rest, and the injury report. None of these are outcomes.
PREGAME = [
    "implied_team_total", "home_implied", "away_implied",
    "spread_line", "total_line", "div_game",
    "is_home", "rest_days", "temp", "wind",
    "own_questionable", "own_status_missing",
    "teammates_out_same_pos", "team_out_total",
    "n_games_hist",
]
# Encoded by the caller, never passed raw. report_status joined this list
# on 2026-09-20: it is a string ("Questionable", "Out", …) and slipped the
# first dtype check because pandas 3 stores strings as an arrow type rather
# than object. The numeric assertion in build() is the real guard now — a
# check that depends on knowing which dtype a library chose this year is not
# a check.
CATEGORICAL = ["position", "roof", "surface", "report_status"]

# Identity, carried through for joining and for walk-forward splitting.
# Never fed to a model.
KEYS = ["player_id", "player_display_name", "position", "team", "opponent_team",
        "season", "week", "game_id", "report_status"]

# The target row's own outcomes. The label is built from the first two; none
# of them may ever appear as a feature. Listed explicitly so the assertion
# below has something concrete to check rather than a guess at a prefix.
OUTCOMES = [
    "rushing_tds", "receiving_tds", "passing_tds", "rz_rush_tds", "rz_rec_tds",
    "carries", "rushing_yards", "rushing_first_downs", "rushing_epa",
    "receptions", "targets", "receiving_yards", "receiving_air_yards",
    "receiving_yards_after_catch", "receiving_epa", "racr", "target_share",
    "air_yards_share", "wopr", "fantasy_points_ppr",
    "completions", "attempts", "passing_yards", "passing_interceptions",
    "sacks_suffered", "passing_air_yards", "passing_epa", "passing_cpoe",
    "rz_carries", "rz10_carries", "rz_targets", "rz10_targets",
    "offense_snaps", "offense_pct",
]


def feature_columns(df):
    cols = [c for c in df.columns if c.startswith((HISTORY_PREFIX, DVP_PREFIX))]
    cols += [c for c in PREGAME if c in df.columns]
    return sorted(cols)


def build(min_games_hist=3, positions=("RB", "WR", "TE", "QB")):
    """Label every player-game with whether that player scored, plus features.

    min_games_hist drops rows whose history is too thin to describe them. The
    upstream builder already requires 3; keeping the parameter here makes the
    choice visible at the point it affects the model rather than buried a
    directory away.
    """
    if not FEATURES_PARQUET.exists():
        sys.exit(f"missing {FEATURES_PARQUET} — run scripts/eda/build_features.py first")
    df = pd.read_parquet(FEATURES_PARQUET)

    df = df[df.position.isin(positions)].copy()
    if "n_games_hist" in df.columns:
        df = df[df.n_games_hist >= min_games_hist].copy()

    # Anytime TD: rushing or receiving. Passing touchdowns are not the
    # player scoring -- a QB throwing four of them scored none of them, and
    # the market prices it that way too.
    df["td"] = ((df.rushing_tds.fillna(0) + df.receiving_tds.fillna(0)) > 0).astype(int)

    feats = feature_columns(df)
    keep = [c for c in KEYS if c in df.columns] + feats + ["td"]
    out = df[keep].copy()

    leaked = sorted(set(feats) & set(OUTCOMES))
    assert not leaked, f"outcome columns reached the feature set: {leaked}"
    for cat in CATEGORICAL:
        assert cat not in feats, f"{cat} is categorical; encode it, do not pass it raw"
    nonnum = [c for c in feats if not pd.api.types.is_numeric_dtype(out[c])]
    assert not nonnum, f"non-numeric columns in the feature set: {nonnum}"

    return out, feats


if __name__ == "__main__":
    df, feats = build()
    print(f"rows      {len(df):,}")
    print(f"features  {len(feats)}")
    print(f"label     td rate {df.td.mean():.4f}")
    print(f"seasons   {df.season.min()}–{df.season.max()}")
    print()
    print("feature set:")
    for f in feats:
        print(f"  {f}")
