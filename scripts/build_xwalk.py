#!/usr/bin/env python
"""Phase 2A, Job 1 — player identity crosswalk.

Builds `player_xwalk` in data/features.db from nflreadpy.load_players().
This is the bridge table that lets snap_counts.pfr_player_id (a
Pro-Football-Reference id) join to player_games.player_id (a GSIS id) — a
join that matches ZERO of 43,672 rows when attempted directly, because the
two tables speak different id systems.

Read-only against the live database (data/nflprops.db). All writes go to
the new data/features.db. See projects/nfl-props/prediction-engine/
decisions.md and phase2a-ingest.md for the governing decisions.

Run with the project venv: .venv/bin/python scripts/build_xwalk.py
"""
import pathlib
import sqlite3

import nflreadpy as nfl
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
LIVE_DB = ROOT / "data" / "nflprops.db"
FEATURES_DB = ROOT / "data" / "features.db"

ACCEPTANCE_TARGET = 43630
ACCEPTANCE_FLOOR = 43000


def load_live_readonly():
    """Open the live DB strictly read-only and pull the two tables needed
    for the crosswalk and its acceptance test."""
    con = sqlite3.connect(f"file:{LIVE_DB}?mode=ro", uri=True)
    player_games = pd.read_sql_query(
        "SELECT player_id, player_display_name, position, season, week, "
        "game_id, team FROM player_games",
        con,
    )
    snap_counts = pd.read_sql_query(
        "SELECT game_id, team, pfr_player_id, player FROM snap_counts", con
    )
    con.close()
    return player_games, snap_counts


def build_xwalk_frame() -> pd.DataFrame:
    """Pull load_players() and shape it into the player_xwalk schema.

    NOTE ON A SPEC/DATA GAP: the spec (phase2a-ingest.md Job 1) lists
    `sleeper_id` as a column sourced from `load_players()`. It is not.
    nflreadpy.load_players() returns 39 columns and none of them is
    sleeper_id (verified directly). sleeper_id lives in the separate
    `nflreadpy.load_ff_playerids()` table, which is keyed differently
    (mfl_id-centric, fantasy-platform ids) and was not in scope here.
    Per the spec's own note that sleeper_id is "carried for later
    sources, unused now," this column is created and left NULL rather
    than silently backfilled from a different id space or silently
    dropped. Flagged here and in the run report.
    """
    players = nfl.load_players().to_pandas()
    xwalk = players[
        ["gsis_id", "pfr_id", "esb_id", "display_name", "position"]
    ].copy()
    xwalk["sleeper_id"] = None
    xwalk = xwalk[
        ["gsis_id", "pfr_id", "esb_id", "sleeper_id", "display_name", "position"]
    ]
    return xwalk


def write_features_db(xwalk: pd.DataFrame):
    con = sqlite3.connect(FEATURES_DB)
    xwalk.to_sql("player_xwalk", con, if_exists="replace", index=False)
    con.execute(
        "CREATE INDEX IF NOT EXISTS ix_xwalk_gsis ON player_xwalk(gsis_id)"
    )
    con.execute(
        "CREATE INDEX IF NOT EXISTS ix_xwalk_pfr ON player_xwalk(pfr_id)"
    )
    con.commit()
    con.close()


def run_acceptance_test(player_games: pd.DataFrame, snap_counts: pd.DataFrame,
                         xwalk: pd.DataFrame):
    """Re-run the snap-count-to-player_games join routed through
    player_xwalk and count matches against the 43,630 target."""
    total = len(player_games)

    # player_games.player_id (gsis) -> xwalk.gsis_id -> xwalk.pfr_id
    pg = player_games.merge(
        xwalk[["gsis_id", "pfr_id"]],
        left_on="player_id",
        right_on="gsis_id",
        how="left",
    )

    # xwalk.pfr_id -> snap_counts.pfr_player_id, keyed to game_id + pfr_id
    # only — NOT team. Verified directly: player_games.team uses the
    # current franchise code retroactively (e.g. Raiders = 'LV' for their
    # 2019 games under Oakland), while snap_counts.team is contemporaneous
    # ('OAK' for the same 2019 games). Requiring team-code equality drops
    # 178 genuinely-matched Raiders rows for no reason connected to player
    # identity. game_id + pfr_id is sufficient and unambiguous: a given
    # pfr_id can belong to only one of the two teams in a given game_id.
    # This one change is what takes the match count from 43,430 to the
    # 43,630 target exactly.
    sc = snap_counts.rename(columns={"pfr_player_id": "pfr_id"})
    sc_dedup = sc.drop_duplicates(subset=["game_id", "pfr_id"])

    merged = pg.merge(
        sc_dedup[["game_id", "pfr_id"]].assign(_sc_hit=True),
        on=["game_id", "pfr_id"],
        how="left",
    )

    matched = merged["_sc_hit"].fillna(False).astype(bool)
    match_count = int(matched.sum())

    residual = merged.loc[~matched].copy()

    def reason(row):
        if pd.isna(row["pfr_id"]):
            return "no_xwalk_entry_for_gsis_id"
        return "xwalk_pfr_id_not_in_snap_counts_for_this_game"

    residual["reason"] = residual.apply(reason, axis=1)
    residual_out = residual[
        [
            "player_id",
            "player_display_name",
            "position",
            "season",
            "week",
            "game_id",
            "team",
            "pfr_id",
            "reason",
        ]
    ]

    return total, match_count, residual_out


def write_residual(residual: pd.DataFrame):
    con = sqlite3.connect(FEATURES_DB)
    residual.to_sql("xwalk_residual", con, if_exists="replace", index=False)
    con.commit()
    con.close()


def summarize_residual(residual: pd.DataFrame):
    print("\n--- residual breakdown ---")
    print(residual["reason"].value_counts().to_string())
    print("\nby season:")
    print(residual["season"].value_counts().sort_index().to_string())
    print("\nby position (top 10):")
    print(residual["position"].value_counts().head(10).to_string())


def main():
    print(f"Reading live DB read-only: {LIVE_DB}")
    player_games, snap_counts = load_live_readonly()
    print(f"player_games rows: {len(player_games)}")
    print(f"snap_counts rows: {len(snap_counts)}")

    print("\nBuilding player_xwalk from nflreadpy.load_players() ...")
    xwalk = build_xwalk_frame()
    print(f"player_xwalk rows: {len(xwalk)}")
    print(f"  gsis_id null: {xwalk['gsis_id'].isna().sum()}")
    print(f"  pfr_id null: {xwalk['pfr_id'].isna().sum()}")
    print("  sleeper_id: NULL for all rows — not present in load_players(), see script docstring")

    write_features_db(xwalk)
    print(f"\nWrote player_xwalk to {FEATURES_DB}")

    total, match_count, residual = run_acceptance_test(player_games, snap_counts, xwalk)
    write_residual(residual)

    print("\n=== ACCEPTANCE TEST ===")
    print(f"Target: {ACCEPTANCE_TARGET} of {total} (floor: {ACCEPTANCE_FLOOR})")
    print(f"Actual: {match_count} of {total}")
    if match_count >= ACCEPTANCE_TARGET:
        print("PASS — meets or exceeds target.")
    elif match_count >= ACCEPTANCE_FLOOR:
        print("BELOW TARGET but above floor — report plainly, do not adjust target.")
    else:
        print("FAIL — below floor. Crosswalk is incomplete.")

    summarize_residual(residual)
    print(f"\nresidual rows written to xwalk_residual: {len(residual)}")


if __name__ == "__main__":
    main()
