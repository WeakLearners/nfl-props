#!/usr/bin/env python
"""Phase 2A, Job 2 — depth charts, two schemas into one table.

nflreadpy.load_depth_charts() returns a different shape either side of
2025 (verified directly, matching the spec):

  2019-2024 ("weekly"):   season, club_code, week, game_type, depth_team,
                           position, depth_position, gsis_id
                           ~37k rows/season, one row per player per week.

  2025-2026 ("snapshot"): dt, team, pos_abb, pos_rank, pos_slot, gsis_id
                           ~554k rows/season, ~2 snapshots/day.

Builds `depth_charts` in data/features.db with every snapshot kept (no
collapsing to one row per player-week — decision #7). Reads the live DB
read-only, ONLY to borrow game_spine for mapping 2025+ timestamps onto
season/week; never writes to it.

Run with the project venv: .venv/bin/python scripts/build_depth_charts.py
"""
import pathlib
import sqlite3

import nflreadpy as nfl
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
FEATURES_DB = ROOT / "data" / "features.db"

WEEKLY_SEASONS = list(range(2019, 2025))
SNAPSHOT_SEASONS = [2025, 2026]

# Weekly-schema team code -> the code schedules/game_spine actually use.
# Verified directly: 2019 depth-chart rows carry 'OAK' (contemporaneous),
# but schedules/player_games/game_spine all retroactively label every
# Raiders game (2019 included) 'LV'. Same root cause as the OAK/LV drift
# found in Job 1's snap_counts join, showing up in a second table.
TEAM_CODE_FIX = {"OAK": "LV"}

# pos_abb (2025+, side-specific: LDE/RDE, LCB/RCB, ...) -> the coarse
# position group weekly's `position` column already uses. Built by hand
# from the value lists (verified: weekly `position` has 22 clean values;
# snapshot pos_abb has 31, including special-teams slots weekly doesn't
# carry as a position at all). Flagged in the run report as a judgment
# call — the spec's own text names `pos_abb` as the source to normalize
# into pos_group, but does not supply the mapping.
POS_ABB_TO_GROUP = {
    "LDE": "DE", "RDE": "DE",
    "LDT": "DT", "RDT": "DT",
    "LG": "G", "RG": "G",
    "LT": "T", "RT": "T",
    "LILB": "ILB", "RILB": "ILB",
    "WLB": "OLB", "SLB": "OLB",
    "LCB": "CB", "RCB": "CB",
    "NB": "DB",
    "PK": "K",
    "KR": "ST", "PR": "ST", "H": "ST",
    # everything else (C, FB, FS, SS, MLB, QB, RB, WR, TE, K, P, LS) is
    # already a coarse label shared with the weekly schema's `position`.
}


def load_weekly() -> pd.DataFrame:
    df = nfl.load_depth_charts(seasons=WEEKLY_SEASONS).to_pandas()
    out = pd.DataFrame({
        "gsis_id": df["gsis_id"],
        "season": df["season"].astype("Int64"),
        "week": df["week"].astype("Int64"),
        "team": df["club_code"].replace(TEAM_CODE_FIX),
        "pos_group": df["position"],
        "depth_rank": pd.to_numeric(df["depth_team"], errors="coerce").astype("Int64"),
        "observed_at": None,
        "source_schema": "weekly",
        "game_type": df["game_type"],
    })
    return out


def load_week_windows(con) -> pd.DataFrame:
    """Per (season, week) kickoff windows from game_spine, used to bucket
    2025+ snapshot timestamps. Read-only borrow of a table Job 5 owns."""
    windows = pd.read_sql_query(
        "SELECT season, week, kickoff_utc FROM game_spine "
        "WHERE kickoff_utc IS NOT NULL",
        con,
    )
    windows["kickoff_utc"] = pd.to_datetime(windows["kickoff_utc"], utc=True)
    grouped = (
        windows.groupby(["season", "week"])["kickoff_utc"]
        .agg(win_start="min", win_end="max")
        .reset_index()
        .sort_values(["season", "week"])
    )
    return grouped


def map_dt_to_season_week(dt: pd.Series, windows: pd.DataFrame):
    """For each timestamp, find the season it belongs to and, within
    that season, the earliest week whose kickoff window ends at or after
    the timestamp.

    Season assignment uses the SAME rule at the season level as at the
    week level: a timestamp belongs to the first season whose season
    finale (last week's kickoff) is at or after it. That makes an
    offseason snapshot (e.g. a March roster chart, or an August
    preseason chart) belong to the UPCOMING season it is building
    toward, not the one that just ended — it falls after the previous
    season's Super Bowl and before the next one's.

    An earlier version of this function used each season's week-1 START
    as the boundary instead of the PRIOR season's finale. That silently
    routed ~93k rows of August-2025 preseason snapshots into season
    2024, week 22 (searchsorted clipping them to the last known week)
    instead of season 2025, week=NULL. Caught by eyeballing the row
    counts per season, which is exactly why that check is in the
    acceptance criteria.

    A timestamp before its season's week 1 gets week=NULL — there is no
    game week it belongs to yet. Vectorized with np.searchsorted; a
    per-row Python loop over ~1.1M rows was too slow."""
    import numpy as np

    season_end = (
        windows.groupby("season")["win_end"].max().sort_index()
    )
    seasons_sorted = season_end.index.to_numpy()
    ends_sorted = season_end.to_numpy()

    dt_vals = dt.to_numpy()
    # which season each timestamp belongs to: first season whose finale
    # (last week's kickoff) is >= the timestamp
    season_pos = np.searchsorted(ends_sorted, dt_vals, side="left")
    season_pos = np.clip(season_pos, 0, len(seasons_sorted) - 1)
    out_season = pd.array(seasons_sorted[season_pos], dtype="Int64")
    out_season = pd.Series(out_season, index=dt.index)
    # timestamps after the very last known season's finale have no
    # season at all (shouldn't occur given our seasons list, but guard)
    out_season[dt_vals > ends_sorted[-1]] = pd.NA

    out_week = pd.Series(pd.NA, index=dt.index, dtype="Int64")
    for s in seasons_sorted:
        mask = (out_season == s).to_numpy()
        if not mask.any():
            continue
        season_windows = windows[windows["season"] == s].sort_values("week")
        wk_starts = season_windows["win_start"].to_numpy()
        wk_ends = season_windows["win_end"].to_numpy()
        wk_nums = season_windows["week"].to_numpy()

        d = dt_vals[mask]
        # index of first week whose win_end >= d (weeks sorted ascending
        # by week number, dates increase with week number)
        pos = np.searchsorted(wk_ends, d, side="left")
        pos = np.clip(pos, 0, len(wk_ends) - 1)
        weeks = wk_nums[pos]
        weeks = weeks.astype("float")
        weeks[d < wk_starts[0]] = np.nan  # preseason/offseason: week stays NULL
        out_week.loc[mask] = pd.array(weeks, dtype="Int64")

    return out_season, out_week


def load_snapshot(con) -> pd.DataFrame:
    df = nfl.load_depth_charts(seasons=SNAPSHOT_SEASONS).to_pandas()
    dt = pd.to_datetime(df["dt"], utc=True)

    windows = load_week_windows(con)
    season, week = map_dt_to_season_week(dt, windows)

    pos_group = df["pos_abb"].map(lambda x: POS_ABB_TO_GROUP.get(x, x))

    out = pd.DataFrame({
        "gsis_id": df["gsis_id"],
        "season": season,
        "week": week,
        "team": df["team"],
        "pos_group": pos_group,
        "depth_rank": df["pos_rank"].astype("Int64"),
        "observed_at": dt.dt.strftime("%Y-%m-%d %H:%M:%S+00:00"),
        "source_schema": "snapshot",
        "game_type": None,
    })
    return out


def write_table(df: pd.DataFrame):
    # features.db is shared with a parallel job building other tables in
    # the same file; a generous busy_timeout avoids spurious "database is
    # locked" failures instead of racing it.
    con = sqlite3.connect(FEATURES_DB, timeout=60)
    df.to_sql("depth_charts", con, if_exists="replace", index=False)
    con.execute("CREATE INDEX IF NOT EXISTS ix_depth_gsis ON depth_charts(gsis_id)")
    con.execute("CREATE INDEX IF NOT EXISTS ix_depth_season_week ON depth_charts(season, week)")
    con.commit()
    con.close()


def main():
    print("Loading weekly-schema depth charts (2019-2024) ...")
    weekly = load_weekly()
    print(f"  {len(weekly)} rows")

    print("Loading snapshot-schema depth charts (2025-2026) ...")
    con = sqlite3.connect(FEATURES_DB, timeout=60)  # game_spine lives here, read-only borrow
    snapshot = load_snapshot(con)
    con.close()
    print(f"  {len(snapshot)} rows")

    combined = pd.concat([weekly, snapshot], ignore_index=True)
    assert combined["source_schema"].isna().sum() == 0, "source_schema must never be null"

    write_table(combined)
    print(f"\nWrote depth_charts to {FEATURES_DB}: {len(combined)} rows total")

    print("\n=== row counts per season ===")
    print(combined.groupby("season", dropna=False).size().to_string())

    print("\n=== source_schema split ===")
    print(combined["source_schema"].value_counts().to_string())

    print("\n=== snapshot rows with week unresolved (preseason/offseason) ===")
    snap_only = combined[combined["source_schema"] == "snapshot"]
    print(f"  {snap_only['week'].isna().sum()} of {len(snap_only)}")

    print("\n=== gsis_id null (can't join to player_games) ===")
    print(combined.groupby("source_schema")["gsis_id"].apply(lambda s: s.isna().sum()).to_string())

    print("\n=== weekly rows: REG/postseason week-number collisions ===")
    wk = combined[combined["source_schema"] == "weekly"]
    dupe_check = wk.groupby(["season", "week"])["game_type"].nunique()
    collisions = dupe_check[dupe_check > 1]
    print(f"  season/week combos with >1 game_type: {len(collisions)}")
    print(collisions.to_string() if len(collisions) else "  (none)")


if __name__ == "__main__":
    main()
