"""Pull nflverse data and land it in SQLite.

Everything here is a straight mirror of an upstream nflverse table except
rz_usage, which is aggregated from play-by-play. We aggregate on ingest rather
than storing raw pbp: 8 seasons of pbp is ~400k rows x 372 cols, and the only
thing the TD model needs out of it is who gets the ball inside the 20.
"""
import polars as pl
import nflreadpy as nfl

from .config import FIRST_SEASON, CURRENT_SEASON, PLAYER_GAME_COLS, SKILL_POSITIONS
from .db import connect, write_table


def seasons(first=FIRST_SEASON, last=CURRENT_SEASON):
    return list(range(first, last + 1))


def _pd(df: pl.DataFrame):
    return df.to_pandas()


def player_games(yrs):
    df = nfl.load_player_stats(seasons=yrs)
    df = df.select([c for c in PLAYER_GAME_COLS if c in df.columns])
    # Drop defenders and specialists; they never appear in the props we price.
    return df.filter(pl.col("position").is_in(SKILL_POSITIONS))


def red_zone_usage(yrs):
    """Per player-game touches inside the 20 and inside the 10.

    Inside-10 is the one that actually drives anytime-TD odds -- roughly half
    of all rushing TDs come from there -- so we keep it as its own split
    rather than lumping the whole red zone together.
    """
    frames = []
    for yr in yrs:
        pbp = nfl.load_pbp(seasons=[yr]).select(
            "game_id", "season", "week", "posteam", "yardline_100",
            "rush_attempt", "pass_attempt", "rusher_player_id",
            "receiver_player_id", "rush_touchdown", "pass_touchdown",
        ).filter(pl.col("yardline_100") <= 20)

        rush = (
            pbp.filter(pl.col("rush_attempt") == 1, pl.col("rusher_player_id").is_not_null())
            .group_by(["season", "week", "game_id", "posteam", "rusher_player_id"])
            .agg(
                rz_carries=pl.len(),
                rz10_carries=(pl.col("yardline_100") <= 10).sum(),
                rz_rush_tds=pl.col("rush_touchdown").sum(),
            )
            .rename({"rusher_player_id": "player_id"})
        )
        rec = (
            pbp.filter(pl.col("pass_attempt") == 1, pl.col("receiver_player_id").is_not_null())
            .group_by(["season", "week", "game_id", "posteam", "receiver_player_id"])
            .agg(
                rz_targets=pl.len(),
                rz10_targets=(pl.col("yardline_100") <= 10).sum(),
                rz_rec_tds=pl.col("pass_touchdown").sum(),
            )
            .rename({"receiver_player_id": "player_id"})
        )
        frames.append(
            rush.join(rec, on=["season", "week", "game_id", "posteam", "player_id"], how="full", coalesce=True)
            .fill_null(0)
            .rename({"posteam": "team"})
        )
    return pl.concat(frames, how="vertical_relaxed")


def build(yrs=None, con=None):
    yrs = yrs or seasons()
    counts = {}
    with connect() as con:
        counts["player_games"] = write_table(_pd(player_games(yrs)), "player_games", con, yrs)
        counts["schedules"] = write_table(_pd(nfl.load_schedules(seasons=yrs)), "schedules", con, yrs)
        counts["injuries"] = write_table(_pd(nfl.load_injuries(seasons=yrs)), "injuries", con, yrs)
        counts["snap_counts"] = write_table(_pd(nfl.load_snap_counts(seasons=yrs)), "snap_counts", con, yrs)
        # Weekly rosters answer "who is on this team this week", which offseason
        # movement makes unanswerable from player_games alone at Week 1.
        ros = nfl.load_rosters_weekly(seasons=yrs).select(
            "season", "week", "team", "position", "status", "gsis_id", "full_name")
        counts["rosters"] = write_table(_pd(ros), "rosters", con, yrs)
        counts["rz_usage"] = write_table(_pd(red_zone_usage(yrs)), "rz_usage", con, yrs)
    return counts
