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


# Known-bad values in the upstream nflreadpy feed itself, applied after every
# load so a fresh `build()` re-pulling from nflreadpy can't silently
# reintroduce a bug that was already found and fixed.
#
# Bar for adding an entry here: an upstream roster value is overridden only
# when 2026 game data (player_games/snap_counts -- an actual snap taken)
# contradicts it, never on an assumption about who "should" be on a team.
# Decision #34 (2026-09-24) originally added a Tua Tagovailoa ATL->MIA
# correction here on the reasoning that a 79-game veteran starter landing on
# a new team with zero 2026 snaps looked wrong. Decision #35 (2026-09-24)
# reverted it: Sean confirmed Tua is genuinely on Atlanta in 2026, and the
# "zero 2026 snaps" signal was never actually distinguishing -- Malik Willis
# started both of Miami's 2026 games and Tua's last game anywhere is 2025
# W15, which is exactly the shape of the 20 legitimate veteran-backup moves
# already found and accepted in the same investigation (Andy Dalton, Tyrod
# Taylor, Odell Beckham Jr., etc. -- signed to a new team, no snaps yet).
# There was no contradicting game evidence, only an assumption. Left empty
# on purpose rather than deleted: this is the right general mechanism for
# the next case where nflreadpy is provably wrong, not for cases where a
# player merely looks surprising.
ROSTER_TEAM_CORRECTIONS = {}


def _apply_roster_corrections(df: pl.DataFrame) -> pl.DataFrame:
    for (gsis_id, season), team in ROSTER_TEAM_CORRECTIONS.items():
        df = df.with_columns(
            pl.when((pl.col("gsis_id") == gsis_id) & (pl.col("season") == season))
            .then(pl.lit(team))
            .otherwise(pl.col("team"))
            .alias("team")
        )
    return df


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
        ros = _apply_roster_corrections(ros)
        counts["rosters"] = write_table(_pd(ros), "rosters", con, yrs)
        counts["rz_usage"] = write_table(_pd(red_zone_usage(yrs)), "rz_usage", con, yrs)
    return counts
