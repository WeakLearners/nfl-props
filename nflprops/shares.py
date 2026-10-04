"""Per player-game share columns for the rating system (read-only).

snap_counts is keyed by a PFR id and player_games by a GSIS id, so a direct
join matches zero rows. The crosswalk in features.db (scripts/build_xwalk.py)
bridges them. Nothing is written: the shares are computed on read, so the
polling job and build_db.py are unchanged.
"""
import pandas as pd

from .config import DATA_DIR
from .db import connect

FEATURES_DB = DATA_DIR / "features.db"


def snap_share():
    """One row per (game_id, player_id) with offense_pct as snap_share."""
    import sqlite3
    with connect() as con:
        snaps = pd.read_sql_query(
            "SELECT game_id, pfr_player_id, offense_pct FROM snap_counts", con)
    fcon = sqlite3.connect(f"file:{FEATURES_DB}?mode=ro", uri=True)
    try:
        xw = pd.read_sql_query(
            "SELECT gsis_id AS player_id, pfr_id FROM player_xwalk "
            "WHERE gsis_id IS NOT NULL AND pfr_id IS NOT NULL", fcon)
    finally:
        fcon.close()
    xw = xw.drop_duplicates("pfr_id")
    s = snaps.merge(xw, left_on="pfr_player_id", right_on="pfr_id", how="inner")
    return (s.rename(columns={"offense_pct": "snap_share"})
             [["game_id", "player_id", "snap_share"]]
             .drop_duplicates(["game_id", "player_id"]))


def player_game_shares(first_season=2019):
    """player_games (REG) plus snap_share, rz10_carry_share, rz10_target_share
    and rush_share (player rushing yards / team rushing yards)."""
    with connect() as con:
        pg = pd.read_sql_query(
            "SELECT * FROM player_games WHERE season_type='REG' AND season >= ?",
            con, params=[first_season])
        rz = pd.read_sql_query(
            "SELECT game_id, team, player_id, rz10_carries, rz10_targets FROM rz_usage", con)
    pg = pg.merge(snap_share(), on=["game_id", "player_id"], how="left")
    rz = rz.groupby(["game_id", "team", "player_id"], as_index=False).sum(numeric_only=True)
    tot = rz.groupby(["game_id", "team"])[["rz10_carries", "rz10_targets"]].transform("sum")
    rz["rz10_carry_share"] = rz.rz10_carries / tot.rz10_carries.replace(0, float("nan"))
    rz["rz10_target_share"] = rz.rz10_targets / tot.rz10_targets.replace(0, float("nan"))
    pg = pg.merge(rz[["game_id", "player_id", "rz10_carry_share", "rz10_target_share"]],
                  on=["game_id", "player_id"], how="left")
    # Inside-10 share is 0 when the player had no inside-10 touch, not unknown.
    pg[["rz10_carry_share", "rz10_target_share"]] = pg[["rz10_carry_share", "rz10_target_share"]].fillna(0.0)
    team_rush = pg.groupby(["game_id", "team"]).rushing_yards.transform("sum")
    pg["rush_share"] = pg.rushing_yards / team_rush.where(team_rush > 0)
    return pg


def snap_coverage(first_season=2019, last_season=2025):
    """Share of skill-player rows (with an offensive touch) that have a snap share."""
    pg = player_game_shares(first_season)
    pg = pg[(pg.season <= last_season) &
            ((pg.attempts.fillna(0) + pg.carries.fillna(0) + pg.targets.fillna(0)) > 0)]
    return float(pg.snap_share.notna().mean()), len(pg)
