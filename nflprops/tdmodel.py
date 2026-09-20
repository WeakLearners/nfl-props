"""Anytime-touchdown model: one feature function, used for training AND serving.

That single sentence is the whole design. The first cut of this model trained
on data/eda_features.parquet, which builds a row for a player-game only after
the game has been played. Scoring an upcoming slate would therefore have
needed a second code path, and a second code path computing "the same"
features is how train/serve skew gets in. Skew does not raise. It returns a
plausible number from slightly wrong inputs, and every check a person
naturally runs -- does this look sensible, is the top pick reasonable --
passes. That is the shape of the 2026-09-13 retraction, and it is not worth
repeating for a marker on a card.

So features() below is called with a (season, week) and knows nothing about
whether those games have been played. For training it is called on past
weeks and joined to what happened. For serving it is called on the upcoming
week and joined to nothing. Identical inputs, identical arithmetic.

Everything it reads comes from nflprops.project's own primitives, so the
decay convention here cannot drift from the projection engine's either.
"""
import numpy as np
import pandas as pd

from .db import connect
from .project import (
    _history, _player_baselines, _rz_baselines, _roster,
)

# Volume columns produced by _player_baselines, renamed to say what they are.
# rushing_yards__vol is carries per game and receiving_yards__vol is targets
# per game -- see SPECS in project.py, where the first element of each tuple
# is the volume column. The rename exists because "rushing_yards__vol" for a
# count of carries is the kind of name that misleads a reader six months on.
VOL = {
    "rushing_yards__vol": "f_carries_pg",
    "receiving_yards__vol": "f_targets_pg",
    "passing_yards__vol": "f_attempts_pg",
}

FEATURES = [
    "f_carries_pg", "f_targets_pg", "f_attempts_pg",
    "f_tgt_share", "f_rz_carries_pg", "f_rz_targets_pg",
    "f_implied_total", "f_spread", "f_total", "f_is_home",
    "f_n_games", "f_teammates_out_pos",
]
POSITIONS = ["QB", "RB", "WR", "TE"]


def _game_context(season, week):
    """Implied team total, spread, total and home flag, per team.

    Known before kickoff -- it is the betting market's own view of the game,
    published days ahead. Nothing here is an outcome.
    """
    with connect() as con:
        s = pd.read_sql_query(
            """SELECT game_id, away_team, home_team, spread_line, total_line
               FROM schedules WHERE season=? AND week=?""",
            con, params=[season, week])
    if s.empty:
        return pd.DataFrame(columns=["team", "opponent", "game_id", "f_implied_total",
                                     "f_spread", "f_total", "f_is_home"])
    # spread_line is quoted from the home team's side. A home favourite by 3
    # has home implied = (total + 3) / 2.
    rows = []
    for r in s.itertuples():
        total = float(r.total_line) if pd.notna(r.total_line) else np.nan
        spread = float(r.spread_line) if pd.notna(r.spread_line) else 0.0
        home_imp = (total + spread) / 2 if pd.notna(total) else np.nan
        away_imp = (total - spread) / 2 if pd.notna(total) else np.nan
        rows.append({"team": r.home_team, "opponent": r.away_team, "game_id": r.game_id,
                     "f_implied_total": home_imp, "f_spread": spread,
                     "f_total": total, "f_is_home": 1})
        rows.append({"team": r.away_team, "opponent": r.home_team, "game_id": r.game_id,
                     "f_implied_total": away_imp, "f_spread": -spread,
                     "f_total": total, "f_is_home": 0})
    return pd.DataFrame(rows)


def features(season, week):
    """One row per rostered, available player for (season, week).

    The universe is the ROSTER, not the box score. That matters twice over.
    At serving time a box score does not exist yet. At training time it makes
    a rostered player who never touched the ball a real zero rather than an
    absent row, which is exactly what he is for an anytime-TD market.
    """
    hist = _history(season, week)
    if hist.empty:
        return pd.DataFrame(columns=["player_id"] + FEATURES)

    base = _player_baselines(hist).rename(columns=VOL)
    # _player_baselines returns a frame with NO COLUMNS when every player is
    # still under MIN_GAMES -- the opening weeks of 2019, and any week early
    # enough that nobody has three games behind them. Merging on a column
    # that is not there raises, so give it the shape the merge expects and
    # let those rows carry NaN features. They are real rows: the players
    # existed and the book priced them. What is missing is history, and NaN
    # is what "no history" should look like.
    if base.empty:
        base = pd.DataFrame(columns=["player_id", "n_games", "tgt_share",
                                     *VOL.values()])
    rz = _rz_baselines(season, week).rename(
        columns={"rz_carries_pg": "f_rz_carries_pg", "rz_targets_pg": "f_rz_targets_pg"})
    active, absent = _roster(season, week)
    if active.empty:
        return pd.DataFrame(columns=["player_id"] + FEATURES)

    df = active[["player_id", "full_name", "position", "team"]].copy()
    df = df.merge(base, on="player_id", how="left")
    df = df.merge(rz, on="player_id", how="left")
    df = df.merge(_game_context(season, week), on="team", how="left")

    # How many same-position team-mates are ruled out. The 2026-09-17 finding
    # was that vacated usage is not a small effect hiding in an average: the
    # engine was 9.0% further off on RB rushing in the 611 games where a
    # position-mate was out. A player inherits goal-line work when the man
    # ahead of him is not playing.
    if not absent.empty:
        out_counts = absent.groupby(["team", "position"]).size().rename("f_teammates_out_pos")
        df = df.merge(out_counts, on=["team", "position"], how="left")
    df["f_teammates_out_pos"] = df.get("f_teammates_out_pos", pd.Series(dtype=float)).fillna(0)

    df["f_tgt_share"] = df.get("tgt_share", np.nan)
    df["f_n_games"] = df.get("n_games", np.nan)
    for c in FEATURES:
        if c not in df.columns:
            df[c] = np.nan

    df["season"], df["week"] = season, week
    keep = ["player_id", "full_name", "position", "team", "opponent", "game_id",
            "season", "week"] + FEATURES
    return df[[c for c in keep if c in df.columns]]


def design_matrix(df):
    """Features plus position one-hots, in a stable column order.

    Stable order matters: a model trained with pos_QB in column 13 and served
    a frame with it in column 11 is wrong in a way nothing reports.
    """
    X = df[FEATURES].astype(float).copy()
    for p in POSITIONS:
        X[f"pos_{p}"] = (df["position"] == p).astype(float)
    return X[FEATURES + [f"pos_{p}" for p in POSITIONS]]


def labels(season, week):
    """Did each player score a rushing or receiving touchdown that week.

    Passing touchdowns are excluded: a quarterback throwing four of them
    scored none of them, and the anytime-TD market prices it the same way.
    """
    with connect() as con:
        pg = pd.read_sql_query(
            """SELECT player_id, rushing_tds, receiving_tds FROM player_games
               WHERE season=? AND week=? AND season_type='REG'""",
            con, params=[season, week])
    if pg.empty:
        return pd.DataFrame(columns=["player_id", "td"])
    pg["td"] = ((pg.rushing_tds.fillna(0) + pg.receiving_tds.fillna(0)) > 0).astype(int)
    return pg[["player_id", "td"]].drop_duplicates("player_id")
