"""Leakage-safe feature queries.

Every function here takes an (as_of_season, as_of_week) and returns only data
from games strictly BEFORE that point. This is enforced at the SQL level on
purpose: backtest leakage is the single easiest way to build a model that
looks profitable and isn't, and it is very hard to spot once it's buried in
pandas. If a feature needs the target week's data, it does not belong here.
"""
import pandas as pd

from .db import connect

# Games of evidence at which we stop shrinking a split toward league average.
# Roughly "half a season" -- a defense's 3-game sample is mostly schedule.
SHRINK_K = 4.0

_BEFORE = "(season < ? OR (season = ? AND week < ?))"


def _q(sql, params):
    with connect() as con:
        return pd.read_sql_query(sql, con, params=params)


def player_history(player_id, season, week, limit=None):
    """A player's completed games, most recent first."""
    sql = f"""
        SELECT * FROM player_games
        WHERE player_id = ? AND {_BEFORE} AND season_type = 'REG'
        ORDER BY season DESC, week DESC
    """
    params = [player_id, season, season, week]
    if limit:
        sql += " LIMIT ?"
        params.append(limit)
    return _q(sql, params)


def player_vs_team(player_id, opponent, season, week):
    """Career games against one specific opponent -- the 'history vs team'
    input. Usually a tiny sample (1-2 games/yr), so treat it as a weak prior,
    never as a standalone signal."""
    return _q(
        f"""SELECT * FROM player_games
            WHERE player_id = ? AND opponent_team = ? AND {_BEFORE}
              AND season_type = 'REG'
            ORDER BY season DESC, week DESC""",
        [player_id, opponent, season, season, week],
    )


# Stats a defense "allows", by the position it allows them to.
_ALLOWED = {
    "QB": ["passing_yards", "attempts", "completions", "passing_tds"],
    "RB": ["rushing_yards", "carries", "receptions", "receiving_yards", "rushing_tds"],
    "WR": ["receiving_yards", "receptions", "targets", "receiving_tds"],
    "TE": ["receiving_yards", "receptions", "targets", "receiving_tds"],
}


def defense_vs_position(season, week, lookback_seasons=2):
    """Per-game production each defense allows to each position group, shrunk
    toward the league mean.

    Returned as a multiplier (1.0 = league average, 1.15 = allows 15% more
    than average) because that is how the projection model consumes it: an
    adjustment on a player's own baseline, not a replacement for it.
    """
    raw = _q(
        f"""SELECT opponent_team AS defteam, position, game_id,
                   passing_yards, attempts, completions, passing_tds,
                   rushing_yards, carries, rushing_tds,
                   receptions, receiving_yards, targets, receiving_tds
            FROM player_games
            WHERE {_BEFORE} AND season >= ? AND season_type = 'REG'""",
        [season, season, week, season - lookback_seasons],
    )

    out = []
    for pos, stats in _ALLOWED.items():
        sub = raw[raw["position"] == pos]
        if sub.empty:
            continue
        # Sum each defense's allowed production, then divide by games faced.
        per_game = sub.groupby("defteam").agg(
            games=("game_id", "nunique"), **{s: (s, "sum") for s in stats}
        )
        for s in stats:
            per_game[s] = per_game[s] / per_game["games"]
        league = per_game[stats].mean()
        n = per_game["games"]
        for s in stats:
            # Shrink the ratio, not the raw value: a defense with 2 games of
            # evidence lands ~1/3 of the way from league average to its split.
            ratio = per_game[s] / league[s]
            per_game[s] = (n * ratio + SHRINK_K * 1.0) / (n + SHRINK_K)
        per_game["position"] = pos
        out.append(per_game.reset_index())
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def injury_report(season, week, teams=None):
    """The official report for the target week. This is the one place we read
    the current week on purpose -- injury designations are published before
    kickoff, so using them is information we genuinely have at bet time."""
    sql = "SELECT * FROM injuries WHERE season = ? AND week = ?"
    params = [season, week]
    if teams:
        sql += f" AND team IN ({','.join('?' * len(teams))})"
        params += list(teams)
    return _q(sql, params)


def slate(season, week):
    """Games for a week, with the market's game lines attached."""
    return _q(
        """SELECT game_id, season, week, gameday, weekday, gametime,
                  away_team, home_team, spread_line, total_line, roof,
                  surface, temp, wind, away_rest, home_rest, div_game
           FROM schedules WHERE season = ? AND week = ? ORDER BY gameday, gametime""",
        [season, week],
    )


def red_zone_history(player_id, season, week, limit=None):
    sql = f"""SELECT * FROM rz_usage WHERE player_id = ? AND {_BEFORE}
              ORDER BY season DESC, week DESC"""
    params = [player_id, season, season, week]
    if limit:
        sql += " LIMIT ?"
        params.append(limit)
    return _q(sql, params)
