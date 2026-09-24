"""Rank alternate-line props by how RELIABLE they are, not how +EV.

The premise is different from everything else in this project. We are not
trying to out-price FanDuel -- validated twice, we can't (model AUC 0.495 vs
market 0.543). We take the market's price as the best available probability and
add value in the two places it is weakest for a parlay bettor:

  1. EMPIRICAL FLOOR -- how often has this player actually cleared this number?
     A direct frequency, not a forecast. No mean estimate required.
  2. AVAILABILITY RISK -- the ways a leg dies that have nothing to do with
     yardage: a Questionable tag, an unstable role, a blowout script.

At -200 to -300 the bet is "does he show up and play a normal game", so those
two dominate. Backtest: players carrying a Questionable tag hit 44.1% against a
52.1% implied price -- an 8-point shortfall (n=34, thin, but the mechanism is
not in doubt).
"""
import pandas as pd
import numpy as np

from .db import connect
from .odds import name_key

# Alternate markets are one-sided (Over only), so there is no Under to de-vig
# against. This is the typical two-way hold on FanDuel props, split evenly --
# it converts a quoted price into a rough fair probability. Approximate on
# purpose: it shifts every prop the same way, so the RANKING is unaffected.
# Measured, not assumed: across 777 historical alternate legs priced -180 to
# -320, the raw prices implied 70.5% and the actual hit rate was 62.4%. That is
# an 8-point hold -- nearly double the standard two-way prop market. Alternate
# ladders are where the book takes its margin.
ASSUMED_HOLD = 0.080

MIN_SNAP = 0.30          # a game below this is not evidence about his role

# Starter floor: average share of offensive snaps over the player's last few
# games. Set from validation, not taste -- among top-tercile legs, props on
# players below this hit 70.1% (-0.9% ROI) while those above hit ~77% (+8%).
# The point is not to bet only stars; it is to bet only players who are
# reliably ON THE FIELD, which is what a -250 leg is really wagering on.
STARTER_SNAP_PCT = 0.55
SNAP_LOOKBACK_GAMES = 4
DEFAULT_LOOKBACK = 17    # roughly one season of form


def implied(american):
    a = float(american)
    return (-a) / ((-a) + 100) if a < 0 else 100 / (a + 100)


def fair(american):
    return implied(american) * (1 - ASSUMED_HOLD / 2)


def _history(season, week, lookback):
    """Player-games before the target week, with snap share attached."""
    with connect() as con:
        pg = pd.read_sql_query(
            """SELECT season, week, player_id, player_display_name, position, team,
                      passing_yards, completions, rushing_yards, receptions,
                      receiving_yards, passing_tds, rushing_tds, receiving_tds
               FROM player_games
               WHERE season_type='REG' AND (season < ? OR (season = ? AND week < ?))""",
            con, params=[season, season, week])
        sn = pd.read_sql_query(
            "SELECT season, week, player, offense_pct FROM snap_counts", con)
    sn["player_key"] = sn.player.map(name_key)
    pg["player_key"] = pg.player_display_name.map(name_key)
    pg = pg.merge(sn[["season", "week", "player_key", "offense_pct"]],
                  on=["season", "week", "player_key"], how="left")
    pg = pg.sort_values(["season", "week"], ascending=False)
    return pg.groupby("player_key").head(lookback)


def clear_rate(hist, player_key, stat, line):
    """How often this player actually beat this number, in games he played.

    Games below MIN_SNAP snaps are dropped rather than counted as failures: a
    prop on a player who does not take the field is voided by the book, not
    lost, so counting those as misses would understate reliability. They are
    reported separately as availability risk.
    """
    g = hist[hist.player_key == player_key]
    if g.empty:
        return np.nan, 0, np.nan
    played = g[g.offense_pct.fillna(0) >= MIN_SNAP]
    dnp_rate = 1 - len(played) / len(g)
    if len(played) < 3 or stat not in played:
        return np.nan, len(played), dnp_rate
    return float((played[stat] > line).mean()), len(played), dnp_rate


# "Stars mode": restrict to players who are actually fantasy-relevant at their
# position, and to lines big enough to be worth watching. Costs little in
# reliability -- validation showed high-bar props hit 74.6% in the top tercile,
# identical to low-bar props -- so this is a preference knob, not a trade-off.
# Depth WITHIN a team, not across the league: QB1, RB1/RB2, WR1/2/3, TE1.
# League-wide top-N leaves whole teams with nobody at a position, which is not
# what "the featured guys in this game" means. Rank is by recent usage, since
# published depth charts are loosely maintained.
STAR_DEPTH = {"QB": 1, "RB": 2, "WR": 3, "TE": 1}

# What defines the pecking order at each position.
DEPTH_STAT = {"QB": "attempts", "RB": "carries", "WR": "targets", "TE": "targets"}

# Below these, the bet is not fun to watch even if it is safe.
STAR_MIN_LINE = {"passing_yards": 200.0, "rushing_yards": 30.0,
                 "receiving_yards": 30.0, "receptions": 3.0}


def star_pool(season, week, lookback=DEFAULT_LOOKBACK):
    """The featured players on each team: QB1, RB1/2, WR1/2/3, TE1.

    Depth is measured by recent usage on the player's CURRENT team -- carries
    for backs, targets for pass-catchers, attempts for quarterbacks -- using
    only games before the target week. Usage beats the published depth chart,
    which lists three receivers as starters in a two-receiver formation.
    """
    with connect() as con:
        pg = pd.read_sql_query(
            """SELECT season, week, player_display_name, position, team,
                      attempts, carries, targets
               FROM player_games
               WHERE season_type='REG' AND (season < ? OR (season = ? AND week < ?))""",
            con, params=[season, season, week])
        cur = pd.read_sql_query(
            """SELECT DISTINCT gsis_id, full_name, team AS cur_team, position AS cur_pos
               FROM rosters WHERE season=? AND week=? AND status='ACT'""",
            con, params=[season, week])
        # Whether this player has actually taken the field this season yet
        # (any team). Used below to stop a big career usage number, banked
        # entirely on an old team, from outranking someone who is actually
        # playing right now on the current one.
        played_now = pd.read_sql_query(
            """SELECT DISTINCT player_display_name FROM player_games
               WHERE season = ? AND season_type='REG' AND week < ?""",
            con, params=[season, week])
    pg["player_key"] = pg.player_display_name.map(name_key)
    pg = pg.sort_values(["season", "week"], ascending=False).groupby("player_key").head(lookback)
    played_now_keys = set(played_now.player_display_name.map(name_key))

    cur = cur.dropna(subset=["cur_team"])
    cur["player_key"] = cur.full_name.map(name_key)
    cur = cur[cur.cur_pos.isin(STAR_DEPTH)].drop_duplicates("player_key")

    usage = []
    for pos, col in DEPTH_STAT.items():
        sub = pg[pg.position == pos]
        if sub.empty:
            continue
        # No position column here: the CURRENT roster's position is
        # authoritative, and carrying a second one produced a duplicate column.
        usage.append(sub.groupby("player_key", as_index=False)[col].mean()
                        .rename(columns={col: "usage"}))
    usage = pd.concat(usage, ignore_index=True)

    # Rank on the CURRENT roster, so an offseason move puts a player in his new
    # team's pecking order rather than his old one.
    m = cur.merge(usage, on="player_key", how="left")
    # role_continuity() already flags "changed team" the same way `clean`
    # does downstream in make_report.py -- reuse that exact definition rather
    # than a second one, so this guard and the shortlist screen never
    # disagree about who counts as having changed teams.
    changed = role_continuity(season, week)[["player_key", "changed_team"]].drop_duplicates("player_key")
    m = m.merge(changed, on="player_key", how="left")
    m["changed_team"] = m.changed_team.fillna(False)
    # A player who changed teams AND has zero games this season anywhere is
    # an unconfirmed role -- his usage number describes a job he may not
    # have any more. Zero him out so a teammate who is actually on the field
    # this season ranks into the slot instead, rather than a career-usage
    # number from a different team parking him at "star" with nothing to
    # back it up. Narrow on purpose: a player who changed teams but has
    # already played this season keeps his usage number unchanged, since
    # he's demonstrably playing the role now.
    stale = m.changed_team & ~m.player_key.isin(played_now_keys)
    m.loc[stale, "usage"] = 0.0
    m["usage"] = m.usage.fillna(0.0)
    m["rank"] = m.groupby(["cur_team", "cur_pos"]).usage.rank(ascending=False, method="first")
    keep = m[m.apply(lambda r: r["rank"] <= STAR_DEPTH[r.cur_pos], axis=1)]
    return keep.rename(columns={"cur_team": "team", "cur_pos": "position"})[
        ["player_key", "full_name", "team", "position", "usage", "rank"]]


def snap_share(season, week, lookback=SNAP_LOOKBACK_GAMES):
    """Average offensive snap share over the player's most recent games before
    the target week. This is the operational definition of 'starter' -- NFL
    depth charts are loosely maintained and list three receivers as starters in
    a two-receiver formation, so who is actually on the field beats who is
    listed."""
    with connect() as con:
        sn = pd.read_sql_query(
            """SELECT season, week, player, offense_pct FROM snap_counts
               WHERE season < ? OR (season = ? AND week < ?)""",
            con, params=[season, season, week])
    sn["player_key"] = sn.player.map(name_key)
    sn = sn.sort_values(["season", "week"], ascending=False)
    recent = sn.groupby("player_key").head(lookback)
    return (recent.groupby("player_key").offense_pct.mean()
            .rename("snap_share").reset_index())


def role_continuity(season, week):
    """Flag players whose historical clear rate describes a role they may no
    longer have: a new team, or no games at all in the current season.

    This is the ranker's blind spot and it is worst in Week 1, when every clear
    rate comes from last season. The book prices the player's CURRENT role; the
    clear rate describes his OLD one. Validation ran on Weeks 6/10/14, where
    history is current, so early-season rankings carry risk the backtest never
    measured.
    """
    with connect() as con:
        cur = pd.read_sql_query(
            "SELECT DISTINCT gsis_id AS pid, team FROM rosters WHERE season=? AND week=?",
            con, params=[season, week])
        prev = pd.read_sql_query(
            "SELECT player_id AS pid, team, MAX(season*100+week) AS last_seen "
            "FROM player_games WHERE season < ? GROUP BY player_id",
            con, params=[season])
        names = pd.read_sql_query(
            "SELECT DISTINCT player_id AS pid, player_display_name FROM player_games", con)
    m = cur.merge(prev, on="pid", how="inner", suffixes=("_now", "_then"))
    m = m.merge(names, on="pid", how="left")
    m["player_key"] = m.player_display_name.map(name_key)
    m["changed_team"] = m.team_now != m.team_then
    return m[["player_key", "team_now", "team_then", "changed_team"]]


def risk_flags(season, week):
    """Pre-game knowable hazards, per player and per team."""
    with connect() as con:
        inj = pd.read_sql_query(
            """SELECT gsis_id, full_name, team, report_status, practice_status
               FROM injuries WHERE season=? AND week=?""", con, params=[season, week])
        sched = pd.read_sql_query(
            """SELECT home_team, away_team, spread_line, total_line, roof, wind
               FROM schedules WHERE season=? AND week=?""", con, params=[season, week])
    inj["player_key"] = inj.full_name.map(name_key)
    sp = pd.concat([
        sched.rename(columns={"home_team": "team"})[["team", "spread_line", "total_line", "roof", "wind"]],
        sched.rename(columns={"away_team": "team"})[["team", "spread_line", "total_line", "roof", "wind"]],
    ])
    sp["abs_spread"] = sp.spread_line.abs()
    return inj, sp
