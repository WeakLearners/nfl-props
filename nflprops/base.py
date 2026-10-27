"""Shared data primitives: history, roster, recency-weighted baselines, position means.

These are data helpers, not a projection engine. project.py (the old projection),
tdmodel.py, rating.py, defense_rank.py and the training-table builder all read them.
Moved out of project.py unchanged.
"""
import numpy as np
import pandas as pd

from .db import connect

# Recency weighting. 0.85/game means a game 10 weeks back carries ~20% the
# weight of last week's; prior seasons take a further flat discount on top.
DECAY = 0.85
PRIOR_SEASON_W = 0.55
MIN_GAMES = 3

# Shrinkage denominators. Efficiency shrinks on ATTEMPTS (the actual sample
# behind a rate), volume and variance shrink on GAMES.
K_EFF_ATT = 60.0
K_VOL_GAMES = 3.0
K_VAR_GAMES = 5.0
# Target share stabilises fast -- a receiver's role is visible in a few games.
K_SHARE_GAMES = 4.0
USE_SHARE = True  # three-way receiving decomposition; ablatable

# stat -> (volume column, efficiency numerator, defense-multiplier keys)
SPECS = {
    "passing_yards":   ("attempts", "passing_yards",   "passing_yards", "attempts"),
    "completions":     ("attempts", "completions",     "completions",   "attempts"),
    "rushing_yards":   ("carries",  "rushing_yards",   "rushing_yards", "carries"),
    "receptions":      ("targets",  "receptions",      "receptions",    "targets"),
    "receiving_yards": ("targets",  "receiving_yards", "receiving_yards", "targets"),
}
# How much of the opponent adjustment to apply, per stat. Set from the
# 2023-2025 walk-forward ablation (scripts/backtest.py): team pass-defense
# quality is a real, reasonably stable signal, but "receiving yards allowed to
# WRs" is dominated by which receivers a defense happened to face, and applying
# it cost 12% of MAE. Do not raise these without re-running the backtest.
DEF_WEIGHT = {
    "passing_yards": 1.0,
    "completions": 1.0,
    "rushing_yards": 0.5,
    "receptions": 0.0,
    "receiving_yards": 0.0,
}

POS_STATS = {
    # QB rushing yards was unmodelled until 2026-09-17 even though SPECS has
    # carried a rushing_yards entry all along (shared with RB). The odds feed
    # has been pulling player_rush_yds_alternate the whole time, so QB rushing
    # lines were arriving with nothing to compare them against. Routed through
    # the existing volume x efficiency shape rather than a new regression: a
    # plain h_carries * h_ypc baseline scored 11.68 MAE against 11.84 for a
    # linear feature model, because rushing yards is a product, not a sum.
    "QB": ["passing_yards", "completions", "rushing_yards"],
    "RB": ["rushing_yards", "receptions", "receiving_yards"],
    "WR": ["receptions", "receiving_yards"],
    "TE": ["receptions", "receiving_yards"],
}


def _history(season, week):
    with connect() as con:
        return pd.read_sql_query(
            """SELECT * FROM player_games
               WHERE season_type='REG' AND (season < ? OR (season = ? AND week < ?))""",
            con, params=[season, season, week])


def _roster(season, week):
    with connect() as con:
        # Every status, not just ACT. A player ruled out is marked INA or RES
        # here, so filtering to ACT up front silently discarded exactly the
        # players whose vacated work we need to account for -- and made the
        # injury-report check below dead code for years. Split the frame after
        # the join instead of before it.
        ros = pd.read_sql_query(
            """SELECT gsis_id AS player_id, team, position, full_name, status
               FROM rosters WHERE season=? AND week=?""",
            con, params=[season, week])
        inj = pd.read_sql_query(
            "SELECT gsis_id AS player_id, report_status FROM injuries WHERE season=? AND week=?",
            con, params=[season, week])
    ros = ros[ros.position.isin(POS_STATS)].dropna(subset=["player_id"])
    ros = ros.merge(inj, on="player_id", how="left").drop_duplicates("player_id")
    # Out by either route: the official report says so, or the team has already
    # moved him off the active roster.
    out = ros.report_status.isin(["Out", "Doubtful"]) | ros.status.isin(["INA", "RES"])
    active = ros[(ros.status == "ACT") & ~out]
    # The absent frame is not a leftover. Its baseline volume is the pool that
    # gets redistributed to whoever is still playing.
    absent = ros[out & ros.status.isin(["ACT", "INA", "RES"])]
    return active, absent


def _weights(g):
    """Exponential recency weight, most recent game = 1.0."""
    order = np.arange(len(g))
    w = DECAY ** order
    w = w * np.where(g["season"].values < g["season"].max(), PRIOR_SEASON_W, 1.0)
    return w


def _team_targets(hist):
    """Team pass-target pool per game. This is the denominator target share is
    measured against, and it is what game script actually moves."""
    t = (hist.groupby(["season", "week", "game_id", "team"], as_index=False)
             .agg(team_targets=("targets", "sum")))
    return t


def _team_baselines(hist):
    """Weighted recent pass volume per team. Far more stable than any single
    receiver's target count, which is what makes the share decomposition work."""
    t = _team_targets(hist).sort_values(["season", "week"], ascending=False)
    rows = []
    for team, g in t.groupby("team", sort=False):
        w = _weights(g)
        rows.append({"team": team, "team_targets_base": float((g.team_targets * w).sum() / w.sum())})
    return pd.DataFrame(rows)


def _player_baselines(hist):
    """Per player: weighted volume/game, weighted efficiency, and game-to-game
    sd for each stat."""
    rows = []
    hist = hist.merge(_team_targets(hist), on=["season", "week", "game_id", "team"], how="left")
    hist["tgt_share"] = hist.targets / hist.team_targets.replace(0, np.nan)
    hist = hist.sort_values(["season", "week"], ascending=False)
    for pid, g in hist.groupby("player_id", sort=False):
        if len(g) < MIN_GAMES:
            continue
        w = _weights(g)
        rec = {"player_id": pid, "n_games": len(g), "w_sum": w.sum()}
        for stat, (vol_c, num_c, _, _) in SPECS.items():
            vol = g[vol_c].fillna(0).values
            num = g[num_c].fillna(0).values
            rec[f"{stat}__vol"] = np.average(vol, weights=w)
            tot_v = (vol * w).sum()
            rec[f"{stat}__eff"] = (num * w).sum() / tot_v if tot_v > 0 else np.nan
            rec[f"{stat}__att"] = tot_v
            rec[f"{stat}__sd"] = np.sqrt(np.average((num - np.average(num, weights=w)) ** 2, weights=w))
        sh = g["tgt_share"].fillna(0).values
        rec["tgt_share"] = float(np.average(sh, weights=w))
        rows.append(rec)
    return pd.DataFrame(rows)


def _rz_baselines(season, week):
    """Decayed red-zone carries and targets per game, per player.

    Same recency weighting as everything else, so a player who lost the goal-line
    role in October is not still credited with it in December.
    """
    with connect() as con:
        rz = pd.read_sql_query(
            """SELECT season, week, player_id, rz_carries, rz_targets
               FROM rz_usage WHERE season < ? OR (season = ? AND week < ?)""",
            con, params=[season, season, week])
    if rz.empty:
        return pd.DataFrame(columns=["player_id", "rz_carries_pg", "rz_targets_pg"])
    rz = rz.sort_values(["season", "week"], ascending=False)
    rows = []
    for pid, g in rz.groupby("player_id", sort=False):
        w = _weights(g)
        rows.append({
            "player_id": pid,
            "rz_carries_pg": float(np.average(g.rz_carries.fillna(0), weights=w)),
            "rz_targets_pg": float(np.average(g.rz_targets.fillna(0), weights=w)),
        })
    return pd.DataFrame(rows)


def _position_means(hist):
    """League-average efficiency and dispersion per position, used as the
    shrinkage target for players with thin samples."""
    out = {}
    for pos, stats in POS_STATS.items():
        sub = hist[hist.position == pos]
        out[pos] = {}
        for stat in stats:
            vol_c, num_c, _, _ = SPECS[stat]
            v, n = sub[vol_c].fillna(0), sub[num_c].fillna(0)
            out[pos][stat] = {
                "eff": n.sum() / v.sum() if v.sum() > 0 else 0.0,
                "vol": v.mean(),
                "sd": sub[num_c].std(),
            }
    return out
