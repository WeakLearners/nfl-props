"""Volume x efficiency projection engine.

The decomposition is the point. A player's yardage is noisy because it is a
product of two things with very different stability: how often he touches the
ball (stable, driven by role) and how much he gets per touch (noisy, driven by
matchup and luck). Projecting them separately and multiplying beats projecting
the product directly, and it is what lets injuries and opponent quality enter
at the place they actually act.
"""
import numpy as np
import pandas as pd

from .db import connect
from .features import defense_vs_position

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
    "QB": ["passing_yards", "completions"],
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
        ros = pd.read_sql_query(
            """SELECT gsis_id AS player_id, team, position, full_name
               FROM rosters WHERE season=? AND week=? AND status='ACT'""",
            con, params=[season, week])
        inj = pd.read_sql_query(
            "SELECT gsis_id AS player_id, report_status FROM injuries WHERE season=? AND week=?",
            con, params=[season, week])
    ros = ros[ros.position.isin(POS_STATS)].dropna(subset=["player_id"])
    ros = ros.merge(inj, on="player_id", how="left")
    # v1 injury handling: drop players who will not play. Redistributing their
    # vacated usage to teammates is the meaningful next step and is NOT done
    # here -- so backup-elevation cases will read low until that lands.
    return ros[~ros.report_status.isin(["Out", "Doubtful"])].drop_duplicates("player_id")


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


def project_week(season, week, verbose=False):
    """Return one row per player-stat with a projected mean and sd."""
    hist = _history(season, week)
    if hist.empty:
        raise RuntimeError(f"no history before {season} W{week}")
    base = _player_baselines(hist)
    team_base = _team_baselines(hist)
    pos_mean = _position_means(hist)
    # League mean target share per position, the shrinkage target for players
    # with few games. A WR with 2 games does not get to claim a 30% share.
    pos_share = (hist.merge(_team_targets(hist), on=["season", "week", "game_id", "team"], how="left")
                     .assign(sh=lambda d: d.targets / d.team_targets.replace(0, np.nan))
                     .groupby("position").sh.mean().to_dict())
    dvp = defense_vs_position(season, week)
    roster = _roster(season, week)

    with connect() as con:
        sched = pd.read_sql_query(
            """SELECT game_id, away_team, home_team, gameday, weekday, gametime,
                      spread_line, total_line FROM schedules WHERE season=? AND week=?""",
            con, params=[season, week])

    # team -> (opponent, game_id)
    opp = {}
    for _, r in sched.iterrows():
        opp[r.away_team] = (r.home_team, r.game_id)
        opp[r.home_team] = (r.away_team, r.game_id)

    dvp_ix = dvp.set_index(["defteam", "position"]) if not dvp.empty else None

    rows = []
    merged = roster.merge(base, on="player_id", how="inner").merge(team_base, on="team", how="left")
    for _, p in merged.iterrows():
        if p.team not in opp:
            continue
        defteam, game_id = opp[p.team]
        for stat in POS_STATS[p.position]:
            vol_c, num_c, d_num, d_vol = SPECS[stat]
            pm = pos_mean[p.position][stat]

            # Shrink efficiency on attempts, volume on games.
            att = p[f"{stat}__att"]
            eff = p[f"{stat}__eff"]
            eff = pm["eff"] if pd.isna(eff) else (att * eff + K_EFF_ATT * pm["eff"]) / (att + K_EFF_ATT)
            n = p["n_games"]
            if USE_SHARE and vol_c == "targets" and pd.notna(p.get("team_targets_base")):
                # Three-way decomposition for receiving: team pass volume x the
                # player's share of it x efficiency. Raw targets/game conflates a
                # stable role with a volatile game script, and projecting it
                # directly was the reason receiving lost to a career average.
                sh = (n * p["tgt_share"] + K_SHARE_GAMES * pos_share.get(p.position, 0.12)) / (n + K_SHARE_GAMES)
                vol = p["team_targets_base"] * sh
            else:
                vol = (n * p[f"{stat}__vol"] + K_VOL_GAMES * pm["vol"]) / (n + K_VOL_GAMES)

            # Opponent adjustment. The defense's allowed-yardage multiplier
            # already contains its allowed-volume multiplier, so applying both
            # raw would double count. Split them: volume takes the sqrt of its
            # own multiplier (most of "carries allowed" is game script, not
            # defensive quality), efficiency takes what's left over.
            m_vol = m_num = 1.0
            if dvp_ix is not None and (defteam, p.position) in dvp_ix.index:
                d = dvp_ix.loc[(defteam, p.position)]
                m_vol = float(d.get(d_vol, 1.0)) if d_vol in d else 1.0
                m_num = float(d.get(d_num, 1.0)) if d_num in d else 1.0
            dw = DEF_WEIGHT.get(stat, 1.0)
            m_vol = 1.0 + (m_vol - 1.0) * dw
            m_num = 1.0 + (m_num - 1.0) * dw
            adj_vol = vol * (m_vol ** 0.5)
            adj_eff = eff * (m_num / m_vol if m_vol > 0 else 1.0)

            mu = adj_vol * adj_eff
            sd = p[f"{stat}__sd"]
            sd = pm["sd"] if pd.isna(sd) else (n * sd + K_VAR_GAMES * pm["sd"]) / (n + K_VAR_GAMES)

            rows.append({
                "game_id": game_id, "player_id": p.player_id, "player": p.full_name,
                "position": p.position, "team": p.team, "opponent": defteam,
                "stat": stat, "proj": round(float(mu), 2), "sd": round(float(sd), 2),
                "proj_vol": round(float(adj_vol), 2), "proj_eff": round(float(adj_eff), 3),
                "def_mult": round(float(m_num), 3), "n_games": int(n),
            })
    return pd.DataFrame(rows)
