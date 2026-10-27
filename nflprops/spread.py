"""The slate frame and the per-player sd, built from the shared primitives only.

slate_rows() gives one row per rostered player and stat with at least MIN_GAMES of history,
the same rows project.project_week() produces, with:
  sd       the player's weighted game-to-game sd, shrunk toward his position's sd on
           K_VAR_GAMES (the formula that sat in project_week, moved here unchanged)
  n_games  games behind the baseline
  base_mu  volume x efficiency, shrunk the same way, with no opponent adjustment and no
           calibration. It is the simple baseline used for completions.
"""
import numpy as np
import pandas as pd

from .base import (K_EFF_ATT, K_SHARE_GAMES, K_VAR_GAMES, K_VOL_GAMES, POS_STATS, SPECS, USE_SHARE,
                   _history, _player_baselines, _position_means, _roster, _team_baselines, _team_targets)
from .db import connect


def slate_rows(season, week):
    hist = _history(season, week)
    if hist.empty:
        raise RuntimeError(f"no history before {season} W{week}")
    base = _player_baselines(hist)
    team_base = _team_baselines(hist)
    pos_mean = _position_means(hist)
    pos_share = (hist.merge(_team_targets(hist), on=["season", "week", "game_id", "team"], how="left")
                     .assign(sh=lambda d: d.targets / d.team_targets.replace(0, np.nan))
                     .groupby("position").sh.mean().to_dict())
    roster, _ = _roster(season, week)
    with connect() as con:
        sched = pd.read_sql_query("SELECT game_id, away_team, home_team FROM schedules WHERE season=? AND week=?",
                                  con, params=[season, week])
    opp = {}
    for r in sched.itertuples():
        opp[r.away_team] = (r.home_team, r.game_id)
        opp[r.home_team] = (r.away_team, r.game_id)
    merged = roster.merge(base, on="player_id", how="inner").merge(team_base, on="team", how="left")
    rows = []
    for _, p in merged.iterrows():
        if p.team not in opp:
            continue
        defteam, game_id = opp[p.team]
        for stat in POS_STATS[p.position]:
            vol_c = SPECS[stat][0]
            pm = pos_mean[p.position][stat]
            att, eff, n = p[f"{stat}__att"], p[f"{stat}__eff"], p["n_games"]
            eff = pm["eff"] if pd.isna(eff) else (att * eff + K_EFF_ATT * pm["eff"]) / (att + K_EFF_ATT)
            if USE_SHARE and vol_c == "targets" and pd.notna(p.get("team_targets_base")):
                sh = (n * p["tgt_share"] + K_SHARE_GAMES * pos_share.get(p.position, 0.12)) / (n + K_SHARE_GAMES)
                vol = p["team_targets_base"] * sh
            else:
                vol = (n * p[f"{stat}__vol"] + K_VOL_GAMES * pm["vol"]) / (n + K_VOL_GAMES)
            sd = p[f"{stat}__sd"]
            sd = pm["sd"] if pd.isna(sd) else (n * sd + K_VAR_GAMES * pm["sd"]) / (n + K_VAR_GAMES)
            rows.append({"game_id": game_id, "player_id": p.player_id, "player": p.full_name,
                         "position": p.position, "team": p.team, "opponent": defteam, "stat": stat,
                         "sd": round(float(sd), 2), "n_games": int(n), "base_mu": float(vol * eff)})
    return pd.DataFrame(rows)
