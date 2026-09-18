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

# Vacated usage. When a team-mate at the same position is Out/Doubtful, the
# work he would have taken has to land somewhere, and until 2026-09-17 this
# engine gave it to nobody. Measured against the 2023-2025 walk-forward output,
# the engine's error on RB rushing yards was 9.0% worse in the 611 games where
# a position-mate was out (21.09 MAE vs 19.34). ALPHA damps the redistribution
# because not all vacated work stays inside the position group -- some becomes
# a different play call entirely. Calibrated by scripts/eda/fit_vacated.py.
# MEASURED AND REJECTED 2026-09-17. Walk-forward on 2024-2025, every stat got
# WORSE, and worse the harder the redistribution was pushed: at ALPHA 0.30 the
# error rose 1.8-3.3%, at 0.55 it rose 4.1-12.0%, at 1.00 it rose 7.3-27.1%.
# The premise still holds -- the engine really is 9.0% further off on RB rushing
# in the 611 games where a position-mate is out. Handing that player's volume to
# whoever is left, in proportion to what they already do, is simply the wrong
# fix. It inflates every backup at the position, including the ones who will
# never see the field. Left switched off rather than deleted so nobody proposes
# it again without reading this. Evidence: scripts/eda/sweep.py.
VACATED_ALPHA = 0.30
VACATED_CAP = 1.60  # never more than a 60% volume bump from one absence
USE_VACATED = False

# Red-zone usage as a role signal. Correlates 0.23-0.33 with yardage and only
# 0.30-0.42 with the target-share cluster, so it carries information the
# volume features do not: not how much work a player gets, but how much of it
# comes where the field is short. Enters as a damped multiplier on volume,
# relative to his own position's average rate.
# MEASURED AND REJECTED 2026-09-17, same sweep. Red-zone rate correlates
# 0.23-0.33 with yardage, so it looked like free information. As a multiplier on
# volume it is not: error rose 0.04-0.14% at weight 0.05 and 0.1-0.8% at 0.15,
# every stat, in the same direction. The likely reason is double counting --
# red-zone usage runs 0.30-0.42 correlated with the volume history it is
# adjusting, so the bump re-applies signal the engine already has.
RZ_WEIGHT = 0.05
RZ_CLIP = (0.85, 1.15)
USE_RZ = False

# Calibration. The engine's raw output is biased in a consistent direction, and
# correcting it beats every feature added here: WR receiving yards fell 21.54 ->
# 20.81 MAE on recalibration alone, against 20.73 with every new feature on top.
# Coefficients are actual ~ a + b*proj, fit per stat on completed seasons by
# scripts/eda/fit_calibration.py. They are constants rather than a runtime fit
# because fitting needs past projections, which means re-running the walk-forward.
# REFIT AFTER EVERY SEASON. Fit window is recorded in CALIB_FIT_ON.
USE_CALIB = True
CALIB_FIT_ON = "2023-2025 walk-forward output, fitted 2026-09-17"
CALIB = {
    "completions":     (-0.8345, 1.0133),
    "passing_yards":   (-14.0245, 1.0350),
    "receiving_yards": (-2.0835, 1.0376),
    "receptions":      (-0.1638, 1.0391),
    "rushing_yards":   (-1.6995, 1.0255),
}

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
    roster, absent = _roster(season, week)
    rz_base = _rz_baselines(season, week)

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
    if not rz_base.empty:
        merged = merged.merge(rz_base, on="player_id", how="left")
    for c in ("rz_carries_pg", "rz_targets_pg"):
        if c not in merged:
            merged[c] = np.nan

    # League-average red-zone rate per position, the yardstick a player's own
    # rate is measured against. A rate means nothing without one.
    pos_rz = merged.groupby("position")[["rz_carries_pg", "rz_targets_pg"]].mean().to_dict("index")

    # Vacated usage pools. For each (team, position, volume column): how much
    # baseline volume belongs to players who will not play, and how much
    # belongs to the players who will. The ratio is what gets redistributed.
    vac = {}
    if USE_VACATED and not absent.empty:
        gone = absent.merge(base, on="player_id", how="inner")
        for (team, pos), g_out in gone.groupby(["team", "position"]):
            g_in = merged[(merged.team == team) & (merged.position == pos)]
            if g_in.empty:
                continue
            for stat in POS_STATS.get(pos, []):
                vol_c = SPECS[stat][0]
                col = f"{stat}__vol"
                lost = float(g_out[col].fillna(0).sum())
                held = float(g_in[col].fillna(0).sum())
                if held > 0 and lost > 0:
                    vac[(team, pos, vol_c)] = lost / held

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

            # Red-zone role. A player taking an outsized share of the work near
            # the goal line is the primary option, and primary options get more
            # of everything. Damped hard because red-zone rate already overlaps
            # the volume history it is adjusting.
            if USE_RZ:
                rz_col = {"carries": "rz_carries_pg", "targets": "rz_targets_pg"}.get(vol_c)
                if rz_col:
                    own = p.get(rz_col)
                    avg = (pos_rz.get(p.position) or {}).get(rz_col)
                    if pd.notna(own) and avg and avg > 0:
                        vol *= float(np.clip(1.0 + RZ_WEIGHT * (own / avg - 1.0), *RZ_CLIP))

            # Vacated usage. A position-mate is out, so his share of the work
            # is available. Split it across whoever is left in proportion to
            # what each of them already does.
            if USE_VACATED:
                ratio = vac.get((p.team, p.position, vol_c))
                if ratio:
                    vol *= min(1.0 + VACATED_ALPHA * ratio, VACATED_CAP)

            # Opponent adjustment. The defense's allowed-yardage multiplier
            # already contains its allowed-volume multiplier, so applying both
            # raw would double count. Split them: volume takes the sqrt of its
            # own multiplier (most of "carries allowed" is game script, not
            # defensive quality), efficiency takes what's left over.
            # A missing multiplier means "no information", which is 1.0 -- not
            # NaN. The defence table is a concat of per-position frames, so a
            # position that is not measured on a stat still gets that column,
            # filled with NaN. Reading it blindly wiped every RB receiving
            # projection (190 rows) silently, because 1.0 + (nan-1.0)*0 is nan.
            def _mult(d, col):
                if col not in d:
                    return 1.0
                v = d.get(col)
                try:
                    v = float(v)
                except (TypeError, ValueError):
                    return 1.0
                return 1.0 if pd.isna(v) or v <= 0 else v

            m_vol = m_num = 1.0
            if dvp_ix is not None and (defteam, p.position) in dvp_ix.index:
                d = dvp_ix.loc[(defteam, p.position)]
                m_vol, m_num = _mult(d, d_vol), _mult(d, d_num)
            dw = DEF_WEIGHT.get(stat, 1.0)
            m_vol = 1.0 + (m_vol - 1.0) * dw
            m_num = 1.0 + (m_num - 1.0) * dw
            adj_vol = vol * (m_vol ** 0.5)
            adj_eff = eff * (m_num / m_vol if m_vol > 0 else 1.0)

            mu = adj_vol * adj_eff
            # Calibration. The engine leans in a consistent direction per stat;
            # this is the straight-line correction for that lean, fit offline.
            if USE_CALIB and stat in CALIB:
                a_cal, b_cal = CALIB[stat]
                mu = max(0.0, a_cal + b_cal * mu)
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
