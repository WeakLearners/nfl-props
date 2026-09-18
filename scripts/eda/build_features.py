#!/usr/bin/env python
"""Phase 1: build one lagged, leakage-safe feature table, one row per
player-game, cached to data/eda_features.parquet.

Leakage rule enforced throughout: every feature for a target row (player,
season, week) is built ONLY from that player's/team's/defense's games with
(season, week) strictly earlier than the target's. Schedule/injury features
for the target week itself are the one deliberate exception (see CLAUDE
instructions: those are known pre-kickoff).

Decay convention matches nflprops/project.py: DECAY=0.85 per game (most
recent prior game = weight 1.0), PRIOR_SEASON_W=0.55 flat multiplier applied
once a historical game's season is behind the target's season. MIN_GAMES=3
history games required before a row is kept.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent.parent))
import numpy as np
import pandas as pd
from nflprops.db import connect

DECAY = 0.85
PRIOR_SEASON_W = 0.55
MIN_GAMES = 3

OUT_PATH = pathlib.Path(__file__).resolve().parent.parent.parent / "data" / "eda_features.parquet"


def load_base():
    with connect() as con:
        pg = pd.read_sql_query(
            "SELECT * FROM player_games WHERE season_type='REG'", con)
        sc = pd.read_sql_query(
            "SELECT game_id, team, player, offense_snaps, offense_pct FROM snap_counts", con)
        rz = pd.read_sql_query("SELECT * FROM rz_usage", con)
        sched = pd.read_sql_query(
            """SELECT game_id, season, week, away_team, home_team, spread_line,
                      total_line, div_game, roof, surface, temp, wind,
                      away_rest, home_rest, weekday, gametime
               FROM schedules""", con)
        inj = pd.read_sql_query(
            "SELECT season, week, team, gsis_id, position, report_status FROM injuries", con)
    return pg, sc, rz, sched, inj


def attach_snaps(pg, sc):
    m = pg.merge(sc, left_on=["game_id", "team", "player_display_name"],
                 right_on=["game_id", "team", "player"], how="left")
    return m.drop(columns=["player"])


def attach_rz(pg, rz):
    rz2 = rz.drop(columns=["season", "week", "team"])
    return pg.merge(rz2, on=["game_id", "player_id"], how="left")


def decayed_history_features(pg):
    """For every row, compute decayed-history aggregates from that player's
    STRICTLY PRIOR games only. O(n^2) inside each player's own game list
    (<=~140 games), fine at 43k total rows."""
    pg = pg.sort_values(["player_id", "season", "week"]).reset_index(drop=True)

    num_cols = ["attempts", "completions", "passing_yards", "passing_epa", "passing_cpoe",
                "carries", "rushing_yards", "rushing_epa",
                "targets", "receptions", "receiving_yards", "receiving_epa",
                "receiving_air_yards", "receiving_yards_after_catch", "racr",
                "target_share", "air_yards_share", "wopr",
                "offense_pct", "offense_snaps",
                "rz_carries", "rz10_carries", "rz_targets", "rz10_targets"]
    for c in num_cols:
        if c not in pg.columns:
            pg[c] = np.nan

    out_rows = []
    for pid, g in pg.groupby("player_id", sort=False):
        g = g.reset_index(drop=True)
        n = len(g)
        seasons = g["season"].values
        vals = {c: g[c].fillna(0.0).values for c in num_cols}
        offense_pct_hist = g["offense_pct"].values  # keep NaN for trend calc
        for i in range(n):
            target_season = seasons[i]
            if i < MIN_GAMES:
                continue
            # history = rows 0..i-1, most recent = i-1
            order = np.arange(i - 1, -1, -1)  # 0 = most recent
            decay_w = DECAY ** order
            season_mult = np.where(seasons[:i] < target_season, PRIOR_SEASON_W, 1.0)
            w = decay_w * season_mult
            wsum = w.sum()
            rec = {"player_id": pid, "_row_idx": g.loc[i, "_row_idx"], "n_games_hist": i}

            def dsum(col):
                return float((vals[col][:i] * w).sum())

            def davg(col):
                return dsum(col) / wsum if wsum > 0 else np.nan

            att, cmp_, pyd = dsum("attempts"), dsum("completions"), dsum("passing_yards")
            car, ryd = dsum("carries"), dsum("rushing_yards")
            tgt, rec_n, recyd = dsum("targets"), dsum("receptions"), dsum("receiving_yards")

            rec["h_attempts"] = davg("attempts")
            rec["h_carries"] = davg("carries")
            rec["h_targets"] = davg("targets")
            rec["h_ypa"] = pyd / att if att > 0 else np.nan
            rec["h_cmp_rate"] = cmp_ / att if att > 0 else np.nan
            rec["h_ypc"] = ryd / car if car > 0 else np.nan
            rec["h_ypt"] = recyd / tgt if tgt > 0 else np.nan
            rec["h_catch_rate"] = rec_n / tgt if tgt > 0 else np.nan
            rec["h_passing_epa_pa"] = dsum("passing_epa") / att if att > 0 else np.nan
            rec["h_rushing_epa_pc"] = dsum("rushing_epa") / car if car > 0 else np.nan
            rec["h_receiving_epa_pt"] = dsum("receiving_epa") / tgt if tgt > 0 else np.nan
            rec["h_cpoe"] = davg("passing_cpoe")
            rec["h_target_share"] = davg("target_share")
            rec["h_air_yards_share"] = davg("air_yards_share")
            rec["h_wopr"] = davg("wopr")
            rec["h_racr"] = davg("racr")
            ay = dsum("receiving_air_yards")
            rec["h_adot"] = ay / tgt if tgt > 0 else np.nan
            yac = dsum("receiving_yards_after_catch")
            rec["h_yac_per_rec"] = yac / rec_n if rec_n > 0 else np.nan
            rec["h_snap_pct"] = davg("offense_pct")
            # snap trend: avg of last 3 games' offense_pct minus avg of the 3
            # before that (positive = gaining playing time). Needs >=6 history games.
            pct_hist = offense_pct_hist[:i]
            valid = pct_hist[~pd.isna(pct_hist)]
            if len(valid) >= 6:
                rec["h_snap_trend"] = float(valid[-3:].mean() - valid[-6:-3].mean())
            else:
                rec["h_snap_trend"] = np.nan
            games_n = i
            rec["h_rz_carries_pg"] = dsum("rz_carries") / games_n if games_n > 0 else np.nan
            rec["h_rz10_carries_pg"] = dsum("rz10_carries") / games_n if games_n > 0 else np.nan
            rec["h_rz_targets_pg"] = dsum("rz_targets") / games_n if games_n > 0 else np.nan
            rec["h_rz10_targets_pg"] = dsum("rz10_targets") / games_n if games_n > 0 else np.nan
            out_rows.append(rec)
    return pd.DataFrame(out_rows)


def team_baselines(pg):
    """Decayed team pass/rush volume per game, computed the same leakage-safe
    way, keyed to (team, season, week) so every player on that team/week
    shares it."""
    tg = (pg.groupby(["season", "week", "game_id", "team"], as_index=False)
            .agg(team_targets=("targets", "sum"), team_attempts=("attempts", "sum"),
                 team_carries=("carries", "sum")))
    tg = tg.sort_values(["team", "season", "week"]).reset_index(drop=True)
    out = []
    for team, g in tg.groupby("team", sort=False):
        g = g.reset_index(drop=True)
        n = len(g)
        seasons = g["season"].values
        tv = {c: g[c].values for c in ["team_targets", "team_attempts", "team_carries"]}
        for i in range(n):
            if i == 0:
                continue
            order = np.arange(i - 1, -1, -1)
            decay_w = DECAY ** order
            season_mult = np.where(seasons[:i] < seasons[i], PRIOR_SEASON_W, 1.0)
            w = decay_w * season_mult
            wsum = w.sum()
            rec = {"season": g.loc[i, "season"], "week": g.loc[i, "week"], "team": team}
            for c in tv:
                rec[c + "_base"] = float((tv[c][:i] * w).sum() / wsum) if wsum > 0 else np.nan
            out.append(rec)
    return pd.DataFrame(out)


def defense_vs_position_asof(pg, lookback_seasons=2):
    """Same shrinkage logic as nflprops/features.py::defense_vs_position, but
    precomputed for every (season, week) present in the data at once instead
    of one DB round-trip per row."""
    SHRINK_K = 4.0
    ALLOWED = {
        "QB": ["passing_yards", "attempts", "completions"],
        "RB": ["rushing_yards", "carries", "receptions", "receiving_yards"],
        "WR": ["receiving_yards", "receptions", "targets"],
        "TE": ["receiving_yards", "receptions", "targets"],
    }
    results = {}
    combos = pg[["season", "week"]].drop_duplicates().sort_values(["season", "week"])
    for season, week in combos.itertuples(index=False):
        hist = pg[(pg.season < season) | ((pg.season == season) & (pg.week < week))]
        hist = hist[hist.season >= season - lookback_seasons]
        if hist.empty:
            continue
        for pos, stats in ALLOWED.items():
            sub = hist[hist.position == pos]
            if sub.empty:
                continue
            per_game = sub.groupby("opponent_team").agg(
                games=("game_id", "nunique"), **{s: (s, "sum") for s in stats})
            for s in stats:
                per_game[s] = per_game[s] / per_game["games"]
            league = per_game[stats].mean()
            n = per_game["games"]
            for s in stats:
                ratio = per_game[s] / league[s]
                per_game[s] = (n * ratio + SHRINK_K * 1.0) / (n + SHRINK_K)
            for defteam, row in per_game.iterrows():
                results[(season, week, pos, defteam)] = {s: row[s] for s in stats}
    return results


def implied_totals(sched):
    sched = sched.copy()
    sched["home_implied"] = sched.total_line / 2 - sched.spread_line / 2
    sched["away_implied"] = sched.total_line / 2 + sched.spread_line / 2
    return sched


def main():
    pg, sc, rz, sched, inj = load_base()
    pg = attach_snaps(pg, sc)
    pg = attach_rz(pg, rz)
    pg["_row_idx"] = np.arange(len(pg))

    print("computing decayed player history features (leakage-safe)...")
    hist_feat = decayed_history_features(pg)
    print(f"  -> {len(hist_feat)} rows with n_games_hist >= {MIN_GAMES}")

    print("computing team volume baselines...")
    team_base = team_baselines(pg)

    print("computing defense-vs-position multipliers (as-of each week)...")
    dvp = defense_vs_position_asof(pg)

    df = pg.merge(hist_feat, on=["player_id", "_row_idx"], how="inner")
    df = df.merge(team_base, on=["season", "week", "team"], how="left")

    sched_it = implied_totals(sched)
    df = df.merge(sched_it, on=["game_id", "season", "week"], how="left")
    df["is_home"] = (df.team == df.home_team).astype(int)
    df["implied_team_total"] = np.where(df.is_home == 1, df.home_implied, df.away_implied)
    df["rest_days"] = np.where(df.is_home == 1, df.home_rest, df.away_rest)

    inj_t = inj.rename(columns={"gsis_id": "player_id"})
    df = df.merge(inj_t[["season", "week", "player_id", "report_status"]],
                  on=["season", "week", "player_id"], how="left")
    df["own_questionable"] = (df.report_status == "Questionable").astype(int)
    df["own_status_missing"] = df.report_status.isna().astype(int)  # not on report at all

    # --- vacated usage: count of teammates at same position (or same team,
    # broader) listed Out/Doubtful for the target week, from the injury report
    # for THAT week (known pre-kickoff). ---
    out_players = inj[inj.report_status.isin(["Out", "Doubtful"])][
        ["season", "week", "team", "gsis_id", "position"]].rename(columns={"gsis_id": "out_player_id"})
    same_pos_out = (out_players.groupby(["season", "week", "team", "position"])
                    .size().reset_index(name="teammates_out_same_pos"))
    df = df.merge(same_pos_out, on=["season", "week", "team", "position"], how="left")
    df["teammates_out_same_pos"] = df["teammates_out_same_pos"].fillna(0)
    # exclude self from the count if the player themself is on the out list
    # (shouldn't happen since player_games only has players who played, but be safe)
    team_out_total = out_players.groupby(["season", "week", "team"]).size().reset_index(name="team_out_total")
    df = df.merge(team_out_total, on=["season", "week", "team"], how="left")
    df["team_out_total"] = df["team_out_total"].fillna(0)

    # --- defense-vs-position multiplier lookup ---
    def_cols = {}
    for idx, row in df[["season", "week", "position", "opponent_team"]].drop_duplicates().iterrows():
        pass  # placeholder, real lookup done vectorized below
    keys = list(zip(df.season, df.week, df.position, df.opponent_team))
    stat_names = set()
    for v in dvp.values():
        stat_names.update(v.keys())
    for s in stat_names:
        df[f"dvp_{s}"] = [dvp.get(k, {}).get(s, np.nan) for k in keys]

    before = len(df)
    df = df.drop_duplicates(subset=["player_id", "game_id"], keep="first")
    if len(df) != before:
        print(f"  dropped {before - len(df)} duplicate player-game row(s) "
              f"(snap_counts fan-out on name match)")

    df.to_parquet(OUT_PATH, index=False)
    print(f"wrote {len(df)} rows x {df.shape[1]} cols -> {OUT_PATH}")


if __name__ == "__main__":
    main()
