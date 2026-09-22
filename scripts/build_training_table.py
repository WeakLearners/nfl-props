"""
Phase 2B, Job 2 — build `training_rows` in features.db.

One row per player-game across 2019-2026, every column backed by a PASS row
in `feature_registry` (Job 1). Population screens, baselines, the split, and
model training are downstream jobs and are NOT done here — every player
goes in.

Design notes that matter for anyone reading this later:

- Every rolling feature is `.shift(1)` before `.rolling(...)`, per player
  (or per team, or per defense) ordered by (season, week). player_games has
  no REG/POST week collisions in any season (checked: max REG week always
  < min POST week that season), so sorting by (season, week) alone is a
  safe chronological order.
- Shares are used wherever the source has them (target_share from
  player_games, offense_pct from snap_counts) rather than counts, per
  standing rule #7.
- Cold start (decision #9): history_level is computed once per row from
  how much of the player's OWN prior-game history exists, then that level's
  fallback (team position-group+depth-rank cross-section, else league
  position-group cross-section) backfills every rolling column that is
  NaN for that row. A row's history_level describes the whole row, not a
  single column.

Read-only against nflprops.db (mode=ro). Writes only `training_rows` to
features.db. Does not touch player_xwalk, depth_charts, stadiums, weather,
or game_spine. Single-threaded, timeout=60.
"""

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

LIVE_DB = Path.home() / "Code/nfl-props/data/nflprops.db"
FEATURES_DB = Path.home() / "Code/nfl-props/data/features.db"

WINDOWS = [1, 4, 8]


def read_sql(query, con):
    return pd.read_sql_query(query, con)


def load():
    live = sqlite3.connect(f"file:{LIVE_DB}?mode=ro", uri=True, timeout=60)
    feat = sqlite3.connect(f"file:{FEATURES_DB}?mode=ro", uri=True, timeout=60)

    pg = read_sql("SELECT * FROM player_games", live)
    snaps = read_sql(
        "SELECT game_id, pfr_player_id, season, week, offense_snaps, offense_pct "
        "FROM snap_counts", live,
    )
    injuries = read_sql(
        "SELECT season, game_type, team, week, gsis_id, report_status, date_modified "
        "FROM injuries", live,
    )
    schedules = read_sql(
        "SELECT game_id, season, week, home_team, away_team, home_rest, away_rest, "
        "spread_line, total_line, roof, surface, stadium_id FROM schedules", live,
    )

    xwalk = read_sql("SELECT gsis_id, pfr_id FROM player_xwalk", feat)
    depth = read_sql(
        "SELECT gsis_id, season, week, team, pos_group, depth_rank, observed_at, source_schema, "
        "game_type FROM depth_charts", feat,
    )
    stadiums = read_sql("SELECT stadium_key, roof_type, surface FROM stadiums", feat)
    weather = read_sql(
        "SELECT game_id, value_kind, fetched_at, temperature_f, wind_speed_mph, "
        "wind_gust_mph, precip_probability_pct, precip_amount_in, humidity_pct FROM weather",
        feat,
    )
    spine = read_sql(
        "SELECT game_id, season, week, kickoff_utc, home_team, away_team, stadium_key "
        "FROM game_spine", feat,
    )

    live.close()
    feat.close()
    return pg, snaps, injuries, schedules, xwalk, depth, stadiums, weather, spine


def shifted_roll(g, col, window, min_periods=1):
    return g[col].shift(1).rolling(window, min_periods=min_periods).mean()


def shifted_expanding(g, col):
    return g[col].shift(1).expanding(min_periods=1).mean()


def build_player_rolling(pg):
    """All player-own-history rolling features. One row per player_id per game,
    every value computed only from that player's STRICTLY earlier games."""
    df = pg.sort_values(["player_id", "season", "week"]).copy()

    # row-level per-game inputs the rolling windows draw from
    df["row_yards_per_target"] = np.where(df["targets"] > 0, df["receiving_yards"] / df["targets"], np.nan)
    df["row_yards_per_carry"] = np.where(df["carries"] > 0, df["rushing_yards"] / df["carries"], np.nan)
    df["row_catch_rate"] = np.where(df["targets"] > 0, df["receptions"] / df["targets"], np.nan)
    df["row_adot"] = np.where(df["targets"] > 0, df["receiving_air_yards"] / df["targets"], np.nan)
    df["row_rec_td_rate"] = np.where(df["targets"] > 0, df["receiving_tds"] / df["targets"], np.nan)
    df["row_rush_td_rate"] = np.where(df["carries"] > 0, df["rushing_tds"] / df["carries"], np.nan)

    grouped = df.groupby("player_id", group_keys=False)

    for w in WINDOWS:
        df[f"team_target_share_l{w}"] = grouped.apply(
            lambda g, w=w: shifted_roll(g, "target_share", w)
        )
    df["team_target_share_career"] = grouped.apply(
        lambda g: shifted_expanding(g, "target_share")
    )

    for col, out in [
        ("row_yards_per_target", "yards_per_target"),
        ("row_yards_per_carry", "yards_per_carry"),
        ("row_catch_rate", "catch_rate"),
        ("row_adot", "adot"),
        ("row_rec_td_rate", "rec_td_rate"),
        ("row_rush_td_rate", "rush_td_rate"),
    ]:
        for w in (4, 8):
            df[f"{out}_l{w}"] = grouped.apply(
                lambda g, col=col, w=w: shifted_roll(g, col, w)
            )
        df[f"{out}_career"] = grouped.apply(
            lambda g, col=col: shifted_expanding(g, col)
        )

    # count of strictly-prior games this player has in the dataset at all
    df["player_prior_games"] = grouped.cumcount()

    return df


def build_snap_share(pg, snaps, xwalk):
    """Join snap_counts -> player_xwalk -> game_id/gsis_id, then compute
    shifted rolling snap-share and the trend feature, per player."""
    snaps = snaps.merge(xwalk, left_on="pfr_player_id", right_on="pfr_id", how="left")
    snaps = snaps.rename(columns={"gsis_id": "player_id"})
    snaps = snaps[["game_id", "player_id", "offense_pct"]].dropna(subset=["player_id"])

    df = pg[["player_id", "game_id", "season", "week"]].merge(
        snaps, on=["game_id", "player_id"], how="left"
    )
    df = df.sort_values(["player_id", "season", "week"])
    grouped = df.groupby("player_id", group_keys=False)

    for w in WINDOWS:
        df[f"team_snap_share_l{w}"] = grouped.apply(
            lambda g, w=w: shifted_roll(g, "offense_pct", w)
        )
    df["team_snap_share_career"] = grouped.apply(
        lambda g: shifted_expanding(g, "offense_pct")
    )

    # trend = last known share minus the share 3 games earlier, both already
    # shift(1)-safe because they're built from the same shift(1) series
    shifted = grouped.apply(lambda g: g["offense_pct"].shift(1))
    df["_shifted_share"] = shifted
    df["snap_share_trend_last3"] = df.groupby("player_id")["_shifted_share"].transform(
        lambda s: s - s.shift(3)
    )
    df = df.drop(columns=["_shifted_share", "offense_pct"])
    return df


def build_carry_share(pg):
    team_game = pg.groupby(["season", "week", "team"], as_index=False)["carries"].sum()
    team_game = team_game.rename(columns={"carries": "team_carries"})
    df = pg[["player_id", "game_id", "season", "week", "team", "carries"]].merge(
        team_game, on=["season", "week", "team"], how="left"
    )
    df["row_carry_share"] = np.where(df["team_carries"] > 0, df["carries"] / df["team_carries"], np.nan)
    df = df.sort_values(["player_id", "season", "week"])
    grouped = df.groupby("player_id", group_keys=False)
    for w in WINDOWS:
        df[f"team_carry_share_l{w}"] = grouped.apply(
            lambda g, w=w: shifted_roll(g, "row_carry_share", w)
        )
    df["team_carry_share_career"] = grouped.apply(
        lambda g: shifted_expanding(g, "row_carry_share")
    )
    return df[["player_id", "game_id"] + [f"team_carry_share_l{w}" for w in WINDOWS] + ["team_carry_share_career"]]


def build_team_pace(pg):
    """Team-level plays/pass-rate, shifted by one game, per team."""
    agg = pg.groupby(["season", "week", "team"], as_index=False).agg(
        attempts=("attempts", "sum"), carries=("carries", "sum"), sacks=("sacks_suffered", "sum")
    )
    agg["plays"] = agg["attempts"] + agg["carries"] + agg["sacks"]
    agg["pass_rate"] = np.where(agg["plays"] > 0, agg["attempts"] / agg["plays"], np.nan)
    agg = agg.sort_values(["team", "season", "week"])
    grouped = agg.groupby("team", group_keys=False)
    for w in WINDOWS:
        agg[f"team_plays_per_game_l{w}"] = grouped.apply(
            lambda g, w=w: shifted_roll(g, "plays", w)
        )
        agg[f"team_pass_rate_l{w}"] = grouped.apply(
            lambda g, w=w: shifted_roll(g, "pass_rate", w)
        )
    keep = ["season", "week", "team"] + [
        f"team_plays_per_game_l{w}" for w in WINDOWS
    ] + [f"team_pass_rate_l{w}" for w in WINDOWS]
    return agg[keep]


def build_opponent_defense(pg):
    """For each (team-as-defense, season, week, position_group): yards
    allowed per target/carry that week. Rolled last-4, shifted by one game,
    per (defense_team, position_group)."""
    agg = pg.groupby(["season", "week", "opponent_team", "position_group"], as_index=False).agg(
        targets=("targets", "sum"), rec_yards=("receiving_yards", "sum"),
        carries=("carries", "sum"), rush_yards=("rushing_yards", "sum"),
    )
    agg = agg.rename(columns={"opponent_team": "defense_team"})
    agg["row_ypt_allowed"] = np.where(agg["targets"] > 0, agg["rec_yards"] / agg["targets"], np.nan)
    agg["row_ypc_allowed"] = np.where(agg["carries"] > 0, agg["rush_yards"] / agg["carries"], np.nan)
    agg = agg.sort_values(["defense_team", "position_group", "season", "week"])
    grouped = agg.groupby(["defense_team", "position_group"], group_keys=False)
    agg["opp_yards_allowed_per_target_l4"] = grouped.apply(
        lambda g: shifted_roll(g, "row_ypt_allowed", 4)
    )
    agg["opp_yards_allowed_per_carry_l4"] = grouped.apply(
        lambda g: shifted_roll(g, "row_ypc_allowed", 4)
    )
    return agg[["season", "week", "defense_team", "position_group",
                "opp_yards_allowed_per_target_l4", "opp_yards_allowed_per_carry_l4"]]


def build_games_missed(pg, schedules):
    """For each player, over the 4 team games strictly before this kickoff,
    how many did the player NOT appear in player_games for."""
    team_games = pd.concat([
        schedules[["season", "week", "home_team"]].rename(columns={"home_team": "team"}),
        schedules[["season", "week", "away_team"]].rename(columns={"away_team": "team"}),
    ]).drop_duplicates().sort_values(["team", "season", "week"])

    played = pg[["player_id", "team", "season", "week"]].drop_duplicates()
    played["played"] = 1

    # every player's team-tenure implied by the games they suited up for;
    # build the player's own team's schedule and mark presence/absence
    players = pg[["player_id", "team", "season"]].drop_duplicates()
    sched_by_team = team_games.merge(players, on=["team", "season"], how="inner")
    sched_by_team = sched_by_team.merge(
        played, on=["player_id", "team", "season", "week"], how="left"
    )
    sched_by_team["played"] = sched_by_team["played"].fillna(0)
    sched_by_team = sched_by_team.sort_values(["player_id", "season", "week"])

    grouped = sched_by_team.groupby("player_id", group_keys=False)
    sched_by_team["games_missed_last4"] = grouped.apply(
        lambda g: g["played"].shift(1).rolling(4, min_periods=1).apply(lambda s: (s == 0).sum())
    )

    return sched_by_team[["player_id", "team", "season", "week", "games_missed_last4"]]


def build_depth_rank(pg, depth, spine):
    """For each player-game, the most recent depth_charts row for that
    player strictly before kickoff. Snapshot rows are filtered on
    observed_at < kickoff_utc; weekly rows (no timestamp) are matched by
    season+week and treated as pre-kickoff by the documented assumption in
    feature_registry."""
    pg2 = pg[["player_id", "game_id", "season", "week", "team"]].merge(
        spine[["game_id", "kickoff_utc"]], on="game_id", how="left"
    )

    weekly = depth[depth["source_schema"] == "weekly"][
        ["gsis_id", "season", "week", "depth_rank", "game_type"]
    ].rename(columns={"gsis_id": "player_id"})
    # depth_charts weekly rows reuse week numbers across REG and POST
    # (2A finding: REG week 19 and WC week 19 both exist for all 32 teams).
    # player_games only carries a coarse REG/POST season_type, so map REG
    # exactly and POST to any postseason game_type. This avoids pulling a
    # REG depth-chart row onto a playoff game or vice versa; it does not
    # guarantee the *round* (WC/DIV/CON/SB) lines up when POST week numbers
    # themselves also collide across rounds -- flagged as a known gap.
    pg2 = pg2.merge(pg[["player_id", "game_id", "season_type"]], on=["player_id", "game_id"], how="left")
    weekly_reg = weekly[weekly["game_type"] == "REG"]
    weekly_post = weekly[weekly["game_type"].isin(["WC", "DIV", "CON", "SB"])]
    reg_match = pg2[pg2["season_type"] == "REG"].merge(
        weekly_reg, on=["player_id", "season", "week"], how="inner"
    )
    post_match = pg2[pg2["season_type"] == "POST"].merge(
        weekly_post, on=["player_id", "season", "week"], how="inner"
    )
    weekly_match = pd.concat([reg_match, post_match])
    weekly_match = weekly_match[["player_id", "game_id", "depth_rank"]]
    weekly_match["_src"] = "weekly"

    snap = depth[depth["source_schema"] == "snapshot"][
        ["gsis_id", "observed_at", "depth_rank"]
    ].dropna(subset=["gsis_id"]).rename(columns={"gsis_id": "player_id"})

    cand = pg2[["player_id", "game_id", "kickoff_utc"]].merge(snap, on="player_id", how="inner")
    cand = cand[cand["observed_at"] < cand["kickoff_utc"]]
    cand = cand.sort_values("observed_at").groupby(["player_id", "game_id"], as_index=False).tail(1)
    snap_match = cand[["player_id", "game_id", "depth_rank"]]
    snap_match["_src"] = "snapshot"

    out = pd.concat([weekly_match, snap_match]).drop_duplicates(subset=["player_id", "game_id"])
    return out[["player_id", "game_id", "depth_rank"]]


def build_injury_status(pg, injuries, spine):
    pg2 = pg[["player_id", "game_id", "season", "week", "team"]].merge(
        spine[["game_id", "kickoff_utc"]], on="game_id", how="left"
    )
    inj = injuries.rename(columns={"gsis_id": "player_id"}).copy()
    inj = inj.dropna(subset=["player_id", "date_modified"])
    inj["season"] = inj["season"].astype(int)
    inj["week"] = inj["week"].astype(int)

    cand = pg2.merge(inj[["player_id", "season", "week", "report_status", "date_modified"]],
                      on=["player_id", "season", "week"], how="inner")
    cand = cand[cand["date_modified"] < cand["kickoff_utc"]]
    cand = cand.sort_values("date_modified").groupby(["player_id", "game_id"], as_index=False).tail(1)
    return cand[["player_id", "game_id", "report_status"]].rename(
        columns={"report_status": "injury_report_status"}
    )


def build_context(schedules, stadiums, spine):
    sched = schedules.merge(spine[["game_id", "stadium_key"]], on="game_id", how="left")
    sched = sched.merge(stadiums, on="stadium_key", how="left", suffixes=("", "_stadium"))
    sched["roof_state"] = sched["roof"].where(sched["roof"].notna(), sched["roof_type"])
    sched["surface_final"] = sched["surface"].where(
        sched["surface"].notna() & (sched["surface"] != ""), sched["surface_stadium"]
    )
    sched["weather_no_weather_flag"] = sched["roof_state"].isin(["dome", "closed"])
    return sched


def build_weather(weather):
    fc = weather[weather["value_kind"] == "forecast"].copy()
    fc = fc.sort_values("fetched_at").groupby("game_id", as_index=False).tail(1)
    fc = fc.rename(columns={
        "temperature_f": "weather_temp_f",
        "wind_speed_mph": "weather_wind_mph",
        "wind_gust_mph": "weather_wind_gust_mph",
        "precip_probability_pct": "weather_precip_prob_pct",
        "precip_amount_in": "weather_precip_amt_in",
        "humidity_pct": "weather_humidity_pct",
    })
    return fc[["game_id", "weather_temp_f", "weather_wind_mph", "weather_wind_gust_mph",
               "weather_precip_prob_pct", "weather_precip_amt_in", "weather_humidity_pct"]]


ROLLING_HISTORY_COLS = (
    [f"team_snap_share_l{w}" for w in WINDOWS] + ["team_snap_share_career"]
    + [f"team_target_share_l{w}" for w in WINDOWS] + ["team_target_share_career"]
    + [f"team_carry_share_l{w}" for w in WINDOWS] + ["team_carry_share_career"]
    + [f"{n}_l{w}" for n in ("yards_per_target", "yards_per_carry", "catch_rate", "adot",
                              "rec_td_rate", "rush_td_rate") for w in (4, 8)]
    + [f"{n}_career" for n in ("yards_per_target", "yards_per_carry", "catch_rate", "adot",
                                "rec_td_rate", "rush_td_rate")]
)


def apply_cold_start(df):
    """decision #9: player history -> team position-group+depth-rank
    cross-section -> league position-group cross-section -> none.
    history_level is one column per row; it governs the fallback applied
    to every column in ROLLING_HISTORY_COLS for that row."""
    df = df.copy()
    has_player_history = df["player_prior_games"] >= 1

    # cross-sectional (same season+week) means among OTHER players who DO
    # have their own history, grouped by team+position_group+depth_rank,
    # and by position_group alone (league). These use only already
    # shift(1)-safe values, so averaging across players at the same week
    # introduces no additional leakage.
    hist_rows = df[has_player_history]

    team_bucket = hist_rows.groupby(
        ["season", "week", "team", "position_group", "depth_rank"]
    )[ROLLING_HISTORY_COLS].transform("mean")
    team_bucket.index = hist_rows.index

    league_bucket = hist_rows.groupby(
        ["season", "week", "position_group"]
    )[ROLLING_HISTORY_COLS].transform("mean")
    league_bucket.index = hist_rows.index

    team_bucket_full = df.groupby(
        ["season", "week", "team", "position_group", "depth_rank"]
    )[ROLLING_HISTORY_COLS].transform(
        lambda s: s[has_player_history.reindex(s.index)].mean()
    )
    league_bucket_full = df.groupby(
        ["season", "week", "position_group"]
    )[ROLLING_HISTORY_COLS].transform(
        lambda s: s[has_player_history.reindex(s.index)].mean()
    )

    history_level = pd.Series("player", index=df.index)
    needs_fallback = ~has_player_history

    for col in ROLLING_HISTORY_COLS:
        col_is_nan = df[col].isna()
        fill_needed = needs_fallback & col_is_nan

        team_vals = team_bucket_full[col]
        use_team = fill_needed & team_vals.notna()
        df.loc[use_team, col] = team_vals[use_team]
        history_level.loc[use_team & (history_level == "player")] = "team_position_depth"

        still_needed = fill_needed & df[col].isna()
        league_vals = league_bucket_full[col]
        use_league = still_needed & league_vals.notna()
        df.loc[use_league, col] = league_vals[use_league]
        history_level.loc[use_league & (history_level == "player")] = "league_position"

    # rows that never got a player value to begin with, and got no fallback
    # value for ANY column either (only possible at the very start of the
    # dataset, before any cross-section exists)
    still_all_nan = needs_fallback & df[ROLLING_HISTORY_COLS].isna().all(axis=1)
    history_level.loc[needs_fallback & (history_level == "player")] = "team_position_depth"
    history_level.loc[still_all_nan] = "none"
    # rows with real player history keep "player"; everything else already
    # set above based on which tier actually filled a value. Rows where
    # `needs_fallback` is True but no column ever got a "player" label
    # collapse correctly because history_level starts at "player" only for
    # has_player_history rows in the first place — fix that seed:
    history_level.loc[has_player_history] = "player"
    history_level.loc[still_all_nan] = "none"

    df["history_level"] = history_level
    return df


def main():
    pg, snaps, injuries, schedules, xwalk, depth, stadiums, weather, spine = load()

    base = pg[["player_id", "player_display_name", "position", "position_group", "season", "week",
               "season_type", "game_id", "team", "opponent_team"]].copy()

    # labels (actuals) -- not features, not gated, this is what Phase 3 predicts
    labels = pg[["player_id", "game_id", "targets", "carries", "receptions",
                 "receiving_yards", "rushing_yards", "receiving_tds", "rushing_tds",
                 "target_share", "air_yards_share", "wopr"]].rename(columns={
        "targets": "y_targets", "carries": "y_carries", "receptions": "y_receptions",
        "receiving_yards": "y_receiving_yards", "rushing_yards": "y_rushing_yards",
        "receiving_tds": "y_receiving_tds", "rushing_tds": "y_rushing_tds",
        "target_share": "y_target_share_row", "air_yards_share": "y_air_yards_share_row",
        "wopr": "y_wopr_row",
    })

    player_roll = build_player_rolling(pg)
    player_roll_cols = ["player_id", "game_id", "player_prior_games"] + [
        c for c in player_roll.columns if c.startswith((
            "team_target_share_l", "team_target_share_career",
            "yards_per_target", "yards_per_carry", "catch_rate", "adot",
            "rec_td_rate", "rush_td_rate",
        ))
    ]
    player_roll = player_roll[player_roll_cols]

    snap_share = build_snap_share(pg, snaps, xwalk)
    carry_share = build_carry_share(pg)
    team_pace = build_team_pace(pg)
    opp_def = build_opponent_defense(pg)
    games_missed = build_games_missed(pg, schedules)
    depth_rank = build_depth_rank(pg, depth, spine)
    injury_status = build_injury_status(pg, injuries, spine)
    context = build_context(schedules, stadiums, spine)
    weather_fc = build_weather(weather)

    df = base.merge(labels, on=["player_id", "game_id"], how="left")
    df = df.merge(player_roll, on=["player_id", "game_id"], how="left")
    df = df.merge(
        snap_share.drop(columns=["season", "week"]), on=["player_id", "game_id"], how="left"
    )
    df = df.merge(carry_share, on=["player_id", "game_id"], how="left")
    df = df.merge(team_pace, on=["season", "week", "team"], how="left")
    df = df.merge(
        opp_def.rename(columns={"defense_team": "opponent_team"}),
        on=["season", "week", "opponent_team", "position_group"], how="left",
    )
    df = df.merge(
        games_missed, on=["player_id", "team", "season", "week"], how="left"
    )
    df = df.merge(depth_rank, on=["player_id", "game_id"], how="left")
    df = df.merge(injury_status, on=["player_id", "game_id"], how="left")

    spine_min = spine[["game_id", "kickoff_utc"]]
    df = df.merge(spine_min, on="game_id", how="left")

    ctx_cols = ["game_id", "home_team", "away_team", "home_rest", "away_rest",
                "spread_line", "total_line", "roof_state", "surface_final",
                "weather_no_weather_flag"]
    df = df.merge(context[ctx_cols], on="game_id", how="left")
    df["is_home"] = (df["team"] == df["home_team"]).astype(int)
    df["rest_days"] = np.where(df["is_home"] == 1, df["home_rest"], df["away_rest"])
    df["spread_line_team"] = np.where(
        df["is_home"] == 1, df["spread_line"], -df["spread_line"]
    )
    df = df.rename(columns={"surface_final": "surface"})
    df = df.drop(columns=["home_team", "away_team", "home_rest", "away_rest", "spread_line"])

    df = df.merge(weather_fc, on="game_id", how="left")
    df["weather_temp_f"] = df["weather_temp_f"].where(~df["weather_no_weather_flag"])
    df["weather_wind_mph"] = df["weather_wind_mph"].where(~df["weather_no_weather_flag"])
    df["weather_wind_gust_mph"] = df["weather_wind_gust_mph"].where(~df["weather_no_weather_flag"])

    # snap_share_rank_in_group: rank of each player's shift(1) snap share
    # among teammates in the same position_group, same season/week
    ss = pg[["player_id", "game_id", "season", "week", "team", "position_group"]].merge(
        snap_share[["player_id", "game_id"] + [f"team_snap_share_l{w}" for w in WINDOWS]],
        on=["player_id", "game_id"], how="left",
    )
    ss["snap_share_rank_in_group"] = ss.groupby(
        ["season", "week", "team", "position_group"]
    )["team_snap_share_l1"].rank(ascending=False, method="min")
    df = df.merge(
        ss[["player_id", "game_id", "snap_share_rank_in_group"]],
        on=["player_id", "game_id"], how="left",
    )

    df = apply_cold_start(df)

    con = sqlite3.connect(FEATURES_DB, timeout=60)
    registry = pd.read_sql_query("SELECT feature_name, status FROM feature_registry", con)
    passing = set(registry[registry["status"] == "PASS"]["feature_name"])

    key_cols = {"player_id", "player_display_name", "position", "position_group", "season",
                "week", "season_type", "game_id", "team", "opponent_team", "kickoff_utc",
                "player_prior_games"}
    label_cols = {c for c in df.columns if c.startswith("y_")}
    feature_cols = [c for c in df.columns if c not in key_cols and c not in label_cols]

    ungated = [c for c in feature_cols if c not in passing]
    if ungated:
        raise RuntimeError(
            f"{len(ungated)} column(s) have no passing feature_registry row and cannot enter "
            f"training_rows: {ungated}"
        )

    df.to_sql("training_rows", con, if_exists="replace", index=False)
    con.commit()

    n_rows, n_cols = df.shape
    print(f"training_rows: {n_rows} rows, {n_cols} columns")
    print(f"seasons: {sorted(df['season'].unique().tolist())}")
    print(f"feature columns (all gated PASS): {len(feature_cols)}")
    print(f"ungated columns rejected: {len(ungated)}")
    print("\nhistory_level distribution:")
    print(df["history_level"].value_counts())

    con.close()


if __name__ == "__main__":
    main()
