"""
Phase 2B, Job 1 — the leakage gate, built before any feature.

This is not an audit run after the fact. It is a declared catalog: every
feature the training table is allowed to carry gets one row here, with the
expression that fixes *when* it was known (`observed_at`) and whether that
expression clears kickoff. `build_training_table.py` reads this table back
and refuses to write any column that does not have a passing row here by
exact name match — that is what makes this a gate rather than documentation.

Two rules the gate cannot check by inspecting a single row, so they are
enforced procedurally in build_training_table.py and simply asserted here:

1. Every rolling window is shifted by one game (last-4 for week 9 = weeks
   5-8, never week 9). Enforced with pandas .shift(1) before .rolling(),
   for every player/team/defense rolling feature.
2. A forecast is available; a measurement is not. `weather` rows with
   value_kind='measured' are recorded here and marked FAIL — their
   observed_at (valid_time) sits at kickoff, not before it. Only
   value_kind='forecast' rows pass.

Read-only against nflprops.db. Writes only `feature_registry` in
features.db. Single-threaded, timeout=60.
"""

import sqlite3
from pathlib import Path

FEATURES_DB = Path.home() / "Code/nfl-props/data/features.db"

# Each row: (feature_name, layer, source_table, observed_at_expr, shift_games,
#            status, reason)
#
# layer: 'usage' | 'conversion' | 'context' | 'meta'
# status: 'PASS' | 'FAIL'
# shift_games: games of lag baked into the rolling window before it is used
#              (1 for anything computed from "prior games"; NULL for
#              same-game context that is fixed before kickoff by nature,
#              e.g. a posted spread)

FEATURES = [
    # ---------------- usage (layer 1) ----------------
    ("team_snap_share_l1", "usage", "snap_counts (via player_xwalk)",
     "kickoff_utc of the single prior game used, shift(1) before window(1)", 1, "PASS", None),
    ("team_snap_share_l4", "usage", "snap_counts (via player_xwalk)",
     "kickoff_utc of prior games used, shift(1) before window(4)", 1, "PASS", None),
    ("team_snap_share_l8", "usage", "snap_counts (via player_xwalk)",
     "kickoff_utc of prior games used, shift(1) before window(8)", 1, "PASS", None),
    ("team_snap_share_career", "usage", "snap_counts (via player_xwalk)",
     "kickoff_utc of all prior games, shift(1) before expanding mean", 1, "PASS", None),

    ("team_target_share_l1", "usage", "player_games.target_share",
     "kickoff_utc of the single prior game used, shift(1) before window(1)", 1, "PASS", None),
    ("team_target_share_l4", "usage", "player_games.target_share",
     "kickoff_utc of prior games used, shift(1) before window(4)", 1, "PASS", None),
    ("team_target_share_l8", "usage", "player_games.target_share",
     "kickoff_utc of prior games used, shift(1) before window(8)", 1, "PASS", None),
    ("team_target_share_career", "usage", "player_games.target_share",
     "kickoff_utc of all prior games, shift(1) before expanding mean", 1, "PASS", None),

    ("team_carry_share_l1", "usage", "player_games.carries / team-game sum(carries)",
     "kickoff_utc of the single prior game used, shift(1) before window(1)", 1, "PASS", None),
    ("team_carry_share_l4", "usage", "player_games.carries / team-game sum(carries)",
     "kickoff_utc of prior games used, shift(1) before window(4)", 1, "PASS", None),
    ("team_carry_share_l8", "usage", "player_games.carries / team-game sum(carries)",
     "kickoff_utc of prior games used, shift(1) before window(8)", 1, "PASS", None),
    ("team_carry_share_career", "usage", "player_games.carries / team-game sum(carries)",
     "kickoff_utc of all prior games, shift(1) before expanding mean", 1, "PASS", None),

    ("team_route_share_l4", "usage", "none — no routes table ingested", "n/a", None, "FAIL",
     "Decision #4: load_participation() cannot serve 2026. No routes-run table exists in "
     "features.db or nflprops.db. The spec names this family; no source backs it. Excluded, "
     "not approximated. Study A (Job 6, out of scope here) is the only sanctioned use of routes."),

    ("depth_rank", "usage", "depth_charts",
     "snapshot rows: depth_charts.observed_at (dt of fetch), must be < kickoff_utc. "
     "weekly rows (2019-2024): no per-row timestamp exists; approximated as pre-kickoff by "
     "construction (a week-keyed depth chart describes the week ahead of it) — documented "
     "assumption, not a measured guarantee.", 0, "PASS",
     "Weekly-schema rows pass by assumption, not by measured timestamp. See phase2b report."),

    ("position_group", "usage", "player_games.position_group", "fixed attribute, not time-varying",
     None, "PASS", None),

    ("snap_share_rank_in_group", "usage", "snap_counts, ranked within team+position_group",
     "built from the same shift(1) snap-share values as team_snap_share_l4", 1, "PASS", None),

    ("injury_report_status", "usage", "injuries.report_status",
     "injuries.date_modified of the most recent report strictly before kickoff_utc", 0, "PASS", None),

    ("games_missed_last4", "usage", "player_games presence vs. team schedule",
     "counted over the 4 team games strictly before this kickoff_utc", 1, "PASS", None),

    ("snap_share_trend_last3", "usage", "snap_counts, shift(1) diff over 3 games",
     "kickoff_utc of prior games used, shift(1) before diff(3)", 1, "PASS", None),

    ("team_plays_per_game_l1", "usage", "player_games, team-game sum(attempts+carries+sacks_suffered)",
     "kickoff_utc of prior team games used, shift(1) before window(1)", 1, "PASS", None),
    ("team_plays_per_game_l4", "usage", "player_games, team-game sum(attempts+carries+sacks_suffered)",
     "kickoff_utc of prior team games used, shift(1) before window(4)", 1, "PASS", None),
    ("team_plays_per_game_l8", "usage", "player_games, team-game sum(attempts+carries+sacks_suffered)",
     "kickoff_utc of prior team games used, shift(1) before window(8)", 1, "PASS", None),

    ("team_pass_rate_l1", "usage", "player_games, team-game attempts / plays",
     "kickoff_utc of prior team games used, shift(1) before window(1)", 1, "PASS", None),
    ("team_pass_rate_l4", "usage", "player_games, team-game attempts / plays",
     "kickoff_utc of prior team games used, shift(1) before window(4)", 1, "PASS", None),
    ("team_pass_rate_l8", "usage", "player_games, team-game attempts / plays",
     "kickoff_utc of prior team games used, shift(1) before window(8)", 1, "PASS", None),

    ("team_seconds_per_play", "usage", "none — no play-clock/game-duration table ingested", "n/a",
     None, "FAIL",
     "Spec names this family. Neither nflprops.db nor features.db carries drive length, "
     "game duration, or a play-by-play timestamp. Not derivable from player_games or "
     "schedules. Excluded rather than fabricated."),

    # ---------------- game context (shared by both layers) ----------------
    ("spread_line_team", "context", "schedules.spread_line, signed to this row's team",
     "posted pregame line; verified available before kickoff (decisions.md, 16/16 spot check)",
     None, "PASS", None),
    ("total_line", "context", "schedules.total_line",
     "posted pregame line, available before kickoff", None, "PASS", None),
    ("is_home", "context", "schedules.home_team/away_team", "fixed by schedule", None, "PASS", None),
    ("rest_days", "context", "schedules.home_rest/away_rest", "fixed by schedule", None, "PASS", None),
    ("roof_state", "context", "schedules.roof, falls back to stadiums.roof_type",
     "fixed by schedule/stadium; known pregame", None, "PASS", None),
    ("surface", "context", "schedules.surface, falls back to stadiums.surface",
     "fixed by schedule/stadium; known pregame", None, "PASS", None),

    ("weather_temp_f", "context", "weather (value_kind='forecast')",
     "weather.fetched_at, 2.5-3.5h before kickoff_utc by fetcher design", 0, "PASS", None),
    ("weather_wind_mph", "context", "weather (value_kind='forecast')",
     "weather.fetched_at, 2.5-3.5h before kickoff_utc by fetcher design", 0, "PASS", None),
    ("weather_wind_gust_mph", "context", "weather (value_kind='forecast')",
     "weather.fetched_at, 2.5-3.5h before kickoff_utc by fetcher design", 0, "PASS", None),
    ("weather_precip_prob_pct", "context", "weather (value_kind='forecast')",
     "weather.fetched_at, 2.5-3.5h before kickoff_utc by fetcher design", 0, "PASS", None),
    ("weather_precip_amt_in", "context", "weather (value_kind='forecast')",
     "weather.fetched_at, 2.5-3.5h before kickoff_utc by fetcher design", 0, "PASS", None),
    ("weather_humidity_pct", "context", "weather (value_kind='forecast')",
     "weather.fetched_at, 2.5-3.5h before kickoff_utc by fetcher design", 0, "PASS", None),
    ("weather_no_weather_flag", "context", "roof_state derived",
     "dome/closed-roof games: state, not a missing forecast", None, "PASS", None),

    ("weather_measured_*", "context", "weather (value_kind='measured')",
     "weather.valid_time ~= kickoff_utc, i.e. at or after kickoff", None, "FAIL",
     "Standing rule #1 / phase2b rule 2: a measurement is not available pregame. These rows "
     "exist only to score forecast error later (Phase 3+). Never a training feature."),

    # ---------------- conversion (layer 2) ----------------
    ("yards_per_target_l4", "conversion", "player_games.receiving_yards/targets",
     "kickoff_utc of prior games used, shift(1) before window(4)", 1, "PASS", None),
    ("yards_per_target_l8", "conversion", "player_games.receiving_yards/targets",
     "kickoff_utc of prior games used, shift(1) before window(8)", 1, "PASS", None),
    ("yards_per_target_career", "conversion", "player_games.receiving_yards/targets",
     "kickoff_utc of all prior games, shift(1) before expanding mean", 1, "PASS", None),

    ("yards_per_carry_l4", "conversion", "player_games.rushing_yards/carries",
     "kickoff_utc of prior games used, shift(1) before window(4)", 1, "PASS", None),
    ("yards_per_carry_l8", "conversion", "player_games.rushing_yards/carries",
     "kickoff_utc of prior games used, shift(1) before window(8)", 1, "PASS", None),
    ("yards_per_carry_career", "conversion", "player_games.rushing_yards/carries",
     "kickoff_utc of all prior games, shift(1) before expanding mean", 1, "PASS", None),

    ("catch_rate_l4", "conversion", "player_games.receptions/targets",
     "kickoff_utc of prior games used, shift(1) before window(4)", 1, "PASS", None),
    ("catch_rate_l8", "conversion", "player_games.receptions/targets",
     "kickoff_utc of prior games used, shift(1) before window(8)", 1, "PASS", None),
    ("catch_rate_career", "conversion", "player_games.receptions/targets",
     "kickoff_utc of all prior games, shift(1) before expanding mean", 1, "PASS", None),

    ("adot_l4", "conversion", "player_games.receiving_air_yards/targets",
     "kickoff_utc of prior games used, shift(1) before window(4)", 1, "PASS", None),
    ("adot_l8", "conversion", "player_games.receiving_air_yards/targets",
     "kickoff_utc of prior games used, shift(1) before window(8)", 1, "PASS", None),
    ("adot_career", "conversion", "player_games.receiving_air_yards/targets",
     "kickoff_utc of all prior games, shift(1) before expanding mean", 1, "PASS", None),

    ("rec_td_rate_l4", "conversion", "player_games.receiving_tds/targets",
     "kickoff_utc of prior games used, shift(1) before window(4)", 1, "PASS", None),
    ("rec_td_rate_l8", "conversion", "player_games.receiving_tds/targets",
     "kickoff_utc of prior games used, shift(1) before window(8)", 1, "PASS", None),
    ("rec_td_rate_career", "conversion", "player_games.receiving_tds/targets",
     "kickoff_utc of all prior games, shift(1) before expanding mean", 1, "PASS", None),

    ("rush_td_rate_l4", "conversion", "player_games.rushing_tds/carries",
     "kickoff_utc of prior games used, shift(1) before window(4)", 1, "PASS", None),
    ("rush_td_rate_l8", "conversion", "player_games.rushing_tds/carries",
     "kickoff_utc of prior games used, shift(1) before window(8)", 1, "PASS", None),
    ("rush_td_rate_career", "conversion", "player_games.rushing_tds/carries",
     "kickoff_utc of all prior games, shift(1) before expanding mean", 1, "PASS", None),

    ("opp_yards_allowed_per_target_l4", "conversion",
     "player_games grouped by opponent_team x position_group",
     "defense's own prior weekly aggregates, shift(1) before window(4)", 1, "PASS", None),
    ("opp_yards_allowed_per_carry_l4", "conversion",
     "player_games grouped by opponent_team x position_group",
     "defense's own prior weekly aggregates, shift(1) before window(4)", 1, "PASS", None),

    # ---------------- meta ----------------
    ("history_level", "meta", "derived from player/team/league rolling coverage",
     "computed entirely from each row's own prior-game features", None, "PASS", None),
]


def main():
    con = sqlite3.connect(FEATURES_DB, timeout=60)
    con.execute("""
        CREATE TABLE IF NOT EXISTS feature_registry (
            feature_name  TEXT PRIMARY KEY,
            layer         TEXT NOT NULL,
            source_table  TEXT NOT NULL,
            observed_at   TEXT NOT NULL,
            shift_games   INTEGER,
            status        TEXT NOT NULL CHECK (status IN ('PASS','FAIL')),
            reason        TEXT
        )
    """)
    con.execute("DELETE FROM feature_registry")
    con.executemany(
        "INSERT INTO feature_registry "
        "(feature_name, layer, source_table, observed_at, shift_games, status, reason) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        FEATURES,
    )
    con.commit()

    total = len(FEATURES)
    passed = sum(1 for f in FEATURES if f[5] == "PASS")
    failed = total - passed
    print(f"Wrote {total} feature_registry rows ({passed} PASS, {failed} FAIL) to {FEATURES_DB}")
    print("\nRejected features and why:")
    for f in FEATURES:
        if f[5] == "FAIL":
            print(f"  - {f[0]}: {f[6]}")

    con.close()


if __name__ == "__main__":
    main()
