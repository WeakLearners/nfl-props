"""
Job 5 (Phase 2A) — build the `game_spine` table in features.db.

One row per game: game_id, season, week, kickoff_utc, home_team, away_team,
stadium_key. This is what makes standing rule #1 ("nothing available only
after kickoff enters training") a query instead of a judgment call — every
other table can join here and compare its own observed_at to kickoff_utc.

Finding, verified against known kickoff times (see report to orchestrator):
`schedules.gametime` is always US/Eastern, regardless of the stadium's own
timezone — including international games (a 9:30 London kickoff is recorded
as "09:30", which is 9:30am ET / 2:30pm local London time, matching the real
early-window London broadcast slot). So kickoff_utc is computed as
(gameday, gametime) interpreted in America/New_York and converted to UTC.
No per-stadium timezone is needed for this step; the `tz` column on
`stadiums` is kept for Job 4's Open-Meteo calls, not for this conversion.

Read-only against nflprops.db. Writes only `game_spine` in features.db.
"""

import sqlite3
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

LIVE_DB = Path.home() / "Code/nfl-props/data/nflprops.db"
FEATURES_DB = Path.home() / "Code/nfl-props/data/features.db"

ET = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")


def compute_kickoff_utc(gameday, gametime):
    if not gameday or not gametime:
        return None
    try:
        local_dt = datetime.strptime(f"{gameday} {gametime}", "%Y-%m-%d %H:%M").replace(tzinfo=ET)
    except ValueError:
        return None
    return local_dt.astimezone(UTC).strftime("%Y-%m-%d %H:%M:%S")


def main():
    con_live = sqlite3.connect(f"file:{LIVE_DB}?mode=ro", uri=True)

    cur = con_live.execute(
        "SELECT game_id, season, week, gameday, gametime, home_team, away_team, stadium_id "
        "FROM schedules"
    )
    games = cur.fetchall()

    rows = []
    missing_kickoff = []
    for game_id, season, week, gameday, gametime, home_team, away_team, stadium_id in games:
        kickoff_utc = compute_kickoff_utc(gameday, gametime)
        if kickoff_utc is None:
            missing_kickoff.append((game_id, season, week, gameday, gametime))
        rows.append((game_id, season, week, kickoff_utc, home_team, away_team, stadium_id))

    con_feat = sqlite3.connect(FEATURES_DB)
    con_feat.execute("""
        CREATE TABLE IF NOT EXISTS game_spine (
            game_id      TEXT PRIMARY KEY,
            season       INTEGER NOT NULL,
            week         INTEGER NOT NULL,
            kickoff_utc  TEXT,
            home_team    TEXT NOT NULL,
            away_team    TEXT NOT NULL,
            stadium_key  TEXT NOT NULL
        )
    """)
    con_feat.execute("DELETE FROM game_spine")
    con_feat.executemany(
        "INSERT INTO game_spine (game_id, season, week, kickoff_utc, home_team, away_team, stadium_key) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        rows,
    )
    con_feat.commit()

    print(f"Wrote {len(rows)} game_spine rows to {FEATURES_DB}")

    cur = con_feat.execute(
        "SELECT season, COUNT(*), SUM(CASE WHEN kickoff_utc IS NULL THEN 1 ELSE 0 END) "
        "FROM game_spine GROUP BY season ORDER BY season"
    )
    print("season | games | null_kickoff_utc")
    for season, n, nulls in cur.fetchall():
        print(f"{season} | {n} | {nulls}")

    if missing_kickoff:
        print("Games with unresolved kickoff_utc:")
        for g in missing_kickoff:
            print("  ", g)
    else:
        print("No games with unresolved kickoff_utc.")

    con_live.close()
    con_feat.close()


if __name__ == "__main__":
    main()
