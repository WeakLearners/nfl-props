"""SQLite store. One file, no server, trivially backed up and inspected."""
import sqlite3
from contextlib import contextmanager

from .config import DB_PATH, DATA_DIR

# Indexes matter here: every model lookup is "this player, seasons before now"
# or "this defense, seasons before now". Without them the projection loop does
# full scans on a ~100k row table for every prop.
INDEXES = {
    "player_games": [
        ("ix_pg_player", "player_id, season, week"),
        ("ix_pg_opp", "opponent_team, season, week"),
        ("ix_pg_game", "game_id"),
    ],
    "schedules": [("ix_sch_week", "season, week")],
    "injuries": [("ix_inj_player", "gsis_id, season, week")],
    "snap_counts": [("ix_snap_player", "pfr_player_id, season, week")],
    "rz_usage": [("ix_rz_player", "player_id, season, week")],
    "rosters": [("ix_ros", "season, week, team"), ("ix_ros_p", "gsis_id, season, week")],
}


def _has_table(con, name):
    return con.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)
    ).fetchone() is not None


@contextmanager
def connect():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(DB_PATH)
    try:
        yield con
        con.commit()
    finally:
        con.close()


def write_table(df, name, con, seasons=None):
    """Write a table.

    With `seasons`, only those seasons are replaced and the rest of the history
    is left alone. Without it, the whole table is replaced. This distinction is
    not cosmetic: an earlier version always replaced, so refreshing a single
    current season silently deleted every prior year -- and the reliability
    ranker, which reads 17 games of history, went quietly empty rather than
    erroring.
    """
    if seasons and _has_table(con, name) and "season" in df.columns:
        con.execute(f"DELETE FROM {name} WHERE season IN "
                    f"({','.join('?' * len(seasons))})", list(seasons))
        df.to_sql(name, con, if_exists="append", index=False)
    else:
        df.to_sql(name, con, if_exists="replace", index=False)
    for ix_name, cols in INDEXES.get(name, []):
        con.execute(f"CREATE INDEX IF NOT EXISTS {ix_name} ON {name} ({cols})")
    return len(df)


def team_abbr_map():
    """Full team name -> the abbreviation this database actually uses.

    nflverse's teams table carries one row per franchise *era*, so a club that
    moved appears more than once. The Rams appear twice under the identical
    current name, "Los Angeles Rams": once as LA and once as LAR. Building the
    lookup with a plain dict(zip(...)) lets the later row win, which silently
    hands back LAR -- a code that appears zero times in player_games, rosters
    or schedules. Every Rams player then fails the team filter and the game's
    report comes out half empty, with a filename the listing page cannot match
    to its schedule row. That happened to 2026 W2 NYG @ LA on 2026-09-21.

    So the schedules table decides, not the teams table. Any abbreviation the
    schedule has never used is dropped before the dict is built. That kills
    LAR, STL, SD and OAK on its own and needs no hand-kept alias list.
    """
    import nflreadpy as nfl
    with connect() as con:
        live = {r[0] for r in con.execute(
            "SELECT away_team FROM schedules UNION SELECT home_team FROM schedules")}
    t = nfl.load_teams().to_pandas()
    t = t[t.team_abbr.isin(live)]
    dupes = t.team_name.value_counts()
    dupes = sorted(dupes[dupes > 1].index)
    if dupes:
        # Loud on purpose. A silent last-row-wins is exactly what caused the
        # bug this function exists to prevent, and it cost a whole report.
        raise ValueError(f"team name maps to more than one live abbr: {dupes}")
    return dict(zip(t.team_name, t.team_abbr))
