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
