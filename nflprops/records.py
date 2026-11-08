"""Regular-season W-L(-T) record of a team as of a game's kickoff.

Reads the schedules table the repo already loads. No new source, no network.
"""
import sqlite3
from html import escape

from .config import DB_PATH


def _when(game):
    """Sortable kickoff stamp from gameday and gametime (ET). A missing time sorts first."""
    return f"{game.get('gameday') or ''} {game.get('gametime') or '00:00'}"


def record_before(games, team, season, kickoff):
    """(wins, losses, ties) for `team` from finished REG games of `season` that kicked off before `kickoff`.

    games: dicts with season, game_type, gameday, gametime, home_team, away_team, home_score, away_score.
    kickoff: "YYYY-MM-DD HH:MM" stamp of the game being shown. That game and later ones are left out.
    """
    w = l = t = 0
    for g in games:
        if g.get("season") != season or g.get("game_type") != "REG" or team not in (g["home_team"], g["away_team"]):
            continue
        hs, as_ = g.get("home_score"), g.get("away_score")
        if hs is None or as_ is None or hs != hs or as_ != as_:  # missing or NaN
            continue
        if _when(g) >= kickoff:
            continue
        mine, theirs = (hs, as_) if g["home_team"] == team else (as_, hs)
        if mine > theirs:
            w += 1
        elif mine < theirs:
            l += 1
        else:
            t += 1
    return w, l, t


def fmt_record(w, l, t):
    return f"{w}-{l}-{t}" if t else f"{w}-{l}"


def game_records(season, week, away, home, db_path=None):
    """(away_record, home_record) strings as of this game's kickoff, or (None, None) if the game is not found."""
    con = sqlite3.connect(f"file:{db_path or DB_PATH}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    try:
        games = [dict(r) for r in con.execute(
            "SELECT season, week, game_type, gameday, gametime, home_team, away_team, home_score, away_score "
            "FROM schedules WHERE season=?", (season,))]
    finally:
        con.close()
    this = [g for g in games if g["game_type"] == "REG" and g["week"] == week
            and {g["home_team"], g["away_team"]} == {away, home}]
    if not this:
        return None, None
    k = _when(this[0])
    return (fmt_record(*record_before(games, away, season, k)),
            fmt_record(*record_before(games, home, season, k)))


def headline(away_name, home_name, season, week, away_abbr, home_abbr):
    """The report <h1>. A record sits right of each team name; if it cannot be worked out the name stands alone."""
    try:
        ra, rh = game_records(season, week, away_abbr, home_abbr)
    except Exception:  # noqa: BLE001 -- a header without records beats no report
        ra = rh = None
    # Inline style: old saved reports carry an older stylesheet, and the header is the only part rewritten in them.
    st = "color:var(--ink-3);margin-left:.4em;white-space:nowrap"
    rec = lambda r: f' <span class="rec" data-inspect-id="header-record" style="{st}">({escape(r)})</span>' if r else ""  # noqa: E731
    return f'<h1>{away_name}{rec(ra)} <span class="at">at</span> {home_name}{rec(rh)}</h1>'
