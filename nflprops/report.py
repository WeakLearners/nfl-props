"""The standard production-ranges report.

One code path builds the page whether it is rendered interactively or by a
scheduled job, so a report posted at 7am is the same artifact as one built by
hand. Everything visual lives in templates/report.html.
"""
import io, json, contextlib, pathlib, sys, urllib.request
from statistics import NormalDist

import pandas as pd
import numpy as np

from . import project as P, reliability as R
from .project import defense_vs_position, SPECS, DEF_WEIGHT
from .config import require, ROOT, save_board
from .odds import name_key
from .db import connect

# scripts/vol_predict.py is not part of the nflprops package (it lives beside
# it, not inside it) -- same import shape scripts/make_report.py uses.
_SCRIPTS = str(pathlib.Path(__file__).resolve().parent.parent / "scripts")
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)
from vol_predict import phase3_vol  # noqa: E402

# Errors ran ~10% larger than the model's own sd across 2023-2025, so the raw
# spread understates how wrong a projection can be.
SD_INFLATE = 1.10
# Situational screens. A leg failing any of these is shown but never shortlisted.
MIN_SNAP, MIN_GAMES = 0.55, 10
DEFAULT_TARGET = -300

ALT = {"player_pass_yds_alternate": "passing_yards",
       "player_rush_yds_alternate": "rushing_yards",
       "player_reception_yds_alternate": "receiving_yards",
       "player_receptions_alternate": "receptions"}
BASE = "https://api.the-odds-api.com/v4/sports/americanfootball_nfl"


def upcoming_events():
    """Pre-game events on the board, with kickoff in ET. Free call."""
    k = require("ODDS_API_KEY")
    with urllib.request.urlopen(f"{BASE}/events?apiKey={k}", timeout=30) as r:
        ev = pd.DataFrame(json.load(r))
    if ev.empty:
        return ev
    ev["ct"] = pd.to_datetime(ev.commence_time, utc=True)
    ev = ev[ev.ct > pd.Timestamp.now(tz="UTC")].copy()
    ev["et"] = ev.ct.dt.tz_convert("America/New_York")
    return ev


def event_lines(event_id, target=DEFAULT_TARGET, season=None, week=None, teams=None):
    """The rung closest to `target` for each player-stat in one game.

    decision #37: when the caller passes season/week/teams, the raw response
    is saved to data/boards/ unmodified before any filtering -- zero extra
    API calls, since this response was already fetched.
    """
    k = require("ODDS_API_KEY")
    u = (f"{BASE}/events/{event_id}/odds?apiKey={k}&bookmakers=fanduel"
         f"&markets={','.join(ALT)}&oddsFormat=american")
    with urllib.request.urlopen(u, timeout=30) as r:
        d = json.load(r)
        rem = r.headers.get("x-requests-remaining")
    if season is not None and week is not None and teams is not None:
        save_board(season, week, teams, d)
    rows = []
    for bm in d.get("bookmakers", []):
        for m in bm.get("markets", []):
            stat = ALT.get(m["key"])
            if not stat:
                continue
            for o in m.get("outcomes", []):
                if o.get("name") == "Over":
                    rows.append({"stat": stat, "player": o.get("description"),
                                 "line": o.get("point"), "price": o.get("price")})
    if not rows:
        return pd.DataFrame(), rem
    df = pd.DataFrame(rows)
    df["player_key"] = df.player.map(name_key)
    df["d"] = (df.price - target).abs()
    return df.sort_values("d").drop_duplicates(["player_key", "stat"]), rem


def _imp(a):
    if pd.isna(a):
        return np.nan
    a = float(a)
    return (-a) / ((-a) + 100) if a < 0 else 100 / (a + 100)


def build(season, week, teams, lines, n_picks=6):
    """Return (rows, shortlist, qb_notes) for one game. `lines` comes from
    event_lines. `qb_notes` is the list of decision #36 QB1-override strings
    (empty when the sportsbook line agreed with the usage rank, or didn't
    narrow to exactly one priced QB)."""
    with contextlib.redirect_stdout(io.StringIO()):
        pr = P.project_week(season, week)
        pool = R.star_pool(season, week)
        snaps = R.snap_share(season, week)
        cont = R.role_continuity(season, week)
    inj, _ = R.risk_flags(season, week)

    pr["player_key"] = pr["player"].map(name_key)

    # Decision #36: when FanDuel prices passing yards for exactly one QB on a
    # team, that QB is the featured QB1 -- replacing star_pool's usage-ranked
    # pick if it disagrees. Same rule as scripts/make_report.py; duplicated
    # here because this is the code path the scheduled job actually runs.
    qb_notes = []
    if not lines.empty:
        qb_priced = set(lines.loc[lines.stat == "passing_yards", "player_key"])
        qb_universe = pr[(pr.position == "QB") & pr.team.isin(teams)][
            ["player_key", "team", "player"]].drop_duplicates("player_key")
        for _team in teams:
            priced = qb_universe[(qb_universe.team == _team) & qb_universe.player_key.isin(qb_priced)]
            if len(priced) != 1:
                continue
            new_key, new_name = priced.iloc[0].player_key, priced.iloc[0].player
            cur = pool[(pool.team == _team) & (pool.position == "QB")]
            if not cur.empty and cur.iloc[0].player_key == new_key:
                continue
            old_name = cur.iloc[0].full_name if not cur.empty else None
            pool = pool[~((pool.team == _team) & (pool.position == "QB"))]
            pool = pd.concat([pool, pd.DataFrame([{
                "player_key": new_key, "full_name": new_name, "team": _team,
                "position": "QB", "usage": np.nan, "rank": 1.0}])], ignore_index=True)
            qb_notes.append(f"QB set by sportsbook line: {new_name} ({_team})"
                             + (f", not {old_name}" if old_name else ""))

    pr = pr[pr.team.isin(teams) & pr.player_key.isin(set(pool.player_key))].copy()
    if pr.empty:
        return [], pr, qb_notes
    rank = {r.player_key: int(r["rank"]) for _, r in pool.iterrows()}

    # Decision #31: Phase 3's ridge volume/yardage models, shown additively
    # next to this engine's own number for receiving/rushing yards. Never
    # merged into one figure, never claimed as better -- see decision #37.
    try:
        p3 = phase3_vol(season, week)
    except Exception as e:
        print(f"  ! phase3_vol failed, skipping: {e}")
        p3 = pd.DataFrame(columns=["player_id"])
    pr = pr.merge(p3, on="player_id", how="left") if "player_id" in pr.columns else pr
    for _c in ("p3_receiving_yards", "p3_rushing_yards"):
        if _c not in pr.columns:
            pr[_c] = np.nan

    dvp = defense_vs_position(season, week)
    dvp_ix = dvp.set_index(["defteam", "position"]) if not dvp.empty else None

    def raw_mult(row):
        if dvp_ix is None or (row.opponent, row.position) not in dvp_ix.index:
            return np.nan
        d = dvp_ix.loc[(row.opponent, row.position)]
        col = SPECS[row.stat][2]
        if col not in d:
            return np.nan
        try:
            return float(d.get(col))
        except (TypeError, ValueError):
            return np.nan

    pr["raw_def"] = pr.apply(raw_mult, axis=1)
    pr["applied"] = pr.stat.map(lambda s: DEF_WEIGHT.get(s, 1.0) > 0)
    pr["sd_adj"] = pr.sd * SD_INFLATE
    pr["lo"] = (pr.proj - pr.sd_adj).clip(lower=0)
    pr["hi"] = pr.proj + pr.sd_adj

    if not lines.empty:
        pr = pr.merge(lines[["player_key", "stat", "line", "price"]],
                      on=["player_key", "stat"], how="left")
    else:
        pr["line"] = np.nan
        pr["price"] = np.nan

    pr["bp"] = pr.price.map(_imp)
    pr["mp"] = [np.nan if pd.isna(l) else 1 - NormalDist(float(m), float(s)).cdf(float(l))
                for l, m, s in zip(pr.line, pr.proj, pr.sd_adj)]
    pr["z"] = (pr.line - pr.proj) / pr.sd_adj
    # Never claim more than the more pessimistic of model and market.
    pr["conf"] = pr[["mp", "bp"]].min(axis=1)

    pr = (pr.merge(snaps, on="player_key", how="left")
            .merge(cont[["player_key", "changed_team"]].drop_duplicates("player_key"),
                   on="player_key", how="left")
            .merge(inj[["player_key", "report_status"]].drop_duplicates("player_key"),
                   on="player_key", how="left"))
    with connect() as con:
        seen = pd.read_sql_query(
            "SELECT player_display_name, COUNT(*) n FROM player_games "
            "WHERE season_type='REG' GROUP BY 1", con)
    seen["player_key"] = seen.player_display_name.map(name_key)
    pr = pr.merge(seen[["player_key", "n"]], on="player_key", how="left")
    pr["snap_share"] = pr.snap_share.fillna(0)
    pr["changed_team"] = pr.changed_team.fillna(False)
    pr["clean"] = (~pr.changed_team
                   & ~pr.report_status.isin(["Questionable", "Doubtful", "Out"])
                   & (pr.snap_share >= MIN_SNAP)
                   & (pr.n.fillna(0) >= MIN_GAMES))

    # Shortlist: strictly by confidence among legs passing the screens, one per
    # player. Two legs on one player die together if he is hurt or benched,
    # which is the concentration a multi-leg slip exists to avoid.
    short = (pr[pr.clean & pr.line.notna()]
             .sort_values("conf", ascending=False)
             .drop_duplicates("player_key")
             .head(n_picks))
    pr["pick"] = None
    for i, ix in enumerate(short.index, 1):
        pr.loc[ix, "pick"] = i

    rows = []
    for r in pr.itertuples():
        rows.append({
            "p": r.player, "role": r.position + str(rank.get(r.player_key, 1)),
            "tm": r.team, "vs": r.opponent, "s": r.stat,
            "mu": round(float(r.proj), 1), "sd": round(float(r.sd_adj), 1),
            "lo": round(float(r.lo), 1), "hi": round(float(r.hi), 1),
            "def": None if pd.isna(r.raw_def) else round(float(r.raw_def), 2),
            "app": bool(r.applied),
            "line": None if pd.isna(r.line) else float(r.line),
            "price": None if pd.isna(r.price) else int(r.price),
            "z": None if pd.isna(r.z) else round(float(r.z), 2),
            "mp": None if pd.isna(r.mp) else round(float(r.mp), 3),
            "bp": None if pd.isna(r.bp) else round(float(r.bp), 3),
            "pick": None if r.pick is None else int(r.pick),
            "p3": (round(float(r.p3_receiving_yards), 1) if r.stat == "receiving_yards"
                                                           and pd.notna(r.p3_receiving_yards)
                   else round(float(r.p3_rushing_yards), 1) if r.stat == "rushing_yards"
                                                              and pd.notna(r.p3_rushing_yards)
                   else None),
        })
    return rows, pr.loc[short.index], qb_notes


def render(rows, away, home, team_a, week, season, kick=None, same_game=True,
           target=DEFAULT_TARGET, qb_notes=None):
    corr = ("All of these are same-game legs, so they move together &mdash; a shootout "
            "carries them over, a grind carries them under. That makes the true joint "
            "chance <em>higher</em> than the naive figure, and it is also why FanDuel "
            "will pay less than the straight payout for them as one same-game parlay."
            if same_game else
            "These legs sit in different games, so they are close to independent &mdash; "
            "the naive joint chance is about right, and the straight payout is what you "
            "would be offered.")
    if qb_notes:
        corr = ("<strong>" + "</strong><br><strong>".join(qb_notes) + "</strong><br>" + corr)
    meta = {"teamA": team_a, "nameA": away, "nameB": home, "corrnote": corr}
    tpl = (ROOT / "templates" / "report.html").read_text()
    head = f'<h1>{away} <span class="at">at</span> {home}</h1>'
    # Kickoff time, not just the weekday. A Sunday slate has games at 1:00,
    # 4:05, 4:25 and 8:20, and "Sunday" cannot tell you which one you have
    # open. The listing sorts by this same time, read from the schedules
    # table (see scripts/serve_reports.py).
    eyebrow = (f"{kick:%A %-I:%M %p} ET · Week {week} · {season}" if kick is not None
               else f"Week {week} · {season}")
    title = f"{away.split()[-1]}–{home.split()[-1]} Production Ranges"
    return (tpl.replace("__DATA__", json.dumps(rows))
               .replace("__META__", json.dumps(meta))
               .replace("__TITLE__", title)
               .replace("__EYEBROW__", eyebrow)
               .replace("__HEADLINE__", head)
               .replace("__TARGET__", json.dumps(target))
               .replace("__TARGET_ABS__", str(abs(target))))
