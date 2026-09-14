#!/usr/bin/env python
"""Build the standard production-ranges report for a game.

This is the canonical report format. Everything renders from templates/report.html
so every game gets the same page, and the shortlist is chosen by one rule rather
than by eye.

Usage: make_report.py SEASON WEEK TEAM_A,TEAM_B [--picks 6] [--target -200]
"""
import sys, pathlib, io, json, contextlib, urllib.request
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import pandas as pd, numpy as np
from statistics import NormalDist
from nflprops import project as P, reliability as R
from nflprops.project import defense_vs_position, SPECS, DEF_WEIGHT
from nflprops.config import require, ROOT
from nflprops.odds import name_key
from nflprops.db import connect

SEASON, WEEK = int(sys.argv[1]), int(sys.argv[2])
TEAMS = sys.argv[3].split(",")
def arg(f, d):
    return type(d)(sys.argv[sys.argv.index(f) + 1]) if f in sys.argv else d
N_PICKS, TARGET = arg("--picks", 6), arg("--target", -200)

# Errors ran ~10% larger than the model's own sd across 2023-2025, so the raw
# spread understates how wrong a projection can be.
SD_INFLATE = 1.10
# Situational screens. A leg failing any of these is shown but never shortlisted.
MIN_SNAP, MIN_GAMES = 0.55, 10

ALT = {"player_pass_yds_alternate": "passing_yards",
       "player_rush_yds_alternate": "rushing_yards",
       "player_reception_yds_alternate": "receiving_yards",
       "player_receptions_alternate": "receptions"}
BASE = "https://api.the-odds-api.com/v4/sports/americanfootball_nfl"


def fetch_lines(teams):
    """The rung closest to TARGET for each player-stat, from the live board."""
    k = require("ODDS_API_KEY")
    with urllib.request.urlopen(f"{BASE}/events?apiKey={k}", timeout=30) as r:
        ev = pd.DataFrame(json.load(r))
    ev["ct"] = pd.to_datetime(ev.commence_time, utc=True)
    ev = ev[ev.ct > pd.Timestamp.now(tz="UTC")]
    rows, meta = [], None
    for e in ev.itertuples():
        u = (f"{BASE}/events/{e.id}/odds?apiKey={k}&bookmakers=fanduel"
             f"&markets={','.join(ALT)}&oddsFormat=american")
        try:
            with urllib.request.urlopen(u, timeout=30) as r:
                d = json.load(r)
                rem = r.headers.get("x-requests-remaining")
        except Exception:
            continue
        got = False
        for bm in d.get("bookmakers", []):
            for m in bm.get("markets", []):
                stat = ALT.get(m["key"])
                if not stat:
                    continue
                for o in m.get("outcomes", []):
                    if o.get("name") == "Over":
                        rows.append({"stat": stat, "player": o.get("description"),
                                     "line": o.get("point"), "price": o.get("price")})
                        got = True
        if got and meta is None:
            meta = {"away": e.away_team, "home": e.home_team,
                    "kick": e.ct.tz_convert("America/New_York"), "rem": rem}
    if not rows:
        sys.exit("no alternate lines on the board for that game yet")
    df = pd.DataFrame(rows)
    df["player_key"] = df.player.map(name_key)
    df["d"] = (df.price - TARGET).abs()
    return df.sort_values("d").drop_duplicates(["player_key", "stat"]), meta


with contextlib.redirect_stdout(io.StringIO()):
    pr = P.project_week(SEASON, WEEK)
    pool = R.star_pool(SEASON, WEEK)
    snaps = R.snap_share(SEASON, WEEK)
    cont = R.role_continuity(SEASON, WEEK)
inj, _ = R.risk_flags(SEASON, WEEK)

pr["player_key"] = pr["player"].map(name_key)
pr = pr[pr.team.isin(TEAMS) & pr.player_key.isin(set(pool.player_key))].copy()
rank = {r.player_key: int(r["rank"]) for _, r in pool.iterrows()}

dvp = defense_vs_position(SEASON, WEEK)
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

lines, meta = fetch_lines(TEAMS)
pr = pr.merge(lines[["player_key", "stat", "line", "price"]], on=["player_key", "stat"], how="left")


def imp(a):
    if pd.isna(a):
        return np.nan
    a = float(a)
    return (-a) / ((-a) + 100) if a < 0 else 100 / (a + 100)


pr["bp"] = pr.price.map(imp)
pr["mp"] = [np.nan if pd.isna(l) else 1 - NormalDist(float(m), float(s)).cdf(float(l))
            for l, m, s in zip(pr.line, pr.proj, pr.sd_adj)]
pr["z"] = (pr.line - pr.proj) / pr.sd_adj
# Never claim more than the more pessimistic of model and market.
pr["conf"] = pr[["mp", "bp"]].min(axis=1)

pr = pr.merge(snaps, on="player_key", how="left").merge(
    cont[["player_key", "changed_team"]].drop_duplicates("player_key"), on="player_key", how="left").merge(
    inj[["player_key", "report_status"]].drop_duplicates("player_key"), on="player_key", how="left")
with connect() as con:
    seen = pd.read_sql_query(
        "SELECT player_display_name, COUNT(*) n FROM player_games WHERE season_type='REG' GROUP BY 1", con)
seen["player_key"] = seen.player_display_name.map(name_key)
pr = pr.merge(seen[["player_key", "n"]], on="player_key", how="left")
pr["snap_share"] = pr.snap_share.fillna(0)
pr["changed_team"] = pr.changed_team.fillna(False)

pr["clean"] = (~pr.changed_team
               & ~pr.report_status.isin(["Questionable", "Doubtful", "Out"])
               & (pr.snap_share >= MIN_SNAP)
               & (pr.n.fillna(0) >= MIN_GAMES))

# Shortlist: strictly by confidence among legs passing the screens, one per
# player. Two legs on one player die together if he is hurt or benched, which
# is the concentration a multi-leg slip is trying to avoid.
short = (pr[pr.clean & pr.line.notna()]
         .sort_values("conf", ascending=False)
         .drop_duplicates("player_key")
         .head(N_PICKS))
pr["pick"] = None
for i, ix in enumerate(short.index, 1):
    pr.loc[ix, "pick"] = i

teamA = TEAMS[0]
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
    })

same_game = pr[pr.pick.notna()].game_id.nunique() <= 1
corr = ("All of these are same-game legs, so they move together &mdash; a shootout carries "
        "them over, a grind carries them under. That makes the true joint chance "
        "<em>higher</em> than the naive figure, and it is also why FanDuel will pay less "
        "than the straight payout for them as one same-game parlay."
        if same_game else
        "These legs sit in different games, so they are close to independent &mdash; the "
        "naive joint chance is about right, and the straight payout is what you would be "
        "offered.")

meta_js = {"teamA": teamA, "nameA": meta["away"] if meta else TEAMS[0],
           "nameB": meta["home"] if meta else TEAMS[1], "corrnote": corr}
tpl = (ROOT / "templates" / "report.html").read_text()
head = (f'<h1>{meta["away"]} <span class="at">at</span> {meta["home"]}</h1>'
        if meta else f'<h1>{TEAMS[0]} <span class="at">at</span> {TEAMS[1]}</h1>')
eyebrow = (f'{meta["kick"]:%A} Night · Week {WEEK} · {SEASON}' if meta
           else f'Week {WEEK} · {SEASON}')
title = (f'{meta["away"].split()[-1]}–{meta["home"].split()[-1]} Production Ranges'
         if meta else f'Week {WEEK} Production Ranges')

out_html = (tpl.replace("__DATA__", json.dumps(rows))
               .replace("__META__", json.dumps(meta_js))
               .replace("__TITLE__", title)
               .replace("__EYEBROW__", eyebrow)
               .replace("__HEADLINE__", head))
out = ROOT / "reports" / f"report_{SEASON}_w{WEEK}_{'-'.join(TEAMS)}.html"
out.write_text(out_html)

print(f"{meta['away']} @ {meta['home']}" if meta else "")
print(f"\n{'#':>2}  {'player':<19}{'tm':<5}{'stat':<16}{'line':>7}{'price':>7}{'conf':>8}")
print("-" * 66)
for r in short.itertuples():
    print(f"{int(pr.loc[r.Index,'pick']):>2}  {r.player[:18]:<19}{r.team:<5}"
          f"{r.stat.replace('_',' '):<16}{r.line:>7g}{int(r.price):>7}{r.conf*100:>7.1f}%")
print(f"\n-> {out.name}   credits remaining {meta['rem'] if meta else '?'}")
