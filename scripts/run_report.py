#!/usr/bin/env python
"""Generate and post a slate report to Slack.

Usage: run_report.py [thursday|sunday|snf|mnf] [--dry]

Slates map to how the parlays get built:
  thursday / snf / mnf  -> one game, top legs from it (a same-game parlay)
  sunday                -> early (1pm) window, late (4pm) window, and combined,
                           one leg per game so the legs stay uncorrelated
"""
import sys, pathlib, json, urllib.request, time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import pandas as pd
from nflprops import rank as K
from nflprops.config import require
from nflprops.db import connect
from nflprops.slack import post, section, header, divider

SLATE = (sys.argv[1] if len(sys.argv) > 1 else "sunday").lower()

# Legs per parlay. Kept small on purpose: each leg carries ~2.2% vig and it
# compounds, and survival falls off a cliff. At a 75% leg hit rate a 4-leg slip
# survives ~32%, six legs ~18%, eight legs ~10%. One leg per game still applies
# within a window, so these are four different games.
MAX_LEGS = 4

# The combined slip is deliberately the opposite trade: one leg per game across
# the whole slate, played as a longshot. Survival is low and the vig compounds
# hard (~2.2% per leg), so this is a lottery ticket by design, not the value
# play. The two window parlays are where the sensible money goes.
COMBINED_MAX_LEGS = 12
DRY = "--dry" in sys.argv
# Stars mode is the DEFAULT for scheduled runs: big names on lines worth
# watching. It costs almost nothing in reliability (high bars hit 74.6% in the
# top tercile, same as low bars) and it is the difference between a slip you
# enjoy and one you endure. Pass --no-stars for the full pool.
STARS = "--no-stars" not in sys.argv

# Questionable players are excluded outright, not merely penalised. Backtest:
# they hit 8 points below their own price, and the whole point of the starter
# filter was to stop betting on whether someone takes the field.
ALLOW_QUESTIONABLE = "--allow-questionable" in sys.argv
BASE = "https://api.the-odds-api.com/v4/sports/americanfootball_nfl"
ALT = {"player_pass_yds_alternate": "passing_yards",
       "player_rush_yds_alternate": "rushing_yards",
       "player_reception_yds_alternate": "receiving_yards",
       "player_receptions_alternate": "receptions"}


def current_week():
    with connect() as con:
        s = pd.read_sql_query(
            "SELECT season, week, gameday FROM schedules WHERE gameday >= date('now') "
            "ORDER BY gameday LIMIT 1", con)
    if s.empty:
        sys.exit("no upcoming games in schedules -- run build_db.py")
    return int(s.season[0]), int(s.week[0])


def fetch(events):
    key = require("ODDS_API_KEY")
    rows, rem = [], None
    for e in events.itertuples():
        u = (f"{BASE}/events/{e.id}/odds?apiKey={key}&bookmakers=fanduel"
             f"&markets={','.join(ALT)}&oddsFormat=american")
        try:
            with urllib.request.urlopen(u, timeout=30) as r:
                d = json.load(r); rem = r.headers.get("x-requests-remaining")
        except Exception as ex:
            print(f"  ! {e.away_team}@{e.home_team}: {ex}", file=sys.stderr); continue
        for bm in d.get("bookmakers", []):
            for m in bm.get("markets", []):
                stat = ALT.get(m["key"])
                if not stat:
                    continue
                for o in m.get("outcomes", []):
                    if o.get("name") == "Over":
                        rows.append({"stat": stat, "player_book": o.get("description"),
                                     "line": o.get("point"), "price": o.get("price"),
                                     "game": f"{e.away_team} @ {e.home_team}",
                                     "kick": e.ct})
        time.sleep(0.2)
    return pd.DataFrame(rows), rem


season, week = current_week()
with urllib.request.urlopen(f"{BASE}/events?apiKey={require('ODDS_API_KEY')}", timeout=30) as r:
    ev = pd.DataFrame(json.load(r))
ev["ct"] = pd.to_datetime(ev.commence_time, utc=True)
now = pd.Timestamp.now(tz="UTC")
ev = ev[ev.ct > now]
if ev.empty:
    sys.exit("no pre-game events")

# Pick the games this slate covers.
ev["et"] = ev.ct.dt.tz_convert("America/New_York")
# Select by WEEKDAY and KICKOFF TIME, not "the next game on the board". An
# earlier version took ev.head(1) for every single-game slate, so asking for
# MNF on a Sunday afternoon returned that night's SNF game with an MNF header.
# The slate name has to actually constrain which game is chosen.
if SLATE in ("thursday", "snf", "mnf"):
    if SLATE == "thursday":
        pick = ev[ev.et.dt.dayofweek == 3]
    elif SLATE == "snf":
        pick = ev[(ev.et.dt.dayofweek == 6) & (ev.et.dt.hour >= 19)]
    else:
        pick = ev[ev.et.dt.dayofweek == 0]
    if pick.empty:
        sys.exit(f"no {SLATE.upper()} game found in the upcoming slate")
    ev = pick.sort_values("ct").head(1)
    g = ev.iloc[0]
    print(f"{SLATE.upper()}: {g.away_team} @ {g.home_team} "
          f"{g.et:%a %-I:%M %p ET}")
    windows = [(SLATE.upper(), ev)]
else:
    sun = ev[ev.et.dt.dayofweek == 6]
    early = sun[sun.et.dt.hour < 16]
    late = sun[(sun.et.dt.hour >= 16) & (sun.et.dt.hour < 19)]
    ev = pd.concat([early, late]).drop_duplicates("id")
    windows = [("EARLY (1pm)", early), ("LATE (4pm)", late)]

alt, rem = fetch(ev)
if alt.empty:
    sys.exit("no alternate props returned")

blocks = [header(f"NFL Props — {season} W{week} · {SLATE.upper()}"
                 + (" · STARS" if STARS else ""))]
combined, combined_pool = [], []
for label, games in windows:
    if games.empty:
        continue
    sub = alt[alt.game.isin({f"{g.away_team} @ {g.home_team}" for g in games.itertuples()})]
    best = K.best_legs(sub, season, week, n=MAX_LEGS,
                       one_per_game=(SLATE == "sunday"), stars=STARS,
                       allow_questionable=ALLOW_QUESTIONABLE)
    if best.empty:
        continue
    combined.append(best)
    combined_pool.append(K.best_legs(sub, season, week, n=COMBINED_MAX_LEGS,
                                     one_per_game=True, stars=STARS,
                                     allow_questionable=ALLOW_QUESTIONABLE))
    # Echo to stdout as well as Slack -- the launchd log is the only record of
    # what a scheduled run actually recommended.
    _m, _k, _p = K.survival(best)
    print(f"\n{label}:  pays ~{_p-1:.1f}x · survival model {_m*100:.0f}% vs book {_k*100:.0f}%")
    for i, r in enumerate(best.itertuples(), 1):
        n = K.notes(r)
        print(f"  {i}. {r.player_book:<21}{r.stat.replace('_',' '):<17} over {r.line:<6g}"
              f"{int(r.price):>6}   cleared {r.hit_rate*100:.0f}% of {int(r.n_games)}"
              + (f"   [{n}]" if n else ""))
    lines = []
    for i, r in enumerate(best.itertuples(), 1):
        note = K.notes(r)
        lines.append(f"`{i}.` *{r.player_book}* — {r.stat.replace('_',' ')} over {r.line:g}  "
                     f"`{int(r.price)}`\n     _{r.game}_ · cleared *{r.hit_rate*100:.0f}%* of last "
                     f"{int(r.n_games)}" + (f" · ⚠️ {note}" if note else ""))
    mdl, mkt, pay = K.survival(best)
    blocks += [section(f"*{label}* — {len(best)} legs · pays ~{pay-1:.1f}x · "
                       f"survival: model *{mdl*100:.0f}%* vs book *{mkt*100:.0f}%*"),
               section("\n".join(lines)), divider()]

if SLATE == "sunday" and len(combined) == 2:
    # Longshot: one leg per game across the entire slate, both windows.
    tot = (pd.concat(combined_pool).sort_values("score", ascending=False)
             .drop_duplicates("game").head(COMBINED_MAX_LEGS))
    mdl, mkt, payout = K.survival(tot)
    lines = [f"`{i}.` *{r.player_book}* — {r.stat.replace('_',' ')} over {r.line:g}  "
             f"`{int(r.price)}`  _{r.game}_" for i, r in enumerate(tot.itertuples(), 1)]
    blocks += [section(f"*COMBINED LONGSHOT* — {len(tot)} legs, one per game"),
               section("\n".join(lines)),
               section(f"Pays about *{payout-1:.0f}x* (+{(payout-1)*100:.0f})\n"
                       f"Survival — model says *{mdl*100:.1f}%*, FanDuel prices it at "
                       f"*{mkt*100:.1f}%*.\n"
                       f"_Do not read that gap as edge. The model is extrapolating well "
                       f"past where it was validated, and a 12-way product compounds that "
                       f"error twelve times. Treat this as a lottery ticket. The 4-leg "
                       f"window slips above are the defensible bets._")]
    print(f"\nCOMBINED LONGSHOT — {len(tot)} legs, pays ~{payout-1:.0f}x, "
          f"survival model {mdl*100:.1f}% vs book {mkt*100:.1f}%")
    for i, r in enumerate(tot.itertuples(), 1):
        print(f"  {i}. {r.player_book:<21}{r.stat.replace('_',' '):<17} over {r.line:<6g}"
              f"{int(r.price):>6}   {r.game}")

blocks.append(section(
    "_Ranked by reliability, not edge. Validated on 777 historical legs: top-tercile "
    "clear rate hits *74.4%* vs *52.6%* bottom tercile at the same price. "
    "The alternate-line band holds ~8 points, so fewer legs = less compounding vig._"))

if DRY:
    print(json.dumps(blocks, indent=1)[:2000])
    print(f"\n[dry run] {sum(len(c) for c in combined)} legs, credits remaining {rem}")
else:
    print(post(blocks=blocks, text=f"NFL Props — W{week} {SLATE}"))
    print(f"posted; credits remaining {rem}")
