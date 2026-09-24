#!/usr/bin/env python
"""Build a production-ranges report for every game on a slate, post to Slack.

Usage: run_reports.py [thursday|sunday|snf|mnf] [--picks 6] [--dry]

One HTML report per game lands in reports/. Slack gets a single digest with
each game's shortlist -- one message per slate, not one per game, because a
13-game Sunday would otherwise bury the channel.

Note: a scheduled job cannot publish to claude.ai. The HTML is written to disk;
publishing it as an artifact is a manual step.
"""
import sys, pathlib, json
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import pandas as pd
from nflprops import report as RP
from nflprops.config import ROOT
from nflprops.db import connect, team_abbr_map
from nflprops.slack import post, section, header, divider

SLATE = (sys.argv[1] if len(sys.argv) > 1 else "sunday").lower()
def arg(f, d):
    return type(d)(sys.argv[sys.argv.index(f) + 1]) if f in sys.argv else d
N_PICKS, DRY = arg("--picks", 6), "--dry" in sys.argv


def current_week():
    with connect() as con:
        s = pd.read_sql_query(
            "SELECT season, week FROM schedules WHERE gameday >= date('now') "
            "ORDER BY gameday LIMIT 1", con)
    if s.empty:
        sys.exit("no upcoming games -- run build_db.py")
    return int(s.season[0]), int(s.week[0])


def pick_games(ev):
    """Which games this slate covers, by weekday and kickoff hour."""
    if SLATE == "thursday":
        return ev[ev.et.dt.dayofweek == 3].sort_values("ct").head(1)
    if SLATE == "mnf":
        return ev[ev.et.dt.dayofweek == 0].sort_values("ct").head(1)
    if SLATE == "snf":
        return ev[(ev.et.dt.dayofweek == 6) & (ev.et.dt.hour >= 19)].sort_values("ct").head(1)
    return ev[(ev.et.dt.dayofweek == 6) & (ev.et.dt.hour < 19)].sort_values("ct")


def dec(a):
    return 1 + (100 / (-a) if a < 0 else a / 100)


season, week = current_week()
ev = RP.upcoming_events()
if ev.empty:
    sys.exit("no pre-game events on the board")
games = pick_games(ev)
if games.empty:
    sys.exit(f"no {SLATE.upper()} game found")

abbr = team_abbr_map()

print(f"{SLATE.upper()}: {len(games)} game(s), {season} W{week}")
blocks = [header(f"Production ranges — {season} W{week} · {SLATE.upper()}")]
made, rem = [], None
for e in games.itertuples():
    a, h = abbr.get(e.away_team), abbr.get(e.home_team)
    if not a or not h:
        print(f"  ! unknown team abbr for {e.away_team} @ {e.home_team}"); continue
    lines, rem = RP.event_lines(e.id, season=season, week=week, teams=[a, h])
    if lines.empty:
        print(f"  ! no alternate lines yet: {a}@{h}"); continue
    rows, short, qb_notes = RP.build(season, week, [a, h], lines, n_picks=N_PICKS)
    if not rows or short.empty:
        print(f"  ! nothing cleared the screens: {a}@{h}"); continue

    html = RP.render(rows, e.away_team, e.home_team, a, week, season,
                     kick=e.et, same_game=True, qb_notes=qb_notes)
    out = ROOT / "reports" / f"report_{season}_w{week}_{a}-{h}.html"
    out.write_text(html)
    made.append(out.name)

    pay = short.price.map(dec).prod()
    joint = short.conf.prod()
    print(f"  {a} @ {h}: {len(short)} legs, pays {pay-1:.2f}x -> {out.name}")
    lines_txt = []
    for i, r in enumerate(short.itertuples(), 1):
        lines_txt.append(
            f"`{i}.` *{r.player}* — {r.stat.replace('_',' ')} over {r.line:g}  "
            f"`{int(r.price)}`  *{r.conf*100:.0f}%*")
    blocks += [
        section(f"*{e.away_team} at {e.home_team}* · {e.et:%a %-I:%M %p ET}"),
        section("\n".join(lines_txt)),
        section(f"_All {len(short)} straight: {pay-1:.2f}× · naive joint chance "
                f"{joint*100:.0f}%. Same-game legs move together, so the true joint "
                f"chance is higher and FanDuel will pay less than {pay-1:.2f}× as an SGP._"),
        divider(),
    ]

if not made:
    sys.exit("nothing to post")

blocks.append(section(
    "_Confidence is the lower of the model's and FanDuel's own number, so it is never "
    "the more optimistic of the two. It measures how likely a leg is to *happen* — not "
    "that it is mispriced. Tested against real lines, this model was the less accurate "
    "of the two._"))
blocks.append(section(f"_{len(made)} report(s) written to `reports/`. "
                      f"Credits remaining: {rem}._"))

if DRY:
    print(f"\n[dry] would post {len(blocks)} blocks; {len(made)} reports written")
else:
    print(post(blocks=blocks, text=f"Production ranges — W{week} {SLATE}"))
    print(f"posted; credits remaining {rem}")
