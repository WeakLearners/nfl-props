#!/usr/bin/env python
"""Build the standard production-ranges report for one game, by hand.

Thin CLI over nflprops/report.py, the single report path the scheduled jobs
(run_reports.py) use too, so a report built here is the same artifact as the
one posted at 7am. Everything renders from templates/report.html.

Usage: make_report.py SEASON WEEK TEAM_A,TEAM_B [--picks 6] [--target -300] [--dry]

--dry builds and prints the shortlist but writes nothing: no HTML, no saved board.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

SEASON, WEEK = int(sys.argv[1]), int(sys.argv[2])
TEAMS = sys.argv[3].split(",")


def arg(f, d):
    return type(d)(sys.argv[sys.argv.index(f) + 1]) if f in sys.argv else d


def main():
    from nflprops import report as RP
    from nflprops.db import team_abbr_map

    n_picks, target, dry = arg("--picks", 6), arg("--target", -300), "--dry" in sys.argv
    if len(TEAMS) != 2:
        sys.exit("TEAM_A,TEAM_B must name exactly two teams")

    ev = RP.upcoming_events()
    if ev.empty:
        sys.exit("no pre-game events on the board")
    abbr = team_abbr_map()
    mine = ev[[{abbr.get(e.away_team), abbr.get(e.home_team)} == set(TEAMS)
               for e in ev.itertuples()]]
    if mine.empty:
        sys.exit(f"no pre-game event for {TEAMS[0]} / {TEAMS[1]} on the board")
    e = next(mine.itertuples())

    save = {} if dry else dict(season=SEASON, week=WEEK, teams=TEAMS)
    lines, rem = RP.event_lines(e.id, target=target, **save)
    if lines.empty:
        sys.exit("no alternate lines on the board for that game yet")
    rows, short, qb_notes = RP.build(SEASON, WEEK, TEAMS, lines, n_picks=n_picks)
    if not rows or short.empty:
        sys.exit("nothing cleared the screens")

    html = RP.render(rows, e.away_team, e.home_team, TEAMS[0], WEEK, SEASON,
                     kick=e.et, same_game=True, target=target, qb_notes=qb_notes)
    out = RP.report_path(SEASON, WEEK, *TEAMS)
    if not dry:
        out.write_text(html)

    print(f"{e.away_team} @ {e.home_team}")
    print(f"\n{'#':>2}  {'player':<19}{'tm':<5}{'stat':<16}{'line':>7}{'price':>7}{'conf':>8}")
    print("-" * 66)
    for i, r in enumerate(short.itertuples(), 1):
        print(f"{i:>2}  {r.player[:18]:<19}{r.team:<5}"
              f"{r.stat.replace('_',' '):<16}{r.line:>7g}{int(r.price):>7}{r.conf*100:>7.1f}%")
    print(f"\n-> {out.name}{' (dry, not written)' if dry else ''}   credits remaining {rem}")


if __name__ == "__main__":
    main()
