#!/usr/bin/env python
"""Post a correction to the phase 2 findings already sent to Slack."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from nflprops.slack import post, section, header, divider

TABLE = """```
stat                 n    model   naive   vs naive
rushing_yards    2,477    24.93   27.93     +10.8%
passing_yards    1,571    67.58   72.59      +6.9%
receiving_yards  5,858    24.83   26.60      +6.6%
completions      1,632     5.63    6.01      +6.4%
receptions       7,302     1.67    1.78      +6.3%
```"""

BUCKETS = """```
baseline     n     model   naive   bias
0-10     3,255     10.73    7.61  +4.19   <- unbettable
10-25    5,204     15.15   14.33  +2.23   <- unbettable
25-45    3,110     21.35   22.46  +0.55   <- model wins
45-70    2,012     26.87   29.70  -0.85   <- model wins
70+        730     32.87   35.70  -1.06   <- model wins
```"""

blocks = [
    header("CORRECTION — NFL Props phase 2"),
    section("*My earlier post was wrong.* I reported that receiving props had no "
            "skill (−11.9% vs a career average) and cut WR/TE yardage from v1. "
            "That conclusion does not survive a closer look — receiving props are "
            "back in."),
    divider(),
    section("*What went wrong:* I scored the model on 8,459 rows no book would "
            "ever price — deep-bench receivers projected for single-digit yards. "
            "Shrinkage inflates those players, and they swamped the average.\n\n"
            "Broken out by how big the player's baseline is:"),
    section(BUCKETS),
    section("The model beats a career average in *every bucket where a prop "
            "actually exists*, and loses only where one doesn't. Same mistake as "
            "the backup-QB trap, wearing a different hat."),
    divider(),
    section("*Corrected results — bettable universe only, n=18,840:*"),
    section(TABLE),
    section("Consistent *+6% to +11% on all five stats.* Nothing is disqualified."),
    divider(),
    section("*The caveat that governs all of it:* beating a career average is "
            "_not_ beating FanDuel. A lot of this edge is the pseudo-line being "
            "stale for declining players — Kupp, Adams, Godwin, Andrews all "
            "project UNDER their career averages and the model is right. But "
            "FanDuel already prices that decline.\n\n"
            "So this is a real *projection* improvement and says nothing yet "
            "about *betting* edge. Only real lines can answer that."),
    section("Week 1 2026 retrospective is now 14/19 — still meaningless, same "
            "reason.\n*Blocked on `ODDS_API_KEY`.*"),
]

print(post(blocks=blocks, text="CORRECTION — NFL Props phase 2 findings"))
