#!/usr/bin/env python
"""Post the phase 2 backtest findings to Slack."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from nflprops.slack import post, section, header, divider

TABLE = """```
stat              n       model   naive    vs naive
passing_yards     1,840   67.92   73.11      +7.1%
rushing_yards     4,361   19.56   21.03      +7.0%
completions       1,840    5.67    6.03      +6.0%
receptions       14,500    1.38    1.38      -0.0%
receiving_yards  14,500   19.72   17.62     -11.9%
```"""

blocks = [
    header("NFL Props — Phase 2 findings"),
    section("Walk-forward backtest, 2023–2025. Every week re-projected using "
            "only prior data. *n = 37,041* player-stats.\n"
            "Baseline = player's career average (roughly where a book hangs a line). "
            "Lower MAE is better."),
    section(TABLE),
    divider(),
    section("*Receiving props are cut from v1.*\n"
            "The model is *worse than a dumb career average* on receiving yards "
            "(−11.9%) and flat on receptions. That removes WR/TE yardage — the "
            "largest prop category — until the volume model is fixed. Shipping a "
            "fake edge is worse than shipping nothing, because parlays multiply a "
            "negative edge just as hard as a positive one."),
    section("*Opponent adjustment had to go per-stat.*\n"
            "Applied uniformly it destroyed receiving accuracy. Team pass-defense "
            "quality is a real signal; _receiving yards allowed to WRs_ is mostly a "
            "record of which receivers a defense happened to face. Now weighted "
            "pass 1.0 / rush 0.5 / receiving 0.0."),
    section("*The backup-QB trap.*\n"
            "The biggest apparent edges were all backup QBs — career averages full "
            "of starts against recent DNPs, so the model screamed UNDER. Correct, "
            "and worthless: there's no line on a QB who won't take a snap. Kenny "
            "Pickett projected 7.0 completions, actual 0, scored as a _win_. Now "
            "gated behind a projected-volume floor."),
    divider(),
    section("*A number to discard.*\n"
            "An earlier _60% directional accuracy_ figure came from a backtest "
            "harness that printed results irreproducible from its own returned "
            "data. Scorer was rebuilt from scratch; the table above is from the "
            "clean path. The old harness is quarantined."),
    section("*Week 1 2026 retrospective:* 12/17 correct — *not meaningful.* "
            "17 samples, and the pseudo-line is a career average, far softer than "
            "anything FanDuel would hang."),
    divider(),
    section("*Next:* fix the receiving volume model (target share + routes run, "
            "not raw targets). Needs no API key.\n"
            "*Blocked:* phase 3 needs `ODDS_API_KEY`. Until real lines are wired "
            "in, no hit rate from this system means anything."),
]

print(post(blocks=blocks, text="NFL Props — Phase 2 findings"))
