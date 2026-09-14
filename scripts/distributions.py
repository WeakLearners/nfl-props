#!/usr/bin/env python
"""Production distributions for stars in a game. No lines, no market.

For each player-stat: the projected mean, the spread around it, and the
opponent adjustment shown separately so you can see what the matchup did.

The spread is widened by 10% over the model's raw estimate. Measured on
2023-2025, actual errors were about 10% larger than the model's own sd -- the
raw number understates how wrong a projection can be, because it only counts a
player's game-to-game variation and not the error in the projection itself.
"""
import sys, pathlib, io, contextlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import pandas as pd, numpy as np
from nflprops import project as P, reliability as R
from nflprops.project import defense_vs_position, SPECS, DEF_WEIGHT
from nflprops.db import connect

SEASON, WEEK = int(sys.argv[1]), int(sys.argv[2])
TEAMS = sys.argv[3].split(",") if len(sys.argv) > 3 else None
SD_INFLATE = 1.10

with contextlib.redirect_stdout(io.StringIO()):
    pr = P.project_week(SEASON, WEEK)
    stars = set(R.star_pool(SEASON, WEEK).player_key)

from nflprops.odds import name_key
pr["player_key"] = pr["player"].map(name_key)
pr = pr[pr.player_key.isin(stars)]
depth = R.star_pool(SEASON, WEEK).set_index("player_key")

# The RAW matchup rating, shown whether or not the model applies it. For
# receiving the model uses a weight of 0 -- applying it measurably hurt
# accuracy, because "yards allowed to WRs" mostly records which receivers a
# defence happened to face. Worth seeing even when it is not used.
dvp = defense_vs_position(SEASON, WEEK)
dvp_ix = dvp.set_index(["defteam", "position"]) if not dvp.empty else None


def raw_mult(row):
    if dvp_ix is None or (row.opponent, row.position) not in dvp_ix.index:
        return float("nan")
    d = dvp_ix.loc[(row.opponent, row.position)]
    col = SPECS[row.stat][2]
    if col not in d:
        return float("nan")
    try:
        v = float(d.get(col))
    except (TypeError, ValueError):
        return float("nan")
    return v


pr["raw_def"] = pr.apply(raw_mult, axis=1)
pr["applied"] = pr.stat.map(lambda s: DEF_WEIGHT.get(s, 1.0) > 0)
if TEAMS:
    pr = pr[pr.team.isin(TEAMS)]

pr["sd_adj"] = pr.sd * SD_INFLATE
pr["lo"] = (pr.proj - pr.sd_adj).clip(lower=0)
pr["hi"] = pr.proj + pr.sd_adj

order = {"passing_yards": 0, "completions": 1, "rushing_yards": 2,
         "receiving_yards": 3, "receptions": 4}
pr = pr.sort_values(["team", "stat", "proj"], key=lambda c: c.map(order) if c.name == "stat" else c,
                    ascending=[True, True, False])

print(f"\n{'='*94}")
print(f"  PRODUCTION RANGES — {SEASON} W{WEEK}   (mean, and the middle ~68% of outcomes)")
print(f"{'='*94}")
hdr = (f"{'role':<5}{'player':<20}{'vs':<5}{'stat':<16}{'expected':>10}"
       f"{'-1 sd':>8}{'+1 sd':>8}{'matchup':>22}")
print(hdr); print("-" * len(hdr))
for t, g in pr.groupby("team"):
    for r in g.itertuples():
        d = depth.loc[r.player_key] if r.player_key in depth.index else None
        role = f"{r.position}{int(d['rank'])}" if d is not None else r.position
        if pd.isna(r.raw_def):
            mtag = "no rating"
        else:
            word = ("easier" if r.raw_def > 1.03 else
                    "tougher" if r.raw_def < 0.97 else "average")
            mtag = f"{r.raw_def:.2f} {word}" + ("" if r.applied else " (not applied)")
        print(f"{role:<5}{r.player[:19]:<20}{r.opponent:<5}{r.stat.replace('_',' '):<16}"
              f"{r.proj:>10.1f}{r.lo:>8.1f}{r.hi:>8.1f}  {mtag:<22}")
    print()

out = pathlib.Path(__file__).parent.parent / f"reports/dist_{SEASON}_w{WEEK}.csv"
pr.to_csv(out, index=False)
print(f"-> {out.name}")
print("\nmatchup = what this defence gives up to the position vs league average.")
print("1.15 = allows 15% more than average; 0.85 = 15% less.")
print("'not applied' means the model measured this adjustment and found it made")
print("projections WORSE for that stat, so it is shown for context only.")
