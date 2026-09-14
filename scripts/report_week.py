#!/usr/bin/env python
"""Per-matchup recommendations for a week, with actuals if the games are done.

Only stats that beat the naive baseline in walk-forward backtest are eligible.
receiving_yards and receptions are excluded: the model is worse than a career
average on them (-11.9% / -0.0% MAE over 2023-2025, n=14,500 each), so any
edge it claims there is noise. Re-enable by fixing the model, not the filter.
"""
import sys, pathlib, io, contextlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import pandas as pd
from nflprops import project as P
from score import _actuals, _baseline

ELIGIBLE = ["passing_yards", "rushing_yards", "completions",
            "receiving_yards", "receptions"]

# Minimum projected volume for a prop to plausibly EXIST at the book. Without
# this the model's biggest "edges" are all backup QBs whose career average
# includes starts but whose recent games are DNPs -- it screams UNDER, and
# there is no line to bet because the player will not take a snap. Filtering
# on projected volume is a v1 proxy for depth-chart-aware starter detection.
MIN_VOL = {"passing_yards": 20.0, "completions": 20.0, "rushing_yards": 7.0,
           "receiving_yards": 3.0, "receptions": 3.0}

# Lowest number FanDuel plausibly hangs. Two filters are needed and they catch
# different things: MIN_VOL uses the PROJECTION (kills backup QBs, whose career
# average is high but who will not play), BETTABLE uses the BASELINE (kills
# deep-bench scrubs, who never had a market in the first place).
BETTABLE = {"passing_yards": 150.0, "completions": 12.0, "rushing_yards": 25.0,
            "receiving_yards": 25.0, "receptions": 2.0}
SEASON, WEEK = int(sys.argv[1]), int(sys.argv[2])

with contextlib.redirect_stdout(io.StringIO()):
    pr = P.project_week(SEASON, WEEK)
pr = pr[pr.stat.isin(ELIGIBLE)]
pr = pr[pr.apply(lambda r: r.proj_vol >= MIN_VOL[r.stat], axis=1)]

df = (pr.merge(_baseline(SEASON, WEEK), on=["player_id", "stat"], how="left")
        .merge(_actuals(SEASON, WEEK), on=["player_id", "stat"], how="left"))
df = df[df.base.notna() & (df.n_prior >= 5)].copy()
df = df[df.apply(lambda r: r.base >= BETTABLE[r.stat], axis=1)]

# Edge proxy until real lines are wired in: how far the projection sits from
# where a book would roughly hang the number. Scaled by the player's own game
# to game sd so a 20-yard gap on a volatile RB is not treated like a 20-yard
# gap on a metronome QB.
df["gap"] = df.proj - df.base
df["edge_z"] = df.gap / df.sd.clip(lower=1)
df["side"] = df.gap.apply(lambda g: "OVER" if g > 0 else "UNDER")
df["hit"] = df.apply(
    lambda r: "" if pd.isna(r.actual) else
    ("WIN " if (r.actual > r.base) == (r.gap > 0) else "loss"), axis=1)

with __import__("nflprops.db", fromlist=["connect"]).connect() as con:
    sched = pd.read_sql_query(
        "SELECT game_id, away_team, home_team, gameday, weekday FROM schedules WHERE season=? AND week=?",
        con, params=[SEASON, WEEK])

print(f"\n{'='*78}\n  {SEASON} WEEK {WEEK} — top 2 props per matchup\n{'='*78}")
rows = []
for _, gm in sched.iterrows():
    g = df[df.game_id == gm.game_id]
    if g.empty:
        continue
    top = g.reindex(g.edge_z.abs().sort_values(ascending=False).index).head(2)
    print(f"\n{gm.away_team} @ {gm.home_team}  ({gm.weekday} {gm.gameday})")
    for _, r in top.iterrows():
        act = "—" if pd.isna(r.actual) else f"{r.actual:.0f}"
        print(f"   {r.hit:<5}{r['player']:<22}{r.stat:<17}{r.side:<6}"
              f"proj {r.proj:>6.1f}  line~{r.base:>6.1f}  actual {act:>5}")
        rows.append(r)

res = pd.DataFrame(rows)
done = res[res.hit != ""]
if len(done):
    w = (done.hit == "WIN ").sum()
    print(f"\n{'='*78}")
    print(f"  {w}/{len(done)} correct ({w/len(done)*100:.0f}%) on games with final data")
    print(f"  NOTE: 'line~' is the naive career average, NOT a FanDuel line.")
    print(f"  A real book line is far sharper, so this overstates true hit rate.")
    print(f"{'='*78}")
