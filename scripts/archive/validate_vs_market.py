#!/usr/bin/env python
"""THE decisive test: is my projection better than FanDuel's line?

Pulls historical FanDuel props from just before kickoff on sampled Sundays,
joins them to the projection the model would have made with only prior data,
and to what actually happened. Then asks two questions:

  1. Whose point estimate is closer to the actual result -- mine or the line?
  2. If I had bet every prop where the model claimed an edge, what is the ROI
     at the prices actually offered?

Question 2 is the only one that pays. A model can be less accurate than the
market overall and still profit on a subset, but if it loses BOTH it is done.
"""
import sys, pathlib, io, contextlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import pandas as pd, numpy as np
from nflprops import histodds as H, odds as O, project as P
from nflprops.db import connect, team_abbr_map
from scripts_util import MIN_VOL

# Sunday snapshots at 16:45Z = 12:45pm ET, just before the 1pm kickoffs.
SNAPSHOTS = [("2025-10-12T16:45:00Z", 2025, 6),
             ("2025-11-09T16:45:00Z", 2025, 10),
             ("2025-12-07T16:45:00Z", 2025, 14)]
MIN_EDGE = float(sys.argv[1]) if len(sys.argv) > 1 else 0.03


def actuals(season, week):
    with connect() as con:
        a = pd.read_sql_query("SELECT * FROM player_games WHERE season=? AND week=? AND season_type='REG'",
                              con, params=[season, week])
    return pd.concat([a[["player_display_name", s]].rename(columns={s: "actual"}).assign(stat=s)
                      for s in O.MARKETS.values()], ignore_index=True)


tm = team_abbr_map()
all_rows, spent = [], 0
for snap, season, week in SNAPSHOTS:
    ev, hit = H.events(snap)
    if ev.empty:
        print(f"{snap}: no events"); continue
    ev["commence_time"] = pd.to_datetime(ev.commence_time, utc=True)
    ev = ev[ev.commence_time > pd.Timestamp(snap)]          # pre-game only
    print(f"{snap}  season {season} W{week}: {len(ev)} pre-game events")

    frames = []
    for e in ev.itertuples():
        df, cached = H.props(e.id, snap)
        spent += 0 if cached else 1
        if not df.empty:
            frames.append(df)
    if not frames:
        continue
    raw = pd.concat(frames, ignore_index=True)

    dv = O.devig(raw)
    dv["player_key"] = dv.player_book.map(O.name_key)

    with contextlib.redirect_stdout(io.StringIO()):
        pr = P.project_week(season, week)
    pr = pr[pr.apply(lambda r: r.proj_vol >= MIN_VOL.get(r.stat, 0), axis=1)].copy()
    pr["player_key"] = pr["player"].map(O.name_key)

    act = actuals(season, week)
    act["player_key"] = act.player_display_name.map(O.name_key)

    m = O.edges(pr[["player_key", "stat", "proj", "sd", "player", "team", "opponent"]], dv)
    m = m.merge(act[["player_key", "stat", "actual"]], on=["player_key", "stat"], how="inner")
    m["season"], m["week"] = season, week
    all_rows.append(m)

d = pd.concat(all_rows, ignore_index=True).dropna(subset=["actual"])
d.to_csv("reports/validate_vs_market.csv", index=False)
print(f"\n{len(d):,} props matched to lines AND actuals   (new API calls: {spent})")

print(f"\n{'='*68}\n  Q1: whose point estimate is closer to reality?\n{'='*68}")
print(f"{'stat':<17}{'n':>6}{'model MAE':>11}{'line MAE':>10}{'verdict':>12}")
print("-" * 56)
for st, g in d.groupby("stat"):
    mm, lm = (g.proj - g.actual).abs().mean(), (g.line - g.actual).abs().mean()
    print(f"{st:<17}{len(g):>6,}{mm:>11.2f}{lm:>10.2f}{('MODEL' if mm < lm else 'LINE'):>12}")
mm, lm = (d.proj - d.actual).abs().mean(), (d.line - d.actual).abs().mean()
print("-" * 56)
print(f"{'ALL':<17}{len(d):>6,}{mm:>11.2f}{lm:>10.2f}{('MODEL' if mm < lm else 'LINE'):>12}")

print(f"\n{'='*68}\n  Q2: ROI of betting every edge >= {MIN_EDGE:.0%}\n{'='*68}")
b = d[d.edge >= MIN_EDGE].copy()
b["won"] = np.where(b.side == "OVER", b.actual > b.line, b.actual < b.line)
b["push"] = b.actual == b.line
b = b[~b.push]
if len(b):
    b["profit"] = np.where(b.won, b.price.map(O.decimal) - 1, -1.0)
    print(f"  bets            {len(b):,}")
    print(f"  hit rate        {b.won.mean()*100:.1f}%   (break-even ~52.4% at -110)")
    print(f"  ROI per $1      {b.profit.mean()*100:+.1f}%")
    print(f"  total P&L       {b.profit.sum():+.1f} units")
    print(f"\n  by stat:")
    for st, g in b.groupby("stat"):
        print(f"    {st:<17}{len(g):>5} bets   {g.won.mean()*100:>5.1f}% hit   {g.profit.mean()*100:>+6.1f}% ROI")
else:
    print("  no bets cleared the threshold")
