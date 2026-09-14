#!/usr/bin/env python
"""Reconstruct what the Sunday-morning report would have recommended.

Uses the historical snapshot so this is exactly what the tool would have seen
at 12:45pm ET -- no hindsight in the odds, and clear rates computed from prior
weeks only.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import pandas as pd
from nflprops import histodds as H, rank as K
from nflprops.db import connect

SNAP = sys.argv[1] if len(sys.argv) > 1 else "2026-09-13T16:45:00Z"
SEASON, WEEK = int(sys.argv[2]), int(sys.argv[3])

ev, _ = H.events(SNAP)
ev["ct"] = pd.to_datetime(ev.commence_time, utc=True)
ev = ev[(ev.ct > pd.Timestamp(SNAP)) & (ev.ct < pd.Timestamp(SNAP) + pd.Timedelta(hours=5))]
print(f"{len(ev)} games kicking off in the window after {SNAP}")

frames = []
for e in ev.itertuples():
    df, cached = H.alt_props(e.id, SNAP)
    if not df.empty:
        df["game"] = f"{e.away_team} @ {e.home_team}"
        frames.append(df)
if not frames:
    sys.exit("no alternate props at that snapshot")
alt = pd.concat(frames, ignore_index=True)
print(f"{len(alt)} alternate legs fetched")

best = K.best_legs(alt, SEASON, WEEK, n=12, one_per_game=True)
print(f"\n{'='*104}")
print(f"  SUNDAY MORNING RECS — {SEASON} W{WEEK}   (one leg per game, ranked by reliability)")
print(f"{'='*104}")
hdr = f"{'#':>3}  {'player':<21}{'game':<24}{'the bet':<32}{'price':>7}{'cleared':>10}  notes"
print(hdr); print("-" * len(hdr))
for i, r in enumerate(best.itertuples(), 1):
    bet = f"{r.stat.replace('_',' ')} over {r.line:g}"
    cl = f"{r.hit_rate*100:.0f}% of {int(r.n_games)}"
    print(f"{i:>3}  {r.player_book[:20]:<21}{r.game[:23]:<24}{bet:<32}{int(r.price):>7}{cl:>10}  {K.notes(r)}")

best.to_csv(pathlib.Path(__file__).parent.parent / f"reports/morning_{SEASON}_w{WEEK}.csv", index=False)

# grade if box scores have landed
with connect() as con:
    pg = pd.read_sql_query(
        "SELECT player_display_name, passing_yards, rushing_yards, receiving_yards, receptions "
        "FROM player_games WHERE season=? AND week=?", con, params=[SEASON, WEEK])
from nflprops.odds import name_key
pg["player_key"] = pg.player_display_name.map(name_key)
graded = 0
for r in best.itertuples():
    row = pg[pg.player_key == r.player_key]
    if len(row) and pd.notna(row.iloc[0].get(r.stat)):
        graded += 1
print(f"\nbox scores available for {graded}/{len(best)} of these legs"
      + ("" if graded else "  -- nflverse posts them overnight; grade tomorrow"))
