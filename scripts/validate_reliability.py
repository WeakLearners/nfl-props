#!/usr/bin/env python
"""Does historical clear rate add anything BEYOND the price?

At a fixed price the market has already equalised its own view, so the only
way the ranking in safest.py is worth anything is if hist% predicts outcomes
within a price band. If it does not, picking randomly among same-priced legs is
just as good and the tool should say so.

Hard-capped on API spend. Alternate props cost ~40 credits per game.
"""
import sys, pathlib, urllib.request
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import pandas as pd, numpy as np
from nflprops import histodds as H, reliability as R
from nflprops.config import require
from nflprops.odds import name_key
from nflprops.db import connect

CREDIT_CAP = 1500
LO, HI = -320, -180
SNAPS = [("2025-10-12T16:45:00Z", 2025, 6),
         ("2025-11-09T16:45:00Z", 2025, 10),
         ("2025-12-07T16:45:00Z", 2025, 14)]
MAX_GAMES = 12          # per snapshot; ~40 credits each


def remaining():
    with urllib.request.urlopen(
            f"https://api.the-odds-api.com/v4/sports/?apiKey={require('ODDS_API_KEY')}",
            timeout=20) as r:
        return int(r.headers.get("x-requests-remaining"))


start = remaining()
print(f"credits at start: {start:,}   cap for this run: {CREDIT_CAP:,}")

rows = []
for snap, season, week in SNAPS:
    ev, _ = H.events(snap)
    ev["ct"] = pd.to_datetime(ev.commence_time, utc=True)
    # same-day games only, so the snapshot is genuinely just-before-kickoff
    ev = ev[(ev.ct > pd.Timestamp(snap)) &
            (ev.ct < pd.Timestamp(snap) + pd.Timedelta(hours=10))].head(MAX_GAMES)
    got = 0
    for e in ev.itertuples():
        if start - remaining() > CREDIT_CAP:
            print("  ! credit cap reached, stopping"); break
        df, cached = H.alt_props(e.id, snap)
        if not df.empty:
            df["season"], df["week"] = season, week
            rows.append(df); got += len(df)
    print(f"  {snap[:10]} W{week}: {len(ev)} games, {got} alternate legs")

alt = pd.concat(rows, ignore_index=True)
alt = alt[(alt.price >= LO) & (alt.price <= HI)].copy()
alt["player_key"] = alt.player_book.map(name_key)
print(f"\n{len(alt)} alternate legs in the {LO}..{HI} band")

# actual outcomes
with connect() as con:
    pg = pd.read_sql_query(
        "SELECT season, week, player_display_name, passing_yards, rushing_yards, "
        "receiving_yards, receptions FROM player_games WHERE season=2025 AND season_type='REG'", con)
pg["player_key"] = pg.player_display_name.map(name_key)
act = pg.melt(id_vars=["season", "week", "player_key"],
              value_vars=["passing_yards", "rushing_yards", "receiving_yards", "receptions"],
              var_name="stat", value_name="actual")
alt = alt.merge(act, on=["season", "week", "player_key", "stat"], how="inner").dropna(subset=["actual"])
alt["won"] = (alt.actual > alt.line).astype(int)
alt["fair"] = alt.price.map(R.fair)

# hist% as of that week -- computed with prior data only
hist_frames = {w: R._history(2025, w, R.DEFAULT_LOOKBACK) for w in alt.week.unique()}
hr = alt.apply(lambda r: R.clear_rate(hist_frames[r.week], r.player_key, r.stat, r.line),
               axis=1, result_type="expand")
alt[["hist", "n_games", "dnp"]] = hr
alt = alt[alt.n_games >= 6]
alt.to_csv("reports/validate_reliability.csv", index=False)

spent = start - remaining()
print(f"{len(alt)} legs with both an outcome and a usable history   (credits spent: {spent:,})")

print(f"\n{'='*66}\n  Calibration of the band\n{'='*66}")
print(f"  mean price implies {alt.fair.mean()*100:.1f}%   actual hit rate {alt.won.mean()*100:.1f}%   n={len(alt)}")

def auc(score, y):
    o = np.argsort(score); y = np.asarray(y)[o]
    n1 = y.sum(); n0 = len(y) - n1
    if n1 == 0 or n0 == 0: return np.nan
    r = np.arange(1, len(y) + 1)
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)

print(f"\n{'='*66}\n  Does hist% rank outcomes? (AUC, 0.5 = useless)\n{'='*66}")
print(f"  price alone   AUC = {auc(alt.fair.values, alt.won.values):.3f}")
print(f"  hist% alone   AUC = {auc(alt.hist.fillna(alt.hist.mean()).values, alt.won.values):.3f}")
blend = 0.65 * alt.fair + 0.35 * alt.hist.fillna(alt.fair)
print(f"  blended       AUC = {auc(blend.values, alt.won.values):.3f}")

print(f"\n{'='*66}\n  Within a NARROW price band, does hist% still separate?\n{'='*66}")
narrow = alt[(alt.price >= -280) & (alt.price <= -200)]
print(f"  band -200..-280, n={len(narrow)}, mean implied {narrow.fair.mean()*100:.1f}%")
for b, g in narrow.groupby(pd.qcut(narrow.hist, 3, duplicates="drop"), observed=True):
    p = g.won.mean(); se = np.sqrt(p * (1 - p) / max(len(g), 1))
    print(f"    hist {str(b):<16} n={len(g):>4}  actual {p*100:>5.1f}%  (+/- {se*100:.1f})")

print(f"\n{'='*66}\n  Availability screens\n{'='*66}")
for lbl, mask in [("misses >25% of games", alt.dnp > 0.25), ("plays reliably", alt.dnp <= 0.25)]:
    g = alt[mask]
    if len(g):
        print(f"  {lbl:<24} n={len(g):>4}  hit {g.won.mean()*100:>5.1f}%  implied {g.fair.mean()*100:>5.1f}%"
              f"  delta {(g.won.mean()-g.fair.mean())*100:>+5.1f}")
