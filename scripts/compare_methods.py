#!/usr/bin/env python
"""Fixed-price vs price-band selection, on the 2025 validation weeks.

Uses only cached historical odds -- no new API spend. This is the sample that
can actually separate the two methods; a single Week 1 slate cannot.
"""
import sys, pathlib, io, contextlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import pandas as pd, numpy as np
from nflprops import histodds as H, rank as K
from nflprops.db import connect
from nflprops.odds import name_key

SNAPS = [("2025-10-12T16:45:00Z", 2025, 6),
         ("2025-11-09T16:45:00Z", 2025, 10),
         ("2025-12-07T16:45:00Z", 2025, 14)]
STATS = ["passing_yards", "rushing_yards", "receiving_yards", "receptions"]


def dec(a):
    return 1 + (100 / (-a) if a < 0 else a / 100)


def load(snap, season, week):
    ev, _ = H.events(snap)
    ev["ct"] = pd.to_datetime(ev.commence_time, utc=True)
    ev = ev[(ev.ct > pd.Timestamp(snap)) & (ev.ct < pd.Timestamp(snap) + pd.Timedelta(hours=10))]
    frames = []
    for e in ev.itertuples():
        df, _ = H.alt_props(e.id, snap)
        if not df.empty:
            df["game"] = f"{e.away_team} @ {e.home_team}"
            frames.append(df)
    if not frames:
        return None, None
    alt = pd.concat(frames, ignore_index=True)
    with contextlib.redirect_stdout(io.StringIO()):
        r = K.rank(alt, season, week)
    r = r[r.n_games >= K.MIN_HISTORY_GAMES]
    r = K.starters_only(r)
    r = r[~r.report_status.isin(["Questionable", "Doubtful", "Out"])]
    with contextlib.redirect_stdout(io.StringIO()):
        r = K.stars_only(r, season, week)
    with connect() as con:
        pg = pd.read_sql_query(
            "SELECT player_display_name, " + ", ".join(STATS) +
            " FROM player_games WHERE season=? AND week=?", con, params=[season, week])
    pg["player_key"] = pg.player_display_name.map(name_key)
    return r, pg


def graded(df, act):
    w = []
    for r in df.itertuples():
        m = act[act.player_key == r.player_key]
        a = m.iloc[0][r.stat] if len(m) and pd.notna(m.iloc[0][r.stat]) else None
        w.append(np.nan if a is None else float(a > r.line))
    df = df.copy(); df["won"] = w
    return df[df.won.notna()]


def nearest(g, target):
    g = g.copy(); g["dist"] = (g.price - target).abs()
    return (g.sort_values("dist").drop_duplicates(["player_key", "stat"])
             .assign(p=lambda d: [K.calibrated_p(K.W_PRICE * m + K.W_HIST * h)
                                  for m, h in zip(d.mkt_fair, d.hit_rate)])
             .drop_duplicates("player_key").sort_values("p", ascending=False))


def band(g, lo=-320, hi=-180):
    return (g[(g.price >= lo) & (g.price <= hi)]
            .drop_duplicates("player_key").sort_values("score", ascending=False))


METHODS = {
    "nearest -200, top 4/game": lambda g: nearest(g, -200).head(4),
    "nearest -250, top 4/game": lambda g: nearest(g, -250).head(4),
    "nearest -300, top 1/game": lambda g: nearest(g, -300).head(1),
    "band, top 4/game":         lambda g: band(g).head(4),
    "band, top 1/game":         lambda g: band(g).head(1),
}

acc = {k: [] for k in METHODS}
for snap, season, week in SNAPS:
    r, act = load(snap, season, week)
    if r is None:
        continue
    print(f"W{week}: {r.game.nunique()} games, {len(r)} eligible star legs")
    for name, sel in METHODS.items():
        picks = [sel(g) for _, g in r.groupby("game")]
        picks = [p for p in picks if len(p)]
        if picks:
            acc[name].append(graded(pd.concat(picks), act))

print(f"\n{'method':<28}{'graded':>8}{'hit':>8}{'avg price':>11}{'ROI':>9}{'se':>7}")
print("-" * 71)
for name, frames in acc.items():
    if not frames:
        continue
    d = pd.concat(frames)
    roi = np.where(d.won == 1, d.price.map(dec) - 1, -1.0)
    p = d.won.mean(); se = np.sqrt(p * (1 - p) / len(d))
    print(f"{name:<28}{len(d):>8}{p*100:>7.1f}%{d.price.mean():>11.0f}"
          f"{roi.mean()*100:>+8.1f}%{se*100:>6.1f}")
