#!/usr/bin/env python
"""Closing-line value: does the model pick the side the market moves toward?

Bets are priced at the THURSDAY line (the softest real market -- props do not
post before then) and graded against the SUNDAY pre-kickoff line. If the model
has genuine information, the line should drift toward its side. CLV is the
professional standard for edge because it converges far faster than win/loss:
a 55% CLV rate over a few hundred props means more than a 55% win rate does.

Null hypothesis is 50%. Anything under ~52% is not worth acting on.
"""
import sys, pathlib, io, contextlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import pandas as pd, numpy as np
from nflprops import histodds as H, odds as O, project as P
from scripts_util import MIN_VOL

WEEKS = [("2025-10-09T18:00:00Z", "2025-10-12T16:45:00Z", 2025, 6),
         ("2025-11-06T18:00:00Z", "2025-11-09T16:45:00Z", 2025, 10),
         ("2025-12-04T18:00:00Z", "2025-12-07T16:45:00Z", 2025, 14)]


def snap_props(snapshot, cutoff):
    """All FanDuel props at one snapshot, for games kicking off after cutoff."""
    ev, _ = H.events(snapshot)
    if ev.empty:
        return pd.DataFrame()
    ev["ct"] = pd.to_datetime(ev.commence_time, utc=True)
    ev = ev[ev.ct > pd.Timestamp(cutoff)]
    frames = []
    for e in ev.itertuples():
        df, _ = H.props(e.id, snapshot)
        if not df.empty:
            frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


rows = []
for thu, sun, season, week in WEEKS:
    early = snap_props(thu, sun)      # only games still unplayed at the Sunday snapshot
    late = snap_props(sun, sun)
    if early.empty or late.empty:
        print(f"W{week}: missing snapshot data"); continue

    e_dv, l_dv = O.devig(early), O.devig(late)
    for df in (e_dv, l_dv):
        df["player_key"] = df.player_book.map(O.name_key)

    with contextlib.redirect_stdout(io.StringIO()):
        pr = P.project_week(season, week)
    pr = pr[pr.apply(lambda r: r.proj_vol >= MIN_VOL.get(r.stat, 0), axis=1)].copy()
    pr["player_key"] = pr["player"].map(O.name_key)

    # Model picks a side at the THURSDAY line.
    m = O.edges(pr[["player_key", "stat", "proj", "sd", "player", "team", "opponent"]], e_dv)
    m = m.rename(columns={"line": "line_early", "price": "price_early",
                          "fair_over": "fair_over_early"})
    late_cols = l_dv[["event_id", "stat", "player_key", "line", "fair_over",
                      "price_over", "price_under"]].rename(
        columns={"line": "line_close", "fair_over": "fair_over_close",
                 "price_over": "price_over_close", "price_under": "price_under_close"})
    m = m.merge(late_cols, on=["event_id", "stat", "player_key"], how="inner")
    m["season"], m["week"] = season, week
    rows.append(m)
    print(f"W{week}: {len(early)} early rows, {len(late)} close rows -> {len(m)} matched props")

d = pd.concat(rows, ignore_index=True)

# Did the market move toward our side?
#   OVER  bet wins CLV if the line rose  (we need fewer yards than the close asks)
#   UNDER bet wins CLV if the line fell
moved = np.where(d.side == "OVER", d.line_close - d.line_early, d.line_early - d.line_close)
d["line_move"] = moved
# When the number did not move, fall back to the price: a shorter price on our
# side at the close means the market came to us.
fair_now = np.where(d.side == "OVER", d.fair_over_close, 1 - d.fair_over_close)
fair_then = np.where(d.side == "OVER", d.fair_over_early, 1 - d.fair_over_early)
d["prob_move"] = fair_now - fair_then
d["beat_close"] = np.where(moved != 0, moved > 0, d.prob_move > 0)
d = d[~((moved == 0) & (d.prob_move == 0))]          # genuinely unchanged: no information
d.to_csv("reports/clv_test.csv", index=False)

print(f"\n{'='*64}\n  CLV — did the line move toward the model's side?\n{'='*64}")
print(f"{'stat':<17}{'n':>6}{'beat close':>12}{'z vs 50%':>10}{'avg move':>10}")
print("-" * 55)
for st, g in d.groupby("stat"):
    p = g.beat_close.mean(); se = np.sqrt(p * (1 - p) / len(g))
    print(f"{st:<17}{len(g):>6}{p*100:>11.1f}%{(p-.5)/se:>10.2f}{g.line_move.mean():>10.2f}")
p = d.beat_close.mean(); se = np.sqrt(p * (1 - p) / len(d))
print("-" * 55)
print(f"{'ALL':<17}{len(d):>6}{p*100:>11.1f}%{(p-.5)/se:>10.2f}{d.line_move.mean():>10.2f}")

print(f"\n  by claimed edge:")
for b, g in d.groupby(pd.cut(d.edge, [-1, .03, .06, .10, 1]), observed=True):
    pp = g.beat_close.mean()
    print(f"    edge {str(b):<14}{len(g):>5} props   beat close {pp*100:>5.1f}%")
