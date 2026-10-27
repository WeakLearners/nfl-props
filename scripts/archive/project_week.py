#!/usr/bin/env python
"""Project a week, and optionally score it against what actually happened.

Usage:  project_week.py SEASON WEEK [--score] [--top N]
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import numpy as np, pandas as pd
from nflprops.project import project_week, SPECS, POS_STATS, _history
from nflprops.db import connect

pd.set_option("display.width", 200)


def actuals(season, week):
    with connect() as con:
        a = pd.read_sql_query(
            "SELECT * FROM player_games WHERE season=? AND week=? AND season_type='REG'",
            con, params=[season, week])
    out = []
    for stat in SPECS:
        s = a[["player_id", stat]].rename(columns={stat: "actual"})
        s["stat"] = stat
        out.append(s)
    return pd.concat(out, ignore_index=True)


def naive_baseline(season, week):
    """Unweighted mean of the player's prior games -- no recency weighting, no
    opponent adjustment. This is the bar the model has to clear to justify
    existing, and it is roughly where a sportsbook hangs its line."""
    hist = _history(season, week)
    rows = []
    for stat in SPECS:
        g = hist.groupby("player_id")[stat].agg(["mean", "count"]).reset_index()
        g["stat"] = stat
        rows.append(g.rename(columns={"mean": "baseline"}))
    return pd.concat(rows, ignore_index=True)[["player_id", "stat", "baseline", "count"]]


def score(proj, season, week):
    df = (proj.merge(actuals(season, week), on=["player_id", "stat"], how="inner")
               .merge(naive_baseline(season, week), on=["player_id", "stat"], how="left"))
    # Only score props that would plausibly be offered: a book does not hang a
    # receiving-yards line on a player projected for 4 yards.
    df = df[df.proj >= 15] if False else df
    print(f"\n=== SCORING {season} W{week} — {df.game_id.nunique()} games, {len(df)} player-stats ===\n")

    hdr = f"{'stat':<17}{'n':>5}{'MAE':>9}{'base MAE':>10}{'improve':>9}{'corr':>7}{'dir%':>7}"
    print(hdr); print("-" * len(hdr))
    agg = []
    for stat, g in df.groupby("stat"):
        g = g[g.baseline.notna() & (g["count"] >= 3)]
        # A book's line sits near the naive average, so "did we correctly call
        # which side of that average the player landed on" is a usable stand-in
        # for over/under accuracy until real lines are wired in.
        live = g[(g.proj > 1) & (g.baseline > 1) & (abs(g.proj - g.baseline) > 0.05 * g.baseline)]
        dir_ok = ((live.proj > live.baseline) == (live.actual > live.baseline)).mean() * 100
        mae, bmae = (g.proj - g.actual).abs().mean(), (g.baseline - g.actual).abs().mean()
        corr = g[["proj", "actual"]].corr().iloc[0, 1]
        agg.append((stat, len(g), mae, bmae, dir_ok, len(live)))
        print(f"{stat:<17}{len(g):>5}{mae:>9.2f}{bmae:>10.2f}{(bmae-mae)/bmae*100:>8.1f}%{corr:>7.2f}{dir_ok:>6.1f}%")

    tot_n = sum(a[1] for a in agg)
    w_dir = sum(a[4] * a[5] for a in agg) / sum(a[5] for a in agg)
    print("-" * len(hdr))
    print(f"{'WEIGHTED':<17}{tot_n:>5}{'':>9}{'':>10}{'':>9}{'':>7}{w_dir:>6.1f}%")
    print("\ndir% = share of calls where the model picked the correct side of the naive")
    print("average. 50% is a coin flip; ~53.5% is the break-even against -115 juice.")
    return df


if __name__ == "__main__":
    season, week = int(sys.argv[1]), int(sys.argv[2])
    proj = project_week(season, week)
    print(f"projected {len(proj)} player-stats across {proj.game_id.nunique()} games")
    if "--score" in sys.argv:
        df = score(proj, season, week)
        df.to_csv(pathlib.Path(__file__).parent.parent / f"reports/score_{season}_w{week}.csv", index=False)
    else:
        n = int(sys.argv[sys.argv.index("--top") + 1]) if "--top" in sys.argv else 25
        print(proj.sort_values("proj", ascending=False).head(n).to_string(index=False))
