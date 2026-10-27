#!/usr/bin/env python
"""Walk-forward backtest. Re-projects every week using only data available
before it, then scores. One week is an anecdote; this is the actual answer."""
import sys, pathlib, io, contextlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import pandas as pd
from nflprops import project as P
from project_week import actuals, naive_baseline

WEEKS = [(s, w) for s in (2023, 2024, 2025) for w in range(1, 19)]


def collect(no_def=False):
    real = P.defense_vs_position
    if no_def:
        P.defense_vs_position = lambda *a, **k: pd.DataFrame()
    frames = []
    try:
        for s, w in WEEKS:
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    pr = P.project_week(s, w)
                df = (pr.merge(actuals(s, w), on=["player_id", "stat"])
                        .merge(naive_baseline(s, w), on=["player_id", "stat"], how="left"))
                df["season"], df["week"] = s, w
                frames.append(df)
            except Exception as e:
                print(f"  skip {s} W{w}: {e}", file=sys.stderr)
    finally:
        P.defense_vs_position = real
    return pd.concat(frames, ignore_index=True)


def report(df, label):
    df = df[df.baseline.notna() & (df["count"] >= 5)]
    print(f"\n### {label}  (n={len(df):,})")
    hdr = f"{'stat':<17}{'n':>7}{'MAE':>8}{'base':>8}{'improve':>9}{'dir%':>8}{'live n':>8}"
    print(hdr); print("-" * len(hdr))
    tot_c = tot_l = 0
    for stat, g in df.groupby("stat"):
        live = g[(g.proj > 1) & (g.baseline > 1) & (abs(g.proj - g.baseline) > 0.05 * g.baseline)]
        hit = ((live.proj > live.baseline) == (live.actual > live.baseline))
        mae, bmae = (g.proj - g.actual).abs().mean(), (g.baseline - g.actual).abs().mean()
        tot_c += hit.sum(); tot_l += len(live)
        print(f"{stat:<17}{len(g):>7,}{mae:>8.2f}{bmae:>8.2f}{(bmae-mae)/bmae*100:>8.1f}%{hit.mean()*100:>7.1f}%{len(live):>8,}")
    print("-" * len(hdr))
    print(f"{'ALL':<17}{len(df):>7,}{'':>8}{'':>8}{'':>9}{tot_c/tot_l*100:>7.1f}%{tot_l:>8,}")
    return tot_c / tot_l * 100


if __name__ == "__main__":
    full = collect(no_def=False)
    nod = collect(no_def=True)
    full.to_csv("reports/backtest_full.csv", index=False)
    a = report(full, "full model (with opponent adjustment)")
    b = report(nod, "no opponent adjustment")
    print(f"\nbreak-even vs -115 juice = 53.5%")
    print(f"full {a:.1f}%   no-def {b:.1f}%")
