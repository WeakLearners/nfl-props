#!/usr/bin/env python
"""Phase: the three named hypotheses, tested explicitly against the lagged
feature table."""
import pathlib
import numpy as np
import pandas as pd

DATA = pathlib.Path(__file__).resolve().parent.parent.parent / "data" / "eda_features.parquet"
df = pd.read_parquet(DATA)
pd.set_option("display.width", 160)

print("="*70, "\nH1: VACATED USAGE\n", "="*70, sep="")
# Does a teammate at the same position being Out/Doubtful for the target week
# predict a bump in the remaining players' target/carry SHARE that game?
for pos, share_col, vol_col in [("WR", "target_share", "targets"),
                                 ("RB", "target_share", "carries"),
                                 ("TE", "target_share", "targets")]:
    sub = df[df.position == pos].copy()
    grp = sub.groupby(sub.teammates_out_same_pos > 0)[share_col].agg(["mean", "count"])
    print(f"\n{pos} target_share by teammate-out-same-position flag:")
    print(grp)
    # also raw volume
    grp2 = sub.groupby(sub.teammates_out_same_pos > 0)[vol_col].agg(["mean", "count"])
    print(f"{pos} {vol_col} by teammate-out-same-position flag:")
    print(grp2)

print("\n" + "="*70, "\nH2: WEATHER\n", "="*70, sep="")
qb = df[df.position == "QB"].dropna(subset=["passing_yards", "wind", "temp"])
print(f"QB passing_yards vs wind, ALL roof types, n={len(qb)}: r={qb.wind.corr(qb.passing_yards):.4f}")
out = qb[qb.roof.isin(["outdoors"])]
print(f"QB passing_yards vs wind, OUTDOOR ONLY, n={len(out)}: r={out.wind.corr(out.passing_yards):.4f}")
print(f"QB passing_yards vs temp, OUTDOOR ONLY, n={len(out)}: r={out.temp.corr(out.passing_yards):.4f}")
# bucketed wind effect, outdoor only
out2 = out.copy()
out2["wind_bucket"] = pd.cut(out2.wind, [-1, 5, 10, 15, 20, 100],
                              labels=["0-5", "6-10", "11-15", "16-20", "20+"])
print("\nQB passing_yards by wind bucket (outdoor games only):")
print(out2.groupby("wind_bucket", observed=True).passing_yards.agg(["mean", "count"]))
print("\ncompletions and h_ypa (efficiency) too:")
print(out2.groupby("wind_bucket", observed=True)[["completions", "h_ypa"]].agg(["mean", "count"]))

print("\n" + "="*70, "\nH3: RED ZONE USAGE vs YARDAGE\n", "="*70, sep="")
for pos, targets in [("RB", ["rushing_yards", "receiving_yards"]),
                     ("WR", ["receiving_yards"]), ("TE", ["receiving_yards"])]:
    sub = df[df.position == pos]
    for t in targets:
        for rz in ["h_rz_carries_pg", "h_rz10_carries_pg", "h_rz_targets_pg", "h_rz10_targets_pg"]:
            s = sub[[rz, t]].dropna()
            if len(s) < 30:
                continue
            r = s[rz].corr(s[t])
            print(f"{pos:<3} {t:<16} vs {rz:<20} r={r:+.4f}  n={len(s)}")

print("\n" + "="*70, "\nQB RUSHING YARDS: mobile/pocket split check\n", "="*70, sep="")
qbr = df[df.position == "QB"].dropna(subset=["rushing_yards", "h_carries"])
qb_avg_carries = qbr.groupby("player_id").h_carries.mean()
mobile_ids = qb_avg_carries[qb_avg_carries >= 4].index
pocket_ids = qb_avg_carries[qb_avg_carries < 4].index
print(f"n QBs mobile (avg hist carries >= 4): {len(mobile_ids)}, pocket (<4): {len(pocket_ids)}")
pooled_r = qbr.h_carries.corr(qbr.rushing_yards)
print(f"POOLED r(h_carries, rushing_yards) across all QBs: {pooled_r:.4f}  n={len(qbr)}")
m = qbr[qbr.player_id.isin(mobile_ids)]
p = qbr[qbr.player_id.isin(pocket_ids)]
print(f"MOBILE-only r: {m.h_carries.corr(m.rushing_yards):.4f}  n={len(m)}, "
      f"mean rushing_yards={m.rushing_yards.mean():.1f}")
print(f"POCKET-only r: {p.h_carries.corr(p.rushing_yards):.4f}  n={len(p)}, "
      f"mean rushing_yards={p.rushing_yards.mean():.1f}")
print(f"share of rows with rushing_yards==0: {(qbr.rushing_yards==0).mean():.3f}")
print(f"pocket share with rushing_yards==0: {(p.rushing_yards==0).mean():.3f}")
print(f"mobile share with rushing_yards==0: {(m.rushing_yards==0).mean():.3f}")
