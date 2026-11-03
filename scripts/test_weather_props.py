#!/usr/bin/env python3
"""Do kickoff weather inputs help the props projections? 2026 weeks 1-8, out of sample (week 8's pool is its published report; rescored.csv ends at week 7).

The shipped models (trained through 2025) already take weather_* columns. This scores the same
players (the report's card pool, data/rescored.csv) three ways:
  as built   the columns as features.db holds them (measured weeks 1-2, forecasts made weeks 3-8, gaps missing)
  filled     gaps filled from the cache (data/weather/game_weather.csv)
  off        every weather column missing
Writes data/props_weather_test.json.
"""
import json, os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "scripts"))
import numpy as np, pandas as pd
from nflprops import weather as W, rescored as RS, spread as SP
from nflprops.odds import name_key
import grade_week as GW, grade_game as GG
from vol_predict import phase3_vol

season = 2026
pool = RS.load().query("season == @season")[["week", "player", "stat", "actual"]]
# data/rescored.csv stops at week 7. Week 8's card pool is the published week-8 reports.
from nflprops import model_quality as MQ
_w8 = MQ.collect(season, [8])
if len(_w8):
    pool = pd.concat([pool, _w8[["week", "player", "stat", "actual"]].drop_duplicates(["week", "player", "stat"])], ignore_index=True)
wx = W.load().drop_duplicates("game_id").set_index("game_id")
rows = []
for w in sorted(int(x) for x in pool.week.unique()):
    sl = SP.slate_rows(season, w)
    if sl.empty:
        print("no slate rows for week", w); continue
    sl["gid"] = sl["game_id"]
    pw = pool[pool.week == w]
    res = {}
    for mode in (None, "fill", "off"):
        p3 = phase3_vol(season, w, weather=mode).set_index("player_id")
        res[mode] = p3
    for r in sl.itertuples():
        for mode, p3 in res.items():
            if r.stat == "completions":
                mu = r.base_mu
            else:
                mu = p3[f"p3_{r.stat}"].get(r.player_id, np.nan) if r.player_id in p3.index else np.nan
            rows.append(dict(week=w, player=r.player, stat=r.stat, mode=str(mode), mu=mu, gid=r.gid))
df = pd.DataFrame(rows).merge(pool, on=["week", "player", "stat"], how="inner").dropna(subset=["mu"])
df["err"] = (df.actual - df.mu).abs()
df["wind_k"] = df.gid.map(wx["wind_k"]); df["precip_k"] = df.gid.map(wx["precip_k"])
df["rough"] = (df.wind_k >= 15) | (df.precip_k >= 1.0)
out = {}
for st, g in df.groupby("stat"):
    t = g.pivot_table(index=["week", "player"], columns="mode", values="err")
    r = g[g["mode"] == "None"].set_index(["week", "player"]).rough
    t = t.join(r)
    out[st] = dict(n=int(len(t)),
                   mae_as_built=float(t["None"].mean()), mae_filled=float(t["fill"].mean()), mae_off=float(t["off"].mean()),
                   n_rough=int(t.rough.sum()),
                   mae_as_built_rough=float(t[t.rough]["None"].mean()), mae_off_rough=float(t[t.rough]["off"].mean()),
                   mae_filled_rough=float(t[t.rough]["fill"].mean()))
json.dump(out, open(os.path.join(ROOT, "data", "props_weather_test.json"), "w"), indent=1)
print(f"{'stat':16s} n    as_built  filled   off   | rough n  as_built filled off")
for st, v in out.items():
    print(f"{st:16s} {v['n']:4d} {v['mae_as_built']:.3f} {v['mae_filled']:.3f} {v['mae_off']:.3f} | {v['n_rough']:3d} {v['mae_as_built_rough']:.3f} {v['mae_filled_rough']:.3f} {v['mae_off_rough']:.3f}")
df.to_pickle(os.environ.get("TMPDIR", "/tmp") + "/props_weather_df.pkl")
