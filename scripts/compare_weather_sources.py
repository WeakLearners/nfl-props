#!/usr/bin/env python3
"""Props projections with the old weather source (features.db) against the merged source (cache + forecast log).

For each 2026 week: how many player-stats have a different mu, and by how much. sd comes from the
player's own history (spread.py) and never reads weather, so sd cannot differ. Writes
data/weather/source_compare.json.
"""
import json, os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "scripts"))
import numpy as np, pandas as pd
from vol_predict import phase3_vol
from build_training_table import pregame_features
from nflprops import weather as W

out = {}
for wk in range(1, 10):
    try:
        a = phase3_vol(2026, wk, weather="features").set_index("player_id")
        b = phase3_vol(2026, wk, weather="cache").set_index("player_id")
    except Exception as e:  # noqa: BLE001
        out[wk] = {"error": str(e)[:120]}; continue
    if a.empty or b.empty:
        out[wk] = {"note": "no rows"}; continue
    cols = [c for c in a.columns if c.startswith("p3_")]
    d = (b[cols] - a[cols].reindex(b.index)).stack()
    d = d.dropna()
    res = dict(n=int(len(d)), n_differ=int((d.abs() > 0.05).sum()),
               max_abs=float(d.abs().max()) if len(d) else 0.0,
               mean_abs=float(d.abs().mean()) if len(d) else 0.0)
    per = {}
    for c in cols:
        x = (b[c] - a[c].reindex(b.index)).dropna()
        per[c] = dict(n_differ=int((x.abs() > 0.05).sum()), max_abs=float(x.abs().max()) if len(x) else 0.0)
    res["by_stat"] = per
    out[wk] = res
    print(wk, res["n"], res["n_differ"], round(res["max_abs"], 2), flush=True)
json.dump(out, open(os.path.join(ROOT, "data", "weather", "source_compare.json"), "w"), indent=1)
