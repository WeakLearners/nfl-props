#!/usr/bin/env python3
"""Walk-forward test: do kickoff weather inputs lower total MAE or team-score MAE?

Windows: 2024-2025 (the tuning window; measured weather is the only kind) and 2026 weeks 1-8 (the test set).
For 2026 two readings of the weather input:
  forecast-where-made: a game with a stored pre-kickoff forecast uses it (weeks 3-8), else the measured value
  measured only:       hindsight weather. An upper bound, not a backtest.
Writes data/gamelines/weather_test.json.
"""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from nflprops import gamelines as G
from nflprops.config import DATA_DIR

VARIANTS = {"none": (), "wind": ("wind",), "rain": ("rain",), "wind+rain": ("wind", "rain"), "heavy": ("heavy",),
            "rain+heavy": ("rain", "heavy"), "wind+rain+heavy": ("wind", "rain", "heavy")}
out = {}
for mode, uf in (("forecast-where-made", True), ("measured only", False)):
    G.USE_FORECAST = uf
    g = G.load_games()
    for name, wx in VARIANTS.items():
        for win, seasons, weeks in (("2024-25", [2024, 2025], None), ("2026 wk1-8", [2026], range(1, 9))):
            if win == "2024-25" and not uf:
                continue
            o = G.walk_forward(g, seasons, {"wx": wx}, weeks=weeks)
            o = o[o.played]
            if win == "2024-25":
                o = o[o.week >= 3]
            sc = np.concatenate([(o.home_score - o.mu_home).abs(), (o.away_score - o.mu_away).abs()]).mean()
            out[f"{name} | {win} | {mode if win != '2024-25' else 'measured'}"] = dict(
                n=len(o), mae_margin=float((o.margin - o.pred_margin).abs().mean()),
                mae_total=float((o.total - o.pred_total).abs().mean()), mae_score=float(sc),
                n_heavy=int((o.w_heavy > 0).sum()),
                n_forecast_rows=int((o.w_src == "forecast").sum()))
json.dump(out, open(DATA_DIR / "gamelines" / "weather_test.json", "w"), indent=1)
print(f"{'variant | window | weather':52s} n   margin  total  score  heavy fc")
for k, v in out.items():
    print(f"{k:52s} {v['n']:3d} {v['mae_margin']:.3f} {v['mae_total']:.3f} {v['mae_score']:.3f}  {v['n_heavy']:3d} {v['n_forecast_rows']:3d}")
