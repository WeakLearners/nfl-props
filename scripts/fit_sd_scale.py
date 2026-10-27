#!/usr/bin/env python
"""Fit the per-stat scale of the new-path sd on out-of-fold residuals.

The Phase 3 artifacts are trained through 2025, so every 2026 week is out of sample.
For each finished 2026 week, on the player-stats that have a card in the published reports: new mu (Phase 3, or the attempts x rate baseline for
completions) against the actual, divided by the new-path sd (spread.py, no 1.10).
scale = the 68.3rd percentile of |actual - mu| / sd, so that mu +/- scale*sd covers
68.3% of the games. Writes data/sd_scale.json. Does not touch calibration.json.

usage: fit_sd_scale.py [--season 2026] [--dry]
"""
import argparse
import json
import pathlib
import sys

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from nflprops import engine_source as ES, spread as SP  # noqa: E402
from nflprops.config import CURRENT_SEASON  # noqa: E402
from nflprops.odds import name_key  # noqa: E402
import grade_game as G  # noqa: E402
import grade_week as W  # noqa: E402
from vol_predict import phase3_vol  # noqa: E402


def collect(season, weeks):
    """One row per (week, player, stat): new mu, new-path sd (unscaled), actual."""
    out = []
    for w in weeks:
        act = G.actuals(season, w)
        if act.empty:
            continue
        by = act.set_index("key")
        sl = SP.slate_rows(season, w)
        p3 = phase3_vol(season, w).set_index("player_id")
        for r in sl.itertuples():
            if r.stat == "completions":
                mu = r.base_mu
            else:
                mu = p3[f"p3_{r.stat}"].get(r.player_id, np.nan) if r.player_id in p3.index else np.nan
            k = name_key(r.player)
            if k not in by.index or pd.isna(mu):
                continue
            a = by.loc[k, r.stat]
            a = a.iloc[0] if isinstance(a, pd.Series) else a
            if pd.isna(a):
                continue
            out.append({"week": w, "player": r.player, "stat": r.stat, "mu_new": float(mu), "sd_new": r.sd,
                        "actual": float(a)})
    return pd.DataFrame(out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, default=CURRENT_SEASON)
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--save", help="also save the collected frame (pickle)")
    a = ap.parse_args()
    weeks = W.finished_weeks(a.season)
    df = collect(a.season, weeks)
    if a.save:
        df.to_pickle(a.save)
    # Fit on the rows that get a card (the report's player pool), not on every rostered player.
    # On the whole roster the cards' coverage came out 63% for receiving yards at the 68% scale.
    from nflprops import model_quality as MQ
    pub = MQ.collect(a.season, weeks)[["week", "player", "stat"]].drop_duplicates()
    df = df.merge(pub, on=["week", "player", "stat"], how="inner")
    df["u"] = (df.actual - df.mu_new).abs() / df.sd_new
    scale = {s: round(float(np.quantile(g.u, 0.683)), 3) for s, g in df.groupby("stat")}
    for s, g in df.groupby("stat"):
        print(f"{s:16s} n={len(g):5d} scale={scale[s]:.3f}  coverage at 1.0={float((g.u <= 1).mean()):.3f}")
    if not a.dry:
        ES.SCALE_PATH.write_text(json.dumps({"fitted_through_week": int(max(weeks)), "season": a.season,
                                             "n": {s: int(len(g)) for s, g in df.groupby("stat")}, **scale}, indent=1))
        print("wrote", ES.SCALE_PATH)
