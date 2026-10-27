#!/usr/bin/env python
"""Refit calibration.json (f and the per-stat mu bias) on NEW-PATH probabilities.

Same fit function the Tuesday grade calls (calibration.fit), fed with every graded leg
re-scored by the new path: new mu (Phase 3, or the attempts x rate baseline for
completions) and the new sd (spread.py, scaled by data/sd_scale.json). Edge-chosen rung
legs stay excluded from the fit (e5dfbf4). Does not run grade_week.

usage: refit_new_path.py            # dry: print the result
       refit_new_path.py --write    # write data/calibration.json
"""
import argparse
import pathlib
import sys

import numpy as np
import pandas as pd
from scipy.stats import norm

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from nflprops import calibration as CAL, engine_source as ES  # noqa: E402
from nflprops.config import CURRENT_SEASON  # noqa: E402
import fit_sd_scale as F  # noqa: E402
import grade_week as W  # noqa: E402


def new_path_inputs(season=CURRENT_SEASON, weeks=None):
    weeks = weeks or W.finished_weeks(season)
    df = F.collect(season, weeks)
    df["sd_new"] = df.sd_new * df.stat.map(ES.sd_scale())
    return df


def rescored_legs(legs, df):
    """Graded legs with mu, sd and the raw probability replaced by the new path's. Unmatched legs drop."""
    ix = df.set_index(["week", "player", "stat"])
    out = []
    for l in legs:
        k = (l["week"], l["p"], l["s"])
        if k not in ix.index or l.get("line") is None:
            continue
        r = ix.loc[k]
        r = r.iloc[0] if isinstance(r, pd.DataFrame) else r
        p = float(1 - norm.cdf((l["line"] - r.mu_new) / r.sd_new))
        out.append(dict(l, mu=float(r.mu_new), mu_raw=float(r.mu_new), sd=float(r.sd_new), mp_raw=p, mp=p))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()
    df = new_path_inputs()
    legs = rescored_legs(CAL._all_graded(), df)
    rows = df.rename(columns={"mu_new": "mu_raw"})[["stat", "mu_raw", "actual"]]
    res = CAL.fit(write=a.write, legs=legs, rows=rows)
    for k, v in res.items():
        print(k, v)
