#!/usr/bin/env python
"""Score weeks again on the new path, as of each week, and write data/rescored.csv.

For every week up to --through (default 7): mu from the Phase 3 model (completions: the
attempts x rate baseline), sd from spread.py scaled by data/sd_scale.json. Each week uses only
data before that week (the artifacts are trained through 2025). The published reports are not
touched. The old-path numbers stay in the file as mu_pub, sd_pub, mp_raw_pub.

usage: rescore_weeks.py [--season 2026] [--through 7]
"""
import argparse
import pathlib
import sys

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from nflprops import engine_source as ES, model_quality as MQ, rescored as RS  # noqa: E402
from nflprops.config import CURRENT_SEASON  # noqa: E402
import fit_sd_scale as F  # noqa: E402
import grade_week as W  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, default=CURRENT_SEASON)
    ap.add_argument("--through", type=int, default=7)
    a = ap.parse_args()
    weeks = [w for w in W.finished_weeks(a.season) if w <= a.through]
    new = F.collect(a.season, weeks)
    new["sd"] = (new.sd_new * new.stat.map(ES.sd_scale())).round(3)
    new = new.rename(columns={"mu_new": "mu"}).drop(columns=["sd_new"])
    new["mu"] = new.mu.round(3)
    new.insert(0, "season", a.season)
    pub = MQ.collect(a.season, weeks)[["week", "player", "stat", "mu", "sd", "mp_raw"]]
    pub = pub.rename(columns={"mu": "mu_pub", "sd": "sd_pub", "mp_raw": "mp_raw_pub"})
    out = new.merge(pub.drop_duplicates(["week", "player", "stat"]), on=["week", "player", "stat"], how="left")
    out.to_csv(RS.PATH, index=False)
    print("wrote", RS.PATH, len(out), "rows")
    print(out.groupby("week").size().to_string())
