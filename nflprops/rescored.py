"""Weeks scored again on the new path, for the calibration fit and model quality.

Weeks 1 to 7 were published with the old projection (and the old sd). data/rescored.csv holds
the same player-stats scored again on the new path as of each week: mu from the Phase 3 model
(completions: attempts x rate), sd from spread.py scaled by data/sd_scale.json. The published
reports are never touched. The old-path values stay in the file as mu_pub, sd_pub, mp_raw_pub
(NaN where the player-stat had no card). Written by scripts/rescore_weeks.py.
"""
import pandas as pd

from .config import DATA_DIR

PATH = DATA_DIR / "rescored.csv"
_CACHE = {"mtime": None, "df": None}


def load():
    """The rescored frame (empty if the file is absent)."""
    try:
        m = PATH.stat().st_mtime
    except OSError:
        return pd.DataFrame(columns=["season", "week", "player", "stat", "mu", "sd", "actual"])
    if _CACHE["mtime"] != m:
        _CACHE["df"], _CACHE["mtime"] = pd.read_csv(PATH), m
    return _CACHE["df"]


def max_week(season):
    d = load()
    d = d[d.season == season]
    return int(d.week.max()) if len(d) else 0


def index(season):
    d = load()
    d = d[d.season == season]
    return d.set_index(["week", "player", "stat"])
