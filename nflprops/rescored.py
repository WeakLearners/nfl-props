"""Weeks scored again on the new path, for the calibration fit and model quality.

Weeks 1 to 7 were published with the old projection (and the old sd). data/rescored.csv holds
the same player-stats scored again on the new path as of each week: mu from the Phase 3 model
(completions: attempts x rate), sd from spread.py scaled by data/sd_scale.json. The published
reports are never touched. The old-path values stay in the file as mu_pub, sd_pub, mp_raw_pub
(NaN where the player-stat had no card). Written by scripts/rescore_weeks.py.
"""
import pandas as pd

from .config import DATA_DIR
from .odds import name_key

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
    """Rows keyed by (week, name_key(player), stat). Look up with key(week, player, stat)."""
    d = load()
    d = d[d.season == season]
    return d.assign(pk=d.player.map(name_key)).set_index(["week", "pk", "stat"])


def key(week, player, stat):
    """The lookup key for index(): the player name is normalized, so 'Estimé' finds 'Estime'."""
    return (week, name_key(player), stat)


def report_unmatched(label, missed, total):
    """Print how many rows found no match in a lookup by player name, so a lost row is visible."""
    print(f"lookup {label}: {total - missed} of {total} rows matched, {missed} with no match")
    return missed


def join_on_name(left, right, how, label):
    """Join two frames on (week, player, stat) with a normalized player key.

    Prints how many rows on each side have no match, so a lost row is visible. `right` must have
    unique (week, player, stat) keys after normalization (the caller drops duplicates).
    """
    l = left.assign(_pk=left.player.map(name_key))
    r = right.assign(_pk=right.player.map(name_key)).drop(columns=["player"])
    r = r.drop_duplicates(["week", "_pk", "stat"])
    m = l.merge(r, on=["week", "_pk", "stat"], how=how)
    lk = set(zip(l.week, l._pk, l.stat))
    rk = set(zip(r.week, r._pk, r.stat))
    only_l = int(sum(1 for k in zip(l.week, l._pk, l.stat) if k not in rk))
    only_r = len(rk - lk)
    print(f"join {label}: {len(lk & rk)} keys matched, {only_l} left rows with no match, {only_r} right rows with no match")
    return m.drop(columns=["_pk"])
