"""Which source supplies mu and sd for each stat.

mu per stat:
  "old"    the old projection (project.py)
  "gated"  the new model after the 2026-10-11 16:00 kickoff, the old one before (today's rule)
  "new"    the new model whenever it has a number (completions: the attempts x rate baseline
           in spread.py), the old one only where the new one is missing
sd per stat:
  "old"    project.py sd x 1.10 (SD_INFLATE)
  "new"    spread.py sd (same per-player sd shrunk toward the position, no 1.10) x the
           per-stat scale in data/sd_scale.json

The defaults are the new path for every stat (since 2026-10-27). LEGACY is the earlier default. data/engine_source.json,
if present, overrides them key by key. td is not a switch: tdmodel.py reads only the
shared primitives in base.py, not the old projection.
"""
import json

from .config import DATA_DIR

STATS = ("passing_yards", "completions", "rushing_yards", "receiving_yards", "receptions")
PATH = DATA_DIR / "engine_source.json"
SCALE_PATH = DATA_DIR / "sd_scale.json"
# Flipped 2026-10-27: one model for every stat (Sean). Before this the defaults were
# mu {passing, rushing: "gated"; receiving, receptions, completions: "old"} and sd "old".
DEFAULT = {
    "mu": {s: "new" for s in STATS},
    "sd": {s: "new" for s in STATS},
}
LEGACY = {
    "mu": {"passing_yards": "gated", "rushing_yards": "gated", "receiving_yards": "old",
           "receptions": "old", "completions": "old"},
    "sd": {s: "old" for s in STATS},
}


def load():
    cfg = {"mu": dict(DEFAULT["mu"]), "sd": dict(DEFAULT["sd"])}
    try:
        over = json.loads(PATH.read_text())
    except (OSError, ValueError):
        return cfg
    for k in ("mu", "sd"):
        cfg[k].update(over.get(k, {}))
    return cfg


def sd_scale():
    """{stat: scale} for the new sd. 1.0 where a stat has no fitted scale."""
    try:
        d = json.loads(SCALE_PATH.read_text())
    except (OSError, ValueError):
        d = {}
    return {s: float(d.get(s, 1.0)) for s in STATS}


def needs_slate(cfg):
    return any(v == "new" for v in cfg["sd"].values()) or cfg["mu"].get("completions") == "new"
