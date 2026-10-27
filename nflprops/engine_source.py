"""The scale of the projection spread, per stat. Record of the engine switch.

History (2026-10-27, Sean: one model for every stat). Until then mu came from the old
projection (project.py) for receiving yards, receptions and completions, and from the
Phase 3 model for passing and rushing yards from the 2026-10-11 16:00 kickoff. The sd
always came from project.py x 1.10. Now every stat uses the Phase 3 model (completions:
the attempts x rate baseline in spread.py) and the spread from spread.py, scaled by
data/sd_scale.json. The source switch was removed in the retirement commit. project.py
stays only so the research scripts that compare engines still run.
"""
import json

from .config import DATA_DIR

STATS = ("passing_yards", "completions", "rushing_yards", "receiving_yards", "receptions")
SCALE_PATH = DATA_DIR / "sd_scale.json"


def sd_scale():
    """{stat: scale} for the sd. 1.0 where a stat has no fitted scale."""
    try:
        d = json.loads(SCALE_PATH.read_text())
    except (OSError, ValueError):
        d = {}
    return {s: float(d.get(s, 1.0)) for s in STATS}
