"""Weekly self-adjustment: one global fraction f that moves the model
probability toward the book probability, shrunk = mp + f * (bp - mp).

The fraction is refit after each Tuesday grade (scripts/grade_week.py) and
stored in data/calibration.json. A missing file means f = 0, no change.
Entertainment only; no edge claim (decision #46).
"""
import json, pathlib, datetime

PATH = pathlib.Path(__file__).resolve().parent.parent / "data" / "calibration.json"
GRID = [i / 10 for i in range(11)]


def load():
    try:
        return json.loads(PATH.read_text())
    except Exception:
        return None


def fraction():
    c = load()
    try:
        return min(1.0, max(0.0, float(c["fraction"]))) if c else 0.0
    except (KeyError, TypeError, ValueError):
        return 0.0


def raw_mp(leg):
    """The model probability before the adjustment (fit input)."""
    return leg["mp_raw"] if leg.get("mp_raw") is not None else leg["mp"]


def graded_legs():
    import sys
    sys.path.insert(0, str(PATH.parent.parent / "scripts"))
    import serve_reports as S
    return [l for l in S.explore_legs()[0]
            if l["s"] != "anytime_td" and l.get("mp") is not None
            and l.get("bp") is not None and l["hit"] is not None]


def brier(legs, f):
    return sum((raw_mp(l) + f * (l["bp"] - raw_mp(l)) - l["hit"]) ** 2 for l in legs) / len(legs)


def fit(write=True):
    """Pick f on the grid with the lowest Brier on all graded legs to date."""
    legs = graded_legs()
    if not legs:
        return None
    best = min(GRID, key=lambda f: brier(legs, f))
    prev = load()
    res = {"fraction": best, "fit_through_week": max(l["week"] for l in legs),
           "legs": len(legs), "brier_before": round(brier(legs, 0.0), 4),
           "brier_after": round(brier(legs, best), 4),
           # a rerun in the same week keeps the earlier week's fraction as "previous"
           "previous_fraction": (None if not prev else prev.get("previous_fraction")
                                 if prev.get("fit_through_week") == max(l["week"] for l in legs)
                                 else prev["fraction"]),
           "updated": datetime.datetime.now().isoformat(timespec="seconds")}
    if write:
        PATH.write_text(json.dumps(res, indent=1))
    return res
