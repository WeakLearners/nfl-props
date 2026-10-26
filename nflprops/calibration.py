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


BIAS_SHRINK_N = 50


def mu_bias():
    """{stat: additive bias} from calibration.json. Missing file or stat = 0."""
    c = load() or {}
    try:
        return {k: float(v) for k, v in (c.get("mu_bias") or {}).items()}
    except (TypeError, ValueError):
        return {}


def bias_n():
    return (load() or {}).get("mu_bias_n") or {}


def bias_for(stat):
    return mu_bias().get(stat, 0.0)


def shrink_bias(mean_err, n):
    """Shrink a mean (actual - mu) toward 0: b * n / (n + 50)."""
    return mean_err * n / (n + BIAS_SHRINK_N)


def fit_bias(rows=None):
    """({stat: bias}, {stat: n}) over ALL projected players in finished games.

    Uses the raw mu (before any bias), so a refit does not chase its own
    correction. `rows` is a DataFrame with stat, mu_raw, actual (tests pass one).
    """
    if rows is None:
        import sys
        sys.path.insert(0, str(PATH.parent.parent / "scripts"))
        import grade_week as GW
        from .config import CURRENT_SEASON
        rows = GW.projection_rows(CURRENT_SEASON, GW.finished_weeks(CURRENT_SEASON))
    if rows is None or len(rows) == 0:
        return {}, {}
    rows = rows.assign(e=rows.actual - rows.mu_raw)
    g = rows.groupby("stat").e.agg(["mean", "count"])
    return ({s: round(shrink_bias(float(r["mean"]), int(r["count"])), 2) for s, r in g.iterrows()},
            {s: int(r["count"]) for s, r in g.iterrows()})


def with_bias(legs, bias):
    """Copies of legs whose raw model probability uses mu_raw + bias.

    Only a leg with a nonzero bias for its stat, and with mu, sd and line, is
    recomputed. Others keep their stored probability.
    """
    from statistics import NormalDist
    out = []
    for l in legs:
        b = bias.get(l["s"], 0.0)
        if b and l.get("mu") is not None and l.get("sd") and l.get("line") is not None:
            m = (l["mu"] if l.get("mu_raw") is None else l["mu_raw"]) + b
            l = dict(l, mp_raw=1 - NormalDist(float(m), float(l["sd"])).cdf(float(l["line"])))
        out.append(l)
    return out


def raw_mp(leg):
    """The model probability before the adjustment (fit input)."""
    return leg["mp_raw"] if leg.get("mp_raw") is not None else leg["mp"]


def _all_graded():
    import sys
    sys.path.insert(0, str(PATH.parent.parent / "scripts"))
    import serve_reports as S
    return [l for l in S.explore_legs()[0]
            if l["s"] != "anytime_td" and l.get("mp") is not None
            and l.get("bp") is not None and l["hit"] is not None]


def _fit_pool(legs):
    # Edge-chosen rungs (week 8 on) are picked because raw mp > bp, so they regress; fitting on them pushes f toward 1, which empties every shortlist.
    return [l for l in legs if l.get("rung") != "chosen"]


def graded_legs():
    return _fit_pool(_all_graded())


def brier(legs, f):
    return sum((raw_mp(l) + f * (l["bp"] - raw_mp(l)) - l["hit"]) ** 2 for l in legs) / len(legs)


def fit(write=True):
    """Pick f on the grid with the lowest Brier on all graded legs to date."""
    every = _all_graded()
    # Bias first, then f on probabilities that use the bias-corrected mu.
    bias, bias_cnt = fit_bias()
    legs = with_bias(_fit_pool(every), bias)
    if not legs:
        return None
    best = min(GRID, key=lambda f: brier(legs, f))
    prev = load()
    res = {"fraction": best, "fit_through_week": max(l["week"] for l in legs),
           "legs": len(legs), "chosen_excluded": len(every) - len(legs),
           "mu_bias": bias, "mu_bias_n": bias_cnt, "brier_before": round(brier(legs, 0.0), 4),
           "brier_after": round(brier(legs, best), 4),
           # a rerun in the same week keeps the earlier week's fraction as "previous"
           "previous_fraction": (None if not prev else prev.get("previous_fraction")
                                 if prev.get("fit_through_week") == max(l["week"] for l in legs)
                                 else prev["fraction"]),
           "updated": datetime.datetime.now().isoformat(timespec="seconds")}
    if write:
        PATH.write_text(json.dumps(res, indent=1))
    return res
