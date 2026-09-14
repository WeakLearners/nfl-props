#!/usr/bin/env python
"""Turn each modeling component off in turn and re-score. Anything that does
not beat its own absence is noise being sold as signal."""
import sys, pathlib, io, contextlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import pandas as pd, numpy as np
from nflprops import project as P
from project_week import actuals, naive_baseline

SEASON, WEEK = int(sys.argv[1]), int(sys.argv[2])


def run(label, **over):
    no_def = over.pop("_no_def", False)
    saved = {k: getattr(P, k) for k in over}
    for k, v in over.items():
        setattr(P, k, v)
    real = P.defense_vs_position
    if no_def:
        real = P.defense_vs_position
        P.defense_vs_position = lambda *a, **k: pd.DataFrame()
    try:
        proj = P.project_week(SEASON, WEEK)
    finally:
        for k, v in saved.items():
            setattr(P, k, v)
        P.defense_vs_position = real

    df = (proj.merge(actuals(SEASON, WEEK), on=["player_id", "stat"])
              .merge(naive_baseline(SEASON, WEEK), on=["player_id", "stat"], how="left"))
    df = df[df.baseline.notna() & (df["count"] >= 3)]
    out = {"config": label}
    for grp, name in [(df[df.position == "QB"], "QB"), (df[df.position != "QB"], "skill")]:
        live = grp[(grp.proj > 1) & (grp.baseline > 1) & (abs(grp.proj - grp.baseline) > 0.05 * grp.baseline)]
        mae = (grp.proj - grp.actual).abs().mean()
        bmae = (grp.baseline - grp.actual).abs().mean()
        out[f"{name}_vs_base"] = f"{(bmae-mae)/bmae*100:+.1f}%"
        out[f"{name}_dir"] = f"{((live.proj > live.baseline) == (live.actual > live.baseline)).mean()*100:.1f}%"
        out[f"{name}_n"] = len(live)
    return out


rows = []
with contextlib.redirect_stdout(io.StringIO()):
    rows.append(run("full model"))
    rows.append(run("no opponent adj", _no_def=True))
    rows.append(run("no recency decay", DECAY=1.0, PRIOR_SEASON_W=1.0))
    rows.append(run("no def + no decay", _no_def=True, DECAY=1.0, PRIOR_SEASON_W=1.0))
    rows.append(run("heavy shrink eff", K_EFF_ATT=200.0))
print(pd.DataFrame(rows).to_string(index=False))
print("\nvs_base = MAE improvement over the naive average (positive is good)")
