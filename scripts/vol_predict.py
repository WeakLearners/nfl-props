"""Decision #31, step 3: serve the shipped Phase 3 ridge volume/yardage
models for an upcoming slate, through pregame_features() -- the one-code-path
rule decision #30 established for the TD models applies here unchanged.

Returns predictions for the same four stats scripts/retrain_vol_models.py
ships: targets, carries, receiving_yards, rushing_yards. Each column is
inference-time reconstruction of rung2_ridge.fit_predict()'s standardize
step, using the TRAIN median/mu/sd saved in the artifact -- never refit,
never recomputed from the pregame rows themselves (that would be exactly
the kind of leak decision #9/#17 exist to prevent).

Read-only against features.db and nflprops.db. Writes nothing. Makes zero
Odds API calls.
"""
import pathlib
import sys

import joblib
import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from build_training_table import pregame_features  # noqa: E402
from rung2_ridge import build_design  # noqa: E402

SHIPPING = ROOT / "data" / "shipping"
STATS = ["targets", "carries", "receiving_yards", "rushing_yards"]


def phase3_vol(season, week):
    """P3 projection per rostered player for each shipped vol/yards stat.
    Empty frame (not an error) if artifacts are missing -- same convention
    as scripts/td/write_markers.py's phase3_combined()."""
    paths = {s: SHIPPING / f"{s}.pkl" for s in STATS}
    missing = [s for s, p in paths.items() if not p.exists()]
    if missing:
        print(f"  ! data/shipping/{{{','.join(missing)}}}.pkl not found -- skipping phase3 vol numbers")
        return pd.DataFrame(columns=["player_id"] + [f"p3_{s}" for s in STATS])

    pf = pregame_features(season, week)
    if pf.empty:
        return pd.DataFrame(columns=["player_id"] + [f"p3_{s}" for s in STATS])
    X, _ = build_design(pf)
    X = X.astype(float)

    out = pf[["player_id"]].copy()
    for stat, path in paths.items():
        art = joblib.load(path)
        Xs = X.reindex(columns=art["columns"], fill_value=0.0)
        Xs = Xs.fillna(art["median"]).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        Xs = ((Xs - art["mu"]) / art["sd"]).to_numpy(float)
        # Clipped at serving time only -- a real value can't be negative.
        # The head-to-head script (vol_head_to_head.py) leaves predictions
        # raw, matching rung2_ridge.py's own unclipped validation protocol,
        # so the two don't score differently for cosmetic reasons.
        out[f"p3_{stat}"] = np.clip(art["model"].predict(Xs), 0, None)
    return out
