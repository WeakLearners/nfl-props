"""Decision #31, step 3: serve the shipped Phase 3 ridge volume/yardage
models for an upcoming slate, through pregame_features() -- the one-code-path
rule decision #30 established for the TD models applies here unchanged.

Returns predictions for the four stats scripts/retrain_vol_models.py ships
their own artifact for -- targets, carries, receiving_yards, rushing_yards
-- plus receptions (decision #38), which has no artifact of its own: it is
targets.pkl's own prediction times the player's pregame catch_rate_career
(a leak-gated feature already in features.db), with a population-mean
fallback for a player with no games yet (rookies), read from
vol_metadata.json's catch_rate_career_fill_if_missing. That routing is
data, not new math -- vol_receptions_compare.py found ridge-direct-on-
receptions and targets x rate a statistical coin flip on 2021-2025
walk-forward MAE, and targets x rate won pooled and 3 of 5 seasons.

Each artifact-backed column is inference-time reconstruction of
rung2_ridge.fit_predict()'s standardize step, using the TRAIN median/mu/sd
saved in the artifact -- never refit, never recomputed from the pregame
rows themselves (that would be exactly the kind of leak decision #9/#17
exist to prevent).

Read-only against features.db and nflprops.db. Writes nothing. Makes zero
Odds API calls.
"""
import json
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
STATS = ["targets", "carries", "receiving_yards", "rushing_yards", "passing_yards"]


def phase3_vol(season, week):
    """P3 projection per rostered player for each shipped vol/yards stat.
    Empty frame (not an error) if artifacts are missing -- same convention
    as scripts/td/write_markers.py's phase3_combined()."""
    paths = {s: SHIPPING / f"{s}.pkl" for s in STATS}
    missing = [s for s, p in paths.items() if not p.exists()]
    if missing:
        print(f"  ! data/shipping/{{{','.join(missing)}}}.pkl not found -- skipping phase3 vol numbers")
        return pd.DataFrame(columns=["player_id"] + [f"p3_{s}" for s in STATS] + ["p3_receptions"])

    pf = pregame_features(season, week)
    if pf.empty:
        return pd.DataFrame(columns=["player_id"] + [f"p3_{s}" for s in STATS] + ["p3_receptions"])
    # include_passing=True so the passing_yards artifact's own columns (which
    # DO include PASSING_COLS) get real pregame values on reindex below,
    # rather than the 0.0 fill_value a missing column would get. Every other
    # artifact's "columns" list doesn't mention PASSING_COLS, so reindexing
    # down to those still drops them for those stats -- this is safe for all
    # five artifacts, not just passing_yards.
    X, _ = build_design(pf, include_passing=True)
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

    # receptions (decision #38): targets.pkl's own prediction x catch_rate_career.
    meta_path = SHIPPING / "vol_metadata.json"
    fill = 0.65  # only reached if vol_metadata.json predates decision #38
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        fill = meta.get("stats", {}).get("receptions", {}).get("catch_rate_career_fill_if_missing", fill)
    cr = pf["catch_rate_career"].reindex(out.index).fillna(fill).clip(0, 1).to_numpy(float)
    out["p3_receptions"] = np.clip(out["p3_targets"], 0, None) * cr
    return out
