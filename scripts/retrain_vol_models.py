#!/usr/bin/env python
"""Decision #31, step 2: ship the four Phase 3 ridge volume/yardage models
(targets, carries, receiving_yards, rushing_yards) as versioned artifacts,
same convention as scripts/retrain_td_models.py.

WHY RIDGE, NOT LIGHTGBM. phase3-model.md's "PHASE 3 -- CLOSED" table
labels all four "model: LightGBM" and "ships: yes", scored against
baseline only. Decision #20, recorded the same day, is the one that
actually resolves which model ships: LightGBM (rung 3) failed to beat
ridge (rung 2) by the required 3% margin on all four stats -- and lost
outright on rushing yards in 2025 (-1.28%). Decision #20's own words:
"Ridge is the production model for targets, carries, receiving yards and
rushing yards." This script ships that model, not the one the stale table
names.

WHAT'S FIT. rung2_ridge.py's exact "raw" variant (no log1p -- decision #18
found the transform wasn't the lever) and the "direct" final-stat route,
not "composed" (decision #23: composed vs. direct are statistically
indistinguishable against the 3% bar; direct is simpler to serve). Same
RidgeCV alpha search, same train-only median-impute + standardize, same
per-stat qualifying screen (qual_targets / qual_carries) rung2_ridge.py's
own SPECS table already defines -- imported, not re-typed.

TRAINING WINDOW: 2019-2025, all seven complete seasons, fixed -- matches
retrain_td_models.py's TRAIN_SEASONS. 2026 stays sealed (decision #24):
this script filters it out before anything else touches the frame.

READ-ONLY against features.db -- this script fits on `training_rows`
exactly as it already exists. It does NOT call build_training_table.py's
main() and does NOT rebuild training_rows. (Decision #30/#31: a fresh
rebuild silently drops historical weather signal once
apply_weather_measured.py has reclassified a played game's weather from
forecast to measured -- confirmed by a controlled test this morning.
Fitting on the live table sidesteps that regression entirely.) Writes
only under data/shipping/.

SCHEDULE: on demand, not launchd -- same reasoning as retrain_td_models.py.
"""
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from rung2_ridge import load, add_derived, build_design, SPECS, ALPHAS
from sklearn.linear_model import Ridge, RidgeCV

LATEST_COMPLETE_SEASON = 2025
TRAIN_SEASONS = list(range(2019, LATEST_COMPLETE_SEASON + 1))

OUT_DIR = Path.home() / "Code/nfl-props/data/shipping"
OUT_DIR.mkdir(exist_ok=True)

# (stat, target column, screen) -- direct final-stat route, and the two
# usage stats. Pulled from rung2_ridge.SPECS rather than re-typed, so the
# screen/target pairing can never drift from the validated protocol.
WANT = {"targets", "carries", "receiving_yards", "rushing_yards"}
STATS = [(stat, tcol, screen) for _layer, stat, tcol, screen, _cnt in SPECS if stat in WANT]


def git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=Path(__file__).resolve().parent
        ).decode().strip()
    except Exception:
        return "unknown"


def fit_ship(Xtr, ytr):
    """Same shape as rung2_ridge.fit_predict, minus the held-out predict --
    this is the shipping fit, so everything needed at inference time
    (median, mu, sd, alpha, model) is returned to be saved, not thrown away."""
    med = Xtr.median()
    a = Xtr.fillna(med)
    a = a.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    mu, sd = a.mean(), a.std().replace(0, 1.0)
    a_std = ((a - mu) / sd).to_numpy(float)
    m = RidgeCV(alphas=ALPHAS).fit(a_std, ytr)
    alpha = float(m.alpha_)
    return {"model": Ridge(alpha=alpha).fit(a_std, ytr), "alpha": alpha,
            "median": med, "mu": mu, "sd": sd}


def main():
    df = add_derived(load())
    df = df[df.season.between(2019, 2025)].reset_index(drop=True)  # 2026 never enters
    X_all, _ = build_design(df)
    X_all = X_all.astype(float)
    in_tr = df.season.isin(TRAIN_SEASONS).to_numpy()

    meta = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "train_seasons": TRAIN_SEASONS,
        "model": "ridge (decision #20 -- LightGBM/rung3 failed to beat this by its 3% bar)",
        "holdout": "2026 -- not fit, not scored, not read by this script (decision #24)",
        "stats": {},
    }

    for stat, tcol, screen in STATS:
        tr = in_tr & df[screen].to_numpy() & df[tcol].notna().to_numpy()
        n = int(tr.sum())
        Xtr, ytr = X_all[tr], df.loc[tr, tcol].to_numpy(float)
        fit = fit_ship(Xtr, ytr)

        path = OUT_DIR / f"{stat}.pkl"
        joblib.dump({
            "model": fit["model"], "columns": list(X_all.columns), "screen": screen,
            "median": fit["median"], "mu": fit["mu"], "sd": fit["sd"], "alpha": fit["alpha"],
        }, path)

        p = fit["model"].predict(((Xtr.fillna(fit["median"]).replace([np.inf, -np.inf], np.nan).fillna(0.0)
                                    - fit["mu"]) / fit["sd"]).to_numpy(float))
        mae_insample = float(np.abs(ytr - p).mean())
        meta["stats"][stat] = {
            "n_train_rows": n, "alpha": fit["alpha"],
            "train_mae_insample": round(mae_insample, 4),
            "artifact": str(path),
        }
        print(f"{stat:<17} n={n:,}  alpha={fit['alpha']:.1f}  in-sample MAE={mae_insample:.4f}  -> {path.name}")

    meta_path = OUT_DIR / "vol_metadata.json"
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"\nwritten -> {meta_path}")
    print(f"trained through {LATEST_COMPLETE_SEASON}; 2026 holdout untouched")


if __name__ == "__main__":
    main()
