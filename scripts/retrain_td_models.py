#!/usr/bin/env python
"""Routine retraining for the prediction engine's touchdown models. Decision
#27 (prediction-engine/decisions.md), steps 1 and 2.

ONE COMMAND, TWO OUTPUTS refit together on purpose: retraining and
calibration are the same act here. Isotonic calibration is fitted ON the
training rows (via held-out folds), so any time the training window moves,
the correction has to move with it or it is calibrating a model that no
longer exists.

WHAT SHIPS AND WHY ONLY HALF OF IT IS CALIBRATED.
  receiving_tds -- RAW LightGBM. #27 found this stat was STALE, not
                   miscalibrated: one more season of training pulled its
                   worst gap from 0.074 to 0.047, into the noise floor. A
                   correction layer on top of that would be sanding a
                   surface that was already flat.
  rushing_tds   -- CalibratedClassifierCV(method="isotonic"), same protocol
                   td_calibrate.py validated: the classifier is fit on 2/3
                   of the training rows, the correction learns from
                   predictions on the held-out 1/3, and neither ever sees a
                   validation or holdout row. #27 found this stat's top
                   bucket asserts ~50% every season while reality tops out
                   near 37% -- a ceiling, not staleness -- and isotonic is
                   what learns a ceiling.

TRAINING WINDOW. Every complete regular season through LATEST_COMPLETE_SEASON.
Bump that constant by hand once a season has finished -- do not infer it from
"the latest season with any rows", because 2026 has rows (two weeks of them)
and none of that season's outcomes may enter training. The 2026 holdout stays
sealed per decision #24: this script never fits on it, never scores against
it, never looks at it.

THIS IS A RESEARCH-ARCHITECTURE ARTIFACT, NOT YET A LIVE ONE. It trains on
`training_rows` in features.db, which scripts/build_training_table.py builds
only from games that have already been played (see that script's `load()`).
There is no equivalent of nflprops/tdmodel.py's features(season, week) for
this design matrix -- no code path that can build one of these rows for a
game that has not been played yet. So this script produces a real, versioned,
reusable model artifact, and that artifact answers "what does the calibrated
rushing-TD model say about a played week" -- but nothing currently serves its
output into the live report. The live report's own touchdown badge is a
SEPARATE system (nflprops/tdmodel.py + scripts/td/), built on a different,
combined anytime-TD target that #27 never tested and this script does not
touch. Wiring THIS artifact into a live slate would require building that
missing pre-game feature function first -- new scope, not part of #27.

SCHEDULE: on demand, not launchd. Unlike the daily data refresh (pure ETL,
safe to run unattended), retraining changes what the model asserts. Every
other model change in this project's history has gone through a reviewed
decision entry first; a silent nightly reflit would be the one exception, and
noticing a bad refit after the fact costs more than remembering to run one
command before a week matters. Run it manually before opening the holdout, or
whenever a new season completes.

Read-only against features.db. Writes only under data/shipping/. Does not
call build_db.py, does not touch any table in features.db.
"""
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import joblib
import lightgbm as lgb
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss, roc_auc_score

from rung2_ridge import load, add_derived, build_design

LATEST_COMPLETE_SEASON = 2025          # bump by hand once a season finishes
TRAIN_SEASONS = list(range(2019, LATEST_COMPLETE_SEASON + 1))

OUT_DIR = Path.home() / "Code/nfl-props/data/shipping"
OUT_DIR.mkdir(exist_ok=True)

PARAMS = dict(objective="binary", n_estimators=300, learning_rate=0.05,
              num_leaves=31, min_child_samples=50, subsample=0.8, subsample_freq=1,
              colsample_bytree=0.8, reg_lambda=1.0, n_jobs=-1, verbose=-1, random_state=17)

# (stat name, outcome column, qualifying screen, calibrate?)
STATS = [
    ("receiving_tds", "y_receiving_tds", "qual_targets", False),
    ("rushing_tds",   "y_rushing_tds",   "qual_carries", True),
]


def git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=Path(__file__).resolve().parent
        ).decode().strip()
    except Exception:
        return "unknown"


def main():
    df = add_derived(load())
    X = build_design(df)[0].astype(float)
    in_tr = df.season.isin(TRAIN_SEASONS).to_numpy()

    meta = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "train_seasons": TRAIN_SEASONS,
        "holdout": "2026 -- not fit, not scored, not read by this script (decision #24)",
        "stats": {},
    }

    for stat, col, screen, calibrate in STATS:
        hit = col + "_hit"
        df[hit] = (df[col] >= 1).astype(float)
        tr = in_tr & df[screen].to_numpy() & df[col].notna().to_numpy()
        n = int(tr.sum())
        Xtr, ytr = X[tr], df.loc[tr, hit]

        if calibrate:
            model = CalibratedClassifierCV(
                lgb.LGBMClassifier(**PARAMS), method="isotonic", cv=3
            ).fit(Xtr, ytr)
            kind = "lightgbm+isotonic"
        else:
            model = lgb.LGBMClassifier(**PARAMS).fit(Xtr, ytr)
            kind = "lightgbm (raw, no calibration -- decision #27)"

        path = OUT_DIR / f"{stat}.pkl"
        joblib.dump({"model": model, "columns": list(X.columns), "screen": screen}, path)

        # In-sample sanity only -- NOT a validation score. The real validation
        # numbers for this exact protocol are in data/td_calibration.json,
        # produced by td_calibrate.py's walk-forward eval on 2024/2025.
        p = model.predict_proba(Xtr)[:, 1]
        meta["stats"][stat] = {
            "kind": kind, "n_train_rows": n,
            "train_brier_insample": round(float(brier_score_loss(ytr, p)), 5),
            "train_auc_insample": round(float(roc_auc_score(ytr, p)), 4),
            "artifact": str(path),
        }
        print(f"{stat:<16} {kind:<40} n={n:,}  -> {path.name}")

    meta_path = OUT_DIR / "metadata.json"
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"\nwritten -> {meta_path}")
    print(f"trained through {LATEST_COMPLETE_SEASON}; 2026 holdout untouched")


if __name__ == "__main__":
    main()
