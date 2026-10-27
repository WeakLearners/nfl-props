#!/usr/bin/env python
"""EXPERIMENT (offline, proposal only -- no live model/report/rank change).

Decision #39's open item: passing_yards shipped as ridge without a rung-3
(LightGBM) comparison, unlike the other four vol/yardage stats which all lost
to ridge at rung 3 (rung3_lightgbm.py). This asks the same question for
passing_yards, on the same protocol passing_head_to_head.py used to certify
ridge: train on every prior season (not the fixed 2019-2023 window rung2/rung3
use for the other four stats -- that asymmetry is passing_head_to_head.py's
choice, kept here for a like-for-like comparison against its own numbers), one
refit per test season, scored on that whole season. Test seasons 2021-2025.
2019-2020 are training data only. 2026 is never read.

Same LightGBM settings as rung3_lightgbm.py: fixed hyperparameters, chosen
before this run, not tuned against these seasons. Objective regression_l1 --
passing_yards is continuous, not a count, so there is no Poisson variant to
compare (see rung3_lightgbm.py's COUNT_STATS distinction).

Read-only against features.db and nflprops.db. Writes only
data/passing_lightgbm_head_to_head.csv. No API calls, no 2026 outcomes.
"""
import pathlib
import sys
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from nflprops import project as P  # noqa: E402
from nflprops.db import connect  # noqa: E402
from rung2_ridge import load, add_derived, build_design  # noqa: E402
from rung3_lightgbm import PARAMS  # noqa: E402 -- same fixed settings, not re-tuned here

TEST_SEASONS = [2021, 2022, 2023, 2024, 2025]

# Reference ridge/old-engine numbers this experiment is scored against
# (data/passing_head_to_head.csv, decision #39) -- same population, same join,
# recomputed here only as a cross-check that the join reproduces them exactly.
REF_OLD_ENGINE = {2021: 68.1188, 2022: 63.3586, 2023: 67.5734, 2024: 69.0597, 2025: 67.205}
REF_RIDGE = {2021: 64.66, 2022: 60.442, 2023: 64.0952, 2024: 62.2261, 2025: 61.3207}


def old_predictions(season):
    """project.py's passing_yards projection for every played week of
    `season`, walk-forward. Copied from passing_head_to_head.py so the join
    population matches exactly."""
    with connect() as con:
        weeks = pd.read_sql_query(
            "SELECT DISTINCT week FROM schedules WHERE season=? AND game_type='REG' ORDER BY week",
            con, params=[season])["week"].tolist()
    frames = []
    for w in weeks:
        try:
            df = P.project_week(season, w)
        except RuntimeError:
            continue
        df = df[df.stat == "passing_yards"]
        if not df.empty:
            frames.append(df[["player_id", "game_id", "proj"]])
    if not frames:
        return pd.DataFrame(columns=["player_id", "game_id", "proj"])
    return pd.concat(frames, ignore_index=True)


def main():
    t0 = time.time()
    tr = add_derived(load())
    tr = tr[tr.season.between(2019, 2025)].reset_index(drop=True)  # 2026 never enters
    X_all, _ = build_design(tr, include_passing=True)
    X_all = X_all.astype(float)  # no scaling/imputation -- see rung3_lightgbm.py docstring

    rows = []
    for season in TEST_SEASONS:
        old = old_predictions(season)
        tr_mask = (tr.season < season).to_numpy() & tr["qual_attempts"].to_numpy() & tr["y_passing_yards"].notna().to_numpy()
        te_mask = (tr.season == season).to_numpy() & tr["qual_attempts"].to_numpy() & tr["y_passing_yards"].notna().to_numpy()
        if te_mask.sum() == 0 or tr_mask.sum() < 200:
            print(f"  {season}: insufficient rows, skipped (train={int(tr_mask.sum())}, test={int(te_mask.sum())})")
            continue

        m = lgb.LGBMRegressor(objective="regression_l1", **PARAMS)
        m.fit(X_all[tr_mask], tr.loc[tr_mask, "y_passing_yards"].to_numpy(float))
        yhat = m.predict(X_all[te_mask])

        new_df = pd.DataFrame({
            "player_id": tr.loc[te_mask, "player_id"].to_numpy(),
            "game_id": tr.loc[te_mask, "game_id"].to_numpy(),
            "y": tr.loc[te_mask, "y_passing_yards"].to_numpy(float),
            "yhat_lightgbm": yhat,
        })
        joined = new_df.merge(old.rename(columns={"proj": "yhat_old"}),
                               on=["player_id", "game_id"], how="inner")
        if joined.empty:
            print(f"  {season}: no overlap with project.py, skipped")
            continue

        mae_lgb = float((joined.y - joined.yhat_lightgbm).abs().mean())
        mae_old = float((joined.y - joined.yhat_old).abs().mean())
        n = len(joined)

        rows.append(dict(
            season=season, stat="passing_yards", n=n,
            mae_lightgbm=round(mae_lgb, 4), mae_old_projectpy=round(mae_old, 4),
            mae_ridge_reference=REF_RIDGE[season],
            lightgbm_beats_ridge=bool(mae_lgb < REF_RIDGE[season]),
            lightgbm_beats_old_engine=bool(mae_lgb < mae_old),
        ))
        print(f"  {season}: n={n}  lightgbm={mae_lgb:.4f}  ridge_ref={REF_RIDGE[season]}  "
              f"old_engine={mae_old:.4f} (ref {REF_OLD_ENGINE[season]})  "
              f"({time.time()-t0:.0f}s elapsed)")

    res = pd.DataFrame(rows)
    res.to_csv(ROOT / "data" / "passing_lightgbm_head_to_head.csv", index=False)

    if not res.empty:
        n_tot = res.n.sum()
        lgb_w = (res.mae_lightgbm * res.n).sum() / n_tot
        ridge_w = (res.mae_ridge_reference * res.n).sum() / n_tot
        old_w = (res.mae_old_projectpy * res.n).sum() / n_tot
        n_won_ridge = int(res.lightgbm_beats_ridge.sum())
        n_won_old = int(res.lightgbm_beats_old_engine.sum())
        print(f"\npooled 2021-2025 (n-weighted mean MAE, same population as passing_head_to_head.csv):")
        print(f"  lightgbm {lgb_w:.4f}   ridge {ridge_w:.4f}   old_engine(project.py) {old_w:.4f}")
        print(f"  lightgbm beats ridge in {n_won_ridge}/5 seasons; "
              f"beats old engine in {n_won_old}/5 seasons")
        print(f"  lightgbm vs ridge: {(ridge_w - lgb_w) / ridge_w * 100:+.2f}%")

    print(f"\nwritten -> data/passing_lightgbm_head_to_head.csv  ({time.time()-t0:.0f}s total)")


if __name__ == "__main__":
    main()
