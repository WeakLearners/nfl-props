#!/usr/bin/env python
"""Phase 3 receptions: pick ridge-direct vs. targets x catch-rate.

Same walk-forward protocol as scripts/vol_head_to_head.py: refit once per
test season (2021-2025) on every prior season (2019..season-1), scored on
that season only. 2026 is never touched.

Two candidates for the NEW receptions number, both pregame-only:
  (a) ridge-direct  -- fit_predict() straight on y_receptions, screen
                        qual_targets (a player who doesn't see enough
                        targets to have a reception forecast worth making).
  (b) targets x rate -- fit_predict() on y_targets (identical tr/te split,
                        identical screen, so this is an apples-to-apples
                        refit of the shipped targets model, not the
                        2019-2025-fixed shipping artifact) multiplied by
                        catch_rate_career -- the leak-gated expanding-mean
                        catch rate already in features.db (shift(1) before
                        the expanding window, per feature_registry), i.e.
                        the player's own shrunk historical catch rate,
                        pregame information only. Missing (rookie) catch
                        rate is filled with the screened population's
                        train-season mean, same convention build_design()
                        uses for missingness elsewhere in this codebase.

Whichever wins on pooled 2021-2025 MAE is what scripts/retrain_vol_models.py
and scripts/vol_predict.py ship. Also runs project.py's own receptions
projection (stat="receptions", column "proj") through the same join used by
vol_head_to_head.py, so the report's old-vs-new head-to-head number for
receptions comes from this script, not a separate estimate.

Read-only against features.db and nflprops.db. Writes only
data/vol_receptions_compare.csv. Zero Odds API calls. 2026 holdout untouched.
"""
import pathlib
import sys
import time

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from nflprops import project as P  # noqa: E402
from nflprops.db import connect  # noqa: E402
from rung2_ridge import load, add_derived, build_design, fit_predict  # noqa: E402

TEST_SEASONS = [2021, 2022, 2023, 2024, 2025]


def old_predictions(season):
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
        df = df[df.stat == "receptions"]
        if not df.empty:
            frames.append(df[["player_id", "game_id", "proj"]].rename(columns={"proj": "yhat_old"}))
    if not frames:
        return pd.DataFrame(columns=["player_id", "game_id", "yhat_old"])
    return pd.concat(frames, ignore_index=True)


def main():
    t0 = time.time()
    tr = add_derived(load())
    tr = tr[tr.season.between(2019, 2025)].reset_index(drop=True)
    X_all, _ = build_design(tr)
    X_all = X_all.astype(float)

    rows = []
    for season in TEST_SEASONS:
        old = old_predictions(season)
        tr_mask = (tr.season < season).to_numpy() & tr.qual_targets.to_numpy() & tr.y_receptions.notna().to_numpy()
        te_mask = (tr.season == season).to_numpy() & tr.qual_targets.to_numpy() & tr.y_receptions.notna().to_numpy()
        if te_mask.sum() == 0 or tr_mask.sum() < 200:
            print(f"  {season}: skipped, insufficient rows")
            continue

        # (a) ridge direct on receptions
        yhat_a, alpha_a, _ = fit_predict(X_all[tr_mask], tr.loc[tr_mask, "y_receptions"].to_numpy(float),
                                          X_all[te_mask])

        # (b) targets refit (same split) x catch_rate_career
        yhat_targets, alpha_t, _ = fit_predict(X_all[tr_mask], tr.loc[tr_mask, "y_targets"].to_numpy(float),
                                                X_all[te_mask])
        fill = tr.loc[tr_mask, "catch_rate_career"].mean()
        cr = tr.loc[te_mask, "catch_rate_career"].fillna(fill).clip(0, 1).to_numpy(float)
        yhat_b = np.clip(yhat_targets, 0, None) * cr

        y = tr.loc[te_mask, "y_receptions"].to_numpy(float)
        mae_a = float(np.abs(y - yhat_a).mean())
        mae_b = float(np.abs(y - yhat_b).mean())

        new_df = pd.DataFrame({
            "player_id": tr.loc[te_mask, "player_id"].to_numpy(),
            "game_id": tr.loc[te_mask, "game_id"].to_numpy(),
            "y": y, "yhat_a": yhat_a, "yhat_b": yhat_b,
        })
        joined = new_df.merge(old, on=["player_id", "game_id"], how="inner")
        mae_old = float((joined.y - joined.yhat_old).abs().mean()) if not joined.empty else np.nan

        rows.append(dict(season=season, n_new=len(new_df), n_joined=len(joined),
                          mae_ridge_direct=round(mae_a, 4), mae_targets_x_rate=round(mae_b, 4),
                          mae_old_projectpy=round(mae_old, 4) if not np.isnan(mae_old) else None,
                          alpha_direct=alpha_a, alpha_targets=alpha_t))
        print(f"  {season}: n={len(new_df):,} joined={len(joined):,} "
              f"ridge_direct={mae_a:.4f} targets_x_rate={mae_b:.4f} old={mae_old:.4f}")

    res = pd.DataFrame(rows)
    res.to_csv(ROOT / "data" / "vol_receptions_compare.csv", index=False)

    n_tot = res.n_joined.sum()
    old_w = (res.mae_old_projectpy * res.n_joined).sum() / n_tot
    a_w = (res.mae_ridge_direct * res.n_joined).sum() / n_tot
    b_w = (res.mae_targets_x_rate * res.n_joined).sum() / n_tot
    new_winner = "ridge_direct" if a_w <= b_w else "targets_x_rate"
    new_mae = min(a_w, b_w)
    wins_a = int((res.mae_ridge_direct < res.mae_targets_x_rate).sum())
    seasons_new_beats_old = int((res[["mae_ridge_direct", "mae_targets_x_rate"]].min(axis=1) < res.mae_old_projectpy).sum())

    print(f"\npooled 2021-2025 (n={n_tot}, weighted by joined rows):")
    print(f"  ridge_direct    {a_w:.4f}")
    print(f"  targets_x_rate  {b_w:.4f}")
    print(f"  project.py(old) {old_w:.4f}")
    print(f"  -> new-model winner: {new_winner} ({new_mae:.4f}); wins ridge_direct-over-targets_x_rate in {wins_a}/5 seasons")
    print(f"  -> new beats old.py in {seasons_new_beats_old}/5 seasons")
    print(f"\nwritten -> data/vol_receptions_compare.csv  ({time.time()-t0:.0f}s total)")


if __name__ == "__main__":
    main()
