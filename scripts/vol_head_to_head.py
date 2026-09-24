#!/usr/bin/env python
"""Decision #31, step 1: project.py's live projection vs. the Phase 3 ridge
volume/yardage models (decision #20 -- ridge, not LightGBM, is what actually
cleared the ship bar), walk-forward on 2021-2025. 2026 is never touched.

WHY RIDGE, NOT LIGHTGBM. phase3-model.md's "PHASE 3 -- CLOSED" table lists
LightGBM's lift over baseline and marks all four volume/yardage stats
"ships: yes". That table is stale. Decision #20, recorded the same day,
found LightGBM (rung 3) failed to beat ridge (rung 2) by the required 3%
margin on ALL FOUR stats in at least one validation season -- rushing
yards even lost outright in 2025 (-1.28%) -- and ships ridge instead. This
script builds artifacts and runs the comparison on the model the decision
log actually names as production.

WHAT'S COMPARED, for four stats:
  targets          -- project.py's proj_vol on stat="receiving_yards" rows
                       (SPECS: receiving_yards's volume column is targets)
                       vs. Phase 3 ridge's direct targets model.
  carries           -- project.py's proj_vol on stat="rushing_yards" rows
                       vs. Phase 3 ridge's direct carries model.
  receiving_yards   -- project.py's proj on stat="receiving_yards" rows
                       vs. Phase 3 ridge's direct receiving_yards model.
  rushing_yards     -- project.py's proj on stat="rushing_yards" rows
                       vs. Phase 3 ridge's direct rushing_yards model.

PROTOCOL. Phase 3 side: refit once per test season on every prior season
(2019..season-1), scored on that season -- same shape as
scripts/td_head_to_head.py, not rung2_ridge.py's intra-season walk-forward
(that script fixes training at 2019-2023 for both validation years on
purpose, decision #17; reusing that fixed window across 2021-2025 would
leak 2022-2023 into scoring 2021). project.py side: project_week(season,
week) is called once per week actually played that season; it already
walk-forwards internally (_history() reads only season<w's season, or
same season & week<w). Both sides therefore see only data strictly before
what they're scoring.

POPULATION. Each stat's population is the INNER JOIN of (player_id,
game_id) rows where training_rows' leak-gated qualifying screen holds
(qual_targets or qual_carries, per rung2_ridge.SPECS) AND project.py
produced a row for that player-game-stat. n dropped by each side is
printed explicitly, not glossed over.

Read-only against features.db and nflprops.db. Writes only
data/vol_head_to_head.csv. Makes zero Odds API calls.
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

# (label, old proj.py column, old proj.py stat filter, new/tr target column, screen)
STATS = [
    ("targets",         "proj_vol", "receiving_yards", "y_targets",         "qual_targets"),
    ("carries",         "proj_vol", "rushing_yards",   "y_carries",         "qual_carries"),
    ("receiving_yards", "proj",     "receiving_yards", "y_receiving_yards", "qual_targets"),
    ("rushing_yards",   "proj",     "rushing_yards",   "y_rushing_yards",   "qual_carries"),
]


def old_predictions(season):
    """project.py's projection for every played week of `season`, walk-forward,
    only stat rows it actually produces (receiving_yards, rushing_yards)."""
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
        df = df[df.stat.isin(["receiving_yards", "rushing_yards"])]
        if not df.empty:
            frames.append(df[["player_id", "game_id", "stat", "proj", "proj_vol"]])
    if not frames:
        return pd.DataFrame(columns=["player_id", "game_id", "stat", "proj", "proj_vol"])
    return pd.concat(frames, ignore_index=True)


def main():
    t0 = time.time()
    tr = add_derived(load())
    tr = tr[tr.season.between(2019, 2025)].reset_index(drop=True)  # 2026 never enters
    X_all, _ = build_design(tr)
    X_all = X_all.astype(float)

    rows = []
    for season in TEST_SEASONS:
        old = old_predictions(season)
        print(f"  {season}: project.py produced {len(old):,} rows "
              f"({time.time() - t0:.0f}s elapsed)")

        for label, old_col, old_stat, ycol, screen in STATS:
            tr_mask = (tr.season < season).to_numpy() & tr[screen].to_numpy() & tr[ycol].notna().to_numpy()
            te_mask = (tr.season == season).to_numpy() & tr[screen].to_numpy() & tr[ycol].notna().to_numpy()
            if te_mask.sum() == 0 or tr_mask.sum() < 200:
                continue
            yhat, alpha, _ = fit_predict(
                X_all[tr_mask], tr.loc[tr_mask, ycol].to_numpy(float), X_all[te_mask])
            new_df = pd.DataFrame({
                "player_id": tr.loc[te_mask, "player_id"].to_numpy(),
                "game_id": tr.loc[te_mask, "game_id"].to_numpy(),
                "y": tr.loc[te_mask, ycol].to_numpy(float),
                "yhat_new": yhat,
            })
            old_sub = old[old.stat == old_stat][["player_id", "game_id", old_col]].rename(
                columns={old_col: "yhat_old"})

            joined = new_df.merge(old_sub, on=["player_id", "game_id"], how="inner")
            n_new_only = len(new_df) - len(joined)
            n_old_only = len(old_sub) - old_sub.merge(
                new_df[["player_id", "game_id"]], on=["player_id", "game_id"], how="inner").shape[0]

            if joined.empty:
                print(f"    {label:<17} season {season}: no overlap, skipped")
                continue

            mae_old = float((joined.y - joined.yhat_old).abs().mean())
            mae_new = float((joined.y - joined.yhat_new).abs().mean())
            rows.append({
                "season": season, "stat": label, "n": len(joined),
                "n_dropped_new_side": n_new_only, "n_dropped_old_side": n_old_only,
                "mae_old_projectpy": round(mae_old, 4),
                "mae_new_ridge": round(mae_new, 4),
                "ridge_alpha": alpha,
            })

    res = pd.DataFrame(rows)
    res.to_csv(ROOT / "data" / "vol_head_to_head.csv", index=False)

    pd.set_option("display.width", 160)
    print("\nper-season:")
    print(res.to_string(index=False))

    print("\npooled across 2021-2025 (n-weighted mean MAE):")
    for label, *_ in STATS:
        sub = res[res.stat == label]
        if sub.empty:
            continue
        n_tot = sub.n.sum()
        old_w = (sub.mae_old_projectpy * sub.n).sum() / n_tot
        new_w = (sub.mae_new_ridge * sub.n).sum() / n_tot
        winner = "ridge" if new_w < old_w else "project.py"
        lift = (old_w - new_w) / old_w * 100
        print(f"  {label:<17} n={n_tot:>6}  project.py {old_w:.4f}  ridge {new_w:.4f}  "
              f"winner={winner}  (ridge lift over project.py {lift:+.2f}%)")

    print(f"\nwritten -> data/vol_head_to_head.csv  ({time.time() - t0:.0f}s total)")


if __name__ == "__main__":
    main()
