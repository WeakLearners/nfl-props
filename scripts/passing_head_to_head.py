#!/usr/bin/env python
"""Decision #39, step 2: project.py's live passing_yards projection vs. the
Phase 3 ridge passing_yards model, walk-forward on 2021-2025. 2026 is never
touched.

Same protocol as scripts/vol_head_to_head.py -- Phase 3 side refits once per
test season on every prior season (2019..season-1), scored on that season;
project.py side calls project_week(season, week) once per week actually
played, which already walk-forwards internally. Both sides see only data
strictly before what they're scoring.

WHY A SEPARATE SCRIPT, NOT A FIFTH ROW IN vol_head_to_head.py's STATS list.
The passing_yards model uses PASSING_COLS (pass_attempts_l4/l8/career,
pass_completions_*, yards_per_attempt_*, completion_rate_*, pass_adot_*) as
features -- rung2_ridge.build_design() only includes them when called with
include_passing=True (decision #39's guard against those columns silently
entering the targets/carries/receiving_yards/rushing_yards design matrices).
vol_head_to_head.py builds ONE shared X_all for all four of its stats with
the default (include_passing=False); passing_yards needs its own X_all built
with include_passing=True, so it gets its own script rather than a shared
design matrix that would have to serve two different column sets.

POPULATION. INNER JOIN of (player_id, game_id) rows where training_rows'
qual_attempts screen holds AND project.py produced a passing_yards row for
that player-game. n dropped by each side is printed explicitly.

Read-only against features.db and nflprops.db. Writes only
data/passing_head_to_head.csv. Makes zero Odds API calls.
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
    """project.py's passing_yards projection for every played week of
    `season`, walk-forward."""
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
    X_all = X_all.astype(float)

    rows = []
    for season in TEST_SEASONS:
        old = old_predictions(season)
        print(f"  {season}: project.py produced {len(old):,} passing_yards rows "
              f"({time.time() - t0:.0f}s elapsed)")

        tr_mask = (tr.season < season).to_numpy() & tr["qual_attempts"].to_numpy() & tr["y_passing_yards"].notna().to_numpy()
        te_mask = (tr.season == season).to_numpy() & tr["qual_attempts"].to_numpy() & tr["y_passing_yards"].notna().to_numpy()
        if te_mask.sum() == 0 or tr_mask.sum() < 200:
            print(f"    passing_yards season {season}: insufficient rows, skipped "
                  f"(train={int(tr_mask.sum())}, test={int(te_mask.sum())})")
            continue
        yhat, alpha, _ = fit_predict(
            X_all[tr_mask], tr.loc[tr_mask, "y_passing_yards"].to_numpy(float), X_all[te_mask])
        new_df = pd.DataFrame({
            "player_id": tr.loc[te_mask, "player_id"].to_numpy(),
            "game_id": tr.loc[te_mask, "game_id"].to_numpy(),
            "y": tr.loc[te_mask, "y_passing_yards"].to_numpy(float),
            "yhat_new": yhat,
        })
        old_sub = old.rename(columns={"proj": "yhat_old"})

        joined = new_df.merge(old_sub, on=["player_id", "game_id"], how="inner")
        n_new_only = len(new_df) - len(joined)
        n_old_only = len(old_sub) - old_sub.merge(
            new_df[["player_id", "game_id"]], on=["player_id", "game_id"], how="inner").shape[0]

        if joined.empty:
            print(f"    passing_yards season {season}: no overlap, skipped")
            continue

        mae_old = float((joined.y - joined.yhat_old).abs().mean())
        mae_new = float((joined.y - joined.yhat_new).abs().mean())
        rows.append({
            "season": season, "stat": "passing_yards", "n": len(joined),
            "n_dropped_new_side": n_new_only, "n_dropped_old_side": n_old_only,
            "mae_old_projectpy": round(mae_old, 4),
            "mae_new_ridge": round(mae_new, 4),
            "ridge_alpha": alpha,
        })

    res = pd.DataFrame(rows)
    res.to_csv(ROOT / "data" / "passing_head_to_head.csv", index=False)

    pd.set_option("display.width", 160)
    print("\nper-season:")
    print(res.to_string(index=False))

    if not res.empty:
        n_tot = res.n.sum()
        old_w = (res.mae_old_projectpy * res.n).sum() / n_tot
        new_w = (res.mae_new_ridge * res.n).sum() / n_tot
        winner = "ridge" if new_w < old_w else "project.py"
        lift = (old_w - new_w) / old_w * 100
        n_seasons_ridge_won = int((res.mae_new_ridge < res.mae_old_projectpy).sum())
        print(f"\npooled across 2021-2025 (n-weighted mean MAE):")
        print(f"  passing_yards     n={n_tot:>6}  project.py {old_w:.4f}  ridge {new_w:.4f}  "
              f"winner={winner}  (ridge lift over project.py {lift:+.2f}%)  "
              f"ridge won {n_seasons_ridge_won}/{len(res)} seasons")

    print(f"\nwritten -> data/passing_head_to_head.csv  ({time.time() - t0:.0f}s total)")


if __name__ == "__main__":
    main()
