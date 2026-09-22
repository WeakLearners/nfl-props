"""Phase 4 supplement -- does the shrinkage arm's effect (if any) show up
selectively in WR/QB carry rows (the ones that were broken) or does it also
move RB rows (which were already fine)? A fix that moves RB rows too is
suspicious per the brief. carries/rushing_yards only -- these are the stats
where the broken opp_yards_allowed_per_carry_l4 column actually fed in and
where the population mixes RB (healthy denominator) with WR/QB (thin
denominator). Re-fits arm_a_shrinkage and no_defense only, same protocol."""
import sys, json
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb

ROOT = Path.home() / "Code/nfl-props"
sys.path.insert(0, str(ROOT / "scripts"))
from phase4_opponent import (load_pg, fit_shrinkage_k, load_training_rows,
                              make_feature_sets, PARAMS, OLD_OPP_COLS)
from rung2_ridge import add_derived, build_design, TRAIN_SEASONS, VALIDATION_SEASONS

STATS = [("carries", "y_carries", "qual_carries", "regression_l1"),
         ("rushing_yards", "y_rushing_yards", "qual_carries", "regression_l1"),
         ("targets", "y_targets", "qual_targets", "regression_l1"),
         ("receiving_yards", "y_receiving_yards", "qual_targets", "regression_l1")]


def score_by_posgroup(df, label):
    X_all, _ = build_design(df)
    X_all = X_all.astype(float)
    in_train = df.season.isin(TRAIN_SEASONS).to_numpy()
    out = {}
    for season in VALIDATION_SEASONS:
        weeks = sorted(df.loc[df.season == season, "week"].unique())
        for stat, tcol, screen, obj in STATS:
            rows_out = []
            for i in range(1, len(weeks)):
                pw = weeks[i]
                prior = df.season.eq(season) & df.week.isin(weeks[:i])
                tr_mask = (in_train | prior.to_numpy()) & df[screen].to_numpy() & df[tcol].notna().to_numpy()
                te_mask = (df.season.eq(season) & df.week.eq(pw)).to_numpy() & df[screen].to_numpy() & df[tcol].notna().to_numpy()
                if te_mask.sum() == 0 or tr_mask.sum() < 200:
                    continue
                m = lgb.LGBMRegressor(objective=obj, **PARAMS)
                ytr = df.loc[tr_mask, tcol].to_numpy(float)
                m.fit(X_all[tr_mask], np.clip(ytr, 0, None) if obj == "poisson" else ytr)
                yhat = m.predict(X_all[te_mask])
                rows_out.append(pd.DataFrame({
                    "y": df.loc[te_mask, tcol].to_numpy(float), "yhat": yhat,
                    "position_group": df.loc[te_mask, "position_group"].to_numpy(),
                }))
            if not rows_out:
                continue
            p = pd.concat(rows_out, ignore_index=True)
            for pos, grp in p.groupby("position_group"):
                err = grp.y - grp.yhat
                out[(season, stat, pos)] = dict(mae=round(float(err.abs().mean()), 4), n=int(len(grp)))
        print(f"  [{label}] {season}: fitted")
    return out


def main():
    pg = load_pg()
    k_table = fit_shrinkage_k(pg)
    tr = add_derived(load_training_rows())
    sets = make_feature_sets(tr, pg, k_table)

    results = {}
    for label in ("no_defense", "arm_a_shrinkage"):
        print(f"=== {label} ===")
        results[label] = score_by_posgroup(sets[label], label)

    print(f"\n{'season':<7}{'stat':<15}{'pos':<5}{'nodef_MAE':>11}{'arm_a_MAE':>11}{'lift%':>8}{'n':>7}")
    rows = []
    for key in results["no_defense"]:
        if key not in results["arm_a_shrinkage"]:
            continue
        season, stat, pos = key
        b = results["no_defense"][key]["mae"]
        a = results["arm_a_shrinkage"][key]["mae"]
        n = results["no_defense"][key]["n"]
        lift = (b - a) / b * 100
        rows.append((season, stat, pos, b, a, lift, n))
    for r in sorted(rows, key=lambda r: (r[1], r[2], r[0])):
        print(f"{r[0]:<7}{r[1]:<15}{r[2]:<5}{r[3]:>11.4f}{r[4]:>11.4f}{r[5]:>+8.2f}{r[6]:>7}")

    Path(ROOT / "data/phase4_opponent_posgroup.json").write_text(json.dumps(
        {"rows": [{"season": r[0], "stat": r[1], "pos": r[2],
                   "nodef_mae": round(r[3], 4), "arm_a_mae": round(r[4], 4),
                   "lift_pct": round(r[5], 2), "n": r[6]} for r in rows]}, indent=2))


if __name__ == "__main__":
    main()
