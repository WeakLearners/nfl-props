"""
Phase 3, Rung 3 -- LightGBM. Are there interactions worth the extra complexity?

WHAT A BOOSTED TREE IS, briefly. Build a shallow tree that predicts the target
badly. Look at what it got wrong. Build a second tree that predicts THOSE
errors. Keep going. Hundreds of weak trees, each cleaning up after the last.

WHAT IT CAN SAY THAT RIDGE CANNOT. Ridge is one scale with fixed weights: every
player gets the same multiplier on snap share. A tree can say "snap share
matters a lot for a WR1 and barely at all for a WR4". That is an interaction.
Rung 2 already told us the linear part of the signal is worth 3-8% on four
stats. Rung 3 asks whether anything is left on top of that.

THE BAR: beat RUNG 2 -- not the naive baseline -- by >= 2% MAE on a majority of
the six stats, in BOTH validation seasons. Complexity pays rent or it does not
ship.

PROTOCOL is identical to rung 2 and reuses its code: train 2019-2023 fixed,
walk-forward inside each validation season, per-stat screened populations, the
same 132-column design matrix. Anything that differed would make the
comparison a comparison of protocols rather than of models.

THREE THINGS DONE DIFFERENTLY, EACH ON PURPOSE:

1. NO SCALING, NO IMPUTATION. A tree asks "is this value above or below a
   threshold", so stretching an axis changes nothing. LightGBM also routes
   missing values down their own branch natively, which is strictly better than
   replacing them with a median. The __isna flag columns stay anyway -- they
   cost nothing and keep the two rungs on identical columns.

2. OBJECTIVES MATCHED TO THE TARGET. Rung 2's biggest failure was touchdowns,
   and the diagnosis was squared-error loss on a rare near-binary count. So the
   four count stats are fit TWICE: once with a Poisson objective, once with
   plain absolute error. That separates two questions that would otherwise be
   tangled -- "does a tree beat a line?" and "does Poisson beat squared error?"
   Continuous stats get absolute error, which is the metric they are judged on.

3. NO HYPERPARAMETER TUNING. Fixed settings, chosen before the run and written
   below. Tuning against the validation seasons is how a model gets credit for
   a search rather than for a signal. Rung 2's ridge alpha WAS tuned (by
   cross-validation inside training data only), so if anything this handicaps
   rung 3. A win under a handicap is a real win.

ALSO FIXED HERE: rung 2's composed-vs-direct comparison was not clean. The two
routes were scored on different row counts, because the composed route needs a
non-null yards-per-target and a zero-target game has none. Both are now scored
on the INTERSECTION of rows where both are defined.

Read-only against features.db. Writes nothing to any database.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb

from rung2_ridge import (load, add_derived, build_design, SPECS,
                         TRAIN_SEASONS, VALIDATION_SEASONS)

ROOT = Path.home() / "Code/nfl-props"
BASELINES = ROOT / "data/baselines.json"
RUNG2 = ROOT / "data/rung2_ridge.json"
OUT = ROOT / "data/rung3_lightgbm.json"

# Fixed before the run. Not tuned. See point 3 above.
PARAMS = dict(n_estimators=300, learning_rate=0.05, num_leaves=31,
              min_child_samples=50, subsample=0.8, subsample_freq=1,
              colsample_bytree=0.8, reg_lambda=1.0, max_depth=-1,
              n_jobs=-1, verbose=-1, random_state=17)

COUNT_STATS = {"targets", "carries", "receiving_tds", "rushing_tds"}


def fit_predict(Xtr, ytr, Xte, objective):
    m = lgb.LGBMRegressor(objective=objective, **PARAMS)
    m.fit(Xtr, np.clip(ytr, 0, None) if objective == "poisson" else ytr)
    return m.predict(Xte), m


def main():
    df = add_derived(load())
    X_all, _ = build_design(df)
    X_all = X_all.astype(float)          # no scaling, no imputation -- see docstring
    print(f"design matrix: {X_all.shape[0]:,} rows x {X_all.shape[1]} columns (NaN passed through)")

    in_train = df.season.isin(TRAIN_SEASONS).to_numpy()
    preds, importances = {}, {}

    for season in VALIDATION_SEASONS:
        weeks = sorted(df.loc[df.season == season, "week"].unique())
        for layer, stat, tcol, screen, _is_count in SPECS:
            objectives = (["poisson", "regression_l1"] if stat in COUNT_STATS
                          else ["regression_l1"])
            for obj in objectives:
                rows_out = []
                for i in range(1, len(weeks)):
                    prior = df.season.eq(season) & df.week.isin(weeks[:i])
                    tr_mask = (in_train | prior.to_numpy()) & df[screen].to_numpy() & df[tcol].notna().to_numpy()
                    te_mask = (df.season.eq(season) & df.week.eq(weeks[i])).to_numpy() & df[screen].to_numpy() & df[tcol].notna().to_numpy()
                    if te_mask.sum() == 0 or tr_mask.sum() < 200:
                        continue
                    yhat, model = fit_predict(X_all[tr_mask],
                                              df.loc[tr_mask, tcol].to_numpy(float),
                                              X_all[te_mask], obj)
                    rows_out.append(pd.DataFrame({"idx": np.flatnonzero(te_mask),
                                                  "y": df.loc[te_mask, tcol].to_numpy(float),
                                                  "yhat": yhat}))
                    if i == len(weeks) - 1 and season == 2025 and obj == objectives[0]:
                        importances[stat] = pd.Series(
                            model.feature_importances_, index=X_all.columns
                        ).sort_values(ascending=False)
                if rows_out:
                    preds[(season, stat, obj)] = pd.concat(rows_out, ignore_index=True)
        print(f"  {season}: fitted")

    # ---- composed final stats, on a SHARED population -----------------------
    for season in VALIDATION_SEASONS:
        for stat, ucol, ccol, ycol in [
            ("receiving_yards", "targets", "yards_per_target", "y_receiving_yards"),
            ("rushing_yards",   "carries", "yards_per_carry",  "y_rushing_yards")]:
            u = preds.get((season, ucol, "poisson"))
            c = preds.get((season, ccol, "regression_l1"))
            d = preds.get((season, stat, "regression_l1"))
            if u is None or c is None or d is None:
                continue
            m = u.merge(c, on="idx", suffixes=("_u", "_c"))
            m["yhat"] = np.clip(m.yhat_u, 0, None) * np.clip(m.yhat_c, 0, None)
            m["y"] = df.loc[m.idx, ycol].to_numpy(float)
            preds[(season, stat, "composed")] = m[["idx", "y", "yhat"]]
            # direct, restricted to exactly the rows composed could score
            preds[(season, stat, "direct_sharedpop")] = d[d.idx.isin(set(m.idx))]

    # ---- score against BOTH the baseline and rung 2 --------------------------
    base = json.loads(BASELINES.read_text())["baselines"]
    r2 = json.loads(RUNG2.read_text())["results"]
    layer_of = {s[1]: s[0] for s in SPECS}
    out, table = {}, []
    for (season, stat, obj), p in sorted(preds.items()):
        err = p.y - p.yhat
        mae, rmse = float(err.abs().mean()), float(np.sqrt((err ** 2).mean()))
        b = base[str(season)][layer_of[stat]][stat]
        bmae = b[b["winner"]]["mae"]
        r2mae = r2[str(season)][stat].get("raw", {}).get("mae")
        rec = dict(mae=round(mae, 4), rmse=round(rmse, 4), n=int(len(p)),
                   baseline_mae=bmae, lift_vs_baseline_pct=round((bmae - mae) / bmae * 100, 2))
        if r2mae:
            rec["rung2_mae"] = r2mae
            rec["lift_vs_rung2_pct"] = round((r2mae - mae) / r2mae * 100, 2)
        out.setdefault(str(season), {}).setdefault(stat, {})[obj] = rec
        table.append((season, layer_of[stat], stat, obj, mae, r2mae, bmae,
                      rec.get("lift_vs_rung2_pct"), rec["lift_vs_baseline_pct"], len(p)))

    OUT.write_text(json.dumps(dict(
        rung=3, model="lightgbm", params=PARAMS,
        bar="beat rung 2 by >= 2% MAE on a majority of the six stats, both seasons",
        results=out), indent=2))

    print(f"\n{'season':<7}{'stat':<17}{'objective':<18}{'MAE':>9}{'rung2':>9}{'base':>9}{'vsR2%':>8}{'vsBase%':>9}{'n':>7}")
    for r in sorted(table, key=lambda r: (r[2], r[3], r[0])):
        vr = f"{r[7]:+.2f}" if r[7] is not None else "    --"
        r2s = f"{r[5]:.4f}" if r[5] else "    --"
        print(f"{r[0]:<7}{r[2]:<17}{r[3]:<18}{r[4]:>9.4f}{r2s:>9}{r[6]:>9.4f}{vr:>8}{r[8]:>+9.2f}{r[9]:>7}")

    print("\n--- top 12 LightGBM split counts, 2025 ---")
    for stat in ("targets", "receiving_tds", "yards_per_target"):
        s = importances.get(stat)
        if s is None:
            continue
        print(f"\n{stat}:")
        for name, v in s.head(12).items():
            print(f"   {int(v):>6}  {name}")

    print(f"\nwritten -> {OUT}")


if __name__ == "__main__":
    main()
