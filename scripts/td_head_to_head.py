#!/usr/bin/env python
"""Decision #30, part 4: old anytime-TD model vs. the Phase 3 combined model,
walk-forward on 2019-2025 only. 2026 is never touched -- sealed holdout.

TARGETS TURNED OUT TO BE THE SAME EVENT. The task that ordered this eval
assumed nflprops.tdmodel's target might include fumble-return or defensive
scores. It does not: tdmodel.labels() is (rushing_tds + receiving_tds) > 0,
identical to what 1 - (1-p_rush)*(1-p_rec) approximates from the Phase 3
side. So no population restriction was needed to make the targets agree --
they already do. What differs between the two models is POPULATION
(roster vs. box-score-derived qualifying screens) and METHOD (one combined
classifier vs. two independently-modeled legs combined by an independence
assumption). Both are handled here by intersecting on (player_id, game_id)
and scoring both models on that identical intersection.

Protocol matches scripts/td/evaluate2.py (commit d04e53c): walk-forward by
season, TEST_SEASONS 2021-2025, train strictly before each test season,
refit every season, brier/logloss/AUC/top1 + a shuffled-label floor per
season, and a final-season calibration table.

OLD: nflprops.tdmodel.design_matrix() + features(), same
HistGradientBoostingClassifier params as evaluate2.py, trained on
td_train.parquet rows only (roster universe).

NEW: retrain_td_models.py's exact model spec (LightGBM; rushing_tds wrapped
in CalibratedClassifierCV isotonic cv=3, receiving_tds raw), refit every
test season on training_rows (features.db) rows strictly before it, scored
as combined = 1-(1-p_rush)(1-p_rec).

Read-only against features.db, nflprops.db and data/td_train.parquet.
Writes nothing. Makes zero Odds API calls.
"""
import pathlib
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from nflprops.tdmodel import design_matrix  # noqa: E402
from rung2_ridge import load as load_training_rows, add_derived, build_design  # noqa: E402

TEST_SEASONS = [2021, 2022, 2023, 2024, 2025]
TD_TRAIN = ROOT / "data" / "td_train.parquet"

LGB_PARAMS = dict(objective="binary", n_estimators=300, learning_rate=0.05,
                   num_leaves=31, min_child_samples=50, subsample=0.8, subsample_freq=1,
                   colsample_bytree=0.8, reg_lambda=1.0, n_jobs=-1, verbose=-1, random_state=17)


def top1_rate(frame, col):
    idx = frame.groupby("game_id")[col].idxmax()
    return frame.loc[idx, "td"].mean(), len(idx)


def fit_old(X, y, seed=0):
    m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06,
                                       max_leaf_nodes=31, l2_regularization=1.0, random_state=seed)
    return m.fit(X, y)


def fit_new_leg(X, y, calibrate):
    if calibrate:
        return CalibratedClassifierCV(lgb.LGBMClassifier(**LGB_PARAMS), method="isotonic", cv=3).fit(X, y)
    return lgb.LGBMClassifier(**LGB_PARAMS).fit(X, y)


def main():
    # ---- old model's data: td_train.parquet, roster universe -------------
    old_df = pd.read_parquet(TD_TRAIN).dropna(subset=["game_id"])
    old_df = old_df[old_df.season.between(2019, 2025)].reset_index(drop=True)
    X_old_all = design_matrix(old_df)

    # ---- new model's data: training_rows, box-score-qualifying universe --
    tr = add_derived(load_training_rows())
    tr = tr[tr.season.between(2019, 2025)].reset_index(drop=True)
    X_new_all, _ = build_design(tr)
    X_new_all = X_new_all.astype(float)
    tr["rush_hit"] = (tr.y_rushing_tds.fillna(0) >= 1).astype(float)
    tr["rec_hit"] = (tr.y_receiving_tds.fillna(0) >= 1).astype(float)

    # ---- intersection: which (player_id, game_id) both models can score --
    keys_new = set(zip(tr.player_id, tr.game_id))
    old_df["in_new"] = list(zip(old_df.player_id, old_df.game_id))
    old_df["in_new"] = old_df["in_new"].isin(keys_new)
    print(f"old universe: {len(old_df)} rows, {old_df.in_new.sum()} also scorable by the new model "
          f"({old_df.in_new.mean():.1%})")

    rows = []
    cal_rows = None
    for season in TEST_SEASONS:
        # ---- OLD: exact evaluate2.py protocol, refit every season --------
        tr_mask_old = (old_df.season < season).values
        te_mask_old = (old_df.season == season).values
        test_old = old_df[te_mask_old].copy()
        model_old = fit_old(X_old_all[tr_mask_old], old_df.loc[tr_mask_old, "td"])
        test_old["p_old"] = model_old.predict_proba(X_old_all[te_mask_old])[:, 1]
        rng = np.random.default_rng(season)
        shuf = fit_old(X_old_all[tr_mask_old],
                       pd.Series(rng.permutation(old_df.loc[tr_mask_old, "td"].to_numpy()),
                                 index=old_df.index[tr_mask_old]), season)
        test_old["p_shuf"] = shuf.predict_proba(X_old_all[te_mask_old])[:, 1]

        # ---- NEW: retrain both legs strictly before this season ----------
        tr_mask_new = (tr.season < season).values
        te_mask_new = (tr.season == season).values
        m_rush = fit_new_leg(X_new_all[tr_mask_new], tr.loc[tr_mask_new, "rush_hit"], calibrate=True)
        m_rec = fit_new_leg(X_new_all[tr_mask_new], tr.loc[tr_mask_new, "rec_hit"], calibrate=False)
        test_new = tr[te_mask_new].copy()
        test_new["p_rush"] = m_rush.predict_proba(X_new_all[te_mask_new])[:, 1]
        test_new["p_rec"] = m_rec.predict_proba(X_new_all[te_mask_new])[:, 1]
        test_new["p_new"] = 1 - (1 - test_new.p_rush) * (1 - test_new.p_rec)

        # ---- score on the intersection ------------------------------------
        test_old_scorable = test_old[test_old.in_new].copy()
        joined = test_old_scorable.merge(
            test_new[["player_id", "game_id", "p_new"]], on=["player_id", "game_id"], how="inner")
        assert len(joined) == len(test_old_scorable), "intersection join dropped rows unexpectedly"

        for name, col in [("old", "p_old"), ("shuf", "p_shuf"), ("new", "p_new")]:
            p = joined[col].to_numpy()
            y = joined["td"].to_numpy()
            t1, n_games = top1_rate(joined, col)
            rows.append({
                "season": season, "model": name, "n": len(joined), "games": n_games,
                "brier": brier_score_loss(y, p),
                "logloss": log_loss(y, np.clip(p, 1e-6, 1 - 1e-6)),
                "auc": roc_auc_score(y, p) if p.std() > 0 else np.nan,
                "top1": t1,
            })

        if season == TEST_SEASONS[-1]:
            cal_rows = {}
            for name, col in [("old", "p_old"), ("new", "p_new")]:
                b = pd.qcut(joined[col], 10, labels=False, duplicates="drop")
                cal = joined.assign(bin=b).groupby("bin").agg(
                    pred=(col, "mean"), actual=("td", "mean"), n=("td", "size"))
                cal_rows[name] = cal

        print(f"  {season}: old n={len(joined)}, new n={len(joined)} (intersection)")

    res = pd.DataFrame(rows)
    pd.set_option("display.width", 140)
    print("\nper-season, intersection population:")
    print(res.pivot(index="season", columns="model", values=["brier", "auc", "top1"]).round(4).to_string())
    print("\nmean across test seasons:")
    agg = res.groupby("model")[["brier", "logloss", "auc", "top1"]].mean().round(4)
    print(agg.reindex(["old", "shuf", "new"]).to_string())

    print(f"\ncalibration, final test season ({TEST_SEASONS[-1]}):")
    for name, cal in cal_rows.items():
        print(f"  {name}:")
        gaps = []
        for b, r in cal.iterrows():
            gap = abs(r.actual - r.pred)
            gaps.append(gap)
            print(f"    decile {int(b)+1:>2}  predicted {r.pred:.3f}  actual {r.actual:.3f}  n={int(r.n)}")
        print(f"    mean |gap| {np.mean(gaps):.4f}")

    res.to_csv(ROOT / "data" / "td_head_to_head.csv", index=False)
    print(f"\nwritten -> data/td_head_to_head.csv")


if __name__ == "__main__":
    main()
