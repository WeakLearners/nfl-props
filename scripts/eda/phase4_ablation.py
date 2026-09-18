#!/usr/bin/env python
"""Phase 4: does adding the top pruned features improve on the current
engine, out of sample?

Reuses scripts/backtest.py's saved output (reports/backtest_full.csv) for
the CURRENT ENGINE's projection and the naive career-average baseline --
both already walk-forward correct (project_week(s,w) only ever reads
season/week strictly before (s,w)).

For "engine + new features" we do NOT retrain project.py. Instead: fit a
small linear residual-correction model, actual ~ current_proj + [new
features], and compare its out-of-sample MAE to a same-shape control model,
actual ~ current_proj alone (recalibration only, no new information). The
gap between those two isolates what the new features add, cleanly, without
having to reimplement the volume x efficiency machinery. Both are trained
walk-forward: expanding-window folds train on strictly prior seasons.
"""
import pathlib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
bt = pd.read_csv(ROOT / "reports" / "backtest_full.csv")
feat = pd.read_parquet(ROOT / "data" / "eda_features.parquet")

df = bt.merge(feat, on=["player_id", "season", "week", "position"], how="inner",
              suffixes=("", "_f"))

# pruned shortlist per stat -- the survivors from Phase 2/3, one family per
# construct to avoid the wopr/target_share/air_yards_share redundancy.
SHORTLIST = {
    "passing_yards": ["h_snap_pct", "total_line", "h_snap_trend"],
    "completions": ["h_snap_pct", "total_line", "team_carries_base"],
    "rushing_yards_QB": ["h_carries", "h_ypc", "h_rz_carries_pg"],
    "rushing_yards_RB": ["h_snap_pct", "h_rz_carries_pg", "team_out_total"],
    "receptions_RB": ["h_snap_pct", "h_rz_targets_pg"],
    "receiving_yards_RB": ["h_snap_pct", "h_rz_targets_pg"],
    "receptions_WR": ["h_snap_pct", "h_rz_targets_pg", "h_air_yards_share"],
    "receiving_yards_WR": ["h_snap_pct", "h_rz_targets_pg", "h_air_yards_share"],
    "receptions_TE": ["h_snap_pct", "h_rz_targets_pg"],
    "receiving_yards_TE": ["h_snap_pct", "h_rz_targets_pg"],
}

FOLDS = [(2022, 2023), (2023, 2024), (2024, 2025)]  # (train_upto, test_season)


def run_stat(stat, position=None, extra_key=None):
    key = extra_key or stat
    sub = df[df["stat"] == stat].copy()
    if position:
        sub = sub[sub.position == position]
    feats = SHORTLIST[key]
    needed = ["actual", "proj"] + feats
    sub = sub.dropna(subset=needed)
    if len(sub) < 200:
        print(f"{key:<22} skip: n={len(sub)} too small")
        return None

    ctrl_preds, aug_preds, base_mae_all, actual_all = [], [], [], []
    naive_preds = []
    for train_upto, test_season in FOLDS:
        train = sub[sub.season <= train_upto]
        test = sub[sub.season == test_season]
        if len(train) < 100 or len(test) < 20:
            continue

        ctrl = make_pipeline(StandardScaler(), Ridge(alpha=5.0))
        ctrl.fit(train[["proj"]], train["actual"])
        ctrl_pred = ctrl.predict(test[["proj"]])

        aug = make_pipeline(StandardScaler(), Ridge(alpha=5.0))
        aug.fit(train[["proj"] + feats], train["actual"])
        aug_pred = aug.predict(test[["proj"] + feats])

        ctrl_preds.append(pd.Series(ctrl_pred, index=test.index))
        aug_preds.append(pd.Series(aug_pred, index=test.index))
        actual_all.append(test["actual"])
        base_mae_all.append(test["proj"])
        naive_preds.append(test["baseline"])

    if not actual_all:
        print(f"{key:<22} skip: no valid folds")
        return None

    actual = pd.concat(actual_all)
    ctrl_p = pd.concat(ctrl_preds)
    aug_p = pd.concat(aug_preds)
    raw_engine = pd.concat(base_mae_all)
    naive = pd.concat(naive_preds)

    mae_raw_engine = (raw_engine - actual).abs().mean()
    mae_naive = (naive - actual).abs().mean()
    mae_ctrl = (ctrl_p - actual).abs().mean()
    mae_aug = (aug_p - actual).abs().mean()
    n = len(actual)
    delta = mae_ctrl - mae_aug
    print(f"{key:<22} n={n:<6} naive={mae_naive:6.2f}  engine_raw={mae_raw_engine:6.2f}  "
          f"engine_recal={mae_ctrl:6.2f}  engine+features={mae_aug:6.2f}  "
          f"delta={delta:+.3f} ({delta/mae_ctrl*100:+.1f}%)  feats={feats}")
    return dict(stat=key, n=n, naive=mae_naive, engine_raw=mae_raw_engine,
                engine_recal=mae_ctrl, engine_plus_features=mae_aug, delta=delta)


def run_qb_rush():
    """The current engine has NO projection at all for QB rushing_yards --
    POS_STATS["QB"] only lists passing_yards/completions (see project.py).
    So there is no `proj` column to recalibrate; compare a features-only
    model straight against the naive career-average baseline (which project_week.py
    computes for every stat in SPECS, rushing_yards included, whether or not
    the engine actually projects it for QBs)."""
    sub = feat[feat.position == "QB"].dropna(subset=["rushing_yards", "h_carries"])
    # naive baseline: unweighted mean of prior games -- recompute directly since
    # backtest_full.csv has no QB/rushing_yards rows to source it from.
    feats = SHORTLIST["rushing_yards_QB"]
    sub = sub.dropna(subset=feats)
    naive_maes, aug_maes, ns = [], [], []
    for train_upto, test_season in FOLDS:
        train = sub[sub.season <= train_upto]
        test = sub[sub.season == test_season]
        if len(train) < 100 or len(test) < 20:
            continue
        aug = make_pipeline(StandardScaler(), Ridge(alpha=5.0))
        aug.fit(train[feats], train["rushing_yards"])
        pred = aug.predict(test[feats])
        # naive: player's own decayed history mean carries * their h_ypc, i.e.
        # what the existing volume x efficiency shape would give with no
        # opponent/team adjustment -- the fairest "already-available" baseline
        naive_pred = test["h_carries"] * test["h_ypc"].fillna(sub["h_ypc"].mean())
        naive_maes.append((naive_pred - test["rushing_yards"]).abs())
        aug_maes.append(pd.Series(pred, index=test.index) - test["rushing_yards"])
        ns.append(len(test))
    naive_mae = pd.concat(naive_maes).abs().mean()
    aug_mae = pd.concat(aug_maes).abs().mean()
    n = sum(ns)
    print(f"{'rushing_yards_QB (vs naive, no engine baseline exists)':<50} "
          f"n={n}  naive_vol*eff={naive_mae:.2f}  features_model={aug_mae:.2f}  "
          f"delta={naive_mae-aug_mae:+.3f}")


if __name__ == "__main__":
    results = []
    results.append(run_stat("passing_yards"))
    results.append(run_stat("completions"))
    run_qb_rush()
    results.append(run_stat("rushing_yards", position="RB", extra_key="rushing_yards_RB"))
    for pos in ["RB", "WR", "TE"]:
        results.append(run_stat("receptions", position=pos, extra_key=f"receptions_{pos}"))
        results.append(run_stat("receiving_yards", position=pos, extra_key=f"receiving_yards_{pos}"))
