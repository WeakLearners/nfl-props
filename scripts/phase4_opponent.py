"""
Phase 4 -- the opponent instrument, rebuilt two ways, scored as three arms.
Decision #26.

WHAT WAS WRONG. `build_opponent_defense` (build_training_table.py:212) groups
player_games by (defense_team, position_group, week) and divides yards by
attempts. Running backs get ~21 carries in a cell and the number is fine.
Wide receivers get ~1 carry in a cell -- one end-around for -6 yards becomes
"this defense allows -6.0 yards per carry to WRs" and gets averaged forward
four weeks. 1,649 of 9,897 carry cells are outside [0, 8] yards/carry, the
range no real defense has ever posted over a season.

THREE ARMS, FROZEN BEFORE ANY RESULT IS SEEN (decision #26):

  A. Shrinkage. shrunk = (n*rate + k*league_avg) / (n+k), same (defense_team,
     position_group, week) grouping as before, but a thin cell gets pulled
     toward the position group's own league average instead of asserting its
     own noise as fact. k is fit from TRAIN seasons only (2019-2023), never
     from validation lift -- see fit_shrinkage_k() for the method.

  B. EPA. rushing_epa and receiving_epa already exist in player_games. Sum
     per (defense_team, week) -- NOT by position_group, that grouping is the
     bug being fixed -- then shifted-roll 4 exactly like every other rolling
     column in this codebase.

  C. Both A and B together.

THE BAR is the no-defense model: the exact same shipped architecture per
stat (rung 3's LightGBM, matching phase3-model.md's closing table) with the
two broken opponent columns dropped and NOTHING put in their place. The
existing columns already lose to their own absence (phase 3 audit), so
beating them proves nothing -- beating their absence is the actual test.

Six stats, matching decision #24's "four ridge stats plus two touchdown
classifiers": targets, carries, receiving_yards, rushing_yards (MAE, LightGBM
regressor, objective matched to the shipped model) and receiving_tds,
rushing_tds (Brier, LightGBM binary classifier, decision #21/#22 shape).
Every non-opponent column and every modeling choice is held fixed across all
four feature-sets (no-defense, A, B, C) -- only the opponent columns change,
so any score difference is attributable to the opponent instrument and
nothing else.

Protocol identical to rung3_lightgbm.py: train 2019-2023 fixed, walk-forward
inside each validation season (2024, 2025), per-stat screened populations
(decision #14), same 300-tree fixed-hyperparameter LightGBM, no tuning
against validation. Bar per decision #25: swing x 1.36 -- MAE 3%, Brier 1%,
both seasons must clear.

Read-only against nflprops.db and features.db. Writes only
data/phase4_opponent.json.
"""
import json
import sqlite3
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import brier_score_loss, roc_auc_score

warnings.filterwarnings("ignore")

ROOT = Path.home() / "Code/nfl-props"
LIVE_DB = ROOT / "data/nflprops.db"
FEATURES_DB = ROOT / "data/features.db"
BASELINES = ROOT / "data/baselines.json"
OUT = ROOT / "data/phase4_opponent.json"

import sys
sys.path.insert(0, str(ROOT / "scripts"))
from rung2_ridge import (  # noqa: E402
    add_derived, build_design, ID_COLS, CATEGORICAL, TRAIN_SEASONS, VALIDATION_SEASONS,
)

WINDOWS4 = 4
PARAMS = dict(n_estimators=300, learning_rate=0.05, num_leaves=31,
              min_child_samples=50, subsample=0.8, subsample_freq=1,
              colsample_bytree=0.8, reg_lambda=1.0, max_depth=-1,
              n_jobs=-1, verbose=-1, random_state=17)
TD_PARAMS = dict(objective="binary", n_estimators=300, learning_rate=0.05,
                  num_leaves=31, min_child_samples=50, subsample=0.8, subsample_freq=1,
                  colsample_bytree=0.8, reg_lambda=1.0, n_jobs=-1, verbose=-1, random_state=17)

MAE_STATS = [
    # (stat, y_col, screen_col, objective)
    ("targets", "y_targets", "qual_targets", "poisson"),
    ("carries", "y_carries", "qual_carries", "regression_l1"),
    ("receiving_yards", "y_receiving_yards", "qual_targets", "regression_l1"),
    ("rushing_yards", "y_rushing_yards", "qual_carries", "regression_l1"),
]
TD_STATS = [
    ("receiving_tds", "y_receiving_tds", "qual_targets"),
    ("rushing_tds", "y_rushing_tds", "qual_carries"),
]

OLD_OPP_COLS = ["opp_yards_allowed_per_target_l4", "opp_yards_allowed_per_carry_l4"]


def shifted_roll(g, col, window, min_periods=1):
    return g[col].shift(1).rolling(window, min_periods=min_periods).mean()


# ---------------------------------------------------------------------------
# Load raw player_games -- needed to build the two new opponent instruments
# from scratch. training_rows only carries the already-broken columns.
# ---------------------------------------------------------------------------

def load_pg():
    con = sqlite3.connect(f"file:{LIVE_DB}?mode=ro", uri=True, timeout=60)
    pg = pd.read_sql(
        "SELECT season, week, team, opponent_team, position_group, "
        "targets, receiving_yards, carries, rushing_yards, rushing_epa, receiving_epa "
        "FROM player_games", con,
    )
    con.close()
    return pg


def load_training_rows():
    con = sqlite3.connect(f"file:{FEATURES_DB}?mode=ro", uri=True, timeout=60)
    df = pd.read_sql("SELECT * FROM training_rows ORDER BY player_id, season, week", con)
    con.close()
    return df


# ---------------------------------------------------------------------------
# Arm A -- shrinkage. k fit from TRAIN seasons only, method-of-moments
# (within-cell vs. between-cell variance), exactly the "standard way" named
# in the brief. Never touches validation rows.
# ---------------------------------------------------------------------------

def fit_shrinkage_k(pg):
    """One k per (side, position_group), side in {carry, target}.

    Model: cell_rate_i ~ N(true_defense_rate_i, sigma0^2 / n_i)
           true_defense_rate_i ~ N(league_avg, sigma_between^2)
    Posterior mean (the empirical-Bayes shrinkage target) is
        (n_i*rate_i + k*league_avg) / (n_i + k),  k = sigma0^2 / sigma_between^2

    sigma0^2 (the per-unit, "within" variance) is estimated from individual
    player-game rows, weighted by their own attempt count, around the
    position group's weighted mean -- the finest granularity the source
    table offers (no single-play data exists here).

    sigma_between^2 (the "true defense differs from league average" variance)
    is backed out of the weighted spread of CELL rates by method of moments:
    the cells' weighted sum of squares equals sigma_between^2 * sum(n) +
    sigma0^2 * (number of cells), on average, under the model above.
    """
    train = pg[pg.season.isin(TRAIN_SEASONS)].copy()
    train["row_ypc"] = np.where(train.carries > 0, train.rushing_yards / train.carries, np.nan)
    train["row_ypt"] = np.where(train.targets > 0, train.receiving_yards / train.targets, np.nan)

    k_table = {}
    for side, count_col, rate_col in [("carry", "carries", "row_ypc"), ("target", "targets", "row_ypt")]:
        for pos in sorted(train.position_group.dropna().unique()):
            rows = train[(train.position_group == pos) & (train[count_col] > 0)]
            n_row = rows[count_col].to_numpy(float)
            rate_row = rows[rate_col].to_numpy(float)
            mu = float(np.sum(n_row * rate_row) / np.sum(n_row))
            sigma0_sq = float(np.sum(n_row * (rate_row - mu) ** 2) / np.sum(n_row))

            # Cell rate is summed yards / summed count, not the mean of
            # game-level rates -- built directly from the raw columns.
            raw_col = "rushing_yards" if side == "carry" else "receiving_yards"
            cells = rows.groupby(["opponent_team", "season", "week"], as_index=False).agg(
                n=(count_col, "sum"), y=(raw_col, "sum"),
            )
            cells["rate"] = cells.y / cells.n
            n_cell = cells.n.to_numpy(float)
            rate_cell = cells.rate.to_numpy(float)
            K = len(cells)

            Q = float(np.sum(n_cell * (rate_cell - mu) ** 2))
            sigma_between_sq = max(1e-6, (Q - sigma0_sq * K) / np.sum(n_cell))
            # Capped for numerical safety only -- a cap this high never binds
            # in practice at the carry/target counts this data has; it just
            # keeps an effectively-all-noise cell (e.g. QB targets) from
            # producing an unbounded float.
            k = min(sigma0_sq / sigma_between_sq, 1_000_000.0)
            k_table[(side, pos)] = dict(k=round(k, 2), sigma0_sq=round(sigma0_sq, 4),
                                          sigma_between_sq=round(sigma_between_sq, 6),
                                          n_cells=K, league_avg=round(mu, 4))
    return k_table


def build_arm_a(pg, k_table):
    """Same (defense_team, position_group, week) grouping as the broken
    column, but each cell's rate is shrunk toward the position group's own
    league average by that side's k before the shift(1)+roll(4)."""
    agg = pg.groupby(["season", "week", "opponent_team", "position_group"], as_index=False).agg(
        targets=("targets", "sum"), rec_yards=("receiving_yards", "sum"),
        carries=("carries", "sum"), rush_yards=("rushing_yards", "sum"),
    ).rename(columns={"opponent_team": "defense_team"})

    for side, count_col, raw_col, out_row_col in [
        ("carry", "carries", "rush_yards", "row_ypc_shrunk"),
        ("target", "targets", "rec_yards", "row_ypt_shrunk"),
    ]:
        raw_rate = np.where(agg[count_col] > 0, agg[raw_col] / agg[count_col], np.nan)
        k = agg["position_group"].map(lambda p: k_table[(side, p)]["k"])
        league_avg = agg["position_group"].map(lambda p: k_table[(side, p)]["league_avg"])
        n = agg[count_col].to_numpy(float)
        shrunk = np.where(
            n > 0,
            (n * np.nan_to_num(raw_rate) + k.to_numpy(float) * league_avg.to_numpy(float)) / (n + k.to_numpy(float)),
            np.nan,
        )
        agg[out_row_col] = shrunk

    agg = agg.sort_values(["defense_team", "position_group", "season", "week"])
    grouped = agg.groupby(["defense_team", "position_group"], group_keys=False)
    agg["opp_ypt_shrunk_l4"] = grouped.apply(lambda g: shifted_roll(g, "row_ypt_shrunk", WINDOWS4))
    agg["opp_ypc_shrunk_l4"] = grouped.apply(lambda g: shifted_roll(g, "row_ypc_shrunk", WINDOWS4))
    return agg[["season", "week", "defense_team", "position_group", "opp_ypt_shrunk_l4", "opp_ypc_shrunk_l4"]]


# ---------------------------------------------------------------------------
# Arm B -- EPA. Grouped by (defense_team, week) ONLY. No position group.
# ---------------------------------------------------------------------------

def build_arm_b(pg):
    df = pg.copy()
    df["rushing_epa"] = df["rushing_epa"].fillna(0.0)
    df["receiving_epa"] = df["receiving_epa"].fillna(0.0)
    agg = df.groupby(["season", "week", "opponent_team"], as_index=False).agg(
        rush_epa_allowed=("rushing_epa", "sum"), rec_epa_allowed=("receiving_epa", "sum"),
    ).rename(columns={"opponent_team": "defense_team"})
    agg = agg.sort_values(["defense_team", "season", "week"])
    grouped = agg.groupby("defense_team", group_keys=False)
    agg["opp_rush_epa_allowed_l4"] = grouped.apply(lambda g: shifted_roll(g, "rush_epa_allowed", WINDOWS4))
    agg["opp_rec_epa_allowed_l4"] = grouped.apply(lambda g: shifted_roll(g, "rec_epa_allowed", WINDOWS4))
    return agg[["season", "week", "defense_team", "opp_rush_epa_allowed_l4", "opp_rec_epa_allowed_l4"]]


# ---------------------------------------------------------------------------
# Assemble the four feature-sets (no-defense, A, B, C) on top of training_rows
# ---------------------------------------------------------------------------

def make_feature_sets(tr, pg, k_table):
    base = tr.drop(columns=OLD_OPP_COLS)

    arm_a = build_arm_a(pg, k_table)
    arm_b = build_arm_b(pg)

    df_nodef = base.copy()

    df_a = base.merge(
        arm_a.rename(columns={"defense_team": "opponent_team"}),
        on=["season", "week", "opponent_team", "position_group"], how="left",
    )

    df_b = base.merge(
        arm_b.rename(columns={"defense_team": "opponent_team"}),
        on=["season", "week", "opponent_team"], how="left",
    )

    df_c = df_a.merge(
        arm_b.rename(columns={"defense_team": "opponent_team"}),
        on=["season", "week", "opponent_team"], how="left",
    )

    return {"no_defense": df_nodef, "arm_a_shrinkage": df_a, "arm_b_epa": df_b, "arm_c_both": df_c}


# ---------------------------------------------------------------------------
# Scoring -- identical walk-forward protocol to rung3_lightgbm.py / td_probability.py
# ---------------------------------------------------------------------------

def score_mae_stats(df, label):
    X_all, _ = build_design(df)
    X_all = X_all.astype(float)
    in_train = df.season.isin(TRAIN_SEASONS).to_numpy()
    out = {}
    for season in VALIDATION_SEASONS:
        weeks = sorted(df.loc[df.season == season, "week"].unique())
        for stat, tcol, screen, obj in MAE_STATS:
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
                rows_out.append(pd.DataFrame({"y": df.loc[te_mask, tcol].to_numpy(float), "yhat": yhat}))
            if not rows_out:
                continue
            p = pd.concat(rows_out, ignore_index=True)
            err = p.y - p.yhat
            mae, rmse = float(err.abs().mean()), float(np.sqrt((err ** 2).mean()))
            out[(season, stat)] = dict(mae=round(mae, 4), rmse=round(rmse, 4), n=int(len(p)))
        print(f"  [{label}] MAE stats, {season}: fitted")
    return out


def score_td_stats(df, label):
    X_all, _ = build_design(df)
    X_all = X_all.astype(float)
    in_tr = df.season.isin(TRAIN_SEASONS).to_numpy()
    g = df.groupby("player_id", sort=False)
    for stat, col, _ in TD_STATS:
        df[col + "_hit"] = (df[col] >= 1).astype(float)
        df[col + "_career_rate"] = g[col + "_hit"].transform(lambda s: s.shift(1).expanding().mean())

    out = {}
    for season in VALIDATION_SEASONS:
        weeks = sorted(df.loc[df.season == season, "week"].unique())
        for stat, col, screen in TD_STATS:
            hit = col + "_hit"
            rows_out = []
            for i in range(1, len(weeks)):
                prior = (df.season.eq(season) & df.week.isin(weeks[:i])).to_numpy()
                tr = (in_tr | prior) & df[screen].to_numpy() & df[col].notna().to_numpy()
                te = (df.season.eq(season) & df.week.eq(weeks[i])).to_numpy() & df[screen].to_numpy() & df[col].notna().to_numpy()
                if te.sum() == 0 or tr.sum() < 200:
                    continue
                m = lgb.LGBMClassifier(**TD_PARAMS).fit(X_all[tr], df.loc[tr, hit])
                league = float(df.loc[tr, hit].mean())
                rows_out.append(pd.DataFrame({
                    "y": df.loc[te, hit].to_numpy(float),
                    "p_model": m.predict_proba(X_all[te])[:, 1],
                    "p_league": league,
                }))
            if not rows_out:
                continue
            r = pd.concat(rows_out, ignore_index=True)
            p_model = r.p_model.clip(0, 1)
            p_league = r.p_league.clip(0, 1)
            out[(season, stat)] = dict(
                n=int(len(r)), actual_rate=round(float(r.y.mean()), 4),
                brier=round(float(brier_score_loss(r.y, p_model)), 5),
                league_brier=round(float(brier_score_loss(r.y, p_league)), 5),
                auc=round(float(roc_auc_score(r.y, p_model)), 4) if p_model.nunique() > 1 else None,
            )
        print(f"  [{label}] TD stats, {season}: fitted")
    return out


def main():
    pg = load_pg()
    tr = add_derived(load_training_rows())

    print("=== fitting shrinkage k from TRAIN seasons only (2019-2023) ===")
    k_table = fit_shrinkage_k(pg)
    for (side, pos), v in sorted(k_table.items()):
        print(f"  {side:<7}{pos:<5} k={v['k']:>10.2f}  sigma0^2={v['sigma0_sq']:>9.4f}  "
              f"sigma_between^2={v['sigma_between_sq']:>10.6f}  cells={v['n_cells']:>5}  "
              f"league_avg={v['league_avg']:>7.3f}")

    feature_sets = make_feature_sets(tr, pg, k_table)

    results = {"k_table": {f"{s}_{p}": v for (s, p), v in k_table.items()}, "arms": {}}
    for label, df in feature_sets.items():
        print(f"\n=== scoring {label} ===")
        mae = score_mae_stats(df, label)
        td = score_td_stats(df, label)
        results["arms"][label] = {
            "mae": {f"{s}_{st}": v for (s, st), v in mae.items()},
            "td": {f"{s}_{st}": v for (s, st), v in td.items()},
        }

    OUT.write_text(json.dumps(results, indent=2))

    # ---- summary table vs no-defense -------------------------------------
    base_mae = results["arms"]["no_defense"]["mae"]
    base_td = results["arms"]["no_defense"]["td"]
    print(f"\n{'arm':<16}{'season':<8}{'stat':<17}{'metric':<8}{'arm_val':>10}{'nodef_val':>11}{'lift%':>8}{'bar':>6}{'clear':>7}")
    for label in ("arm_a_shrinkage", "arm_b_epa", "arm_c_both"):
        arm_mae = results["arms"][label]["mae"]
        arm_td = results["arms"][label]["td"]
        for key, v in sorted(arm_mae.items()):
            season, stat = key.split("_", 1)
            b = base_mae[key]["mae"]
            lift = (b - v["mae"]) / b * 100
            print(f"{label:<16}{season:<8}{stat:<17}{'MAE':<8}{v['mae']:>10.4f}{b:>11.4f}{lift:>+8.2f}{'3%':>6}{'YES' if lift>=3 else 'no':>7}")
        for key, v in sorted(arm_td.items()):
            season, stat = key.split("_", 1)
            b = base_td[key]["brier"]
            lift = (b - v["brier"]) / b * 100
            print(f"{label:<16}{season:<8}{stat:<17}{'Brier':<8}{v['brier']:>10.5f}{b:>11.5f}{lift:>+8.2f}{'1%':>6}{'YES' if lift>=1 else 'no':>7}")

    print(f"\nwritten -> {OUT}")


if __name__ == "__main__":
    main()
