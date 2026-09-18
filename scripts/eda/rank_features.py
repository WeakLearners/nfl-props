#!/usr/bin/env python
"""Phase 2+3: rank candidate features per position/target by Pearson r and
mutual information, then report a redundancy-pruned shortlist.

Only lagged/pre-kickoff columns are used as features -- this file does not
touch the raw same-game box-score columns (attempts, targets, passing_yards,
...) except as the outcome being predicted.
"""
import pathlib
import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_regression

DATA = pathlib.Path(__file__).resolve().parent.parent.parent / "data" / "eda_features.parquet"

FEATURES = [
    "h_attempts", "h_carries", "h_targets", "n_games_hist",
    "h_ypa", "h_ypc", "h_ypt", "h_cmp_rate", "h_catch_rate",
    "h_target_share", "h_air_yards_share", "h_wopr", "h_racr", "h_adot", "h_yac_per_rec",
    "h_passing_epa_pa", "h_rushing_epa_pc", "h_receiving_epa_pt", "h_cpoe",
    "h_snap_pct", "h_snap_trend",
    "h_rz_carries_pg", "h_rz10_carries_pg", "h_rz_targets_pg", "h_rz10_targets_pg",
    "team_targets_base", "team_attempts_base", "team_carries_base",
    "spread_line", "total_line", "implied_team_total", "div_game",
    "temp", "wind", "rest_days", "is_home",
    "own_questionable", "own_status_missing", "teammates_out_same_pos", "team_out_total",
    "dvp_passing_yards", "dvp_attempts", "dvp_completions",
    "dvp_rushing_yards", "dvp_carries",
    "dvp_receiving_yards", "dvp_receptions", "dvp_targets",
]

TARGETS = {
    "QB": ["passing_yards", "completions", "rushing_yards"],
    "RB": ["rushing_yards", "receptions", "receiving_yards"],
    "WR": ["receptions", "receiving_yards"],
    "TE": ["receptions", "receiving_yards"],
}


def rank_one(df, pos, target):
    sub = df[df.position == pos].copy()
    cols = [c for c in FEATURES if c in sub.columns]
    sub = sub.dropna(subset=[target])
    rows = []
    X_valid_cols = []
    for c in cols:
        s = sub[[c, target]].dropna()
        n = len(s)
        if n < 30 or s[c].std() == 0:
            continue
        r = s[c].corr(s[target])
        rows.append({"feature": c, "n": n, "pearson_r": r})
        X_valid_cols.append(c)
    rank_df = pd.DataFrame(rows)
    if rank_df.empty:
        return rank_df

    # mutual information needs a shared, complete-case matrix
    mi_sub = sub[X_valid_cols + [target]].dropna()
    if len(mi_sub) >= 50:
        mi = mutual_info_regression(mi_sub[X_valid_cols].values, mi_sub[target].values,
                                     random_state=0, n_neighbors=5)
        mi_map = dict(zip(X_valid_cols, mi))
        rank_df["mi_n"] = len(mi_sub)
    else:
        mi_map = {}
        rank_df["mi_n"] = len(mi_sub)
    rank_df["mutual_info"] = rank_df.feature.map(mi_map)
    rank_df["abs_r"] = rank_df.pearson_r.abs()
    rank_df = rank_df.sort_values("abs_r", ascending=False)
    return rank_df


def main():
    df = pd.read_parquet(DATA)
    for pos, targets in TARGETS.items():
        for target in targets:
            print(f"\n{'='*70}\n{pos} -> {target}\n{'='*70}")
            r = rank_one(df, pos, target)
            if r.empty:
                print("  no usable features (insufficient n)")
                continue
            r = r[["feature", "n", "pearson_r", "mutual_info", "mi_n"]]
            with pd.option_context("display.max_rows", None, "display.width", 140):
                print(r.to_string(index=False, float_format=lambda x: f"{x:8.4f}"))

            # rank-disagreement flag: normalize both to 0-1 rank and diff
            rr = r.dropna(subset=["mutual_info"]).copy()
            if len(rr) > 3:
                rr["r_rank"] = rr.abs_r.rank(ascending=False) if "abs_r" in rr else rr.pearson_r.abs().rank(ascending=False)
                rr["mi_rank"] = rr.mutual_info.rank(ascending=False)
                rr["rank_gap"] = (rr.r_rank - rr.mi_rank).abs()
                flagged = rr.sort_values("rank_gap", ascending=False).head(3)
                print("\n  biggest Pearson-vs-MI rank disagreements:")
                for _, row in flagged.iterrows():
                    print(f"    {row.feature:<24} r_rank={row.r_rank:.0f}  mi_rank={row.mi_rank:.0f}"
                          f"  (r={row.pearson_r:.3f}, mi={row.mutual_info:.4f})")


if __name__ == "__main__":
    main()
