#!/usr/bin/env python
"""2026 holdout scorer -- decision #42 (amends decision #33's protocol).

Scores BOTH engines per stat on 2026 weeks given by --weeks:
  old engine : nflprops.project.project_week (engine/v1.json, decision #33)
  new model  : scripts/vol_predict.phase3_vol (Phase 3; decisions #31/#38/#40)
for passing_yards, rushing_yards, receiving_yards, receptions. Completions
(old engine vs baseline) and anytime_td (nflprops.tdmodel, Brier) keep their
existing rules. Rules are in prediction-engine/holdout-protocol.md section 8.

The code runs on any season so it can be smoke-tested on a validation year.
A 2026 run needs --confirm-holdout. Makes zero Odds API calls. Writes only the
JSON named by --out.
"""
import argparse
import json
import pathlib
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "td"))

from nflprops import project as P  # noqa: E402
from nflprops import tdmodel as T  # noqa: E402
from nflprops.db import connect  # noqa: E402
from rung2_ridge import load, add_derived  # noqa: E402
from vol_predict import phase3_vol  # noqa: E402
from predict import train_through  # noqa: E402

# stat -> (y column in training_rows, screen column, bar %, baseline rule, has new model)
STATS = {
    "passing_yards":   ("y_passing_yards",   "qual_attempts", 3.0, "better", True),
    "rushing_yards":   ("y_rushing_yards",   "qual_carries",  1.0, "last4",  True),
    "receiving_yards": ("y_receiving_yards", "qual_targets",  2.0, "last4",  True),
    "receptions":      ("y_receptions",      "qual_targets",  3.0, "better", True),
    "completions":     ("y_completions",     "qual_attempts", 3.0, "better", False),
}
BRIER_BAR = 1.0
SHUFFLED_TOP1_FLOOR = 0.2171


def mae(a, b):
    return float(np.mean(np.abs(a - b)))


def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def build_truth(season, weeks):
    tr = add_derived(load())
    with connect() as con:
        comp = pd.read_sql_query(
            "SELECT player_id, game_id, completions AS y_completions FROM player_games "
            "WHERE season=? AND season_type='REG'", con, params=[season])
    tr = tr.merge(comp, on=["player_id", "game_id"], how="left")
    tr = tr.sort_values(["player_id", "season", "week"]).reset_index(drop=True)
    g = tr.groupby("player_id", sort=False)
    for stat, (ycol, *_rest) in STATS.items():
        tr[f"{stat}_career"] = g[ycol].transform(lambda s: s.shift(1).expanding().mean())
        tr[f"{stat}_last4"] = g[ycol].transform(lambda s: s.shift(1).rolling(4, min_periods=1).mean())
    return tr[(tr.season == season) & (tr.week.isin(weeks)) & (tr.season_type == "REG")].copy()


def run_stats(season, weeks, truth):
    old_frames, new_frames = [], []
    for w in weeks:
        old = P.project_week(season, w)
        old = old[old.stat.isin(STATS)][["player_id", "game_id", "stat", "proj"]]
        old["week"] = w
        old_frames.append(old)
        p3 = phase3_vol(season, w)
        p3["week"] = w
        new_frames.append(p3)
    old = pd.concat(old_frames, ignore_index=True)
    p3 = pd.concat(new_frames, ignore_index=True)

    out = {}
    for stat, (ycol, screen, bar, brule, has_new) in STATS.items():
        t = truth[truth[screen] & truth[ycol].notna()]
        o = old[old.stat == stat][["player_id", "game_id", "proj"]].rename(columns={"proj": "yhat_old"})
        d = t.merge(o, on=["player_id", "game_id"], how="inner")
        d = d.dropna(subset=[f"{stat}_career", f"{stat}_last4"])
        res = {"bar_pct": bar, "n_old_rows": int(len(d))}
        y = d[ycol].to_numpy(float)
        res["zero_baseline_mae"] = mae(y, 0.0)
        res["old_mae"], res["old_rmse"] = mae(y, d.yhat_old), rmse(y, d.yhat_old)
        bl = {k: (mae(y, d[f"{stat}_{k}"]), rmse(y, d[f"{stat}_{k}"])) for k in ("career", "last4")}
        pick = "last4" if brule == "last4" else min(bl, key=lambda k: bl[k][0])
        res.update(baseline=pick, baseline_mae=bl[pick][0], baseline_rmse=bl[pick][1],
                   baseline_career_mae=bl["career"][0], baseline_last4_mae=bl["last4"][0])
        res["old_lift_pct"] = (res["baseline_mae"] - res["old_mae"]) / res["baseline_mae"] * 100
        res["old_pass"] = bool(res["old_lift_pct"] >= bar)
        if has_new:
            col = f"p3_{stat}"
            pp = p3[["player_id", "week", col]].rename(columns={col: "yhat_new"})
            h = d.merge(pp, on=["player_id", "week"], how="inner").dropna(subset=["yhat_new"])
            yh = h[ycol].to_numpy(float)
            res["n_h2h"] = int(len(h))
            res["h2h_old_mae"], res["h2h_old_rmse"] = mae(yh, h.yhat_old), rmse(yh, h.yhat_old)
            res["h2h_new_mae"], res["h2h_new_rmse"] = mae(yh, h.yhat_new), rmse(yh, h.yhat_new)
            res["new_vs_old_lift_pct"] = (res["h2h_old_mae"] - res["h2h_new_mae"]) / res["h2h_old_mae"] * 100
            res["verdict"] = "NEW REPLACES OLD" if res["new_vs_old_lift_pct"] >= bar else "OLD STAYS"
        out[stat] = res
    return out


def run_td(season, weeks):
    frames = []
    for w in weeks:
        model, n_train = train_through(season, w)
        f = T.features(season, w)
        lab = T.labels(season, w)
        f = f.merge(lab, on="player_id", how="inner")
        f["p"] = model.predict_proba(T.design_matrix(f))[:, 1]
        import pandas as _pd
        prior = _pd.read_parquet(ROOT / "data" / "td_train.parquet")
        prior = prior[(prior.season < season) | ((prior.season == season) & (prior.week < w))]
        f["base"] = float(prior["td"].mean())
        f["week"] = w
        frames.append(f)
    f = pd.concat(frames, ignore_index=True)
    td = f["td"].to_numpy(float)
    b_m, b_c = brier_score_loss(td, f.p), brier_score_loss(td, f.base)
    top = f.loc[f.groupby("game_id")["p"].idxmax()]
    return {"n": int(len(f)), "brier_model": b_m, "brier_base_rate": b_c,
            "brier_improvement_pct": (b_c - b_m) / b_c * 100,
            "bar_pct": BRIER_BAR, "pass": bool((b_c - b_m) / b_c * 100 >= BRIER_BAR),
            "auc": float(roc_auc_score(td, f.p)), "top1": float(top["td"].mean()),
            "top1_games": int(len(top)), "top1_shuffled_floor": SHUFFLED_TOP1_FLOOR,
            "base_rate_used_mean": float(f.base.mean())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, required=True)
    ap.add_argument("--weeks", required=True, help="e.g. 1-4")
    ap.add_argument("--out", required=True)
    ap.add_argument("--confirm-holdout", action="store_true")
    a = ap.parse_args()
    if a.season == 2026 and not a.confirm_holdout:
        sys.exit("2026 is the sealed holdout; pass --confirm-holdout (decision #42 run only)")
    lo, hi = (int(x) for x in a.weeks.split("-"))
    weeks = list(range(lo, hi + 1))
    truth = build_truth(a.season, weeks)
    res = {"season": a.season, "weeks": weeks, "stats": run_stats(a.season, weeks, truth),
           "anytime_td": run_td(a.season, weeks)}
    pathlib.Path(a.out).write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
