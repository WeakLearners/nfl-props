#!/usr/bin/env python
"""EXPERIMENT (offline, proposal only -- no live rank.py change). Follow-up to
decision #41: the missing out-of-fold check for the changed_team penalty.

decision #41 found the changed_team penalty (currently 0.06 in nflprops.rank)
looks ~3.5x underweighted against 2025 outcomes (fitted 0.204, n=134). This
asks the question that actually matters for the report: does swapping in
0.204 change which legs get PICKED, and does the resulting top-N selection's
out-of-fold hit rate improve?

METHOD. Same 777 2025 legs (reports/validate_reliability.csv), same 5-fold
GroupKFold by player_key as scripts/penalty_refit_2025.py. Score every leg
under two schemes -- CURRENT (changed_team=0.06) and PROPOSED
(changed_team=0.204) -- with every other penalty held at its live value
(Questionable 0.08, Doubtful/Out 0.60, dnp>25% 0.04, abs_spread>9 0.02,
thin-history 0.10; blend weights W_HIST/W_PRICE unchanged). For each fold's
held-out legs, group by (week, event_id) -- the game -- drop to one leg per
player (the higher-scored stat), and take the top min(6, group size) by
score, matching nflprops.rank.best_legs()'s n=6 default. Hit rate = mean(won)
over the picks selected that way, pooled across all 5 folds.

CAVEAT, stated plainly. GroupKFold splits by PLAYER, not by game, so a test
fold rarely holds every player who was priced in a given game -- the
selection here picks the best-N among whichever of that game's players
happen to be in the held-out fold, not the full slate FanDuel actually
offered. This understates a real report's competition for the top-6 slots.
Treat this as a lower bound on how much the selection would actually
change, not the exact number a live swap would produce.

CI: player-block bootstrap (2000 resamples, resampling player_key groups
with replacement, same block structure the CV respects) on the hit-rate
DIFFERENCE (proposed - current), so a picked leg's outcome is never treated
as independent from another leg on the same player.

Read-only. Writes only data/penalty_refit_topN_2025.csv. No API calls, no
2026 outcomes.
"""
import pathlib
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from nflprops.rank import W_HIST, W_PRICE  # noqa: E402
from penalty_refit_2025 import load_legs, attach_risk_features  # noqa: E402

N_PICKS = 6
N_FOLDS = 5
N_BOOT = 2000
RNG = np.random.default_rng(17)

FIXED_PENALTIES = dict(questionable=0.08, doubtful_or_out=0.60, dnp_gt_25pct=0.04,
                       abs_spread_gt_9=0.02, thin_history_lt6=0.10)


def score(df, changed_team_weight):
    blend = np.where(df["hist"].notna(), W_PRICE * df.fair + W_HIST * df["hist"], df.fair)
    penalty = (
        (df.report_status == "Questionable").astype(float) * FIXED_PENALTIES["questionable"]
        + df.report_status.isin(["Doubtful", "Out"]).astype(float) * FIXED_PENALTIES["doubtful_or_out"]
        + (df.dnp.fillna(0) > 0.25).astype(float) * FIXED_PENALTIES["dnp_gt_25pct"]
        + (df.abs_spread.fillna(0) > 9).astype(float) * FIXED_PENALTIES["abs_spread_gt_9"]
        + (df.n_games.fillna(0) < 6).astype(float) * FIXED_PENALTIES["thin_history_lt6"]
        + df.changed_team.fillna(False).astype(float) * changed_team_weight
    )
    return blend - penalty


def pick_top_n(df):
    """One leg per player (higher-scored stat), then top N_PICKS per game."""
    best_per_player = df.sort_values("score", ascending=False).drop_duplicates("player_key")
    picks = []
    for _, g in best_per_player.groupby(["week", "event_id"]):
        picks.append(g.sort_values("score", ascending=False).head(N_PICKS))
    if not picks:
        return best_per_player.iloc[0:0]
    return pd.concat(picks, ignore_index=True)


def main():
    df = load_legs()
    df = attach_risk_features(df)
    print(f"{len(df)} legs, 2025 weeks {sorted(df.week.unique())}\n")

    groups = df.player_key.astype("category").cat.codes
    gkf = GroupKFold(n_splits=N_FOLDS)

    fold_picks = {"current": [], "proposed": []}
    for fold_i, (_, test_idx) in enumerate(gkf.split(df, groups=groups)):
        te = df.iloc[test_idx].copy()
        for name, w in [("current", 0.06), ("proposed", 0.204)]:
            te_s = te.copy()
            te_s["score"] = score(te_s, w)
            picks = pick_top_n(te_s)
            picks["fold"] = fold_i
            fold_picks[name].append(picks)

    all_picks = {name: pd.concat(v, ignore_index=True) for name, v in fold_picks.items()}

    rows = []
    for name, picks in all_picks.items():
        n = len(picks)
        hit = picks.won.mean()
        rows.append(dict(scheme=name, changed_team_weight=(0.06 if name == "current" else 0.204),
                         n_picks=n, hit_rate=round(float(hit), 4)))
        print(f"{name:<10} changed_team={0.06 if name=='current' else 0.204:.3f}  "
              f"n_picks={n}  hit_rate={hit*100:.1f}%")

    # Player-block bootstrap on the DIFFERENCE. Resample player_keys (with
    # replacement) from the set that appears in either scheme's picks, then
    # recompute each scheme's hit rate over the resampled players' picks.
    cur, prop = all_picks["current"], all_picks["proposed"]
    all_players = np.array(sorted(set(cur.player_key) | set(prop.player_key)))
    diffs = []
    for _ in range(N_BOOT):
        samp = RNG.choice(all_players, size=len(all_players), replace=True)
        # weight each player by how many times it was drawn
        w = pd.Series(samp).value_counts()
        c = cur[cur.player_key.isin(w.index)].assign(w=lambda d: d.player_key.map(w))
        p = prop[prop.player_key.isin(w.index)].assign(w=lambda d: d.player_key.map(w))
        if c.empty or p.empty:
            continue
        c_rate = np.average(c.won, weights=c.w)
        p_rate = np.average(p.won, weights=p.w)
        diffs.append(p_rate - c_rate)
    diffs = np.array(diffs)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    mean_diff = diffs.mean()

    print(f"\nDifference (proposed - current) in top-{N_PICKS} pick hit rate:")
    print(f"  mean {mean_diff*100:+.1f}pp   95% bootstrap CI [{lo*100:+.1f}pp, {hi*100:+.1f}pp]   "
          f"n_boot={len(diffs)}")

    res = pd.DataFrame(rows)
    res.to_csv(ROOT / "data" / "penalty_refit_topN_2025.csv", index=False)
    print(f"\nwritten -> data/penalty_refit_topN_2025.csv")


if __name__ == "__main__":
    main()
