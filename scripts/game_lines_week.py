#!/usr/bin/env python3
"""Weekly game-lines run: freeze this week's predictions, then score the record.

Usage: game_lines_week.py [season] [week]   (default: 2026, first week with no result)

Walk-forward, as Sean asked on 2026-11-02. The first fit uses seasons before the
test season only. For each week W the model is refit on every game played before
W, predicts W, and the predictions are written to data/gamelines/frozen/ ONCE.
An existing frozen file is never overwritten, so a later refit cannot rewrite
what was predicted. The "no adjust" comparison is one fit on seasons before the
test season, used for every week.

Frozen files for weeks that were already played when this feature was built are
marked reconstructed=true: they come from the same rule, built after the fact.
"""
import json
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd

from nflprops import gamelines as G
from nflprops.config import DATA_DIR

OUT = DATA_DIR / "gamelines"
FROZEN = OUT / "frozen"
SD_FROM_SEASON = 2023   # out-of-fold residuals start here (2019-2022 only feed the fits)


def _rms(a):
    return float(np.sqrt(np.mean(np.square(a))))


def _sd_before(oof, t):
    d = oof[oof["played"] & (oof["t"] < t) & ~((oof["season"] == SD_FROM_SEASON) & (oof["week"] < 3))]
    return _rms(d["margin"] - d["pred_margin"]), _rms(d["total"] - d["pred_total"])


def _rows(df, sd_m, sd_t, train_through, reconstructed):
    out = []
    for r in df.itertuples():
        out.append({
            "game_id": r.game_id, "away": r.away_team, "home": r.home_team,
            "gameday": r.gameday, "gametime": r.gametime,
            "mu_home": round(r.mu_home, 2), "mu_away": round(r.mu_away, 2),
            "pred_margin": round(r.pred_margin, 2), "pred_total": round(r.pred_total, 2),
            "p_home": round(float(G.win_prob(r.pred_margin, sd_m)), 4),
            "home_qb": r.home_qb_name, "away_qb": r.away_qb_name,
            "home_qb_out": int(r.home_qb_out), "away_qb_out": int(r.away_qb_out),
            "spread_at_freeze": None if pd.isna(r.spread_line) else float(r.spread_line),
            "total_at_freeze": None if pd.isna(r.total_line) else float(r.total_line),
        })
    return {"sd_margin": round(sd_m, 3), "sd_total": round(sd_t, 3), "train_through": train_through,
            "params": G.PARAMS, "reconstructed": reconstructed,
            "frozen_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "games": out}


def load_frozen(season):
    rows = []
    for f in sorted(FROZEN.glob(f"{season}_w*.json")):
        d = json.loads(f.read_text())
        wk = int(f.stem.split("_w")[1])
        for r in d["games"]:
            rows.append({**r, "season": season, "week": wk, "sd_margin": d["sd_margin"],
                         "reconstructed": d["reconstructed"]})
    return pd.DataFrame(rows)


def main():
    season = int(sys.argv[1]) if len(sys.argv) > 1 else 2026
    g = G.load_games()
    cur = g[g["season"] == season]
    if len(sys.argv) > 2:
        target = int(sys.argv[2])
    else:
        by_week = cur.groupby("week")["played"].sum()
        target = int(by_week[by_week == 0].index.min())
    FROZEN.mkdir(parents=True, exist_ok=True)

    # out-of-fold fits for the residual SD: all earlier seasons, then the test season up to target
    oof = G.walk_forward(g[g["season"] <= season], range(SD_FROM_SEASON, season + 1),
                         weeks=None)
    oof = oof[(oof["season"] < season) | (oof["week"] < target)]

    made = []
    for w in range(1, target + 1):
        path = FROZEN / f"{season}_w{w}.json"
        if path.exists():
            continue                                   # frozen: never rewritten
        tgt = cur[cur["week"] == w]
        if tgt.empty:
            continue
        tr = g[(g["t"] < tgt["t"].iloc[0]) & g["played"]]
        pred = G.fit_predict(tr, tgt)
        sd_m, sd_t = _sd_before(oof, tgt["t"].iloc[0])
        recon = bool(tgt["played"].any())
        path.write_text(json.dumps(_rows(pred, sd_m, sd_t, f"{tr['season'].iloc[-1]} wk {tr['week'].iloc[-1]}",
                                         recon), indent=1))
        made.append(w)
    print("froze weeks:", made or "none (all existed)")

    # --- scoring: frozen (weekly adjust) vs one fit on seasons before the test season
    fz = load_frozen(season)
    base_tr = g[(g["season"] < season) & g["played"]]
    base = G.fit_predict(base_tr, cur[cur["week"] <= target])
    sd_m0, sd_t0 = _sd_before(oof, (season - 2000) * 22 + 1)
    base["p_home"] = G.win_prob(base["pred_margin"], sd_m0)
    adj = cur.merge(fz[["game_id", "pred_margin", "pred_total", "p_home", "mu_home", "mu_away"]], on="game_id")
    result = {"season": season, "target_week": target, "params": G.PARAMS,
              "sd_margin_no_adjust": sd_m0, "weeks": [], "cumulative": {}, "coefs": base.attrs["coefs"]}
    for label, df in (("adjust", adj), ("no_adjust", base[base["week"] <= target])):
        df = df.copy()
        df["week_i"] = df["week"]
        for w in sorted(df[df["played"]]["week"].unique()):
            m = G.metrics(df[df["week"] == w])
            result["weeks"].append({"model": label, "week": int(w), **m})
        upto = {}
        for w in sorted(df[df["played"]]["week"].unique()):
            upto[int(w)] = G.metrics(df[df["week"] <= w])
        result.setdefault("cumulative_by_week", {})[label] = upto
        result["cumulative"][label] = G.metrics(df)
    # the market on its own, for the same played games
    pl = cur[cur["played"] & (cur["week"] < target)]
    result["market_note"] = "line columns inside each metrics block are the closing line on the same rows"
    (OUT / "backtest_2026.json").write_text(json.dumps(result, indent=1, default=float))
    print(json.dumps(result["cumulative"], indent=1, default=float))


if __name__ == "__main__":
    main()
