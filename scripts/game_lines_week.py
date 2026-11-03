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
from nflprops.config import DATA_DIR, WINDOW_DISPLAY, WINDOW_RECORD, VALUE_MARGIN

OUT = DATA_DIR / "gamelines"
FROZEN = OUT / "frozen" / G.MODEL_VERSION
SD_FROM_SEASON = 2023   # out-of-fold residuals start here (2019-2022 only feed the fits)


def _rms(a):
    return float(np.sqrt(np.mean(np.square(a))))


def _sd_before(oof, t):
    d = oof[oof["played"] & (oof["t"] < t) & ~((oof["season"] == SD_FROM_SEASON) & (oof["week"] < 3))]
    return _rms(d["margin"] - d["pred_margin"]), _rms(d["total"] - d["pred_total"])


QUANT = ("home_score", "away_score", "margin", "total")


def _resid(d):
    return {"home_score": d["home_score"] - d["mu_home"], "away_score": d["away_score"] - d["mu_away"],
            "margin": d["margin"] - d["pred_margin"], "total": d["total"] - d["pred_total"]}


def _windows_before(oof, t, target=WINDOW_RECORD):
    """Window half-width per quantity: mu +/- scale*sd, scale refit from earlier out-of-fold misses only.

    sd is the RMS of those misses. The scale is the target (WINDOW_RECORD or WINDOW_DISPLAY) percentile of |miss|/sd, so the
    window held that share of earlier results by construction. A window only narrows if the misses shrink.
    """
    d = oof[oof["played"] & (oof["t"] < t) & ~((oof["season"] == SD_FROM_SEASON) & (oof["week"] < 3))]
    out = {}
    for k, r in _resid(d).items():
        sd = _rms(r)
        scale = float(np.quantile(np.abs(r) / sd, target))
        out[k] = {"sd": round(sd, 3), "scale": round(scale, 3), "half": round(scale * sd, 3)}
    return out


def window_record(df, half):
    """Model-only record: share of results inside the model's own window, and mean window width.

    half: dict quantity -> half-width Series aligned with df (or a number). No betting line is used.
    """
    d = df[df["played"]]
    out = {"n": int(len(d))}
    for k, r in _resid(d).items():
        h = half[k].loc[d.index] if hasattr(half[k], "loc") else half[k]
        out[k] = {"coverage": float((r.abs() <= h).mean()), "width": float((2 * h).mean())} if len(d) else None
    return out


def _rows(df, sd_m, sd_t, train_through, reconstructed, windows):
    out = []
    for r in df.itertuples():
        out.append({
            "game_id": r.game_id, "away": r.away_team, "home": r.home_team,
            "gameday": r.gameday, "gametime": r.gametime,
            "mu_home": round(r.mu_home, 2), "mu_away": round(r.mu_away, 2),
            "pred_margin": round(r.pred_margin, 2), "pred_total": round(r.pred_total, 2),
            "p_home": round(float(G.win_prob(r.pred_margin, sd_m)), 4),
            "wx_used": {"source": r.w_src, "forecast_pulled_at": r.w_pulled,
                        "heavy_rain_flag": int(r.w_heavy)},
            "home_qb": r.home_qb_name, "away_qb": r.away_qb_name,
            "home_qb_out": int(r.home_qb_out), "away_qb_out": int(r.away_qb_out),
            "spread_at_freeze": None if pd.isna(r.spread_line) else float(r.spread_line),
            "total_at_freeze": None if pd.isna(r.total_line) else float(r.total_line),
        })
    return {"windows": windows[0], "windows50": windows[1], "target_record": WINDOW_RECORD,
            "target_display": WINDOW_DISPLAY, "sd_margin": round(sd_m, 3), "sd_total": round(sd_t, 3), "train_through": train_through,
            "params": G.PARAMS, "reconstructed": reconstructed,
            "frozen_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "games": out}


def load_frozen(season):
    rows = []
    for f in sorted(FROZEN.glob(f"{season}_w*.json")):
        d = json.loads(f.read_text())
        wk = int(f.stem.split("_w")[1])
        for r in d["games"]:
            rows.append({**r, "season": season, "week": wk, "sd_margin": d["sd_margin"],
                         **{"h_" + k: v["half"] for k, v in d["windows"].items()},
                         **{"g_" + k: v["half"] for k, v in d["windows50"].items()},
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
    pj = FROZEN / "params.json"
    mine = {"version": G.MODEL_VERSION, "params": json.loads(json.dumps(G.PARAMS))}
    if pj.exists():
        if json.loads(pj.read_text())["params"] != mine["params"]:
            sys.exit(f"PARAMS changed but MODEL_VERSION is still {G.MODEL_VERSION}. Bump MODEL_VERSION: "
                     "frozen predictions are never rewritten.")
    else:
        pj.write_text(json.dumps(mine, indent=1))

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
        win = (_windows_before(oof, tgt["t"].iloc[0]), _windows_before(oof, tgt["t"].iloc[0], WINDOW_DISPLAY))
        path.write_text(json.dumps(_rows(pred, sd_m, sd_t, f"{tr['season'].iloc[-1]} wk {tr['week'].iloc[-1]}",
                                         recon, win), indent=1))
        made.append(w)
    print("froze weeks:", made or "none (all existed)")

    # --- scoring: frozen (weekly adjust) vs one fit on seasons before the test season
    fz = load_frozen(season)
    base_tr = g[(g["season"] < season) & g["played"]]
    base = G.fit_predict(base_tr, cur[cur["week"] <= target])
    sd_m0, sd_t0 = _sd_before(oof, (season - 2000) * 22 + 1)
    base["p_home"] = G.win_prob(base["pred_margin"], sd_m0)
    win0 = _windows_before(oof, (season - 2000) * 22 + 1)
    win50 = _windows_before(oof, (season - 2000) * 22 + 1, WINDOW_DISPLAY)
    for k in QUANT:
        base["h_" + k] = win0[k]["half"]
        base["g_" + k] = win50[k]["half"]
    adj = cur.merge(fz[["game_id", "pred_margin", "pred_total", "p_home", "mu_home", "mu_away"] + ["h_" + k for k in QUANT] + ["g_" + k for k in QUANT]], on="game_id")
    result = {"season": season, "target_week": target, "params": G.PARAMS,
              "sd_margin_no_adjust": sd_m0, "target_record": WINDOW_RECORD, "target_display": WINDOW_DISPLAY, "value_margin": VALUE_MARGIN,
              "windows_no_adjust": win0, "windows50_no_adjust": win50, "weeks": [], "cumulative": {}, "coefs": base.attrs["coefs"]}
    for label, df in (("adjust", adj), ("no_adjust", base[base["week"] <= target])):
        df = df.copy()
        df["week_i"] = df["week"]
        half = {k: df["h_" + k] for k in QUANT}
        for w in sorted(df[df["played"]]["week"].unique()):
            result.setdefault("window_weeks", []).append(
                {"model": label, "week": int(w), **window_record(df[df["week"] == w], half)})
        result.setdefault("window_cumulative", {})[label] = window_record(df, half)
        half50 = {k: df["g_" + k] for k in QUANT}
        for w in sorted(df[df["played"]]["week"].unique()):
            rec = window_record(df[df["week"] == w], half50)
            ww = [x for x in result["window_weeks"] if x["model"] == label and x["week"] == int(w)][0]
            ww["w50"] = {k: rec[k] for k in QUANT}
        result.setdefault("window50_cumulative", {})[label] = window_record(df, half50)
        tw = {int(w): G.tallies(df[df["week"] == w], VALUE_MARGIN) for w in sorted(df[df["played"]]["week"].unique())}
        result.setdefault("tallies", {})[label] = {"weeks": tw, "season": G.tallies(df, VALUE_MARGIN)}
        result.setdefault("calibration", {})[label] = G.calibration(df)
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
