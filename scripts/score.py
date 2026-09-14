#!/usr/bin/env python
"""Clean scorer. Deliberately does NOT reuse backtest.report(), which produced
figures that could not be reproduced from its own returned frames."""
import sys, pathlib, io, contextlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import pandas as pd
from nflprops import project as P
from nflprops.db import connect
from nflprops.project import SPECS


def _actuals(s, w):
    with connect() as con:
        a = pd.read_sql_query("SELECT * FROM player_games WHERE season=? AND week=? AND season_type='REG'",
                              con, params=[s, w])
    return pd.concat([a[["player_id", st]].rename(columns={st: "actual"}).assign(stat=st) for st in SPECS],
                     ignore_index=True)


def _baseline(s, w):
    with connect() as con:
        h = pd.read_sql_query("SELECT * FROM player_games WHERE season_type='REG' AND (season<? OR (season=? AND week<?))",
                              con, params=[s, s, w])
    out = []
    for st in SPECS:
        g = h.groupby("player_id")[st].agg(base="mean", n_prior="count").reset_index()
        out.append(g.assign(stat=st))
    return pd.concat(out, ignore_index=True)


def build(weeks):
    fr = []
    for s, w in weeks:
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                pr = P.project_week(s, w)
        except Exception as e:
            print(f"skip {s} W{w}: {e}", file=sys.stderr); continue
        fr.append(pr.merge(_actuals(s, w), on=["player_id", "stat"])
                    .merge(_baseline(s, w), on=["player_id", "stat"], how="left")
                    .assign(season=s, week=w))
    return pd.concat(fr, ignore_index=True)


# Roughly the lowest number FanDuel will hang each prop at. Scoring below this
# measures the model on markets that do not exist -- which is how receiving
# yards first looked 12% worse than a career average when, inside the real
# market, it is better.
BETTABLE = {"passing_yards": 150.0, "completions": 12.0, "rushing_yards": 25.0,
            "receiving_yards": 25.0, "receptions": 2.0}


def table(df, label, bettable=False):
    df = df[df.base.notna() & (df.n_prior >= 5)]
    if bettable:
        df = df[df.apply(lambda r: r.base >= BETTABLE[r.stat], axis=1)]
    print(f"\n### {label}   n={len(df):,}")
    print(f"{'stat':<17}{'n':>7}{'model MAE':>11}{'naive MAE':>11}{'improve':>9}")
    print("-" * 55)
    for st, g in df.groupby("stat"):
        m, b = (g.proj - g.actual).abs().mean(), (g.base - g.actual).abs().mean()
        flag = "" if m < b else "   <-- worse than naive"
        print(f"{st:<17}{len(g):>7,}{m:>11.2f}{b:>11.2f}{(b-m)/b*100:>8.1f}%{flag}")


if __name__ == "__main__":
    weeks = [(s, w) for s in (2023, 2024, 2025) for w in range(1, 19)]
    df = build(weeks)
    df.to_csv("reports/score_backtest.csv", index=False)
    table(df, "walk-forward 2023-2025 — ALL rows")
    table(df, "walk-forward 2023-2025 — BETTABLE universe only", bettable=True)
