"""Model quality: how good the projections were, before any bet is considered.

Measurement only. Nothing here changes a model, a weight, or a screen.

One row per (season, week, stat) in data/model_quality.csv, scope "week", plus one
row per (season, stat), scope "season", for the season to date (its week is the last
week included). Definitions:
  bias       mean of (actual - mu). Positive: the model projected too low.
  mae        mean of |actual - mu|, mu as published on the card.
  naive_mae  mean of |actual - naive| on the same players. Naive is the player's
             plain average of the stat over every regular-season game before that
             week (no weights, no opponent). A player with no earlier game is left
             out of both MAE numbers.
  rel_mae    mae / naive_mae. Below 1.0: the model beats the plain average.
  cov68      share of players with |actual - mu| <= sd. Target is about 68%.
  brier_raw  Brier score of mp_raw (the model alone) against "actual beat the line".
  brier_pub  Brier score of mp (the published number, moved toward the book by f).
  auc / auc_raw  chance that a leg that hit has a higher probability than a leg
             that missed (mp, mp_raw). 0.5 is a coin flip.
  b1..b5     five fixed bins of mp_raw (0-20, 20-40, 40-60, 60-80, 80-100%):
             leg count, mean predicted, share that hit.
Probability numbers use every projected player that has a priced line, not only
the shortlist.
"""
import pathlib

import numpy as np
import pandas as pd

from .config import DATA_DIR, ROOT, CURRENT_SEASON
from .db import connect
from .odds import name_key

CSV = DATA_DIR / "model_quality.csv"
MIN_N = 30
BIN_EDGES = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0001]
BIN_LABELS = ["0-20%", "20-40%", "40-60%", "60-80%", "80-100%"]
STATS = ["passing_yards", "completions", "rushing_yards", "receiving_yards", "receptions"]


def _naive(season, week):
    """{(name_key, stat): plain average over every regular-season game before this week}."""
    with connect() as con:
        pg = pd.read_sql_query(
            """SELECT player_display_name, passing_yards, completions, rushing_yards,
                      receiving_yards, receptions FROM player_games
               WHERE season_type='REG' AND (season < ? OR (season = ? AND week < ?))""",
            con, params=[season, season, week])
    pg["key"] = pg.player_display_name.map(name_key)
    return pg.groupby("key")[STATS].mean()


def collect(season, weeks):
    """One row per projected player-stat in finished games, with the actual and the naive."""
    import sys
    sc = str(ROOT / "scripts")
    if sc not in sys.path:
        sys.path.insert(0, sc)
    import grade_game as G
    rows = []
    for week in weeks:
        final = {(r.away_team, r.home_team) for r in G.played_games(season, week).itertuples()}
        act = G.actuals(season, week)
        if act.empty:
            continue
        by_key = act.set_index("key")
        naive = _naive(season, week)
        for rp in sorted((ROOT / "reports").glob(f"report_{season}_w{week}_*.html")):
            if G.teams_of(rp) not in final:
                continue
            for d in G.load_report(rp):
                k = name_key(d["p"])
                if k not in by_key.index:
                    continue
                a = by_key.loc[k, d["s"]]
                if isinstance(a, pd.Series):
                    a = a.iloc[0]
                if pd.isna(a):
                    continue
                nv = naive.loc[k, d["s"]] if k in naive.index else np.nan
                if isinstance(nv, pd.Series):
                    nv = nv.iloc[0]
                line = d.get("line")
                rows.append({
                    "season": season, "week": week, "stat": d["s"], "player": d["p"],
                    "mu": d["mu"], "sd": d.get("sd"), "actual": float(a), "naive": nv,
                    "line": line, "mp": d.get("mp"), "mp_raw": d.get("mp_raw"),
                    "hit": None if line is None else float(a > line)})
    return pd.DataFrame(rows)


def _auc(p, y):
    """Mann-Whitney AUC. NaN when one class is missing."""
    p, y = np.asarray(p, float), np.asarray(y, float)
    pos, neg = (y == 1).sum(), (y == 0).sum()
    if pos == 0 or neg == 0:
        return np.nan
    r = pd.Series(p).rank().values
    return float((r[y == 1].sum() - pos * (pos + 1) / 2) / (pos * neg))


def _brier(p, y):
    return float(np.mean((np.asarray(p, float) - np.asarray(y, float)) ** 2)) if len(y) else np.nan


def bins(sub):
    """Five-bin table on mp_raw: [(label, n, mean predicted, share hit)]."""
    sub = sub[sub.mp_raw.notna() & sub.hit.notna()]
    out = []
    cut = pd.cut(sub.mp_raw.astype(float), BIN_EDGES, right=False, labels=BIN_LABELS)
    for lab in BIN_LABELS:
        b = sub[cut == lab]
        out.append((lab, len(b), float(b.mp_raw.mean()) if len(b) else np.nan,
                    float(b.hit.mean()) if len(b) else np.nan))
    return out


def summarize(df):
    """One dict of A1-A4 numbers for a frame of one stat (any span of weeks)."""
    n = len(df)
    nv = df[df.naive.notna()]
    cv = df[df.sd.notna()]
    pr = df[df.mp.notna() & df.mp_raw.notna() & df.hit.notna()]
    r = {"n": n,
         "bias": float((df.actual - df.mu).mean()) if n else np.nan,
         "mae": float((df.actual - df.mu).abs().mean()) if n else np.nan,
         "naive_n": len(nv),
         "naive_mae": float((nv.actual - nv.naive).abs().mean()) if len(nv) else np.nan,
         "model_mae_same": float((nv.actual - nv.mu).abs().mean()) if len(nv) else np.nan,
         "cov_n": len(cv),
         "cov68": float(((cv.actual - cv.mu).abs() <= cv.sd).mean()) if len(cv) else np.nan,
         "priced_n": len(pr),
         "brier_raw": _brier(pr.mp_raw, pr.hit), "brier_pub": _brier(pr.mp, pr.hit),
         "auc": _auc(pr.mp, pr.hit) if len(pr) else np.nan,
         "auc_raw": _auc(pr.mp_raw, pr.hit) if len(pr) else np.nan}
    r["rel_mae"] = (r["model_mae_same"] / r["naive_mae"]
                    if len(nv) and r["naive_mae"] > 0 else np.nan)
    for i, (_, bn, bp, bh) in enumerate(bins(pr), 1):
        r[f"b{i}_n"], r[f"b{i}_pred"], r[f"b{i}_real"] = bn, bp, bh
    return r


def table(df, by_week=True):
    """Rows per (season, week, stat), or per (season, stat) when by_week is False."""
    keys = ["season", "week", "stat"] if by_week else ["season", "stat"]
    out = []
    for k, sub in df.groupby(keys):
        row = dict(zip(keys, k if isinstance(k, tuple) else (k,)))
        row.update(summarize(sub))
        out.append(row)
    return pd.DataFrame(out)


def write_csv(season=None, weeks=None, df=None):
    """Recompute every finished week of the season and rewrite its rows in the CSV."""
    import sys
    sc = str(ROOT / "scripts")
    if sc not in sys.path:
        sys.path.insert(0, sc)
    import grade_week as W
    season = season or CURRENT_SEASON
    weeks = weeks or W.finished_weeks(season)
    if df is None:
        df = collect(season, weeks)
    if df.empty:
        return None
    t = table(df, by_week=True)
    t.insert(0, "scope", "week")
    ssn = table(df, by_week=False)  # season to date: one row per stat, week = last week included
    ssn.insert(0, "scope", "season")
    ssn.insert(2, "week", int(df.week.max()))
    t = pd.concat([t, ssn], ignore_index=True).round(4)
    t["low_n"] = (t.n < MIN_N) | (t.priced_n < MIN_N)
    old = pd.read_csv(CSV) if CSV.exists() else pd.DataFrame()
    if not old.empty:
        old = old[old.season != season]
    pd.concat([old, t], ignore_index=True).to_csv(CSV, index=False)
    return t


def load():
    """The CSV as a list of dicts (NaN becomes None). [] if the file is absent."""
    if not CSV.exists():
        return []
    d = pd.read_csv(CSV)
    return d.astype(object).where(d.notna(), None).to_dict("records")
