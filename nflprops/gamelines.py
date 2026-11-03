"""Game lines: spread, total and win probability from one team-points model.

Entertainment only. The model predicts the points each team scores, so the
margin and the total come from the same fit. It reads `schedules` only, and
every fit for week W uses games strictly before W (the features.py rule).
See projects/nfl-props/game-lines-model.md.
"""
import sqlite3

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.linear_model import Ridge

from .config import DB_PATH

TEAM_FIX = {"OAK": "LV", "SD": "LAC", "STL": "LA", "LAR": "LA"}
FIRST_FIT_SEASON = 2019
QB_WINDOW = 8        # the usual starter = most common starter in the last 8 games
QB_MIN_STARTS = 3    # fewer starts than this and the team has no usual starter

# Chosen on 2024-2025 out-of-fold games (scripts/fit_game_lines.py), never on 2026.
PARAMS = {"alpha": 10.0, "half_life": 20.0, "qb": True, "rest": False}


def _t(df):
    """Time order that counts the offseason as extra weeks (22 slots per season)."""
    return (df["season"] - 2000) * 22 + df["week"]


def load_games(db_path=DB_PATH):
    """Regular-season games with lines, scores (NaN if unplayed) and the QB flags."""
    db = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    g = pd.read_sql(
        "SELECT game_id, season, week, gameday, gametime, away_team, home_team, away_score, "
        "home_score, result, spread_line, total_line, home_moneyline, away_moneyline, "
        "away_rest, home_rest, away_qb_id, home_qb_id, away_qb_name, home_qb_name "
        "FROM schedules WHERE game_type='REG' AND season>=?", db, params=(FIRST_FIT_SEASON,))
    db.close()
    for c in ("away_team", "home_team"):
        g[c] = g[c].replace(TEAM_FIX)
    g = g.sort_values(["season", "week", "gameday"]).reset_index(drop=True)
    g["t"] = _t(g)
    g["played"] = g["home_score"].notna()
    g["margin"] = g["home_score"] - g["away_score"]
    g["total"] = g["home_score"] + g["away_score"]
    return _qb_flags(g)


def _qb_flags(g):
    """home_qb_out / away_qb_out = 1 when the team's usual starter is not the starter.

    The starter for an unplayed game is the one nflverse lists. The usual
    starter comes from games strictly before this one.
    """
    hist = {}   # team -> list of starter ids, oldest first
    home_out, away_out = [], []
    for r in g.itertuples():
        flags = []
        for team, qb in ((r.home_team, r.home_qb_id), (r.away_team, r.away_qb_id)):
            recent = hist.get(team, [])[-QB_WINDOW:]
            usual = None
            if recent:
                vals, counts = np.unique(recent, return_counts=True)
                if counts.max() >= QB_MIN_STARTS:
                    usual = vals[counts.argmax()]
            flags.append(int(usual is not None and isinstance(qb, str) and qb != usual))
        home_out.append(flags[0])
        away_out.append(flags[1])
        # only played games add to the history, so a future week never feeds itself
        if r.played:
            for team, qb in ((r.home_team, r.home_qb_id), (r.away_team, r.away_qb_id)):
                if isinstance(qb, str):
                    hist.setdefault(team, []).append(qb)
    g["home_qb_out"], g["away_qb_out"] = home_out, away_out
    return g


def _design(g, teams, qb, rest):
    """Two rows per game: the home team's points, then the away team's."""
    n = len(g)
    ti = {t: i for i, t in enumerate(teams)}
    k = len(teams)
    extra = 1 + (2 if qb else 0) + (1 if rest else 0)
    X = np.zeros((2 * n, 2 * k + extra))
    hr = g["home_team"].map(ti).to_numpy()
    ar = g["away_team"].map(ti).to_numpy()
    idx = np.arange(n)
    X[idx, hr] = 1                # home offence
    X[idx, k + ar] = 1            # away defence
    X[n + idx, ar] = 1            # away offence
    X[n + idx, k + hr] = 1        # home defence
    c = 2 * k
    X[idx, c] = 1                 # home field: +1 on the home row, 0 on the away row
    c += 1
    if qb:
        X[idx, c] = g["home_qb_out"].to_numpy()        # own offence loses its usual QB
        X[n + idx, c] = g["away_qb_out"].to_numpy()
        X[idx, c + 1] = g["away_qb_out"].to_numpy()    # opponent offence loses its usual QB
        X[n + idx, c + 1] = g["home_qb_out"].to_numpy()
        c += 2
    if rest:
        rd = (g["home_rest"] - g["away_rest"]).clip(-7, 7).fillna(0).to_numpy() / 7.0
        X[idx, c] = rd
        X[n + idx, c] = -rd
    return X


def fit_predict(train, target, params=None):
    """Fit on `train` (played games) and return target with mu_home, mu_away columns."""
    p = {**PARAMS, **(params or {})}
    teams = sorted(set(train["home_team"]) | set(train["away_team"]) | set(target["home_team"])
                   | set(target["away_team"]))
    X = _design(train, teams, p["qb"], p["rest"])
    y = np.concatenate([train["home_score"].to_numpy(), train["away_score"].to_numpy()])
    age = train["t"].max() + 1 - train["t"].to_numpy()
    w = 0.5 ** (age / p["half_life"])
    w = np.concatenate([w, w])
    # The columns after the team blocks are not shrunk: scaling them up makes the
    # ridge penalty on them negligible, so only the team ratings are pulled to 0.
    boost = 20.0
    k2 = 2 * len(teams)
    scale = np.ones(X.shape[1])
    scale[k2:] = boost
    m = Ridge(alpha=p["alpha"]).fit(X * scale, y, sample_weight=w)
    coef = m.coef_ * scale
    Xt = _design(target, teams, p["qb"], p["rest"])
    pred = Xt @ coef + m.intercept_
    n = len(target)
    out = target.copy()
    out["mu_home"], out["mu_away"] = pred[:n], pred[n:]
    out["pred_margin"] = out["mu_home"] - out["mu_away"]
    out["pred_total"] = out["mu_home"] + out["mu_away"]
    names = (["home_field"] + (["qb_out_own", "qb_out_opp"] if p["qb"] else [])
             + (["rest_diff"] if p["rest"] else []))
    out.attrs["coefs"] = dict(zip(names, coef[k2:].round(3)))
    return out


def walk_forward(games, seasons, params=None, weeks=None):
    """Fit before each week, predict that week. Returns one row per predicted game."""
    parts = []
    for (s, w), tgt in games[games["season"].isin(seasons)].groupby(["season", "week"]):
        if weeks is not None and w not in weeks:
            continue
        tr = games[(games["t"] < tgt["t"].iloc[0]) & games["played"]]
        if len(tr) < 200:
            continue
        parts.append(fit_predict(tr, tgt, params))
    return pd.concat(parts) if parts else pd.DataFrame()


def sd_from_residuals(oof):
    """Margin and total SD from played out-of-fold games."""
    d = oof[oof["played"]]
    return (float(np.sqrt(((d["margin"] - d["pred_margin"]) ** 2).mean())),
            float(np.sqrt(((d["total"] - d["pred_total"]) ** 2).mean())))


def win_prob(margin, sd):
    return norm.cdf(np.asarray(margin) / sd)


def devig(ml_home, ml_away):
    """Home win probability with the book margin removed, from American odds."""
    def imp(o):
        o = np.asarray(o, dtype=float)
        return np.where(o < 0, -o / (-o + 100), 100 / (o + 100))
    a, b = imp(ml_home), imp(ml_away)
    return a / (a + b)


def metrics(df):
    """Model against the closing line on played games. Pushes are left out of hit rates.

    df needs pred_margin, pred_total and p_home (the model's home win probability).
    """
    d = df[df["played"] & df["spread_line"].notna()].copy()
    if d.empty:
        return {}
    out = {"n": int(len(d))}
    out["mae_margin_model"] = float((d["margin"] - d["pred_margin"]).abs().mean())
    out["mae_margin_line"] = float((d["margin"] - d["spread_line"]).abs().mean())
    out["mae_total_model"] = float((d["total"] - d["pred_total"]).abs().mean())
    out["mae_total_line"] = float((d["total"] - d["total_line"]).abs().mean())
    cover = d["margin"] - d["spread_line"]
    side = d["pred_margin"] - d["spread_line"]
    m = (cover != 0) & (side != 0)
    out["ats_n"] = int(m.sum())
    out["ats_hit"] = float(((cover[m] > 0) == (side[m] > 0)).mean())
    ou = d["total"] - d["total_line"]
    oside = d["pred_total"] - d["total_line"]
    m = (ou != 0) & (oside != 0)
    out["ou_n"] = int(m.sum())
    out["ou_hit"] = float(((ou[m] > 0) == (oside[m] > 0)).mean())
    # win probability: the model, and the book's devigged moneyline
    y = np.where(d["margin"] > 0, 1.0, np.where(d["margin"] < 0, 0.0, 0.5))
    p = np.clip(d["p_home"].to_numpy(), 1e-4, 1 - 1e-4)
    out["brier_model"] = float(((p - y) ** 2).mean())
    out["logloss_model"] = float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())
    mk = d.dropna(subset=["home_moneyline", "away_moneyline"])
    if len(mk):
        ym = np.where(mk["margin"] > 0, 1.0, np.where(mk["margin"] < 0, 0.0, 0.5))
        pm = np.clip(devig(mk["home_moneyline"], mk["away_moneyline"]), 1e-4, 1 - 1e-4)
        out["brier_market"] = float(((pm - ym) ** 2).mean())
        out["logloss_market"] = float(-(ym * np.log(pm) + (1 - ym) * np.log(1 - pm)).mean())
        pmod = np.clip(mk["p_home"].to_numpy(), 1e-4, 1 - 1e-4)
        out["brier_model_same_rows"] = float(((pmod - ym) ** 2).mean())
        out["logloss_model_same_rows"] = float(-(ym * np.log(pmod) + (1 - ym) * np.log(1 - pmod)).mean())
    if "mu_home" in d:
        # per-team points: the miss for each team's score, both teams pooled
        out["mae_score_model"] = float(np.concatenate([
            (d["home_score"] - d["mu_home"]).abs().to_numpy(),
            (d["away_score"] - d["mu_away"]).abs().to_numpy()]).mean())
        # the same miss when each score is rounded to a whole point, as the page shows it
        out["mae_score_model_rounded"] = float(np.concatenate([
            (d["home_score"] - d["mu_home"].round()).abs().to_numpy(),
            (d["away_score"] - d["mu_away"].round()).abs().to_numpy()]).mean())
        # the book's own score is implied by its spread and total (below)
    if d["total_line"].notna().all():
        hl = (d["total_line"] + d["spread_line"]) / 2     # book-implied home points
        al = (d["total_line"] - d["spread_line"]) / 2
        out["mae_score_line"] = float(np.concatenate([
            (d["home_score"] - hl).abs().to_numpy(), (d["away_score"] - al).abs().to_numpy()]).mean())
    out["bias_margin"] = float((d["pred_margin"] - d["margin"]).mean())
    out["bias_total"] = float((d["pred_total"] - d["total"]).mean())
    return out
