"""Data for the Game Lines page. Read-only: frozen predictions + live lines and results.

Predictions come from data/gamelines/frozen/ (written once per week by
scripts/game_lines_week.py). Lines and scores come from the schedules table at
request time, so a finished game grades itself with no rebuild.
"""
import json
import math
import sqlite3

from .config import DATA_DIR, DB_PATH

from .gamelines import MODEL_VERSION

FROZEN = DATA_DIR / "gamelines" / "frozen" / MODEL_VERSION
BACKTEST = DATA_DIR / "gamelines" / "backtest_2026.json"


def _clean(v):
    return None if v is None or (isinstance(v, float) and math.isnan(v)) else v


def lines_blob(season=2026):
    weeks = {}
    try:
        db = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
        db.row_factory = sqlite3.Row
        sched = {r["game_id"]: dict(r) for r in db.execute(
            "SELECT game_id, spread_line, total_line, home_moneyline, away_moneyline, "
            "home_score, away_score FROM schedules WHERE season=? AND game_type='REG'", (season,))}
        db.close()
    except sqlite3.Error:
        sched = {}
    wx = {}
    try:
        from .weather import load as wload
        for r in wload().itertuples():
            wx[r.game_id] = {
                "indoors": not bool(r.applies),
                "roof_unknown": bool(r.roof_unknown),
                "kind": None if isinstance(r.kind, float) or r.kind is None else r.kind,
                "pulled_at": None if isinstance(r.pulled_at, float) else r.pulled_at,
                "temp_f": _clean(r.temp_f), "wind_mph": _clean(r.wind_mph), "gust_mph": _clean(r.gust_mph),
                "precip_mm": _clean(r.precip_mm), "precip_prob": _clean(r.precip_prob)}
    except Exception:  # noqa: BLE001 -- the page works without weather
        wx = {}
    for f in sorted(FROZEN.glob(f"{season}_w*.json"), key=lambda p: int(p.stem.split("_w")[1])):
        wk = int(f.stem.split("_w")[1])
        d = json.loads(f.read_text())
        games = []
        for r in d["games"]:
            s = {k: _clean(v) for k, v in sched.get(r["game_id"], {}).items()}
            games.append({**r, "wx": wx.get(r["game_id"]), **{k: s.get(k) for k in (
                "spread_line", "total_line", "home_moneyline", "away_moneyline",
                "home_score", "away_score")}})
        games.sort(key=lambda g: (g["gameday"] or "", g["gametime"] or ""))
        weeks[wk] = {"games": games, "sd_margin": d["sd_margin"], "sd_total": d["sd_total"],
                     "windows": d.get("windows"), "windows50": d.get("windows50"),
                     "target_record": d.get("target_record"), "target_display": d.get("target_display"),
                     "train_through": d["train_through"], "reconstructed": d["reconstructed"]}
    try:
        bt = json.loads(BACKTEST.read_text())
    except (FileNotFoundError, ValueError):
        bt = None
    cur = max(weeks) if weeks else None
    return {"season": season, "currentWeek": cur, "weeks": weeks, "backtest": bt}
