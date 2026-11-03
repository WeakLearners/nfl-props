"""Data for the Game Lines page. Read-only: frozen predictions + live lines and results.

Predictions come from data/gamelines/frozen/ (written once per week by
scripts/game_lines_week.py). Lines and scores come from the schedules table at
request time, so a finished game grades itself with no rebuild.
"""
import json
import math
import sqlite3

from .config import DATA_DIR, DB_PATH

FROZEN = DATA_DIR / "gamelines" / "frozen"
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
    for f in sorted(FROZEN.glob(f"{season}_w*.json"), key=lambda p: int(p.stem.split("_w")[1])):
        wk = int(f.stem.split("_w")[1])
        d = json.loads(f.read_text())
        games = []
        for r in d["games"]:
            s = {k: _clean(v) for k, v in sched.get(r["game_id"], {}).items()}
            games.append({**r, **{k: s.get(k) for k in (
                "spread_line", "total_line", "home_moneyline", "away_moneyline",
                "home_score", "away_score")}})
        games.sort(key=lambda g: (g["gameday"] or "", g["gametime"] or ""))
        weeks[wk] = {"games": games, "sd_margin": d["sd_margin"], "sd_total": d["sd_total"],
                     "train_through": d["train_through"], "reconstructed": d["reconstructed"]}
    try:
        bt = json.loads(BACKTEST.read_text())
    except (FileNotFoundError, ValueError):
        bt = None
    cur = max(weeks) if weeks else None
    return {"season": season, "currentWeek": cur, "weeks": weeks, "backtest": bt}
