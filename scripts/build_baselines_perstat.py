"""
Decision #14 — baselines scored per-stat, on the population that does that thing.

The combined touches screen made the carries baseline mostly a test of
predicting zero, because that population is mostly pass-catchers. Each stat now
gets the players who actually accumulate it.

Screens stay leak-gated: a player's qualifying average uses games strictly
before the one being scored.

Decision #17 (2026-09-22) added a SECOND validation season. Baselines are now
computed for 2024 and 2025 separately, because a rung must clear its bar on
both. A baseline that exists for only one of the two seasons cannot be a bar
for the other. Output shape is baselines[<season>][<layer>][<stat>].
"""
import json, sqlite3
from pathlib import Path
import numpy as np, pandas as pd

DB = Path.home() / "Code/nfl-props/data/features.db"
OUT = Path.home() / "Code/nfl-props/data/baselines.json"
VALIDATION_SEASONS = (2024, 2025)
MIN_PER_GAME = 3.0

con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True, timeout=60)
df = pd.read_sql("""
    SELECT player_id, season, week, game_id,
           y_targets, y_carries, y_receiving_yards, y_rushing_yards,
           y_receiving_tds, y_rushing_tds
    FROM training_rows ORDER BY player_id, season, week
""", con)
con.close()

df["y_yards_per_target"] = np.where(df.y_targets > 0, df.y_receiving_yards / df.y_targets, np.nan)
df["y_yards_per_carry"]  = np.where(df.y_carries > 0, df.y_rushing_yards / df.y_carries, np.nan)

g = df.groupby("player_id", sort=False)

def prior_mean(col):
    """expanding mean of games strictly before this one"""
    return g[col].transform(lambda s: s.shift(1).expanding().mean())

def prior_last4(col):
    return g[col].transform(lambda s: s.shift(1).rolling(4, min_periods=1).mean())

# leak-gated qualifying screens, one per stat family
df["qual_carries"] = prior_mean("y_carries") >= MIN_PER_GAME
df["qual_targets"] = prior_mean("y_targets") >= MIN_PER_GAME

SPECS = [
    ("usage",      "targets",           "y_targets",           "qual_targets"),
    ("usage",      "carries",           "y_carries",           "qual_carries"),
    ("final",      "receiving_yards",   "y_receiving_yards",   "qual_targets"),
    ("final",      "rushing_yards",     "y_rushing_yards",     "qual_carries"),
    ("final",      "receiving_tds",     "y_receiving_tds",     "qual_targets"),
    ("final",      "rushing_tds",       "y_rushing_tds",       "qual_carries"),
    ("conversion", "yards_per_target",  "y_yards_per_target",  "qual_targets"),
    ("conversion", "yards_per_carry",   "y_yards_per_carry",   "qual_carries"),
]

result, rows = {}, []
for layer, name, col, screen in SPECS:
    df["_career"] = prior_mean(col)
    df["_last4"]  = prior_last4(col)
    for season in VALIDATION_SEASONS:
        v = df[(df.season == season) & df[screen]].dropna(subset=[col, "_career", "_last4"])
        if len(v) == 0:
            continue
        out = {}
        for cand in ("_career", "_last4"):
            err = v[col] - v[cand]
            out[cand.strip("_")] = dict(mae=round(float(err.abs().mean()), 4),
                                        rmse=round(float(np.sqrt((err ** 2).mean())), 4))
        winner = min(("career", "last4"), key=lambda k: out[k]["mae"])
        result.setdefault(str(season), {}).setdefault(layer, {})[name] = dict(
            **out, winner=winner, n=int(len(v)),
            screen=f"prior-game avg >= {MIN_PER_GAME} " + ("carries" if screen == "qual_carries" else "targets"),
            decider="MAE", validation_season=season)
        rows.append((season, layer, name, out["career"]["mae"], out["career"]["rmse"],
                     out["last4"]["mae"], out["last4"]["rmse"], winner, len(v)))

OUT.write_text(json.dumps(dict(
    decision="#14 — per-stat screens; see decisions.md",
    note="Fixed before any model exists. MAE decides; RMSE reported beside it.",
    validation_seasons=list(VALIDATION_SEASONS),
    baselines=result), indent=2))

rows.sort(key=lambda r: (r[1], r[2], r[0]))
print(f"{'season':<8}{'layer':<11}{'stat':<18}{'careerMAE':>10}{'last4MAE':>10}  {'winner':<7}{'n':>6}")
for r in rows:
    print(f"{r[0]:<8}{r[1]:<11}{r[2]:<18}{r[3]:>10.4f}{r[5]:>10.4f}  {r[7]:<7}{r[8]:>6}")
print(f"\nwritten -> {OUT}")
