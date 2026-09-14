#!/usr/bin/env python
"""Post-refresh sanity check. Run after build_db.py; a silently stale mirror
is the failure mode most likely to survive into a live report."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import pandas as pd
from nflprops.db import connect
from nflprops.config import CURRENT_SEASON

CHECKS = []


def check(name, ok, detail=""):
    CHECKS.append((ok, name, detail))


with connect() as con:
    q = lambda s: pd.read_sql_query(s, con)
    pg = q("SELECT season, week, COUNT(*) n, COUNT(DISTINCT player_id) p FROM player_games GROUP BY 1,2")
    latest = pg[pg.season == CURRENT_SEASON]
    check("player_games has current season", not latest.empty,
          f"weeks {sorted(latest.week.tolist())}")
    check("no null player_ids",
          q("SELECT COUNT(*) c FROM player_games WHERE player_id IS NULL").c[0] == 0)
    check("per-week player counts sane",
          latest.p.between(150, 900).all() if not latest.empty else False,
          f"min={latest.p.min() if not latest.empty else 'NA'}")
    # A partial-season refresh used to wipe prior years, and every other check
    # here still passed because they only look at the current season. The
    # reliability ranker needs ~17 games of history, so assert the depth.
    seasons = sorted(pg.season.unique())
    check("history depth intact (>= 5 prior seasons)",
          len(seasons) >= 6, f"seasons {seasons[0]}..{seasons[-1]}")
    check("prior-season volume sane",
          q("SELECT COUNT(*) c FROM player_games WHERE season < %d" % CURRENT_SEASON).c[0] > 30000)
    check("schedules cover current season",
          q(f"SELECT COUNT(*) c FROM schedules WHERE season={CURRENT_SEASON}").c[0] > 250)
    check("injuries present for current season",
          q(f"SELECT COUNT(*) c FROM injuries WHERE season={CURRENT_SEASON}").c[0] > 0)
    check("rz_usage aligns with player_games",
          q("SELECT COUNT(*) c FROM rz_usage r WHERE NOT EXISTS "
            "(SELECT 1 FROM player_games p WHERE p.player_id=r.player_id "
            "AND p.season=r.season AND p.week=r.week)").c[0] < 2000,
          "orphan rz rows are mostly QBs/defenders, a few hundred is normal")

bad = 0
for ok, name, detail in CHECKS:
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"   [{detail}]" if detail else ""))
    bad += not ok
sys.exit(1 if bad else 0)
