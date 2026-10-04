#!/usr/bin/env python3
"""Fit data/trends/holdup_rates.json from 2021-2025 only. Rerun only if the rules change."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import pandas as pd
from nflprops import trends as T
from nflprops.db import connect
from nflprops.shares import player_game_shares

pg = player_game_shares(2020)
pg = pg[pg.season <= 2025]          # the 2026 holdout is never read here
with connect() as con:
    rz = pd.read_sql_query("SELECT game_id, player_id, SUM(rz10_carries) rz10_carries, "
                           "SUM(rz10_targets) rz10_targets FROM rz_usage GROUP BY game_id, player_id", con)
rates = T.fit_and_save(T.entity_frames(pg, rz))
print(len(rates), "buckets ->", T.RATES_PATH)
