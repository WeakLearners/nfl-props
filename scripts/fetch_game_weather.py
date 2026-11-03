#!/usr/bin/env python3
"""Fill data/weather/game_weather.csv from Open-Meteo. The one weather source for props and game lines.

usage: fetch_game_weather.py           past games once; every future game's forecast refreshed
       fetch_game_weather.py --near    only forecasts for games kicking off within 6 hours (the hourly job)

Past rows are kept. Every forecast pull is also appended to data/weather/forecast_log.csv, which is never
rewritten, so what was known before a kickoff can always be read back.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nflprops import weather as W
d = W.fetch(within_hours=6 if "--near" in sys.argv else None)
print(len(d), "rows;", d["kind"].value_counts().to_dict())
