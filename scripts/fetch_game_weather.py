#!/usr/bin/env python3
"""Fill data/weather/game_weather.csv from Open-Meteo. Safe to rerun: past rows are kept, forecasts refresh."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nflprops import weather as W
d = W.fetch()
print(len(d), "rows;", d["kind"].value_counts().to_dict())
