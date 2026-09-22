"""
Decision #13 — populate training weather from MEASURED rows.

Rule 2 of phase2b-features.md said forecast-only. Applied literally that left
weather null on all 43,672 rows, because exactly one forecast exists. Sean
decided 2026-09-22 to train on measured and serve on forecast, with the gap
between them to be measured once the hourly fetcher accumulates paired rows.

This is a knowing, dated deviation. See decision #13 for its expiry.

Dome and closed-roof games keep weather NULL with weather_no_weather_flag = 1 —
that is a state, not missing data.
"""
import sqlite3
from pathlib import Path

DB = Path.home() / "Code/nfl-props/data/features.db"
con = sqlite3.connect(DB, timeout=60)

# one measured row per game: the fetch whose valid_time is nearest kickoff
con.executescript("""
DROP TABLE IF EXISTS _wx_pick;
CREATE TEMP TABLE _wx_pick AS
SELECT game_id, temperature_f, wind_speed_mph, wind_gust_mph,
       precip_probability_pct, precip_amount_in, humidity_pct
FROM (
  SELECT w.*, ROW_NUMBER() OVER (
           PARTITION BY w.game_id
           ORDER BY ABS(julianday(w.valid_time) - julianday(w.kickoff_utc))
         ) AS rn
  FROM weather w
  WHERE w.value_kind = 'measured'
) WHERE rn = 1;
""")

cols = ["temperature_f->weather_temp_f", "wind_speed_mph->weather_wind_mph",
        "wind_gust_mph->weather_wind_gust_mph",
        "precip_probability_pct->weather_precip_prob_pct",
        "precip_amount_in->weather_precip_amt_in",
        "humidity_pct->weather_humidity_pct"]
sets = ", ".join(
    f"{dst} = (SELECT p.{src} FROM _wx_pick p WHERE p.game_id = training_rows.game_id)"
    for src, dst in (c.split("->") for c in cols)
)
con.execute(f"""
    UPDATE training_rows SET {sets}
    WHERE weather_no_weather_flag = 0
      AND game_id IN (SELECT game_id FROM _wx_pick)
""")
con.commit()

q = lambda s: con.execute(s).fetchone()
print("rows with temperature:", q("SELECT COUNT(weather_temp_f) FROM training_rows")[0], "of", q("SELECT COUNT(*) FROM training_rows")[0])
print("no-weather (dome/closed):", q("SELECT COUNT(*) FROM training_rows WHERE weather_no_weather_flag=1")[0])
print("outdoor rows still null:", q("SELECT COUNT(*) FROM training_rows WHERE weather_no_weather_flag=0 AND weather_temp_f IS NULL")[0])
print("temp range:", q("SELECT ROUND(MIN(weather_temp_f),1), ROUND(MAX(weather_temp_f),1) FROM training_rows"))
print("wind range:", q("SELECT ROUND(MIN(weather_wind_mph),1), ROUND(MAX(weather_wind_mph),1) FROM training_rows"))
con.close()
