"""
Job 4 (Phase 2A) — the weather fetcher.

Decision #3 requires four things captured per game: the forecast value, the
timestamp the forecast was made, the kickoff time, and the measured value
after the fact. Standing rule #8 says never overwrite a timestamped source —
store every fetch. Those two requirements together mean the storage shape is
an append-only LONG table: one row per (game, fetch event), tagged with
`value_kind` ('forecast' or 'measured') and `fetched_at`. A 2B query recovers
the "four columns" view decision #3 describes by filtering on value_kind and
comparing `fetched_at`/`valid_time` to `kickoff_utc` from game_spine — the
same table can hold many forecast fetches for one game without ever
clobbering an earlier one.

Source: Open-Meteo. No API key. Two endpoints:
  - https://api.open-meteo.com/v1/forecast       (live forecast)
  - https://archive-api.open-meteo.com/v1/archive (historical measured)

Dome games (roof_type == 'dome') are skipped entirely — decision #3 says a
dome is "no weather" as a STATE, and that state already lives as a non-null
value on `stadiums.roof_type` / reachable via `game_spine`. This table having
zero rows for a dome game is not a missing-data gap; the gap is closed by the
join, not by inserting placeholder nulls here. Retractable-roof stadiums are
fetched like outdoor ones, because the roof's actual state on game day is
architecture-dependent and observable only after the fact (via schedules.roof)
— 2B decides whether to gate on it.

Usage:
    python fetch_weather.py forecast              # run near kickoff-3h, on a schedule
    python fetch_weather.py backfill [--season YYYY]  # historical measured backfill
"""

import argparse
import sqlite3
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests

FEATURES_DB = Path.home() / "Code/nfl-props/data/features.db"

FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

FORECAST_HOURLY_FIELDS = (
    "temperature_2m,wind_speed_10m,wind_gusts_10m,"
    "precipitation_probability,precipitation,relative_humidity_2m"
)
ARCHIVE_HOURLY_FIELDS = (
    "temperature_2m,wind_speed_10m,wind_gusts_10m,"
    "precipitation,relative_humidity_2m"
)  # archive has no precipitation_probability — it's a deterministic record, skipped per spec


def utcnow():
    return datetime.now(timezone.utc)


def ensure_table(con):
    con.execute("""
        CREATE TABLE IF NOT EXISTS weather (
            id                      INTEGER PRIMARY KEY AUTOINCREMENT,
            game_id                 TEXT NOT NULL,
            stadium_key             TEXT NOT NULL,
            kickoff_utc             TEXT NOT NULL,
            value_kind              TEXT NOT NULL CHECK (value_kind IN ('forecast','measured')),
            fetched_at              TEXT NOT NULL,
            valid_time              TEXT NOT NULL,
            temperature_f           REAL,
            wind_speed_mph          REAL,
            wind_gust_mph           REAL,
            precip_probability_pct  REAL,
            precip_amount_in        REAL,
            humidity_pct            REAL,
            source                  TEXT NOT NULL
        )
    """)
    con.execute("CREATE INDEX IF NOT EXISTS idx_weather_game ON weather (game_id, value_kind)")
    con.commit()


def eligible_games(con, dome_excluded=True):
    q = """
        SELECT gs.game_id, gs.stadium_key, gs.kickoff_utc, s.lat, s.lon, s.roof_type
        FROM game_spine gs
        JOIN stadiums s ON s.stadium_key = gs.stadium_key
        WHERE gs.kickoff_utc IS NOT NULL
    """
    if dome_excluded:
        q += " AND s.roof_type != 'dome'"
    return con.execute(q).fetchall()


def nearest_hour_index(hourly_times, target_dt):
    target_hour = target_dt.strftime("%Y-%m-%dT%H:00")
    try:
        return hourly_times.index(target_hour)
    except ValueError:
        return None


def extract_row(hourly, idx):
    def get(field):
        vals = hourly.get(field)
        if vals is None or idx is None or idx >= len(vals):
            return None
        return vals[idx]
    return dict(
        temperature_f=get("temperature_2m"),
        wind_speed_mph=get("wind_speed_10m"),
        wind_gust_mph=get("wind_gusts_10m"),
        precip_probability_pct=get("precipitation_probability"),
        precip_amount_in=get("precipitation"),
        humidity_pct=get("relative_humidity_2m"),
    )


def insert_row(con, game_id, stadium_key, kickoff_utc, value_kind, fetched_at, valid_time, values, source):
    con.execute(
        """INSERT INTO weather
           (game_id, stadium_key, kickoff_utc, value_kind, fetched_at, valid_time,
            temperature_f, wind_speed_mph, wind_gust_mph, precip_probability_pct,
            precip_amount_in, humidity_pct, source)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            game_id, stadium_key, kickoff_utc, value_kind, fetched_at, valid_time,
            values["temperature_f"], values["wind_speed_mph"], values["wind_gust_mph"],
            values["precip_probability_pct"], values["precip_amount_in"], values["humidity_pct"],
            source,
        ),
    )


def run_forecast(con, window_hours_low=2.5, window_hours_high=3.5):
    """
    Fetch a forecast for every non-dome game whose kickoff falls between
    2.5 and 3.5 hours from now — a ~1-hour band centered on the kickoff-minus-3h
    target, wide enough that an hourly cron run never skips a game.
    """
    now = utcnow()
    lo = now + timedelta(hours=window_hours_low)
    hi = now + timedelta(hours=window_hours_high)
    games = eligible_games(con)
    fired = 0
    for game_id, stadium_key, kickoff_utc, lat, lon, roof_type in games:
        kickoff_dt = datetime.strptime(kickoff_utc, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
        if not (lo <= kickoff_dt <= hi):
            continue
        try:
            resp = requests.get(
                FORECAST_URL,
                params=dict(
                    latitude=lat, longitude=lon,
                    hourly=FORECAST_HOURLY_FIELDS,
                    temperature_unit="fahrenheit", wind_speed_unit="mph",
                    precipitation_unit="inch", timezone="UTC",
                    forecast_days=4,
                ),
                timeout=20,
            )
            resp.raise_for_status()
            data = resp.json()
        except requests.RequestException as e:
            print(f"FETCH FAILED forecast game_id={game_id}: {e}", file=sys.stderr)
            continue

        idx = nearest_hour_index(data["hourly"]["time"], kickoff_dt)
        values = extract_row(data["hourly"], idx)
        insert_row(
            con, game_id, stadium_key, kickoff_utc, "forecast",
            now.strftime("%Y-%m-%d %H:%M:%S"),
            data["hourly"]["time"][idx] if idx is not None else "",
            values, "open-meteo-forecast",
        )
        fired += 1
    con.commit()
    print(f"Forecast run: {fired} games fetched (window {window_hours_low}-{window_hours_high}h out).")


def run_backfill(con, season=None):
    """Historical MEASURED weather for games whose kickoff has already passed."""
    now = utcnow()
    games = eligible_games(con)
    done = 0
    skipped_future = 0
    already = 0
    failed = []

    cur = con.execute("SELECT DISTINCT game_id FROM weather WHERE value_kind='measured'")
    have = {r[0] for r in cur.fetchall()}

    for game_id, stadium_key, kickoff_utc, lat, lon, roof_type in games:
        if season is not None and not game_id.startswith(f"{season}_"):
            continue
        kickoff_dt = datetime.strptime(kickoff_utc, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
        if kickoff_dt > now:
            skipped_future += 1
            continue
        if game_id in have:
            already += 1
            continue

        date_str = kickoff_dt.strftime("%Y-%m-%d")
        try:
            resp = requests.get(
                ARCHIVE_URL,
                params=dict(
                    latitude=lat, longitude=lon,
                    start_date=date_str, end_date=date_str,
                    hourly=ARCHIVE_HOURLY_FIELDS,
                    temperature_unit="fahrenheit", wind_speed_unit="mph",
                    precipitation_unit="inch", timezone="UTC",
                ),
                timeout=20,
            )
            resp.raise_for_status()
            data = resp.json()
            idx = nearest_hour_index(data["hourly"]["time"], kickoff_dt)
            values = extract_row(data["hourly"], idx)
            values["precip_probability_pct"] = None  # not provided by archive endpoint
            insert_row(
                con, game_id, stadium_key, kickoff_utc, "measured",
                utcnow().strftime("%Y-%m-%d %H:%M:%S"),
                data["hourly"]["time"][idx] if idx is not None else "",
                values, "open-meteo-archive",
            )
            done += 1
            if done % 100 == 0:
                con.commit()
                print(f"  ...{done} backfilled so far")
        except (requests.RequestException, KeyError, ValueError) as e:
            failed.append((game_id, str(e)))
        time.sleep(0.1)

    con.commit()
    print(f"Backfill: {done} inserted, {already} already had a measured row, "
          f"{skipped_future} skipped (kickoff still in the future).")
    if failed:
        print(f"Backfill failures: {len(failed)}")
        for gid, err in failed[:20]:
            print("  ", gid, err)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["forecast", "backfill"])
    parser.add_argument("--season", type=int, default=None)
    parser.add_argument("--hours-low", type=float, default=2.5,
                         help="forecast mode: lower bound of the kickoff-distance window, in hours")
    parser.add_argument("--hours-high", type=float, default=3.5,
                         help="forecast mode: upper bound of the kickoff-distance window, in hours")
    args = parser.parse_args()

    con = sqlite3.connect(FEATURES_DB)
    ensure_table(con)

    if args.mode == "forecast":
        run_forecast(con, window_hours_low=args.hours_low, window_hours_high=args.hours_high)
    else:
        run_backfill(con, season=args.season)

    con.close()


if __name__ == "__main__":
    main()
