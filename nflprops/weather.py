"""Kickoff weather for every game, from Open-Meteo. One cache file, one fetcher.

data/weather/game_weather.csv has one row per game: the mean over the kickoff
hour and the three hours after it (a game lasts about three hours). Games in a
dome or with a closed roof have applies=0 and no numbers. The cache is not
written to `schedules`. Past games use the archive API (and the forecast API's
past_days for the last few days, which the archive does not hold yet). Future
games use the forecast API and keep the pull time.

Coordinates, roof type and kickoff time come from features.db (stadiums, game_spine).
"""
import sqlite3
import time
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests

from .config import DATA_DIR, DB_PATH

CACHE = DATA_DIR / "weather" / "game_weather.csv"
FEATURES_DB = DATA_DIR / "features.db"
ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"
FORECAST = "https://api.open-meteo.com/v1/forecast"
COMMON = "wind_speed_unit=mph&temperature_unit=fahrenheit&timezone=UTC"
WINDOW_H = 4          # kickoff hour + 3 hours
ARCHIVE_LAG_D = 6     # the archive does not hold the last few days yet
HEAVY_RAIN_MM = 1.0   # mean mm per hour over the window


def games_table():
    """Every regular-season game with stadium, kickoff, and the roof state if known."""
    f = sqlite3.connect(f"file:{FEATURES_DB}?mode=ro", uri=True)
    g = pd.read_sql("SELECT gs.game_id, gs.season, gs.week, gs.kickoff_utc, gs.stadium_key, "
                    "s.lat, s.lon, s.roof_type FROM game_spine gs JOIN stadiums s "
                    "ON s.stadium_key = gs.stadium_key WHERE gs.kickoff_utc IS NOT NULL", f)
    f.close()
    n = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    r = pd.read_sql("SELECT game_id, roof, game_type FROM schedules", n)
    n.close()
    g = g.merge(r, on="game_id", how="left")
    g["kick"] = pd.to_datetime(g["kickoff_utc"], utc=True)
    # applies: weather reaches the field. Closed roof or dome = no weather (a state, not a gap).
    roof = g["roof"].fillna("")
    g["applies"] = ~((g["roof_type"] == "dome") | roof.isin(["dome", "closed"]))
    g["roof_unknown"] = g["roof_type"].eq("retractable") & g["roof"].isna()
    return g


def _hourly(url, lat, lon, start, end, extra=""):
    q = (f"{url}?latitude={lat}&longitude={lon}&hourly=temperature_2m,wind_speed_10m,wind_gusts_10m,"
         f"precipitation{extra}&{COMMON}&{start}&{end}")
    for k in range(6):
        r = requests.get(q, timeout=60)
        if r.status_code == 200:
            h = r.json()["hourly"]
            d = pd.DataFrame(h)
            d["time"] = pd.to_datetime(d["time"], utc=True)
            return d.set_index("time")
        time.sleep(65 if r.status_code == 429 else 3 * (k + 1))
    r.raise_for_status()


def _window(d, kick):
    w = d.loc[kick.floor("h"): kick.floor("h") + timedelta(hours=WINDOW_H - 1)]
    if len(w) < WINDOW_H:
        return None
    k = w.iloc[0]   # the kickoff hour alone: what the props models were trained on (features.db weather)
    return {"temp_k": k["temperature_2m"], "wind_k": k["wind_speed_10m"], "gust_k": k["wind_gusts_10m"],
            "precip_k": k["precipitation"], "temp_f": w["temperature_2m"].mean(), "wind_mph": w["wind_speed_10m"].mean(),
            "gust_mph": w["wind_gusts_10m"].max(), "precip_mm": w["precipitation"].mean(),
            "precip_prob": w["precipitation_probability"].max()
            if "precipitation_probability" in w else None}


def fetch(only_future=False):
    """Fill the cache. Past games are fetched once; future games are refetched every call."""
    g = games_table()
    now = pd.Timestamp.now(tz="UTC")
    have = pd.read_csv(CACHE) if CACHE.exists() else pd.DataFrame(columns=["game_id"])
    done = set(have["game_id"]) if len(have) else set()
    rows = []
    for st, sg in g[g["applies"] | g["roof_unknown"]].groupby("stadium_key"):
        lat, lon = sg["lat"].iloc[0], sg["lon"].iloc[0]
        past = sg[(sg["kick"] < now) & ~sg["game_id"].isin(done)]
        old = past[past["kick"] < now - timedelta(days=ARCHIVE_LAG_D)]
        new = past[past["kick"] >= now - timedelta(days=ARCHIVE_LAG_D)]
        fut = sg[sg["kick"] >= now]
        batches = []
        for _, yr in (old.groupby("season") if not only_future else []):
            d = _hourly(ARCHIVE, lat, lon, f"start_date={yr['kick'].min().date()}",
                        f"end_date={(yr['kick'].max() + timedelta(days=1)).date()}")
            batches.append((yr, d, "measured"))
            time.sleep(0.5)
        if len(new) or len(fut):
            d = _hourly(FORECAST, lat, lon, "past_days=14", "forecast_days=10",
                        extra=",precipitation_probability")
            if len(new):
                batches.append((new, d, "measured"))
            if len(fut):
                batches.append((fut, d, "forecast"))
        for part, d, kind in batches:
            for r in part.itertuples():
                w = _window(d, r.kick)
                if w:
                    rows.append({"game_id": r.game_id, "kind": kind,
                                 "pulled_at": now.strftime("%Y-%m-%d %H:%MZ"), **w})
        _save(have, rows, g)
    return _save(have, rows, g)


def _save(have, rows, g):
    out = pd.DataFrame(rows)
    # forecasts are replaced by each pull; measured rows are kept once written
    keep = have[have["game_id"].isin(set(g["game_id"]))] if len(have) else have
    if len(keep):
        keep = keep[~((keep["kind"] == "forecast") & keep["game_id"].isin(set(out["game_id"]) if len(out) else set()))]
        keep = keep[~keep["game_id"].isin(set(out["game_id"]) if len(out) else set())]
    allw = pd.concat([keep, out], ignore_index=True)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    allw.to_csv(CACHE, index=False)
    return allw


def load():
    """Cache joined to the game table. Columns: applies, roof_unknown, temp_f, wind_mph, precip_mm, kind."""
    g = games_table()[["game_id", "applies", "roof_unknown", "roof_type", "roof"]]
    w = pd.read_csv(CACHE) if CACHE.exists() else pd.DataFrame(columns=["game_id"])
    return g.merge(w, on="game_id", how="left")


def forecasts_made():
    """Pre-kickoff forecasts the hourly job stored in features.db (kickoff minus about 3 h).

    Returns game_id, fc_precip_mm, fc_wind_mph, fc_temp_f. features.db stores precipitation in
    inches per hour, so it is converted to mm to match the cache.
    """
    f = sqlite3.connect(f"file:{FEATURES_DB}?mode=ro", uri=True)
    d = pd.read_sql("SELECT game_id, precip_amount_in, wind_speed_mph, temperature_f, fetched_at FROM weather "
                    "WHERE value_kind='forecast' ORDER BY fetched_at", f)
    f.close()
    d = d.groupby("game_id").last().reset_index()
    return pd.DataFrame({"game_id": d["game_id"], "fc_precip_mm": d["precip_amount_in"] * 25.4,
                         "fc_wind_mph": d["wind_speed_mph"], "fc_temp_f": d["temperature_f"]})


WX_COLS = ["weather_temp_f", "weather_wind_mph", "weather_wind_gust_mph", "weather_precip_prob_pct",
           "weather_precip_amt_in", "weather_humidity_pct"]


def apply_to_features(pf, mode):
    """Change the weather columns of a pregame feature frame, for the props models.

    mode None   leave the frame as built from features.db (measured/forecast rows it holds)
         "off"  weather columns missing for every game, as if no weather were known (test only)
         "fill" fill the columns that are missing for a game that has weather, from the cache
                (kickoff hour: temperature, wind, gust, rain in inches per hour)
    """
    if mode is None or pf.empty or "weather_temp_f" not in pf.columns:
        return pf
    pf = pf.copy()
    if mode == "off":
        for c in WX_COLS:
            pf[c] = float("nan")
        return pf
    w = load().drop_duplicates("game_id").set_index("game_id")
    gid = pf["game_id"]
    miss = pf["weather_temp_f"].isna() & ~pf["weather_no_weather_flag"].fillna(False).astype(bool)
    for col, src, k in (("weather_temp_f", "temp_k", 1), ("weather_wind_mph", "wind_k", 1),
                        ("weather_wind_gust_mph", "gust_k", 1), ("weather_precip_amt_in", "precip_k", 1 / 25.4)):
        v = gid.map(w[src]) * k
        pf.loc[miss, col] = v[miss]
    return pf


def game_summary(game_id):
    """Kickoff weather for one game, for the pages. None if the game is not in the cache.

    A forecast the hourly job stored before kickoff (features.db) wins: it is what the props models saw.
    Else the cache row: a forecast pulled now for a future game, or the measured value for a past one.
    """
    g = load()
    g = g[g["game_id"] == game_id]
    if g.empty:
        return None
    r = g.iloc[0]
    if not bool(r["applies"]):
        return {"indoors": True}
    out = {"indoors": False, "roof_unknown": bool(r["roof_unknown"])}
    try:
        f = sqlite3.connect(f"file:{FEATURES_DB}?mode=ro", uri=True)
        fc = f.execute("SELECT fetched_at, temperature_f, wind_speed_mph, wind_gust_mph, precip_probability_pct, "
                       "precip_amount_in FROM weather WHERE game_id=? AND value_kind='forecast' "
                       "ORDER BY fetched_at DESC LIMIT 1", (game_id,)).fetchone()
        f.close()
    except sqlite3.Error:
        fc = None
    if fc:
        out.update(kind="forecast", pulled_at=str(fc[0]) + "Z", temp_f=fc[1], wind_mph=fc[2], gust_mph=fc[3],
                   precip_prob=fc[4], precip_mm=None if fc[5] is None else fc[5] * 25.4)
    elif pd.notna(r.get("temp_k")):
        out.update(kind=r["kind"], pulled_at=r["pulled_at"], temp_f=float(r["temp_k"]), wind_mph=float(r["wind_k"]),
                   gust_mph=float(r["gust_k"]), precip_prob=None if pd.isna(r["precip_prob"]) else float(r["precip_prob"]),
                   precip_mm=float(r["precip_k"]))
    else:
        out.update(kind=None)
    return out
