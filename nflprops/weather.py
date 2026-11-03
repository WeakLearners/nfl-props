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
FORECAST_LOG = DATA_DIR / "weather" / "forecast_log.csv"   # every forecast ever pulled, append only
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
         f"precipitation,relative_humidity_2m{extra}&{COMMON}&{start}&{end}")
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
            "precip_k": k["precipitation"], "humidity_k": k["relative_humidity_2m"],
            "prob_k": k["precipitation_probability"] if "precipitation_probability" in k else None, "temp_f": w["temperature_2m"].mean(), "wind_mph": w["wind_speed_10m"].mean(),
            "gust_mph": w["wind_gusts_10m"].max(), "precip_mm": w["precipitation"].mean(),
            "precip_prob": w["precipitation_probability"].max()
            if "precipitation_probability" in w else None}


def fetch(only_future=False, within_hours=None):
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
        if within_hours is not None:
            fut = fut[fut["kick"] <= now + timedelta(hours=within_hours)]
            past = past.iloc[0:0]
            old = old.iloc[0:0]
            new = new.iloc[0:0]
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
    _log_forecasts(pd.DataFrame(rows))
    return _save(have, rows, g)


def _log_forecasts(out):
    """Append every forecast row to the log. Nothing is ever removed from it."""
    f = out[out["kind"] == "forecast"] if len(out) else out
    if not len(f):
        return
    cols = ["game_id", "pulled_at", "temp_k", "wind_k", "gust_k", "precip_k", "prob_k", "humidity_k"]
    new = f[cols].assign(source="open-meteo")
    FORECAST_LOG.parent.mkdir(parents=True, exist_ok=True)
    new.to_csv(FORECAST_LOG, mode="a", header=not FORECAST_LOG.exists(), index=False)


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
    """Pre-kickoff forecasts for game lines: game_id, fc_precip_mm, fc_wind_mph, fc_temp_f.

    From kickoff_inputs(), forecast kind only. The log holds the old hourly job's forecasts (imported once)
    and every pull since.
    """
    rows = [(gid, x["precip_mm"], x["wind_mph"], x["temp_f"], x["pulled_at"]) for gid, x in kickoff_inputs().items()
            if x["kind"] == "forecast"]
    return pd.DataFrame(rows, columns=["game_id", "fc_precip_mm", "fc_wind_mph", "fc_temp_f", "fc_pulled_at"])


WX_COLS = ["weather_temp_f", "weather_wind_mph", "weather_wind_gust_mph", "weather_precip_prob_pct",
           "weather_precip_amt_in", "weather_humidity_pct"]


def apply_to_features(pf, mode):
    """Change the weather columns of a pregame feature frame, for the props models.

    mode "cache" the default: every outdoor game takes its inputs from kickoff_inputs() (the one weather
                 source: latest forecast in the log before kickoff, else the cache value)
    mode None / "features"  leave the frame as built from features.db (the old source; kept for comparison)
         "off"  weather columns missing for every game, as if no weather were known (test only)
         "fill" fill the columns that are missing for a game that has weather, from the cache
                (kickoff hour: temperature, wind, gust, rain in inches per hour)
    """
    if mode in (None, "features") or pf.empty or "weather_temp_f" not in pf.columns:
        return pf
    pf = pf.copy()
    if mode == "cache":
        inp = kickoff_inputs()
        nw = pf["weather_no_weather_flag"].fillna(False).astype(bool)
        for i, r in pf[~nw].iterrows():
            x = inp.get(r["game_id"])
            if not x:
                continue
            fc = x["kind"] == "forecast"
            for col, v in (("weather_temp_f", x["temp_f"]), ("weather_wind_mph", x["wind_mph"]),
                           ("weather_wind_gust_mph", x["gust_mph"]),
                           ("weather_precip_amt_in", None if x["precip_mm"] is None else x["precip_mm"] / 25.4),
                           ("weather_precip_prob_pct", x["prob"] if fc else float("nan")),
                           ("weather_humidity_pct", x["humidity"])):
                pf.at[i, col] = v
        return pf
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
    """Kickoff weather for one game, for the pages. None if the game is not known.

    Uses kickoff_inputs(): the latest forecast pulled before kickoff, else the cache value.
    """
    g = games_table()
    g = g[g["game_id"] == game_id]
    if g.empty:
        return None
    r = g.iloc[0]
    if not bool(r["applies"]):
        return {"indoors": True}
    x = kickoff_inputs().get(game_id)
    out = {"indoors": False, "roof_unknown": bool(r["roof_unknown"])}
    if not x:
        out["kind"] = None
        return out
    out.update(kind=x["kind"], pulled_at=x["pulled_at"], temp_f=x["temp_f"], wind_mph=x["wind_mph"], gust_mph=x["gust_mph"],
               precip_prob=x["prob"], precip_mm=x["precip_mm"])
    return out


def import_feature_forecasts():
    """One-off: copy the forecasts the old hourly job stored in features.db into the forecast log.

    They are the only record of what was known before kickoff for weeks 3 and later. Idempotent: a row
    already in the log (same game and pulled_at) is skipped. features.db stores inches per hour; the log
    uses mm per hour like the cache.
    """
    f = sqlite3.connect(f"file:{FEATURES_DB}?mode=ro", uri=True)
    d = pd.read_sql("SELECT game_id, fetched_at, temperature_f, wind_speed_mph, wind_gust_mph, precip_probability_pct, "
                    "precip_amount_in, humidity_pct FROM weather WHERE value_kind='forecast'", f)
    f.close()
    new = pd.DataFrame({"game_id": d["game_id"], "pulled_at": d["fetched_at"].str.slice(0, 16) + "Z",
                        "temp_k": d["temperature_f"], "wind_k": d["wind_speed_mph"], "gust_k": d["wind_gust_mph"],
                        "precip_k": d["precip_amount_in"] * 25.4, "prob_k": d["precip_probability_pct"],
                        "humidity_k": d["humidity_pct"], "source": "features.db import"})
    have = pd.read_csv(FORECAST_LOG) if FORECAST_LOG.exists() else pd.DataFrame(columns=new.columns)
    key = set(zip(have["game_id"], have["pulled_at"]))
    new = new[[(g, p) not in key for g, p in zip(new["game_id"], new["pulled_at"])]]
    FORECAST_LOG.parent.mkdir(parents=True, exist_ok=True)
    new.to_csv(FORECAST_LOG, mode="a", header=not FORECAST_LOG.exists(), index=False)
    return len(new)


def kickoff_inputs():
    """What was known at kickoff for each outdoor game: {game_id: dict}. One source for props and game lines.

    The latest forecast in the log pulled before kickoff, else the cache value (measured, or a forecast
    pulled now for a future game). Fields: kind, pulled_at, temp_f, wind_mph, gust_mph, precip_mm, prob, humidity.
    """
    g = games_table().set_index("game_id")
    c = pd.read_csv(CACHE).drop_duplicates("game_id").set_index("game_id") if CACHE.exists() else pd.DataFrame()
    lg = pd.read_csv(FORECAST_LOG) if FORECAST_LOG.exists() else pd.DataFrame(columns=["game_id", "pulled_at"])
    if len(lg):
        lg["pulled"] = pd.to_datetime(lg["pulled_at"].str.replace("Z", "", regex=False), utc=True, errors="coerce")
        lg = lg.merge(g[["kick"]].reset_index(), on="game_id")
        lg = lg[lg["pulled"] <= lg["kick"]].sort_values("pulled").groupby("game_id").last()
    out = {}
    for gid, r in g.iterrows():
        if not r["applies"] and not r["roof_unknown"]:
            continue
        if gid in lg.index:
            x = lg.loc[gid]
            out[gid] = dict(kind="forecast", pulled_at=x["pulled_at"], temp_f=x["temp_k"], wind_mph=x["wind_k"],
                            gust_mph=x["gust_k"], precip_mm=x["precip_k"], prob=x["prob_k"], humidity=x["humidity_k"])
        elif len(c) and gid in c.index and pd.notna(c.at[gid, "temp_k"]):
            x = c.loc[gid]
            out[gid] = dict(kind=x["kind"], pulled_at=x["pulled_at"], temp_f=x["temp_k"], wind_mph=x["wind_k"],
                            gust_mph=x["gust_k"], precip_mm=x["precip_k"],
                            prob=x["prob_k"] if "prob_k" in x else None,
                            humidity=x["humidity_k"] if "humidity_k" in x else None)
    return out
