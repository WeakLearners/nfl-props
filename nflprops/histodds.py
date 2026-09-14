"""Historical FanDuel props, cached hard to disk.

Historical calls cost 10 credits per market per event -- roughly 30-50 credits
a game. Every response is cached under data/hist/ and never re-fetched, because
an accidental re-run of a full slate is real money off the quota.
"""
import json, time, urllib.request, urllib.error
from pathlib import Path

import pandas as pd

from .config import require, DATA_DIR
from .odds import MARKETS

HIST = DATA_DIR / "hist"
BASE = "https://api.the-odds-api.com/v4/historical/sports/americanfootball_nfl"


def _cached(path, fetch):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return json.loads(path.read_text()), True
    d = fetch()
    path.write_text(json.dumps(d))
    return d, False


def events(snapshot):
    key = require("ODDS_API_KEY")

    def go():
        with urllib.request.urlopen(f"{BASE}/events?apiKey={key}&date={snapshot}", timeout=30) as r:
            return json.load(r)

    d, hit = _cached(HIST / f"events_{snapshot.replace(':','')}.json", go)
    ev = pd.DataFrame(d.get("data", d))
    return ev, hit


def props(event_id, snapshot, book="fanduel"):
    key = require("ODDS_API_KEY")
    mk = ",".join(MARKETS)

    def go():
        u = (f"{BASE}/events/{event_id}/odds?apiKey={key}&date={snapshot}"
             f"&bookmakers={book}&markets={mk}&oddsFormat=american")
        try:
            with urllib.request.urlopen(u, timeout=30) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return {"data": {}}
            raise

    d, hit = _cached(HIST / f"props_{event_id}_{snapshot.replace(':','')}.json", go)
    g = d.get("data", d) or {}
    rows = []
    for bm in g.get("bookmakers", []):
        for m in bm.get("markets", []):
            stat = MARKETS.get(m["key"])
            if not stat:
                continue
            for o in m.get("outcomes", []):
                rows.append({"event_id": event_id, "stat": stat,
                             "player_book": o.get("description"), "side": o.get("name"),
                             "line": o.get("point"), "price": o.get("price")})
    if not hit:
        time.sleep(0.25)
    return pd.DataFrame(rows), hit


ALT_MARKETS = {"player_pass_yds_alternate": "passing_yards",
               "player_rush_yds_alternate": "rushing_yards",
               "player_reception_yds_alternate": "receiving_yards",
               "player_receptions_alternate": "receptions"}


def alt_props(event_id, snapshot, book="fanduel"):
    """Historical ALTERNATE-line props. 10 credits per market, so ~40 per game.
    Cached identically to props() -- never re-fetched."""
    key = require("ODDS_API_KEY")
    mk = ",".join(ALT_MARKETS)

    def go():
        u = (f"{BASE}/events/{event_id}/odds?apiKey={key}&date={snapshot}"
             f"&bookmakers={book}&markets={mk}&oddsFormat=american")
        try:
            with urllib.request.urlopen(u, timeout=30) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return {"data": {}}
            raise

    d, hit = _cached(HIST / f"alt_{event_id}_{snapshot.replace(':','')}.json", go)
    g = d.get("data", d) or {}
    rows = []
    for bm in g.get("bookmakers", []):
        for m in bm.get("markets", []):
            stat = ALT_MARKETS.get(m["key"])
            if not stat:
                continue
            for o in m.get("outcomes", []):
                if o.get("name") != "Over":
                    continue
                rows.append({"event_id": event_id, "stat": stat,
                             "player_book": o.get("description"),
                             "line": o.get("point"), "price": o.get("price")})
    if not hit:
        time.sleep(0.25)
    return pd.DataFrame(rows), hit
