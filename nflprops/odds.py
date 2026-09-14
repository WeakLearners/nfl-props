"""The Odds API client: FanDuel player props, de-vigged.

Quota note: the events endpoint is free; each event-odds call costs one credit
per market requested. Pulling 5 markets for a 16-game slate is ~80 credits, so
a full season of weekly pulls is comfortably inside a 20k allowance. Do not
call this in a loop while iterating -- cache to disk instead.
"""
import json
import urllib.request
from statistics import NormalDist

import pandas as pd

from .config import require

BASE = "https://api.the-odds-api.com/v4/sports/americanfootball_nfl"

# The Odds API market key -> our stat name.
MARKETS = {
    "player_pass_yds": "passing_yards",
    "player_pass_completions": "completions",
    "player_rush_yds": "rushing_yards",
    "player_reception_yds": "receiving_yards",
    "player_receptions": "receptions",
}


def _get(url):
    with urllib.request.urlopen(url, timeout=30) as r:
        return json.load(r), {
            "used": r.headers.get("x-requests-used"),
            "remaining": r.headers.get("x-requests-remaining"),
        }


def events():
    d, q = _get(f"{BASE}/events?apiKey={require('ODDS_API_KEY')}")
    return pd.DataFrame(d), q


def event_props(event_id, book="fanduel"):
    """Raw over/under rows for one game."""
    mk = ",".join(MARKETS)
    d, q = _get(f"{BASE}/events/{event_id}/odds?apiKey={require('ODDS_API_KEY')}"
                f"&bookmakers={book}&markets={mk}&oddsFormat=american")
    rows = []
    for bm in d.get("bookmakers", []):
        for m in bm.get("markets", []):
            stat = MARKETS.get(m["key"])
            if not stat:
                continue
            for o in m.get("outcomes", []):
                rows.append({
                    "event_id": event_id, "book": bm["key"], "stat": stat,
                    "player_book": o.get("description"), "side": o.get("name"),
                    "line": o.get("point"), "price": o.get("price"),
                })
    return pd.DataFrame(rows), q


def implied(american):
    """American price -> implied probability, vig included."""
    a = float(american)
    return (-a) / ((-a) + 100) if a < 0 else 100 / (a + 100)


def decimal(american):
    a = float(american)
    return 1 + (100 / (-a) if a < 0 else a / 100)


def devig(df):
    """Collapse Over/Under pairs into one row per player-stat-line with the
    vig removed proportionally.

    Proportional de-vigging is the standard first pass. It slightly overstates
    the fair probability on heavy favourites (the vig is not actually split
    evenly), which matters for longshot legs -- worth revisiting if the parlay
    legs skew toward long prices.
    """
    p = df.pivot_table(index=["event_id", "stat", "player_book", "line"],
                       columns="side", values="price", aggfunc="first").reset_index()
    p = p.dropna(subset=["Over", "Under"])
    po, pu = p["Over"].map(implied), p["Under"].map(implied)
    p["hold"] = po + pu - 1.0
    p["fair_over"] = po / (po + pu)
    p["price_over"], p["price_under"] = p["Over"], p["Under"]
    return p.drop(columns=["Over", "Under"])


def model_p_over(proj, sd, line):
    """P(stat > line) under a normal approximation.

    Normal is a crude fit for low-count markets like receptions (a 2.5 line on
    a discrete variable), and it will misprice the tails. Adequate for ranking
    edges in v1; replace with a negative binomial before trusting the absolute
    probabilities.
    """
    sd = max(float(sd), 1e-6)
    return 1.0 - NormalDist(float(proj), sd).cdf(float(line))


def edges(proj_df, odds_df):
    """Join projections to de-vigged lines and compute edge + EV per $1."""
    m = odds_df.merge(proj_df, on=["player_key", "stat"], how="inner")
    m["p_over"] = [model_p_over(r.proj, r.sd, r.line) for r in m.itertuples()]
    m["edge_over"] = m.p_over - m.fair_over
    m["side"] = m.edge_over.apply(lambda e: "OVER" if e > 0 else "UNDER")
    m["p_model"] = m.apply(lambda r: r.p_over if r.side == "OVER" else 1 - r.p_over, axis=1)
    m["p_fair"] = m.apply(lambda r: r.fair_over if r.side == "OVER" else 1 - r.fair_over, axis=1)
    m["edge"] = m.p_model - m.p_fair
    m["price"] = m.apply(lambda r: r.price_over if r.side == "OVER" else r.price_under, axis=1)
    m["ev"] = m.p_model * (m.price.map(decimal) - 1) - (1 - m.p_model)
    return m


import re

_SUFFIX = re.compile(r"\b(jr|sr|ii|iii|iv|v)\b")


def name_key(name):
    """Normalise a player name for joining book data to nflverse.

    The book writes 'Chris Godwin Jr.' where nflverse has 'Chris Godwin', and
    punctuation differs on names like 'Ja'Marr'. Unmatched names are surfaced
    rather than dropped silently -- a missed join is a missed bet.
    """
    if not isinstance(name, str):
        return ""
    # Periods are DELETED, not spaced: the book writes "D.J. Moore" where
    # nflverse has "DJ Moore", and spacing the period makes those two names
    # disagree forever. Same for T.J., A.J., J.K.
    s = name.lower().replace(".", "").replace("'", "").replace("-", " ")
    s = _SUFFIX.sub(" ", s)
    return " ".join(s.split())
