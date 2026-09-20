#!/usr/bin/env python
"""How much the book keeps, measured separately on the two shelves it sells from.

Every prop is a bar to clear. FanDuel will put the bar at a player's usual
number and charge a little, or lower the bar and charge a lot. The low bars are
the "alternate" ladder. The project has measured the ladder -- prices implied
70.5%, reality delivered 62.4%, an 8-point cut -- and has never measured the
normal shelf at all. ASSUMED_HOLD carried 4.5% for the normal shelf as an
industry figure, not as anything checked here.

This checks it. Same three Sundays, same games, same method, both shelves.

The ladder is the control. If this script cannot reproduce the known 8 points,
its number for the normal shelf means nothing either.

Reads ONLY the on-disk cache under data/hist/. Never calls the API, so it costs
no credits and can be re-run freely. Cached games it cannot find are reported,
not fetched.

usage:
  measure_hold.py
  measure_hold.py --csv reports/hold.csv
"""
import sys, json, pathlib, argparse

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import pandas as pd

from nflprops.db import connect
from nflprops.odds import MARKETS, name_key
from nflprops.histodds import ALT_MARKETS, HIST

# The three pre-kickoff Sundays the project already paid for. 16:45Z is 12:45pm
# ET, fifteen minutes before the 1pm games -- the prices a bettor could actually
# have taken.
SNAPSHOTS = [("2025-10-12T16:45:00Z", 2025, 6),
             ("2025-11-09T16:45:00Z", 2025, 10),
             ("2025-12-07T16:45:00Z", 2025, 14)]
SHELF = {"standard": (MARKETS, "props"), "ladder": (ALT_MARKETS, "alt")}


def implied(price):
    """What the price says the chance is, vig included. This is the number the
    bettor is charged for, so it is the number the outcome must be judged
    against."""
    return (-price) / ((-price) + 100) if price < 0 else 100 / (price + 100)


def cached(prefix, event_id, snapshot):
    p = HIST / f"{prefix}_{event_id}_{snapshot.replace(':', '')}.json"
    return json.loads(p.read_text()) if p.exists() else None


def priced(blob, market_map, event_id):
    """Every price in one game, one shelf, BOTH sides.

    Both sides matter. The Over is the side people want to buy, and a book
    prices the popular side to suit itself. Measuring only Overs would hide
    that, so the Under is collected too and judged by its own price.
    """
    rows = []
    for bm in (blob.get("data", blob) or {}).get("bookmakers", []):
        for m in bm.get("markets", []):
            stat = market_map.get(m["key"])
            if not stat:
                continue
            for o in m.get("outcomes", []):
                if o.get("name") not in ("Over", "Under") or o.get("point") is None:
                    continue
                rows.append({"event_id": event_id, "stat": stat, "side": o["name"],
                             "player_book": o.get("description"),
                             "line": float(o["point"]), "price": int(o["price"])})
    return rows


def actuals(season, week):
    cols = sorted(set(MARKETS.values()) | set(ALT_MARKETS.values()))
    with connect() as con:
        a = pd.read_sql_query(
            f"SELECT player_display_name, {', '.join(cols)} FROM player_games "
            "WHERE season=? AND week=? AND season_type='REG'", con, params=[season, week])
    long = pd.concat([a[["player_display_name", c]].rename(columns={c: "actual"}).assign(stat=c)
                      for c in cols], ignore_index=True)
    long["key"] = long.player_display_name.map(name_key)
    return long.dropna(subset=["actual"]).drop_duplicates(["key", "stat"])


def collect():
    frames, missing = [], 0
    for snap, season, week in SNAPSHOTS:
        ev = cached("events", "", snap)
        if ev is None:
            # events cache is named without an id; rebuild that one path by hand
            p = HIST / f"events_{snap.replace(':', '')}.json"
            ev = json.loads(p.read_text()) if p.exists() else None
        if ev is None:
            print(f"  {snap}: no cached events, skipping")
            continue
        games = (ev.get("data", ev) or [])
        # Only games where BOTH shelves are cached. Holding the games fixed is
        # the whole point: a difference between shelves must not be a difference
        # between samples.
        ids = [g["id"] for g in games
               if cached("props", g["id"], snap) and cached("alt", g["id"], snap)]
        missing += len(games) - len(ids)
        act = actuals(season, week)
        for shelf, (mmap, prefix) in SHELF.items():
            rows = []
            for gid in ids:
                rows += priced(cached(prefix, gid, snap), mmap, gid)
            if not rows:
                continue
            d = pd.DataFrame(rows)
            d["key"] = d.player_book.map(name_key)
            d = d.merge(act[["key", "stat", "actual"]], on=["key", "stat"], how="inner")
            d["shelf"], d["season"], d["week"] = shelf, season, week
            d["implied"] = d.price.map(implied)
            d["hit"] = (d.actual > d.line) if False else d.apply(lambda r: r.actual > r.line if r.side == "Over" else r.actual < r.line, axis=1)
            frames.append(d)
        print(f"  {snap}  W{week}: {len(ids)} games with both shelves cached")
    if missing:
        print(f"  ({missing} cached games had only one shelf and were left out)")
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def band(p):
    for lo, hi, lbl in [(0, .55, "under 55%"), (.55, .65, "55–65%"),
                        (.65, .75, "65–75%"), (.75, .85, "75–85%"), (.85, 2, "over 85%")]:
        if lo <= p < hi:
            return lbl
    return "?"


def line(name, g):
    imp, act = g.implied.mean(), g.hit.mean()
    # A flat $100 on every one of these, at the price offered.
    pnl = (g.apply(lambda r: (100 * 100 / -r.price if r.price < 0 else float(r.price))
                   if r.hit else -100.0, axis=1)).sum()
    return (f"{name:<14}{len(g):>7,}{imp*100:>11.1f}%{act*100:>10.1f}%"
            f"{(imp-act)*100:>9.1f}{pnl/(100*len(g))*100:>10.1f}%")


HEAD = f"{'':<14}{'bets':>7}{'price says':>12}{'reality':>10}{'cut':>9}{'return':>10}"

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv")
    a = ap.parse_args()

    print("Reading the cache. No API calls, no credits.\n")
    d = collect()
    if d.empty:
        sys.exit("nothing cached to measure")
    if a.csv:
        d.to_csv(a.csv, index=False)
        print(f"\nwrote {a.csv}")

    print(f"\n{'='*62}\n  The two shelves, same games, same kickoffs\n{'='*62}")
    print(HEAD); print("-" * 62)
    for shelf in ("standard", "ladder"):
        for sd in ("Over", "Under"):
            g = d[(d.shelf == shelf) & (d.side == sd)]
            if not g.empty:
                print(line(f"{shelf} {sd}", g))

    print(f"\n{'='*62}\n  The ladder as a control -- reproduce the known 8 points\n{'='*62}")
    lad = d[(d.shelf == "ladder") & d.price.between(-320, -180)]
    print(HEAD); print("-" * 62)
    if not lad.empty:
        print(line("-180..-320", lad))
    print("\nThe July figure on this band was 777 bets, 70.5% implied, 62.4% actual.")

    print(f"\n{'='*62}\n  By how expensive the price is\n{'='*62}")
    d["band"] = d.implied.map(band)
    for shelf in ("standard", "ladder"):
        g = d[d.shelf == shelf]
        if g.empty:
            continue
        print(f"\n{shelf}")
        print(HEAD); print("-" * 62)
        for b in ["under 55%", "55–65%", "65–75%", "75–85%", "over 85%"]:
            s = g[g.band == b]
            if len(s) >= 25:
                print(line(b, s))

    print(f"\n{'='*62}\n  By stat\n{'='*62}")
    for shelf in ("standard", "ladder"):
        g = d[d.shelf == shelf]
        if g.empty:
            continue
        print(f"\n{shelf}")
        print(HEAD); print("-" * 62)
        for st, s in g.groupby("stat"):
            if len(s) >= 25:
                print(line(st.replace("_", " "), s))
