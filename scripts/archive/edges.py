#!/usr/bin/env python
"""Pull live FanDuel props, join to projections, rank by edge.

This is the first point in the project where a number means something: edge is
measured against a real market price, not a stand-in.
"""
import sys, pathlib, io, contextlib, time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import pandas as pd
from nflprops import odds as O
from nflprops.db import team_abbr_map
from nflprops import project as P
from scripts_util import BETTABLE, MIN_VOL  # noqa

SEASON, WEEK = int(sys.argv[1]), int(sys.argv[2])
MIN_EDGE = 0.03          # 3 percentage points over the fair price
CACHE = pathlib.Path(__file__).parent.parent / "data" / f"odds_{SEASON}_w{WEEK}.parquet"


def team_map():
    return team_abbr_map()


def fetch(evs):
    frames, q = [], None
    for e in evs.itertuples():
        try:
            df, q = O.event_props(e.id)
            if not df.empty:
                df["away"], df["home"] = e.away_abbr, e.home_abbr
                frames.append(df)
        except Exception as ex:
            print(f"  ! {e.away_abbr}@{e.home_abbr}: {ex}", file=sys.stderr)
        time.sleep(0.2)
    print(f"  quota used={q['used']} remaining={q['remaining']}" if q else "  no data")
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


tm = team_map()
evs, _ = O.events()
evs["away_abbr"] = evs.away_team.map(tm)
evs["home_abbr"] = evs.home_team.map(tm)
evs = evs.dropna(subset=["away_abbr", "home_abbr"])

# Only games that have not kicked off. In-play odds price the football already
# played -- a receiver sitting on 80 yards carries a 106.5 full-game line -- so
# joining them to a pre-game projection manufactures enormous fake edges. This
# filter is the difference between a signal and an artifact.
evs["commence_time"] = pd.to_datetime(evs.commence_time, utc=True)
now = pd.Timestamp.now(tz="UTC")
live = evs[evs.commence_time <= now]
evs = evs[evs.commence_time > now]
if len(live):
    print(f"skipping {len(live)} in-progress/completed game(s): "
          + ", ".join(f"{r.away_abbr}@{r.home_abbr}" for r in live.itertuples()))
print(f"{len(evs)} pre-game events with odds available")
if evs.empty:
    sys.exit("no pre-game events -- nothing to price")

raw = fetch(evs)
if raw.empty:
    sys.exit("no props returned")
raw.to_parquet(CACHE)
print(f"cached -> {CACHE.name}  ({len(raw)} outcome rows)")

dv = O.devig(raw)
dv["player_key"] = dv.player_book.map(O.name_key)

with contextlib.redirect_stdout(io.StringIO()):
    proj = P.project_week(SEASON, WEEK)
proj = proj[proj.apply(lambda r: r.proj_vol >= MIN_VOL.get(r.stat, 0), axis=1)].copy()
proj["player_key"] = proj["player"].map(O.name_key)

e = O.edges(proj[["player_key", "stat", "proj", "sd", "player", "team", "opponent"]], dv)

# Surface join failures rather than letting them vanish.
book_keys = set(dv.player_key)
matched = set(e.player_key)
miss = sorted(book_keys - matched)
print(f"\njoined {len(e)} props;  {len(miss)} book players unmatched to a projection")
if miss:
    print("  unmatched sample:", ", ".join(miss[:8]))

e = e[e.edge >= MIN_EDGE].sort_values("edge", ascending=False)
print(f"\n{len(e)} props clear the {MIN_EDGE:.0%} edge threshold\n")
cols = ["player", "team", "opponent", "stat", "side", "line", "price", "proj", "p_model", "p_fair", "edge", "ev"]
if len(e):
    out = e[cols].head(25).copy()
    for c in ("p_model", "p_fair", "edge", "ev"):
        out[c] = (out[c] * 100).round(1)
    print(out.to_string(index=False))
e.to_csv(pathlib.Path(__file__).parent.parent / f"reports/edges_{SEASON}_w{WEEK}.csv", index=False)
