"""DEPRECATED -- do not use.

This script selected legs using the historical-clear-rate ranking, which was
shown to come from a validation with a bug in it: the test looked up a player's
record against the wrong number. Recomputed correctly the effect vanishes
(61.9% / 61.4% / 61.0% from weakest to strongest record -- no signal).

Superseded by scripts/run_reports.py, which builds the production-ranges report
and shortlists on confidence rather than on that ranking. Kept for the record.
"""
import sys
sys.exit("deprecated -- see scripts/run_reports.py")

#!/usr/bin/env python
"""Rank the safest alternate-line legs on an upcoming slate.

Usage: safest.py SEASON WEEK [--min-price -320] [--max-price -180] [--top 25]

Output is ordered by a reliability score, NOT by edge. The aim is a string of
legs that hold up, which is a different objective from beating the price.
"""
import sys, pathlib, json, urllib.request, time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import pandas as pd, numpy as np
import nflreadpy as nfl
from nflprops import reliability as R
from nflprops.config import require
from nflprops.odds import name_key

SEASON, WEEK = int(sys.argv[1]), int(sys.argv[2])
def arg(f, d):
    return type(d)(sys.argv[sys.argv.index(f) + 1]) if f in sys.argv else d
LO, HI, TOP = arg("--min-price", -320), arg("--max-price", -180), arg("--top", 25)

ALT = {"player_pass_yds_alternate": "passing_yards",
       "player_rush_yds_alternate": "rushing_yards",
       "player_reception_yds_alternate": "receiving_yards",
       "player_receptions_alternate": "receptions"}
BASE = "https://api.the-odds-api.com/v4/sports/americanfootball_nfl"


def fetch_alt():
    key = require("ODDS_API_KEY")
    with urllib.request.urlopen(f"{BASE}/events?apiKey={key}", timeout=30) as r:
        evs = pd.DataFrame(json.load(r))
    evs["ct"] = pd.to_datetime(evs.commence_time, utc=True)
    evs = evs[evs.ct > pd.Timestamp.now(tz="UTC")]        # pre-game only
    print(f"{len(evs)} upcoming games")
    rows = []
    for e in evs.itertuples():
        u = (f"{BASE}/events/{e.id}/odds?apiKey={key}&bookmakers=fanduel"
             f"&markets={','.join(ALT)}&oddsFormat=american")
        try:
            with urllib.request.urlopen(u, timeout=30) as r:
                d = json.load(r); rem = r.headers.get("x-requests-remaining")
        except Exception as ex:
            print(f"  ! {e.away_team}@{e.home_team}: {ex}"); continue
        for bm in d.get("bookmakers", []):
            for m in bm.get("markets", []):
                stat = ALT.get(m["key"])
                if not stat:
                    continue
                for o in m.get("outcomes", []):
                    if o.get("name") != "Over":
                        continue
                    rows.append({"stat": stat, "player_book": o.get("description"),
                                 "line": o.get("point"), "price": o.get("price"),
                                 "game": f"{e.away_team} @ {e.home_team}"})
        time.sleep(0.2)
    print(f"credits remaining: {rem}")
    return pd.DataFrame(rows)


alt = fetch_alt()
alt = alt[(alt.price >= LO) & (alt.price <= HI)].copy()
print(f"{len(alt)} alternate legs priced {LO} to {HI}")
if alt.empty:
    sys.exit("nothing in that price band yet")

alt["player_key"] = alt.player_book.map(name_key)
hist = R._history(SEASON, WEEK, R.DEFAULT_LOOKBACK)
inj, sp = R.risk_flags(SEASON, WEEK)

# team lookup so we can attach spread/weather
tm = hist[["player_key", "team", "position"]].drop_duplicates("player_key")
alt = alt.merge(tm, on="player_key", how="left").merge(
    sp.drop_duplicates("team"), on="team", how="left").merge(
    inj[["player_key", "report_status", "practice_status"]].drop_duplicates("player_key"),
    on="player_key", how="left")

res = alt.apply(lambda r: R.clear_rate(hist, r.player_key, r.stat, r.line),
                axis=1, result_type="expand")
alt[["hit_rate", "n_games", "dnp_rate"]] = res
alt["mkt_fair"] = alt.price.map(R.fair)

# Reliability score. Weights set by validation on 777 historical alternate legs
# (scripts/validate_reliability.py): empirical clear rate ranks outcomes far
# better than the price does -- AUC 0.628 vs 0.534 -- so history carries the
# larger weight. An earlier version had these reversed, on the assumption the
# market was the better ranker. In the standard two-way market it is; in the
# alternate-line band it is not.
alt["blend"] = np.where(alt.hit_rate.notna(),
                        0.35 * alt.mkt_fair + 0.65 * alt.hit_rate, alt.mkt_fair)
# Thin history is not evidence. Legs below this are shown but never ranked top.
alt.loc[alt.n_games.fillna(0) < 6, "blend"] -= 0.10
alt["penalty"] = (
    alt.report_status.isin(["Questionable"]).astype(float) * 0.08      # backtested -8.0pp
    + alt.report_status.isin(["Doubtful", "Out"]).astype(float) * 0.60  # effectively unplayable
    + (alt.dnp_rate.fillna(0) > 0.25).astype(float) * 0.04             # misses games
    + (alt.abs_spread.fillna(0) > 9).astype(float) * 0.02              # blowout bench risk
    + (alt.n_games.fillna(0) < 6).astype(float) * 0.03                 # thin evidence
)
alt["score"] = alt.blend - alt.penalty
alt = alt.sort_values("score", ascending=False)

# One leg per player: the parlay wants independent-ish legs, and stacking two
# props on the same player concentrates exactly the risk we are trying to avoid.
best = alt.drop_duplicates("player_key").head(TOP)

print(f"\n{'='*104}")
print(f"  SAFEST LEGS — {SEASON} W{WEEK}   (one per player, ranked by reliability)")
print(f"{'='*104}")
hdr = (f"{'player':<22}{'game':<20}{'stat':<17}{'line':>7}{'price':>7}"
       f"{'mkt%':>7}{'hist%':>7}{'n':>4}{'score':>7}  flags")
print(hdr); print("-" * len(hdr))
for r in best.itertuples():
    flags = []
    if r.report_status in ("Questionable", "Doubtful", "Out"): flags.append(r.report_status.upper())
    if pd.notna(r.dnp_rate) and r.dnp_rate > 0.25: flags.append(f"misses {r.dnp_rate:.0%}")
    if pd.notna(r.abs_spread) and r.abs_spread > 9: flags.append(f"spread {r.abs_spread:.0f}")
    if pd.notna(r.n_games) and r.n_games < 6: flags.append(f"only {int(r.n_games)}g")
    hist_s = f"{r.hit_rate*100:>6.0f}%" if pd.notna(r.hit_rate) else "     —"
    print(f"{r.player_book[:21]:<22}{r.game[:19]:<20}{r.stat:<17}{r.line:>7.1f}{int(r.price):>7}"
          f"{r.mkt_fair*100:>6.0f}%{hist_s}{int(r.n_games):>4}{r.score*100:>6.0f}  {', '.join(flags)}")

out = pathlib.Path(__file__).parent.parent / f"reports/safest_{SEASON}_w{WEEK}.csv"
alt.to_csv(out, index=False)
print(f"\nfull list -> {out.name}")
print("\nNOTE: mkt% is FanDuel's price with an assumed 4.5% hold removed. hist% is how")
print("often this player actually cleared this number in games he played (n games).")
print("These are RELIABILITY ranks, not +EV bets -- the market is efficient and every")
print("leg here is priced fairly. Fewer legs = less compounding vig.")
