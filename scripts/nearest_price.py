#!/usr/bin/env python
"""Fixed-price method.

Instead of taking any leg inside a price BAND, pick for each player-stat the
rung closest to a target price, then rank by estimated probability. Holding the
price roughly constant means the market's contribution is near-constant too, so
what separates candidates is the probability estimate rather than a mix of the
two. Cleaner comparison, and it matches how the bet is actually built on the
slider.

  single-game parlay : legs nearest -200, top 4 within one game
  multi-game parlay  : legs nearest -300, top 1 per game
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import pandas as pd, numpy as np
from nflprops import histodds as H, rank as K, reliability as R
from nflprops.db import connect
from nflprops.odds import name_key

SNAP = "2026-09-13T16:45:00Z"
SEASON, WEEK = 2026, 1
SINGLE_TARGET, MULTI_TARGET = -200, -300


def load_alt():
    ev, _ = H.events(SNAP)
    ev["ct"] = pd.to_datetime(ev.commence_time, utc=True)
    ev = ev[(ev.ct > pd.Timestamp(SNAP)) & (ev.ct < pd.Timestamp(SNAP) + pd.Timedelta(hours=8))]
    frames = []
    for e in ev.itertuples():
        df, _ = H.alt_props(e.id, SNAP)
        if not df.empty:
            df["game"] = f"{e.away_team} @ {e.home_team}"
            frames.append(df)
    return pd.concat(frames, ignore_index=True)


def eligible(alt):
    """Apply the validated screens, but NOT a price band -- the price filter is
    the nearest-target selection that happens after."""
    r = K.rank(alt, SEASON, WEEK)
    r = r[r.n_games >= K.MIN_HISTORY_GAMES]
    r = K.starters_only(r)
    r = r[~r.report_status.isin(["Questionable", "Doubtful", "Out"])]
    return K.stars_only(r, SEASON, WEEK)


def nearest(r, target):
    """One rung per player-stat: the price closest to target."""
    r = r.copy()
    r["dist"] = (r.price - target).abs()
    return (r.sort_values("dist")
             .drop_duplicates(["player_key", "stat"])
             .assign(p=lambda d: [K.calibrated_p(K.W_PRICE * m + K.W_HIST * h)
                                  for m, h in zip(d.mkt_fair, d.hit_rate)]))


def actuals():
    with connect() as con:
        pg = pd.read_sql_query(
            "SELECT player_display_name, passing_yards, rushing_yards, "
            "receiving_yards, receptions FROM player_games WHERE season=? AND week=?",
            con, params=[SEASON, WEEK])
    pg["player_key"] = pg.player_display_name.map(name_key)
    return pg


def grade(rows, act):
    out = []
    for r in rows.itertuples():
        m = act[act.player_key == r.player_key]
        a = m.iloc[0][r.stat] if len(m) and pd.notna(m.iloc[0][r.stat]) else None
        out.append(None if a is None else bool(a > r.line))
    rows = rows.copy()
    rows["actual"] = [act[act.player_key == r.player_key].iloc[0][r.stat]
                      if len(act[act.player_key == r.player_key])
                      and pd.notna(act[act.player_key == r.player_key].iloc[0][r.stat])
                      else np.nan for r in rows.itertuples()]
    rows["won"] = out
    return rows


def dec(a):
    return 1 + (100 / (-a) if a < 0 else a / 100)


alt = load_alt()
elig = eligible(alt)
act = actuals()
print(f"{alt.game.nunique()} games · {len(elig)} eligible star legs after screens\n")

# ---------- single-game parlays: nearest -200, top 4 within each game ----------
print("=" * 100)
print(f"  SINGLE-GAME PARLAYS — legs nearest {SINGLE_TARGET}, top 4 by probability")
print("=" * 100)
sg_results = []
for game, g in elig.groupby("game"):
    cand = nearest(g, SINGLE_TARGET).sort_values("p", ascending=False).head(4)
    if len(cand) < 4:
        continue
    cand = grade(cand, act)
    pay = cand.price.map(dec).prod()
    graded = cand[cand.won.notna()]
    hit = graded.won.all() if len(graded) == len(cand) else None
    print(f"\n{game}   pays {pay-1:.1f}x   est. survival {cand.p.prod()*100:.0f}%")
    for r in cand.itertuples():
        a = "—" if pd.isna(r.actual) else f"{r.actual:g}"
        res = "" if r.won is None else ("WIN " if r.won else "LOSS")
        print(f"   {res:<5}{r.player_book[:20]:<21}{r.stat.replace('_',' '):<17}"
              f"over {r.line:<6g}{int(r.price):>6}  p={r.p*100:>3.0f}%  actual {a:>5}")
    if hit is not None:
        print(f"   -> parlay {'HIT' if hit else 'DEAD'}"
              + ("" if hit else f" ({(~graded.won).sum()} leg(s) lost)"))
    sg_results.append({"game": game, "pay": pay, "hit": hit,
                       "legs": len(cand), "won": graded.won.sum(),
                       "graded": len(graded)})

# ---------- multi-game parlay: nearest -300, top 1 per game ----------
print("\n" + "=" * 100)
print(f"  MULTI-GAME PARLAY — legs nearest {MULTI_TARGET}, best 1 per game")
print("=" * 100)
picks = []
for game, g in elig.groupby("game"):
    cand = nearest(g, MULTI_TARGET).sort_values("p", ascending=False)
    if len(cand):
        picks.append(cand.iloc[0])
mg = grade(pd.DataFrame(picks), act)
pay = mg.price.map(dec).prod()
print(f"\n{len(mg)} legs   pays {pay-1:.0f}x   est. survival {mg.p.prod()*100:.1f}%\n")
for r in mg.itertuples():
    a = "—" if pd.isna(r.actual) else f"{r.actual:g}"
    res = "" if r.won is None else ("WIN " if r.won else "LOSS")
    print(f"   {res:<5}{r.player_book[:20]:<21}{r.stat.replace('_',' '):<17}"
          f"over {r.line:<6g}{int(r.price):>6}  p={r.p*100:>3.0f}%  actual {a:>5}   {r.game[:26]}")

g2 = mg[mg.won.notna()]
print(f"\n   graded {len(g2)}/{len(mg)}: {g2.won.sum()} hit, {(~g2.won).sum()} lost")
print(f"   -> parlay {'HIT' if len(g2)==len(mg) and g2.won.all() else 'DEAD'}")

# ---------- summary ----------
print("\n" + "=" * 100)
sg = pd.DataFrame(sg_results)
full = sg[sg.hit.notna()]
tot_legs = sg.graded.sum(); tot_won = sg.won.sum()
print(f"  SINGLE-GAME: {len(sg)} parlays built, {len(full)} fully graded, "
      f"{int(full.hit.sum()) if len(full) else 0} hit")
print(f"               leg level: {tot_won}/{tot_legs} ({tot_won/tot_legs*100:.0f}%)")
print(f"  MULTI-GAME:  {g2.won.sum()}/{len(g2)} legs ({g2.won.sum()/len(g2)*100:.0f}%)")
