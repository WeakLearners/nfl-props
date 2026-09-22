#!/usr/bin/env python
"""Grade every priced leg in every finished game and tally by group.

Usage: python scripts/eda/tally_results.py [--json OUT]

Why this exists, and why it grades every row rather than the shortlist:
scoring only the legs we liked best flatters the model, because it hides every
projection it made and did not rank. The report page widened its own grading
for the same reason on 2026-09-20. This is that widening applied to the whole
season at once.

Two numbers per group, and the second is the one that matters:

  hit rate  -- how often the real number beat the line.
  ROI       -- what $1 flat on every leg returned, at the prices actually
               offered. A 70% hit rate on legs priced -290 loses money. Hit
               rate alone cannot tell you that; ROI can.

Grading rule and the finished-game gate are the project's own, matching
scripts/grade_game.py and serve_reports.outcomes(): a leg hits when the real
number beats the line, and a game counts only once it has a final score. A
receiver with 12 yards at half time has not missed an over-29.5.

Findings as of 2026-09-21 live in the vault at
projects/nfl-props/discrimination-measured.md.
"""
import argparse
import json
import os
import pathlib
import re
import sqlite3
import sys
from collections import Counter, defaultdict
from datetime import datetime

import numpy as np
from scipy.stats import rankdata

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from nflprops.config import ROOT  # noqa: E402
from nflprops.db import DB_PATH  # noqa: E402
from nflprops.odds import name_key  # noqa: E402

REPORTS = ROOT / "reports"
NAME = re.compile(r"report_(\d{4})_w(\d+)_([A-Z]+)-([A-Z]+)\.html$")
DATA = re.compile(r"const DATA = (\[.*?\]);\n", re.S)
STATS = ("passing_yards", "rushing_yards", "receiving_yards",
         "receptions", "completions")


def dec(american):
    """American price -> decimal. -290 pays 1.34 back on 1 staked."""
    return 1 + (100 / (-american) if american < 0 else american / 100)


def load():
    """Every priced leg from a finished game, graded."""
    db = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    done, kick = set(), {}
    for s, w, a, h, score, day, t in db.execute(
            "SELECT season,week,away_team,home_team,away_score,gameday,gametime "
            "FROM schedules"):
        if score is not None:
            done.add((s, w, a, h))
        kick[(s, w, a, h)] = f"{day} {t}" if day and t else None

    acts = {}
    for s, w, nm, *vals in db.execute(
            f"SELECT season,week,player_display_name,{','.join(STATS)} "
            "FROM player_games WHERE season_type='REG'"):
        k = name_key(nm)
        if k:
            acts[(s, w, k)] = dict(zip(STATS, vals))
    db.close()

    legs = []
    for f in sorted(os.listdir(REPORTS)):
        m = NAME.search(f)
        if not m:
            continue
        s, w, a, h = int(m[1]), int(m[2]), m[3], m[4]
        if (s, w, a, h) not in done:
            continue
        blob = DATA.search((REPORTS / f).read_text(encoding="utf-8"))
        if not blob:
            sys.stderr.write(f"no DATA block in {f}\n")
            continue
        for r in json.loads(blob.group(1)):
            if r.get("line") is None or r.get("price") is None:
                continue
            real = acts.get((s, w, name_key(r["p"])), {}).get(r["s"])
            if real is None:
                continue          # did not dress, or no stat line yet
            d = dec(r["price"])
            r.update(season=s, week=w, game=f"{a} @ {h}", kick=kick[(s, w, a, h)],
                     home=(r["tm"] == h), actual=real, dec=d, be=1 / d,
                     hit=bool(real > r["line"]))
            r["pnl"] = (d - 1) if r["hit"] else -1.0
            legs.append(r)
    return legs


def tally(legs, name, keyfn, order=None, minn=1):
    g = defaultdict(list)
    for l in legs:
        k = keyfn(l)
        if k is not None:
            g[k].append(l)
    rows = []
    for k, ls in g.items():
        n = len(ls)
        if n < minn:
            continue
        hits = sum(l["hit"] for l in ls)
        be = sum(l["be"] for l in ls) / n
        rows.append((k, n, hits, hits / n, be, sum(l["pnl"] for l in ls) / n))
    rows.sort(key=(lambda r: order.index(r[0]) if r[0] in order else 99)
              if order else (lambda r: -r[1]))
    print(f"### {name}")
    print(f"{'group':<24}{'n':>5}{'hit':>5}{'rate':>8}{'break-even':>12}"
          f"{'vs BE':>8}{'ROI/$1':>9}")
    for k, n, h, rate, be, roi in rows:
        print(f"{str(k):<24}{n:>5}{h:>5}{rate:>7.1%}{be:>12.1%}"
              f"{rate - be:>+8.1%}{roi:>+9.3f}")
    print()


def pos(l):
    return re.match(r"[A-Z]+", l["role"]).group()


def depth(l):
    m = re.search(r"\d+$", l["role"])
    return "depth " + (m.group() if m else "?")


def window(l):
    """Which sitting a game belongs to. A Sunday is four, not one."""
    if not l["kick"]:
        return None
    d = datetime.strptime(l["kick"], "%Y-%m-%d %H:%M")
    hh = d.hour + d.minute / 60
    return {3: "Thu night", 0: "Mon night"}.get(d.weekday()) or (
        "Sun 1:00 early" if hh < 14 else
        "Sun 4:00 late" if hh < 17 else "Sun night")


def price_band(l):
    p = l["price"]
    return ("a. <= -300" if p <= -300 else "b. -299..-200" if p <= -200 else
            "c. -199..-150" if p <= -150 else "d. -149..-110" if p <= -110 else
            "e. >= -109")


def conf_band(l):
    v = l.get("mp")
    if v is None:
        return None
    for e in (0.6, 0.7, 0.8, 0.9):
        if v < e:
            return f"model {int((e - .1) * 100)}-{int(e * 100)}%"
    return "model 90-100%"


def edge_band(l):
    if l.get("mp") is None or l.get("bp") is None:
        return None
    e = l["mp"] - l["bp"]
    return ("a. model < book by 5%+" if e < -.05 else
            "b. model < book" if e < 0 else
            "c. model > book by <5%" if e < .05 else
            "d. model > book by 5-10%" if e < .10 else
            "e. model > book by 10%+")


def headline(legs):
    n = len(legs)
    y = np.array([l["hit"] for l in legs], dtype=int)
    pnl = np.array([l["pnl"] for l in legs])
    be = np.mean([l["be"] for l in legs])
    rng = np.random.default_rng(0)
    bs = np.array([pnl[rng.integers(0, n, n)].mean() for _ in range(20000)])
    print(f"{n} graded legs across "
          f"{len({(l['season'], l['week'], l['game']) for l in legs})} finished games\n")
    print("### HEADLINE")
    print(f"  hit         {y.sum()}/{n} = {y.mean():.1%}")
    print(f"  break-even  {be:.1%}   gap {y.mean() - be:+.1%}")
    print(f"  $1 flat     {pnl.sum():+.2f}")
    print(f"  ROI per $1  {pnl.mean():+.3f}  95% CI "
          f"[{np.percentile(bs, 2.5):+.3f}, {np.percentile(bs, 97.5):+.3f}]")
    print(f"  P(ROI > 0)  {(bs > 0).mean():.1%}\n")

    # Discrimination, which is the number that decides whether any of the rest
    # of this file means anything. AUC 0.50 = cannot tell a hit from a miss.
    print("### DISCRIMINATION (AUC, 0.50 = coin flip)")
    for nm, key, flip in (("model prob", "mp", 1), ("book prob", "bp", 1),
                          ("z-score", "z", -1)):
        v = np.array([l.get(key) if l.get(key) is not None else np.nan
                      for l in legs], dtype=float)
        ok = ~np.isnan(v)
        a = auc(y[ok], flip * v[ok])
        bs = [auc(y[ok][i], flip * v[ok][i]) for i in
              (rng.integers(0, ok.sum(), ok.sum()) for _ in range(4000))]
        bs = np.array([b for b in bs if b == b])
        print(f"  {nm:<12} n={ok.sum():>4}  AUC={a:.3f}  95% CI "
              f"[{np.percentile(bs, 2.5):.3f}, {np.percentile(bs, 97.5):.3f}]")
    mp = np.array([l["mp"] for l in legs if l.get("mp") is not None])
    print(f"\n  mean model prob {mp.mean():.1%}  vs realised {y.mean():.1%}"
          f"  -> calibrated in aggregate, and that is not the same as useful\n")


def auc(y, s):
    """Probability a random hit is ranked above a random miss."""
    if len(set(y)) < 2:
        return float("nan")
    # rankdata, not argsort: tied scores must share a rank. Two legs the model
    # rates identically cannot be credited as correctly ordered.
    r = rankdata(s)
    npos, nneg = y.sum(), (1 - y).sum()
    return (r[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg)


def parlays(legs):
    """Same-game slips: did every shortlisted leg clear?"""
    g = defaultdict(list)
    for l in legs:
        if l.get("pick"):
            g[(l["week"], l["game"])].append(l)
    won = sum(all(x["hit"] for x in v) for v in g.values())
    print(f"### SAME-GAME SLIPS\n  cleared every leg: {won}/{len(g)}")
    for (w, game), v in sorted(g.items()):
        mark = "  <- CLEARED" if all(x["hit"] for x in v) else ""
        print(f"  W{w} {game:<14} {sum(x['hit'] for x in v)}/{len(v)}{mark}")
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", help="also write the graded legs here")
    a = ap.parse_args()
    legs = load()
    if not legs:
        sys.exit("no graded legs -- no finished game has a report")
    if a.json:
        pathlib.Path(a.json).write_text(json.dumps(legs, indent=1))
    headline(legs)
    tally(legs, "Shortlisted vs not",
          lambda l: "shortlisted" if l.get("pick") else "not shortlisted",
          order=["shortlisted", "not shortlisted"])
    tally(legs, "Shortlist rank",
          lambda l: f"pick #{l['pick']}" if l.get("pick") else None)
    tally(legs, "By stat", lambda l: l["s"])
    tally(legs, "By position", pos)
    tally(legs, "By depth rank", depth)
    tally(legs, "By slate window", window,
          order=["Thu night", "Sun 1:00 early", "Sun 4:00 late",
                 "Sun night", "Mon night"])
    tally(legs, "By price", price_band)
    tally(legs, "Model confidence (calibration)", conf_band,
          order=[f"model {a}-{a + 10}%" for a in (50, 60, 70, 80, 90)])
    tally(legs, "Claimed edge (model minus book)", edge_band)
    tally(legs, "Defence adjustment applied",
          lambda l: "applied" if l.get("app") else "not applied")
    tally(legs, "Home vs away", lambda l: "home" if l["home"] else "away")
    tally(legs, "By week", lambda l: f"{l['season']} W{l['week']}")
    tally(legs, "By game", lambda l: f"{l['game']} W{l['week']}")
    parlays(legs)
    print("Nothing above survived Holm correction across 29 groups on "
          "2026-09-21. Treat every split as noise until it repeats.")


if __name__ == "__main__":
    main()
