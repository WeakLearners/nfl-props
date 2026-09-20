#!/usr/bin/env python
"""Pool every graded game in a week into one write-up, and the season into one line.

A per-game write-up answers "how did Sunday's Carolina game go". Nobody bets one
game, and six legs cannot tell you whether a model works. This pools them: every
leg from every finished game, one hit rate, one calibration table, one return.

The return is the point. A hit rate on its own is a trap -- a leg priced at -300
has to land 75% of the time just to break even, so 70% is not a good week, it is
a losing one. This file always prints both numbers next to each other.

Measurement only. Nothing here changes a model, a weight, or a screen.

usage:
  grade_week.py                     # current season, latest week with finals
  grade_week.py --season 2026 --week 2
  grade_week.py --no-write
"""
import sys, pathlib, argparse, datetime as dt

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import pandas as pd

from nflprops.config import CURRENT_SEASON
from nflprops.db import connect
from nflprops.odds import name_key
import grade_game as G

VAULT = G.VAULT
# Confidence bands for the calibration table. Wide on purpose: at a few dozen
# legs a week, narrower buckets hold two legs each and measure nothing.
BANDS = [(0.50, 0.70), (0.70, 0.80), (0.80, 0.90), (0.90, 1.01)]


def profit(price, hit):
    """Dollars won or lost on a flat $100 stake at an American price."""
    if price is None or hit is None:
        return None
    if not hit:
        return -100.0
    return 100 * 100 / -price if price < 0 else float(price)


def legs(season, week):
    """Every shortlisted leg from every FINISHED game in the week, with outcome."""
    final = {(r.away_team, r.home_team): r
             for r in G.played_games(season, week).itertuples()}
    act = G.actuals(season, week)
    if act.empty:
        return pd.DataFrame(), {}
    by_key = act.set_index("key")
    rows = []
    for rp in sorted((ROOT / "reports").glob(f"report_{season}_w{week}_*.html")):
        tm = G.teams_of(rp)
        if tm not in final:
            continue
        g = final[tm]
        margin = abs(int(g.away_score) - int(g.home_score))
        for d in G.load_report(rp):
            if not d.get("pick"):
                continue
            k = name_key(d["p"])
            a = by_key.loc[k, d["s"]] if k in by_key.index else None
            if isinstance(a, pd.Series):
                a = a.iloc[0]
            line = d.get("line")
            hit = None if (a is None or pd.isna(a) or line is None) else bool(a > line)
            conf = (None if d.get("mp") is None or d.get("bp") is None
                    else min(d["mp"], d["bp"]))
            rows.append({"game": f"{tm[0]} @ {tm[1]}", "margin": margin,
                         "player": d["p"], "stat": d["s"], "line": line,
                         "price": d.get("price"), "conf": conf,
                         "actual": None if a is None or pd.isna(a) else float(a),
                         "hit": hit, "pnl": profit(d.get("price"), hit)})
    return pd.DataFrame(rows), final


def accuracy(season, week):
    """Projection error across EVERY projected player in finished games, not the
    shortlist. Six chosen legs flatter the model; this is the honest sample."""
    final = {(r.away_team, r.home_team)
             for r in G.played_games(season, week).itertuples()}
    act = G.actuals(season, week)
    if act.empty:
        return pd.DataFrame()
    by_key = act.set_index("key")
    rows = []
    for rp in sorted((ROOT / "reports").glob(f"report_{season}_w{week}_*.html")):
        if G.teams_of(rp) not in final:
            continue
        for d in G.load_report(rp):
            k = name_key(d["p"])
            if k not in by_key.index:
                continue
            a = by_key.loc[k, d["s"]]
            if isinstance(a, pd.Series):
                a = a.iloc[0]
            if pd.notna(a):
                rows.append({"stat": d["s"], "err": abs(d["mu"] - float(a))})
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).groupby("stat").err.agg(["mean", "count"])


def finished_weeks(season):
    with connect() as con:
        w = pd.read_sql_query(
            "SELECT DISTINCT week FROM schedules WHERE season=? AND away_score IS NOT NULL "
            "AND game_type='REG' ORDER BY week", con, params=[season])
    return [int(x) for x in w.week] if not w.empty else []


def headline(df):
    """hits, graded, rate, dollars, roi -- or None when nothing is graded."""
    g = df[df.hit.notna()]
    if g.empty:
        return None
    hits, n = int(g.hit.sum()), len(g)
    pnl = g.pnl.sum()
    return hits, n, hits / n, pnl, pnl / (100 * n)


def render(season, week, df, acc, season_rows):
    L = [f"# Week {week} — the whole slate ({dt.date.today()})\n"]
    sched = G.played_games(season, week)
    with connect() as con:
        total = pd.read_sql_query(
            "SELECT COUNT(*) n FROM schedules WHERE season=? AND week=?",
            con, params=[season, week]).n[0]
    reported = len({G.teams_of(p) for p in
                    (ROOT / "reports").glob(f"report_{season}_w{week}_*.html")})
    graded_games = df.game.nunique() if not df.empty else 0
    done = graded_games >= reported
    L.append(f"**Coverage:** {graded_games} of {reported} reported games graded; "
             f"{len(sched)} of {total} games on the schedule are final.\n")
    L.append(f"**Status:** {'final.' if done else 'partial — more games still to play.'}\n")
    L.append("Measurement only. No model, weight, or screen was changed to produce it.\n")

    h = headline(df)
    L.append("## The week in one line\n")
    if not h:
        L.append("Nothing graded yet.\n")
        return "\n".join(L)
    hits, n, rate, pnl, roi = h
    cs = df.conf.dropna()
    claim = f"{cs.mean()*100:.0f}%" if not cs.empty else "n/a"
    # Break-even is what turns a hit rate into a verdict. At these prices most
    # legs need to land three times in four just to stand still.
    be = df[df.hit.notna()].apply(
        lambda r: (-r.price) / ((-r.price) + 100) if r.price and r.price < 0
        else 100 / (r.price + 100), axis=1).mean()
    L.append(f"**{hits} of {n} legs hit — {rate*100:.1f}%.** The reports claimed "
             f"{claim} on average. At the prices actually offered, break-even was "
             f"**{be*100:.1f}%**.\n")
    verdict = "ahead of" if rate > be else "behind"
    L.append(f"A flat $100 on every leg returns **${pnl:+,.0f}** on ${100*n:,} staked "
             f"— **{roi*100:+.1f}%**. The hit rate is {verdict} break-even.\n")
    if rate > be and roi < 0:
        L.append("Note the disagreement: the hit rate beats the average break-even "
                 "while the money loses. That happens when the legs that missed were "
                 "the expensive ones. Price-weighted, not count-weighted, is what pays.\n")

    L.append("## By game\n")
    L.append("| Game | Final | Margin | Hit | $100 flat |")
    L.append("|---|---|---|---|---|")
    for gm, sub in df.groupby("game"):
        s = sched[(sched.away_team == gm.split(" @ ")[0]) &
                  (sched.home_team == gm.split(" @ ")[1])]
        score = (f"{int(s.away_score.iloc[0])}–{int(s.home_score.iloc[0])}"
                 if not s.empty else "—")
        gr = sub[sub.hit.notna()]
        if gr.empty:
            continue
        L.append(f"| {gm} | {score} | {int(sub.margin.iloc[0])} | "
                 f"{int(gr.hit.sum())}/{len(gr)} | ${gr.pnl.sum():+,.0f} |")
    L.append("")

    L.append("## Is the confidence honest\n")
    L.append("Each band holds the legs the reports priced at that confidence. "
             "Claimed is what they said; actual is what happened.\n")
    L.append("| Claimed | Legs | Claimed avg | Actual | $100 flat |")
    L.append("|---|---|---|---|---|")
    gr = df[df.hit.notna() & df.conf.notna()]
    for lo, hi in BANDS:
        b = gr[(gr.conf >= lo) & (gr.conf < hi)]
        if b.empty:
            continue
        L.append(f"| {lo*100:.0f}–{min(hi,1.0)*100:.0f}% | {len(b)} | "
                 f"{b.conf.mean()*100:.0f}% | {b.hit.mean()*100:.0f}% | "
                 f"${b.pnl.sum():+,.0f} |")
    L.append("\nA band where actual sits well under claimed is the model talking "
             "itself into legs. One where actual sits over it is money left behind.\n")

    if not acc.empty:
        L.append("## Projection error, every projected player\n")
        L.append("| Stat | Players | Average miss |")
        L.append("|---|---|---|")
        for stat, r in acc.iterrows():
            L.append(f"| {G.STAT_LABEL.get(stat, stat)} | {int(r['count'])} | {r['mean']:.1f} |")
        L.append("")

    if season_rows is not None and not season_rows.empty:
        sh = headline(season_rows)
        if sh:
            hits, n, rate, pnl, roi = sh
            wk = season_rows.week.nunique()
            L.append("## Season to date\n")
            L.append(f"**{hits} of {n} legs across {wk} week"
                     f"{'s' if wk != 1 else ''} — {rate*100:.1f}%.** "
                     f"Flat $100: **${pnl:+,.0f}**, **{roi*100:+.1f}%**.\n")
            L.append("| Week | Hit | Rate | $100 flat |")
            L.append("|---|---|---|---|")
            for w, sub in season_rows.groupby("week"):
                s = headline(sub)
                if s:
                    L.append(f"| {w} | {s[0]}/{s[1]} | {s[2]*100:.0f}% | ${s[3]:+,.0f} |")
            L.append("")

    L.append("## What this is not\n")
    L.append("A tally, not a conclusion. A band holding a dozen legs cannot "
             "distinguish a 70% model from an 80% one. Read the trend across weeks; "
             "do not change anything on one week's shape.\n")
    return "\n".join(L)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int)
    ap.add_argument("--week", type=int)
    ap.add_argument("--no-write", action="store_true")
    a = ap.parse_args()
    season = a.season or CURRENT_SEASON

    weeks = finished_weeks(season)
    if not weeks:
        print(f"{season}: no finished games")
        sys.exit(0)
    week = a.week or weeks[-1]

    season_rows = []
    for w in weeks:
        d, _ = legs(season, w)
        if not d.empty:
            d = d.assign(week=w)
            season_rows.append(d)
    season_df = pd.concat(season_rows) if season_rows else pd.DataFrame()

    df, _ = legs(season, week)
    out = render(season, week, df, accuracy(season, week), season_df)
    if a.no_write:
        print(out)
    else:
        VAULT.mkdir(parents=True, exist_ok=True)
        p = VAULT / f"{season}-w{week}-week.md"
        p.write_text(out)
        print(f"wrote {p}")
