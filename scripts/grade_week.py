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
            # Rung choice (week 8 on): the same player-stat at the rung the old
            # rule would have used, graded from its stored line.
            old_line, old_price = d.get("old_line"), d.get("old_price")
            old_hit = (None if (a is None or pd.isna(a) or old_line is None)
                       else bool(a > old_line))
            conf = (None if d.get("mp") is None or d.get("bp") is None
                    else min(d["mp"], d["bp"]))
            rows.append({"game": f"{tm[0]} @ {tm[1]}", "margin": margin,
                         "player": d["p"], "stat": d["s"], "line": line,
                         "price": d.get("price"), "conf": conf,
                         "actual": None if a is None or pd.isna(a) else float(a),
                         "hit": hit, "pnl": profit(d.get("price"), hit),
                         "rung": d.get("rung"), "old_line": old_line,
                         "old_price": old_price, "old_hit": old_hit,
                         "old_pnl": profit(old_price, old_hit)})
    return pd.DataFrame(rows), final


def projection_rows(season, weeks):
    """One row per projected player-stat in finished games: the raw mu (before
    any bias), the stored mu, and the actual. Not only shortlisted legs.
    Reports built before the bias carry no mu_raw, and their mu is raw."""
    rows = []
    for week in weeks:
        final = {(r.away_team, r.home_team)
                 for r in G.played_games(season, week).itertuples()}
        act = G.actuals(season, week)
        if act.empty:
            continue
        by_key = act.set_index("key")
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
                    rows.append({"stat": d["s"], "mu": d["mu"],
                                 "mu_raw": d.get("mu_raw", d["mu"]), "actual": float(a)})
    return pd.DataFrame(rows)


def accuracy(season, week):
    """Projection error across EVERY projected player in finished games, not the
    shortlist. Six chosen legs flatter the model; this is the honest sample.
    Miss is the average absolute error of the raw mu, and of the raw mu plus
    the fitted bias (calibration.json), so the bias effect is visible."""
    from nflprops import calibration as CAL
    df = projection_rows(season, [week])
    if df.empty:
        return pd.DataFrame()
    df["b"] = df.stat.map(CAL.bias_for)
    df["err"] = (df.mu_raw - df.actual).abs()
    df["err_adj"] = (df.mu_raw + df.b - df.actual).abs()
    return df.groupby("stat").agg(count=("err", "size"), mean=("err", "mean"),
                                  adj=("err_adj", "mean"), bias=("b", "first"))


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


def _f(x, d=1, pct=False):
    if x is None or pd.isna(x):
        return "n/a"
    if round(x, d + (2 if pct else 0)) == 0:
        x = 0.0
    return f"{x*100:.{d}f}%" if pct else f"{x:.{d}f}"


def model_quality_section(mq, acc, week):
    """Top section: how good the projections were, with no betting line involved.
    mq: collect() frame for the season through `week`. acc: accuracy() table."""
    from nflprops import model_quality as MQ
    L = ["# Model quality (no betting line involved)\n",
         "Definitions. Bias = mean of actual minus mu. MAE = mean absolute error of mu as published. "
         "Naive = the player's plain average of the stat over all earlier regular-season games. "
         "Rel MAE = model MAE / naive MAE on the same players (below 1.0 beats the plain average). "
         "Within 1 sd = share of players with actual inside mu plus or minus sd (target about 68%). "
         "Brier = mean squared error of a probability against \"actual beat the line\" (lower is better, "
         "0.25 is a coin flip). Raw is the model alone (reports before the blend have no stored raw number, so their published number is used, because it was not blended yet). Published is moved toward the book by f. "
         "AUC = chance that a leg that hit ranks above a leg that missed (0.5 is a coin flip). "
         "A star marks a count under 30.\n"]
    if mq is None or mq.empty:
        return "\n".join(L + ["No projections graded yet.\n"])
    wk = mq[mq.week == week]
    for name, d in (("Week " + str(week), wk), ("Season to date", mq[mq.week <= week])):
        if d.empty:
            continue
        t = MQ.table(d, by_week=False)
        L.append(f"## {name}: projection error and spread\n")
        L.append("| Stat | Players | Bias | MAE | Naive MAE | Rel MAE | Within 1 sd |")
        L.append("|---|---|---|---|---|---|---|")
        for _, r in t.iterrows():
            star = "*" if r.n < MQ.MIN_N else ""
            L.append(f"| {G.STAT_LABEL.get(r.stat, r.stat)} | {int(r.n)}{star} | {_f(r.bias)} | {_f(r.mae)} | "
                     f"{_f(r.naive_mae)} | {_f(r.rel_mae, 3)} | {_f(r.cov68, 1, True)} |")
        L.append("")
        L.append(f"## {name}: probability and ranking\n")
        L.append("| Stat | Priced legs | Brier raw | Brier published | AUC raw | AUC published |")
        L.append("|---|---|---|---|---|---|")
        for _, r in t.iterrows():
            star = "*" if r.priced_n < MQ.MIN_N else ""
            L.append(f"| {G.STAT_LABEL.get(r.stat, r.stat)} | {int(r.priced_n)}{star} | {_f(r.brier_raw, 4)} | "
                     f"{_f(r.brier_pub, 4)} | {_f(r.auc_raw, 3)} | {_f(r.auc, 3)} |")
        L.append("")
        pr = d[d.mp.notna() & d.mp_raw.notna() & d.hit.notna()]
        L.append(f"## {name}: predicted vs realized (raw model probability, all stats)\n")
        L.append("| Predicted | Legs | Mean predicted | Hit |")
        L.append("|---|---|---|---|")
        for lab, n, pp, rr in MQ.bins(pr):
            L.append(f"| {lab} | {n}{'*' if n < MQ.MIN_N else ''} | {_f(pp, 1, True)} | {_f(rr, 1, True)} |")
        L.append("")
    L.append("Priced legs are projected players that have a line on their card, not only the shortlist. "
             "Completions has no line. The week-by-week trend is on the Trends page.\n")
    if not acc.empty:
        L.append("## Projection error with the fitted bias, every projected player\n")
        from nflprops import calibration as CAL
        nn = CAL.bias_n()
        L.append("| Stat | Players | Bias (season n) | Miss, raw | Miss, with bias |")
        L.append("|---|---|---|---|---|")
        for stat, r in acc.iterrows():
            L.append(f"| {G.STAT_LABEL.get(stat, stat)} | {int(r['count'])} | "
                     f"{r['bias']:+.1f} ({nn.get(stat, 0)}) | {r['mean']:.1f} | {r['adj']:.1f} |")
        L.append("")
    return "\n".join(L)


def render(season, week, df, acc, season_rows, mq=None):
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
    L.append(model_quality_section(mq, acc, week))

    h = headline(df)
    L.append("# Against the line\n")
    L.append("Shortlisted priced Over legs only. Anytime-TD picks are not in this section.\n")
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


def _breakeven(prices):
    """Average break-even hit rate for a series of American prices."""
    p = [(-x) / ((-x) + 100) if x < 0 else 100 / (x + 100) for x in prices]
    return sum(p) / len(p) if p else None


def _side(df, hit, price, pnl):
    g = df[df[hit].notna()]
    if g.empty:
        return None
    hits, n = int(g[hit].sum()), len(g)
    return hits, n, _breakeven(g[price]), g[pnl].sum()


def rung_section(df, season_df):
    """Chosen rung vs old rung, same player-stats, week and season.

    Only legs whose rung was chosen (week 8 on) have an old rung to compare.
    Fallback legs and weeks before 8 are left out.
    """
    def pick(d):
        if d is None or d.empty or "rung" not in d.columns:
            return pd.DataFrame()
        return d[(d.rung == "chosen") & d.old_line.notna()]

    L = ["## Chosen rung vs old rung\n",
         "Same players and stats on both sides. Chosen is the rung the card bet. "
         "Old is the rung nearest -300. Only legs with a chosen rung count.\n"]
    any_rows = False
    L.append("| Scope | Rung | Legs | Hit | Needed | $100 flat |")
    L.append("|---|---|---|---|---|---|")
    for name, d in (("Week", pick(df)), ("Season", pick(season_df))):
        if d.empty:
            continue
        for label, cols in (("Chosen", ("hit", "price", "pnl")),
                            ("Old", ("old_hit", "old_price", "old_pnl"))):
            r = _side(d, *cols)
            if r:
                any_rows = True
                hits, n, be, pnl = r
                L.append(f"| {name} | {label} | {n} | {hits}/{n} ({hits/n*100:.1f}%) | "
                         f"{be*100:.1f}% | ${pnl:+,.0f} |")
    if not any_rows:
        return "## Chosen rung vs old rung\n\nNo graded legs with a chosen rung yet (starts with week 8).\n"
    L.append("")
    return "\n".join(L)


def adjustment_section(write=True):
    """Refit the weekly fraction (decision #46) and report it in three lines."""
    from nflprops import calibration as CAL
    r = CAL.fit(write=write)
    if r is None:
        return "## Weekly adjustment\n\nNo graded legs yet. Fraction stays 0.\n"
    prev = r["previous_fraction"]
    return ("## Weekly adjustment\n\n"
            f"New fraction: {r['fraction']:.1f} (last week: "
            f"{'none' if prev is None else f'{prev:.1f}'}). Model moves that share of the way to the book.\n\n"
            f"Brier on {r['legs']} legs to date: {r['brier_before']:.4f} before, "
            f"{r['brier_after']:.4f} after.\n\n"
            "Projection bias added to mu (actual minus mu, shrunk by n/(n+50)). "
            "A stat keeps it only if it lowers that stat's Brier:\n\n"
            + "\n".join(
                f"- {k}: candidate {r['mu_bias_candidate'].get(k, 0):+.1f} (n {r['mu_bias_n'].get(k, 0)}), "
                + ("kept" if i["kept"] else "dropped")
                + (f", Brier {i['brier_before']:.4f} to {i['brier_after']:.4f}"
                   if i["brier_before"] is not None else ", no graded legs")
                for k, i in sorted(r["mu_bias_kept"].items())) + "\n\n"
            f"Edge-chosen rung legs left out of the fit: {r.get('chosen_excluded', 0)}.\n")


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
    from nflprops import model_quality as MQ
    mq = MQ.collect(season, [w for w in weeks if w <= week])
    out = render(season, week, df, accuracy(season, week), season_df, mq)
    out += "\n" + rung_section(df, season_df)
    out += "\n" + adjustment_section(write=not a.no_write)
    if not a.no_write:
        MQ.write_csv(season, weeks, MQ.collect(season, weeks))
    if a.no_write:
        print(out)
    else:
        VAULT.mkdir(parents=True, exist_ok=True)
        p = VAULT / f"{season}-w{week}-week.md"
        p.write_text(out)
        print(f"wrote {p}")
