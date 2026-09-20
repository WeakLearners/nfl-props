#!/usr/bin/env python
"""Grade a finished game against the report that was published before it.

Measurement only. This script never changes a model, a weight, or a screen --
one game is an anecdote, and the Week 1 MNF grade is on file as the reason that
rule exists. It reads the published report, pulls what actually happened, and
writes the comparison down.

usage:
  grade_game.py                     # yesterday's games, refresh data first
  grade_game.py --season 2026 --week 2
  grade_game.py --no-refresh --no-write
"""
import sys, re, json, argparse, pathlib, datetime as dt

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import pandas as pd

from nflprops.config import CURRENT_SEASON
from nflprops.db import connect
from nflprops.odds import name_key

VAULT = pathlib.Path("/Users/Sean/second-brain/projects/nfl-props/grades")
BLOWOUT = 17          # margin at which game script, not the model, owns the result

# A write-up is finished only once snap counts exist for its game. Until then it
# is a scoreboard, not an explanation: a miss could be a benched player. The
# runner keys off these exact strings, so a provisional file is redone and a
# final one is left alone.
STATUS_FINAL = "final."
STATUS_PROVISIONAL = ("provisional — graded before snap counts were published. "
                      "Will be re-graded automatically.")
STAT_LABEL = {"passing_yards": "passing yards", "rushing_yards": "rushing yards",
              "receiving_yards": "receiving yards", "receptions": "receptions",
              "completions": "completions"}


def american(p):
    return f"{int(p):+d}" if p is not None else "—"


def load_report(path):
    h = path.read_text()
    data = json.loads(re.search(r"const DATA\s*=\s*(\[.*?\]);", h, re.S).group(1))
    return data


def played_games(season, week):
    with connect() as con:
        return pd.read_sql_query(
            """SELECT game_id, away_team, home_team, away_score, home_score, gameday
               FROM schedules WHERE season=? AND week=? AND away_score IS NOT NULL""",
            con, params=[season, week])


def actuals(season, week):
    with connect() as con:
        a = pd.read_sql_query(
            """SELECT player_display_name, team, position, completions, attempts,
                      passing_yards, carries, rushing_yards, receptions, targets,
                      receiving_yards
               FROM player_games WHERE season=? AND week=? AND season_type='REG'""",
            con, params=[season, week])
    a["key"] = a.player_display_name.map(name_key)
    return a


def snaps(season, week, teams=()):
    """Snap share per player, plus whether the data exists yet.

    Snap counts come from Pro Football Reference, not the live feed, so they
    land a day or more after the box score. An empty result and a genuine 0%
    look identical once they are a dict, and the difference decides whether
    this write-up can claim it checked availability. So the caller is told.
    """
    with connect() as con:
        s = pd.read_sql_query(
            "SELECT player, team, offense_pct FROM snap_counts WHERE season=? AND week=?",
            con, params=[season, week])
    if teams:
        s = s[s.team.isin(list(teams))]
    s["key"] = s.player.map(name_key)
    return s.set_index("key").offense_pct.to_dict(), not s.empty


def teams_of(report_path):
    """('CAR', 'ATL') from report_2026_w2_CAR-ATL.html."""
    return tuple(report_path.stem.split("_")[-1].split("-"))


def grade(season, week, report_path, write=True):
    data = load_report(report_path)
    act = actuals(season, week)
    snap, have_snaps = snaps(season, week, teams_of(report_path))
    if act.empty:
        return None
    sched = played_games(season, week)
    by_key = act.set_index("key")

    rows, flags = [], []
    for d in sorted([x for x in data if x.get("pick")], key=lambda x: x["pick"]):
        k = name_key(d["p"])
        stat = d["s"]
        a = by_key.loc[k, stat] if k in by_key.index else None
        if isinstance(a, pd.Series):
            a = a.iloc[0]
        line = d.get("line")
        hit = None if (a is None or line is None) else bool(a > line)
        note = ""
        if a is not None and line is not None:
            if a < d["lo"]:
                note = "below the model's own low bound"
            elif a > d["hi"]:
                note = "above the model's own high bound"
            else:
                note = "inside modeled range"
        pct = snap.get(k)
        if pct is not None and pct < 0.30:
            flags.append(f"{d['p']} played only {pct*100:.0f}% of snaps")
        rows.append({"n": d["pick"], "player": d["p"], "role": d["role"], "tm": d["tm"],
                     "stat": STAT_LABEL.get(stat, stat), "line": line, "price": d.get("price"),
                     "lo": d["lo"], "hi": d["hi"], "actual": a, "hit": hit, "note": note,
                     "conf": None if d.get("mp") is None or d.get("bp") is None
                             else min(d["mp"], d["bp"])})

    # whole-game accuracy, every projected player, not only the shortlist
    allrows = []
    for d in data:
        k = name_key(d["p"])
        if k not in by_key.index:
            continue
        a = by_key.loc[k, d["s"]]
        if isinstance(a, pd.Series):
            a = a.iloc[0]
        if pd.notna(a):
            allrows.append({"stat": d["s"], "err": abs(d["mu"] - float(a))})
    acc = pd.DataFrame(allrows).groupby("stat").err.agg(["mean", "count"]) if allrows else pd.DataFrame()

    away, home = teams_of(report_path)
    g = sched[(sched.away_team == away) & (sched.home_team == home)]
    if g.empty:
        # No fallback. The old code fell back to the whole week's schedule, so a
        # write-up for one game printed all sixteen final scores under its own
        # title. An unfinished game is not gradable; say so and write nothing.
        print(f"{season} W{week} {away} @ {home}: not final yet, skipping")
        return None
    out = render(season, week, report_path, rows, acc, g, flags, have_snaps)
    if write:
        VAULT.mkdir(parents=True, exist_ok=True)
        p = VAULT / f"{season}-w{week}-{report_path.stem.split('_')[-1]}.md"
        # Don't rewrite an identical file. This runs every half hour during the
        # season, and the vault auto-commits when it goes idle -- a no-op write
        # would spend the day producing commits that change nothing.
        if not p.exists() or p.read_text() != out:
            p.write_text(out)
            print(f"wrote {p}")
    return out


def render(season, week, report_path, rows, acc, sched, flags, have_snaps=True):
    L = []
    title = report_path.stem.split("_")[-1].replace("-", " @ ")
    L.append(f"# Week {week} grade — {title} ({dt.date.today()})\n")
    for r in sched.itertuples():
        margin = abs((r.away_score or 0) - (r.home_score or 0))
        L.append(f"**Final:** {r.away_team} {int(r.away_score)}, {r.home_team} {int(r.home_score)}"
                 f"{'  — a blowout, margin ' + str(int(margin)) if margin >= BLOWOUT else ''}\n")
    L.append(f"**Source:** `{report_path.name}`, published before kickoff. "
             "Outcomes from `player_games`.\n")
    L.append(f"**Status:** {STATUS_FINAL if have_snaps else STATUS_PROVISIONAL}\n")
    L.append("This is a measurement. No model, weight, or screen was changed to produce it.\n")

    L.append("## The legs\n")
    L.append("| # | Player | Stat | Line | Price | Model ±1sd | Actual | Result |")
    L.append("|---|---|---|---|---|---|---|---|")
    hits = graded = 0
    for r in rows:
        if r["hit"] is not None:
            graded += 1
            hits += int(r["hit"])
        res = "—" if r["hit"] is None else ("**HIT**" if r["hit"] else "**MISS**")
        if r["note"] and r["hit"] is False:
            res += f" — {r['note']}"
        act = "—" if r["actual"] is None or pd.isna(r["actual"]) else f"{float(r['actual']):g}"
        L.append(f"| {r['n']} | {r['player']} ({r['tm']} {r['role']}) | {r['stat']} | "
                 f"over {r['line']} | {american(r['price'])} | {r['lo']:.1f} – {r['hi']:.1f} | "
                 f"{act} | {res} |")
    L.append("")
    if graded:
        cs = [r["conf"] for r in rows if r["conf"]]
        claim = f"{min(cs)*100:.0f}–{max(cs)*100:.0f}%" if cs else "n/a"
        L.append(f"**{hits} of {graded} hit.** The report's own priced confidence on these "
                 f"ran {claim} per leg.\n")

    below = [r for r in rows if r["note"] == "below the model's own low bound"]
    L.append("## Automated checks\n")
    L.append(f"- Legs landing below the model's own low bound: **{len(below)} of {graded}**."
             + (" That many is the signature of one shared shock, not independent misses."
                if len(below) >= 3 else ""))
    for r in sched.itertuples():
        if abs((r.away_score or 0) - (r.home_score or 0)) >= BLOWOUT:
            L.append(f"- Margin was {int(abs(r.away_score - r.home_score))}. A blowout moves every "
                     "counting stat in both directions at once. The model projects each player "
                     "alone and has no way to see it coming.")
    for f in flags:
        L.append(f"- Snap-share flag: {f}")
    if not have_snaps:
        L.append("- **Snap share was not checked.** Pro Football Reference had not "
                 "published snap counts for this game when this ran. A missed leg "
                 "here could still be a benched player rather than a bad "
                 "projection. Re-graded automatically once the counts land.")
    elif not flags:
        L.append("- No shortlisted player fell below 30% of snaps.")
    L.append("")

    if not acc.empty:
        L.append("## Whole-game accuracy, every projected player\n")
        L.append("| Stat | Players | Average miss |")
        L.append("|---|---|---|")
        for stat, r in acc.iterrows():
            L.append(f"| {STAT_LABEL.get(stat, stat)} | {int(r['count'])} | {r['mean']:.1f} |")
        L.append("\nThe shortlist is six legs. This table is the honest sample.\n")

    L.append("## Open question for the next session\n")
    L.append("One game decides nothing. Add this to the running tally and look at the shape "
             "across games, not at this one.\n")
    return "\n".join(L)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int)
    ap.add_argument("--week", type=int)
    ap.add_argument("--no-refresh", action="store_true")
    ap.add_argument("--force", action="store_true",
                    help="re-grade even write-ups already marked final")
    ap.add_argument("--no-write", action="store_true")
    a = ap.parse_args()

    if not a.no_refresh:
        from nflprops import ingest
        ingest.build([a.season or CURRENT_SEASON])

    season = a.season or CURRENT_SEASON
    if a.week:
        weeks = [a.week]
    else:
        # Look back several days, not just yesterday. Box scores sometimes land
        # late, and a job that only ever checks yesterday would miss that game
        # forever. Anything already graded is skipped, so re-running is free.
        days = [(dt.date.today() - dt.timedelta(days=d)).isoformat() for d in range(1, 5)]
        with connect() as con:
            w = pd.read_sql_query(
                f"""SELECT DISTINCT week FROM schedules
                    WHERE season=? AND gameday IN ({','.join('?' * len(days))})
                      AND away_score IS NOT NULL""",
                con, params=[season] + days)
        weeks = sorted(int(x) for x in w.week)
        if not weeks:
            print(f"no finished games in the last 4 days; nothing to grade")
            sys.exit(0)

    any_done = False
    for week in weeks:
        if actuals(season, week).empty:
            print(f"{season} W{week}: box scores not in the database yet")
            continue
        final = {(r.away_team, r.home_team) for r in played_games(season, week).itertuples()}
        for rp in sorted((ROOT / "reports").glob(f"report_{season}_w{week}_*.html")):
            if teams_of(rp) not in final:
                continue          # still to be played, or still being played
            dest = VAULT / f"{season}-w{week}-{rp.stem.split('_')[-1]}.md"
            # Skip only what is FINISHED, not merely what exists. The old rule
            # was "the file is there, move on", which meant a Sunday-afternoon
            # pass locked in a write-up that had no snap data and no other game
            # in it, and the later pass skipped right over it. Re-running is
            # cheap; a permanently half-graded week is not.
            if dest.exists() and not a.force and not a.no_write:
                if f"**Status:** {STATUS_FINAL}" in dest.read_text():
                    continue
            out = grade(season, week, rp, write=not a.no_write)
            if out:
                any_done = True
                if a.no_write:
                    print(out)
    sys.exit(0 if any_done else 3)
