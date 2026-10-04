"""Trends & Insights page. Display only. Entertainment, no edge claim.

Each trend row gets an outlook: how often a change of that type and size
carried into the next 3 games in 2021-2025. Rates are fitted on those seasons
only and stored in data/trends/holdup_rates.json. The 2026 holdout is never
read for a rate, and nothing here feeds an engine evaluation. Rules for the
buckets, the "held" test and the label thresholds are fixed in
projects/nfl-props/trends-insights-plan-2026-10-04.md section 12.1.
"""
import html
import json

import numpy as np
import pandas as pd

from .config import DATA_DIR, ROOT

HIST_SEASONS = range(2021, 2026)
RATES_PATH = DATA_DIR / "trends" / "holdup_rates.json"
HOLD_CARRY = 0.5                    # a change "held" if >= half of it carried
HOLD_AT, FADE_AT, MIN_N = 0.60, 0.40, 50
B_ANCHORS = (6, 9, 12, 15)          # type B: games played at the anchor
MIN_PRIOR_GAMES = 8                 # type A needs this many games last season

# metric: (positions, volume column, volume floor, unit, buckets (lo, hi) in unit, title)
METRICS = {
    "target_share": (("WR", "TE"), "targets", 3, "pts", ((2, 5), (5, 10), (10, 999)), "Target share"),
    "carry_share": (("RB",), "carries", 5, "pts", ((2, 5), (5, 10), (10, 999)), "Carry share"),
    "snap_share": (("RB", "WR", "TE"), "snap_pct", 25, "pts", ((2, 5), (5, 10), (10, 999)), "Offense snap share"),
    "rz10": (("RB", "WR", "TE"), None, 0, "per game", ((.25, .5), (.5, 1), (1, 999)), "Inside-10 touches"),
    "pass_rate": (("TEAM",), None, 0, "pts", ((3, 6), (6, 10), (10, 999)), "Team pass rate"),
}


def bucket(metric, delta):
    """Bucket label for |delta| (in the metric's unit), or None below the smallest."""
    a = abs(delta)
    for lo, hi in METRICS[metric][4]:
        if lo <= a < hi:
            return f"{lo:g}+" if hi == 999 else f"{lo:g}-{hi:g}"
    return None


def label(rate):
    """rate: dict with n and hold_share, or None. Returns (label, text). Never hides a row."""
    if not rate or rate["n"] < MIN_N:
        n = rate["n"] if rate else 0
        return "Watch", f"Watch (history too thin, n={n})"
    s, n = rate["hold_share"], rate["n"]
    lab = "Likely to hold" if s >= HOLD_AT else "Likely to fade" if s <= FADE_AT else "Watch"
    mg = f", merged: {rate['merge']}" if rate.get("merge") else ""
    return lab, f"{lab} ({s*100:.1f}%, n={n}{mg})"


# ---- entity frames ---------------------------------------------------------
def entity_frames(pg, rz=None):
    """{metric: DataFrame(entity, name, team, position, season, week, value, vol)}.
    value is in the metric's unit (points for shares). pg: player_game_shares()."""
    pg = pg.copy()
    tc = pg.groupby(["game_id", "team"]).carries.transform("sum")
    pg["carry_share"] = np.where(tc > 0, pg.carries / tc.where(tc > 0), np.nan) * 100
    pg["target_share"] = pg.target_share * 100
    pg["snap_pct"] = pg.snap_share * 100
    pg["snap_share"] = pg.snap_pct
    if rz is not None:
        pg = pg.merge(rz, on=["game_id", "player_id"], how="left")
    for c in ("rz10_carries", "rz10_targets"):
        pg[c] = pg[c].fillna(0) if c in pg else 0.0
    pg["rz10"] = np.where(pg.position == "RB", pg.rz10_carries, pg.rz10_targets)
    out = {}
    for m, (pos, vol, _, _, _, _) in METRICS.items():
        if m == "pass_rate":
            continue
        d = pg[pg.position.isin(pos)].dropna(subset=[m]).copy()
        d["value"], d["vol"] = d[m], d[vol] if vol else 0.0
        out[m] = d.rename(columns={"player_id": "entity", "player_display_name": "name"})[
            ["entity", "name", "team", "position", "season", "week", "value", "vol", "targets", "carries"]]
    t = pg.groupby(["game_id", "team", "season", "week"], as_index=False)[["attempts", "carries"]].sum()
    t["value"] = 100 * t.attempts / (t.attempts + t.carries)
    t = t.dropna(subset=["value"])
    t = t.assign(entity=t.team, name=t.team, position="TEAM", vol=0.0, targets=0, carries=0)
    out["pass_rate"] = t[["entity", "name", "team", "position", "season", "week", "value", "vol", "targets", "carries"]]
    return out


def _windows(g, season):
    """g: one entity's rows (sorted). Yields (type, anchor, last3, base, next3) values for history."""
    cur = g[g.season == season].sort_values("week")
    prior = g[g.season == season - 1]
    v = cur.value.values
    if len(v) >= 6 and len(prior) >= MIN_PRIOR_GAMES:
        yield "A", 3, cur.iloc[:3], prior.value.mean(), v[3:6].mean()
    for a in B_ANCHORS:
        if len(v) >= a + 3:
            yield "B", a, cur.iloc[a - 3:a], v[:a - 3].mean(), v[a:a + 3].mean()


def episodes(frame, metric, seasons=HIST_SEASONS):
    """History cases: one per entity, season and anchor. Columns: pos, type, delta, carry."""
    floor = METRICS[metric][2]
    rows = []
    for _, g in frame.groupby("entity"):
        pos = g.position.iloc[0]
        for s in seasons:
            for typ, a, last, base, nxt in _windows(g, s):
                l3 = last.value.mean()
                if last.vol.mean() < floor or l3 == base:
                    continue
                rows.append((pos, typ, l3 - base, (nxt - base) / (l3 - base)))
    return pd.DataFrame(rows, columns=["pos", "type", "delta", "carry"])


def holdup_rates(frames, seasons=HIST_SEASONS):
    """{key: {n, held, hold_share}} with key 'metric|pos|type|rise/fall|bucket'."""
    rates = {}
    for m, fr in frames.items():
        ep = episodes(fr[fr.season.isin(list(seasons) + [min(seasons) - 1])], m, seasons)
        for r in ep.itertuples():
            b = bucket(m, r.delta)
            if b is None:
                continue
            k = rate_key(m, r.pos, r.type, r.delta, b)
            d = rates.setdefault(k, {"n": 0, "held": 0})
            d["n"] += 1
            d["held"] += int(r.carry >= HOLD_CARRY)
    for d in rates.values():
        d["hold_share"] = round(d["held"] / d["n"], 3)
    return rates


def _bucket_names(metric):
    return [f"{lo:g}+" if hi == 999 else f"{lo:g}-{hi:g}" for lo, hi in METRICS[metric][4]]


def _pool(rates, keys):
    n = sum(rates.get(k, {}).get("n", 0) for k in keys)
    held = sum(rates.get(k, {}).get("held", 0) for k in keys)
    return n, held


def merge_candidates(metric, pos, typ, direction, b):
    """Ordered merge options (list of key lists) for one thin bucket. Same stat, position
    and direction always. Size steps keep the type. Game-number steps pool types A and B."""
    names = _bucket_names(metric)
    i = names.index(b)
    k = lambda t, nm: f"{metric}|{pos}|{t}|{direction}|{nm}"
    pairs = [sorted((i, j)) for j in (i - 1, i + 1) if 0 <= j < len(names)]
    out = []
    for types in ([typ], ["A", "B"]):
        if len(types) == 2:
            out.append([[k(t, names[i]) for t in types]])
        out.append([[k(t, names[x]) for t in types for x in pr] for pr in pairs])
        out.append([[k(t, nm) for t in types for nm in names]])
    return [c for c in out if c and c[0]]


def merged_rate(rates, metric, pos, typ, direction, b):
    """Smallest merge with n >= MIN_N, or None. Within a step, the larger n wins."""
    for step in merge_candidates(metric, pos, typ, direction, b):
        best = max(((_pool(rates, ks), ks) for ks in step), key=lambda x: x[0][0])
        (n, held), ks = best
        if n >= MIN_N:
            return {"n": n, "held": held, "hold_share": round(held / n, 3), "sources": ks,
                    "merge": merge_text(ks, direction)}
    return None


def merge_text(keys, direction):
    parts = [k.split("|") for k in keys]
    types = sorted({p[2] for p in parts})
    sizes = []
    for p in parts:
        if p[4] not in sizes:
            sizes.append(p[4])
    s = " + ".join(sizes) if len(sizes) > 1 else sizes[0]
    return f"{direction} {s}" + (", games 3-5 + 6+" if len(types) > 1 else "")


def all_merges(rates):
    out = {}
    for m, (pos_l, *_rest) in METRICS.items():
        for pos in pos_l:
            for typ in "AB":
                for d in ("rise", "fall"):
                    for b in _bucket_names(m):
                        k = f"{m}|{pos}|{typ}|{d}|{b}"
                        if rates.get(k, {}).get("n", 0) >= MIN_N:
                            continue
                        r = merged_rate(rates, m, pos, typ, d, b)
                        if r:
                            out[k] = r
    return out


def rate_key(metric, pos, typ, delta, b):
    return f"{metric}|{pos}|{typ}|{'rise' if delta > 0 else 'fall'}|{b}"


def fit_and_save(frames):
    rates = holdup_rates(frames)
    RATES_PATH.parent.mkdir(parents=True, exist_ok=True)
    RATES_PATH.write_text(json.dumps({
        "version": 1, "seasons": "2021-2025", "held_if_carry_at_least": HOLD_CARRY,
        "thresholds": {"hold": HOLD_AT, "fade": FADE_AT, "min_n": MIN_N},
        "rates": dict(sorted(rates.items())),
        "merged": dict(sorted(all_merges(rates).items()))}, indent=1))
    return rates


def load_rates():
    """Effective rates: a thin bucket is replaced by its stored merge, if one exists."""
    if not RATES_PATH.exists():
        return {}
    d = json.loads(RATES_PATH.read_text())
    rates = dict(d["rates"])
    for k, m in d.get("merged", {}).items():
        rates[k] = m
    return rates


# ---- live rows -------------------------------------------------------------
def live_rows(frame, metric, season, week, rates):
    """Trend rows for the current season through `week`."""
    floor = METRICS[metric][2]
    rows = []
    for ent, g in frame.groupby("entity"):
        cur = g[(g.season == season) & (g.week <= week)].sort_values("week")
        prior = g[g.season == season - 1]
        n = len(cur)
        if n < 3 or g.iloc[-1].season != season:
            continue
        last = cur.iloc[-3:]
        if n >= 6:
            typ, base, bl = "B", cur.value.iloc[:-3].mean(), "earlier 2026"
        elif len(prior) >= MIN_PRIOR_GAMES:
            typ, base, bl = "A", prior.value.mean(), "2025"
        else:
            continue
        l3 = last.value.mean()
        b = bucket(metric, l3 - base)
        if last.vol.mean() < floor or b is None:
            continue
        pos = g.position.iloc[-1]
        lab, text = label(rates.get(rate_key(metric, pos, typ, l3 - base, b)))
        rows.append(dict(entity=ent, name=g.name.iloc[-1], team=g.team.iloc[-1], pos=pos,
                         last3=round(l3, 2), base=round(base, 2), baseline=bl,
                         prior=round(prior.value.mean(), 2) if len(prior) else None,
                         delta=round(l3 - base, 2), games=3, vol=int(last.vol.sum()),
                         label=lab, outlook=text))
    return sorted(rows, key=lambda r: -abs(r["delta"]))


def next_games(season):
    """{team: (opponent, 'vs'|'at', week)} for each team's first unplayed game."""
    from .db import connect
    with connect() as con:
        s = pd.read_sql_query(
            "SELECT week, home_team, away_team FROM schedules WHERE season=? AND game_type='REG' "
            "AND home_score IS NULL ORDER BY week", con, params=[season])
    out = {}
    for r in s.itertuples():
        out.setdefault(r.home_team, (r.away_team, "vs", r.week))
        out.setdefault(r.away_team, (r.home_team, "at", r.week))
    return out


DVP_STAT = {"target_share": {"WR": "targets", "TE": "targets"}, "carry_share": {"RB": "carries"},
            "snap_share": {"RB": "rushing_yards", "WR": "receiving_yards", "TE": "receiving_yards"},
            "rz10": {"RB": "rushing_tds", "WR": "receiving_tds", "TE": "receiving_tds"}}


def dvp_mult(dvp, opp, pos, stat):
    r = dvp[(dvp.defteam == opp) & (dvp.position == pos)]
    return float(r[stat].iloc[0]) if len(r) and stat in r else None


def _ppg(season, week):
    from .db import connect
    with connect() as con:
        s = pd.read_sql_query(
            "SELECT week, home_team, away_team, home_score, away_score FROM schedules "
            "WHERE season=? AND game_type='REG' AND week<=? AND home_score IS NOT NULL",
            con, params=[season, week])
    h = s.rename(columns={"home_team": "team", "home_score": "pts"})[["team", "week", "pts"]]
    a = s.rename(columns={"away_team": "team", "away_score": "pts"})[["team", "week", "pts"]]
    return pd.concat([h, a]).sort_values(["team", "week"])


def build_data(season=None, week=None):
    """All sections as plain dicts."""
    from .rating import current_ratings, latest_completed_week
    from .shares import player_game_shares
    from .db import connect
    from .features import defense_vs_position
    if season is None:
        season, week = latest_completed_week()
    pg = player_game_shares(2020)
    with connect() as con:
        rz = pd.read_sql_query(
            "SELECT game_id, player_id, SUM(rz10_carries) rz10_carries, SUM(rz10_targets) rz10_targets "
            "FROM rz_usage GROUP BY game_id, player_id", con)
    frames = entity_frames(pg, rz)
    rates = load_rates()
    nxt = next_games(season)
    dvp = defense_vs_position(season, week + 1)
    sec = {}
    for m in METRICS:
        rows = live_rows(frames[m], m, season, week, rates)
        for r in rows:
            o = nxt.get(r["team"] if r["pos"] != "TEAM" else r["entity"])
            r["next"] = f"{o[1]} {o[0]}" if o else "bye or none"
            mult = dvp_mult(dvp, o[0], r["pos"], DVP_STAT[m][r["pos"]]) if o and m in DVP_STAT else None
            r["dvp"] = round(mult, 2) if mult else None
        sec[m] = rows
    # rank movers, week over week
    _, _, now = current_ratings(pg, season, week)
    _, _, prev = current_ratings(pg, season, week - 1)
    pr = prev.set_index("player_id")["rank"]
    mv = now.assign(prev_rank=now.player_id.map(pr))
    mv = mv.dropna(subset=["prev_rank"])
    mv["move"] = (mv.prev_rank - mv["rank"]).astype(int)
    sec["rank"] = [dict(name=r["name"], team=r.team, pos=r.position, rank=int(r["rank"]),
                        prev=int(r.prev_rank), move=int(r.move), games=int(r.n_games),
                        next=("%s %s" % nxt[r.team][1::-1]) if r.team in nxt else "bye or none")
                   for _, r in mv[mv.move != 0].iterrows()]
    sec["rank"].sort(key=lambda r: -abs(r["move"]))
    # defense versus position, week over week
    dprev = defense_vs_position(season, week)
    stat = {"QB": "passing_yards", "RB": "rushing_yards", "WR": "receiving_yards", "TE": "receiving_yards"}
    sec["dvp"] = {}
    for pos, st in stat.items():
        a = dvp[dvp.position == pos].set_index("defteam")
        b = dprev[dprev.position == pos].set_index("defteam")[st]
        sec["dvp"][pos] = sorted(
            [dict(team=t, stat=st, now=round(float(a.at[t, st]), 2), prev=round(float(b.get(t, np.nan)), 2),
                  games=int(a.at[t, "games"])) for t in a.index],
            key=lambda r: -r["now"])
    # team scoring
    pp = _ppg(season, week)
    sec["scoring"] = {t: dict(last3=round(float(g.pts.iloc[-3:].mean()), 1), season=round(float(g.pts.mean()), 1),
                              games=len(g)) for t, g in pp.groupby("team") if len(g) >= 3}
    return dict(season=season, week=week, sections=sec)


def _e(x):
    return html.escape(str(x))


def _row(r, unit, ctx):
    sign = "+" if r["delta"] > 0 else ""
    d = f"{sign}{r['delta']:.1f} {unit}" if unit == "pts" else f"{sign}{r['delta']:.2f} per game"
    fm = (lambda x: f"{x:.1f}%") if unit == "pts" else (lambda x: f"{x:.2f}")
    pr = f" · 2025: {fm(r['prior'])}" if r.get("prior") is not None else ""
    dv = f" · defense factor {r['dvp']:.2f}" if r.get("dvp") else ""
    return (f'<li class="trow" data-pos="{r["pos"]}" data-inspect-id="trends-row">'
            f'<div class="t1"><span class="nm">{_e(r["name"])}</span> <span class="tm">{_e(r["team"])} {r["pos"]}</span>'
            f'<span class="dl">{d}</span></div>'
            f'<div class="t2">{fm(r["last3"])} last 3 vs {fm(r["base"])} {r["baseline"]}{pr}</div>'
            f'<div class="t2">3-game change, small sample · {r["vol"]} {ctx} in 3 games</div>'
            f'<div class="t2 ol ol-{r["label"].split()[-1].lower()}" data-inspect-id="trends-outlook">'
            f'Outlook: {_e(r["outlook"])} · next: {_e(r["next"])}{dv}</div></li>')


def _usage_card(data, metric, ctx, sid):
    rows = data["sections"][metric]
    unit = METRICS[metric][3]
    parts = []
    for pos in METRICS[metric][0]:
        pr = [r for r in rows if r["pos"] == pos]
        up, dn = [r for r in pr if r["delta"] > 0][:6], [r for r in pr if r["delta"] < 0][:6]
        parts.append(f'<h4 data-pos="{pos}" data-inspect-id="trends-{sid}-{pos.lower()}">{pos}: risers</h4><ul data-pos="{pos}">'
                     + ("".join(_row(r, unit, ctx) for r in up) or '<li class="none">None at this size.</li>')
                     + f'</ul><h4 data-pos="{pos}">{pos}: fallers</h4><ul data-pos="{pos}">'
                     + ("".join(_row(r, unit, ctx) for r in dn) or '<li class="none">None at this size.</li>') + "</ul>")
    return "".join(parts)


def render(data):
    s, w, sec = data["season"], data["week"], data["sections"]
    cards = []

    def card(cid, title, note, body):
        cards.append(f'<details class="card" data-inspect-id="trends-card-{cid}"><summary>{title}</summary>'
                     f'<p class="note">{note}</p>{body}</details>')

    base_note = ("Change = last 3 games against the baseline shown. Outlook = how often a change of this type and "
                 "size carried into the next 3 games in 2021-2025.")
    card("usage", "1 Usage risers and fallers", base_note + " Target share for WR and TE, carry share for RB.",
         _usage_card(data, "target_share", "targets", "usage-t") + _usage_card(data, "carry_share", "carries", "usage-c"))
    card("snap", "2 Snap share movers", base_note, _usage_card(data, "snap_share", "snap pts", "snap"))
    rk = []
    for pos in ("QB", "RB", "WR", "TE"):
        pr = [r for r in sec["rank"] if r["pos"] == pos]
        items = "".join(
            f'<li class="trow" data-pos="{pos}" data-inspect-id="trends-rank-row"><div class="t1"><span class="nm">{_e(r["name"])}</span> '
            f'<span class="tm">{_e(r["team"])} {pos}</span><span class="dl">{"+" if r["move"] > 0 else ""}{r["move"]}</span></div>'
            f'<div class="t2">#{r["prev"]} to #{r["rank"]} · {r["games"]} games · next: {_e(r["next"])}</div>'
            f'<div class="t2 ol">Outlook: Watch (no history rate for rank moves)</div></li>'
            for r in (pr[:6] if pr else []))
        rk.append(f'<h4 data-pos="{pos}" data-inspect-id="trends-rank-{pos.lower()}">{pos}</h4><ul data-pos="{pos}">{items or "<li class=none>No moves.</li>"}</ul>')
    card("rank", "3 Rank movers",
         'Week-over-week change in the position rank. Full lists: <a href="/rankings.html" data-inspect-id="trends-rankings-link">Player rankings</a>. '
         "The rank is recent production. It is not the depth chart.", "".join(rk))
    card("rz", "4 Inside-10 role changes",
         base_note + " RB: carries inside the 10. WR and TE: targets inside the 10. Apart from yardage.",
         _usage_card(data, "rz10", "inside-10 touches", "rz"))
    dv = []
    for pos, rows in sec["dvp"].items():
        trs = "".join(f'<tr data-pos="{pos}"><td>{r["team"]}</td><td class="num">{r["now"]:.2f}</td>'
                      f'<td class="num">{r["prev"]:.2f}</td><td class="num">{r["games"]}</td></tr>' for r in rows[:5] + rows[-5:])
        dv.append(f'<h4 data-pos="{pos}" data-inspect-id="trends-dvp-{pos.lower()}">{pos}: {rows[0]["stat"].replace("_", " ")} allowed (5 easiest, 5 hardest)</h4>'
                  f'<div class="tablewrap" data-pos="{pos}"><table><thead><tr><th>Defense</th><th class="num">Now</th>'
                  f'<th class="num">Last wk</th><th class="num">Games</th></tr></thead><tbody>{trs}</tbody></table></div>')
    card("dvp", "5 Defense versus position",
         "Factor 1.00 = league average. Above 1.00 allows more. Shrunk toward 1.00 for few games. Games = two-season window. "
         "Outlook: Watch (no history rate for this type).", "".join(dv))
    tr = [r for r in sec["pass_rate"]]
    sc = sec["scoring"]
    items = []
    for r in tr[:8]:
        c = sc.get(r["team"])
        pts = f" · points per game {c['last3']} last 3 vs {c['season']} season" if c else ""
        sign = "+" if r["delta"] > 0 else ""
        items.append(f'<li class="trow" data-inspect-id="trends-team-row"><div class="t1"><span class="nm">{_e(r["team"])}</span>'
                     f'<span class="dl">{sign}{r["delta"]:.1f} pts</span></div>'
                     f'<div class="t2">Pass share {r["last3"]:.1f}% last 3 vs {r["base"]:.1f}% {r["baseline"]}{pts}</div>'
                     f'<div class="t2">3-game change, small sample</div>'
                     f'<div class="t2 ol ol-{r["label"].split()[-1].lower()}" data-inspect-id="trends-team-outlook">Outlook: {_e(r["outlook"])} · next: {_e(r["next"])}</div></li>')
    card("team", "6 Team pass rate and scoring",
         "Pass share = pass attempts divided by attempts plus carries. Points per game is context. It has no history rate.",
         f'<ul>{"".join(items) or "<li class=none>No team moved 3 points or more.</li>"}</ul>')
    tpl = (ROOT / "templates" / "trends.html").read_text()
    return (tpl.replace("__WEEK__", f"{s} Week {w}")
               .replace("__BUILT__", pd.Timestamp.now().strftime("%a %Y-%m-%d %H:%M"))
               .replace("__CARDS__", "\n".join(cards)))


def write_trends():
    """Write reports/trends.html and data/trends/{season}_w{week}.json. Called with write_rankings()."""
    data = build_data()
    p = DATA_DIR / "trends" / f"{data['season']}_w{data['week']}.json"
    p.write_text(json.dumps(data, indent=1, default=str))
    out = ROOT / "reports" / "trends.html"
    out.write_text(render(data), encoding="utf-8")
    return out
