"""Data for the Game Lines page. Read-only: frozen predictions + live lines and results.

Predictions come from data/gamelines/frozen/ (written once per week by
scripts/game_lines_week.py). Lines and scores come from the schedules table at
request time, so a finished game grades itself with no rebuild.
"""
import json
import math
import sqlite3

from .config import DATA_DIR, DB_PATH

from .gamelines import MODEL_VERSION

FROZEN = DATA_DIR / "gamelines" / "frozen" / MODEL_VERSION
BACKTEST = DATA_DIR / "gamelines" / "backtest_2026.json"


def _clean(v):
    return None if v is None or (isinstance(v, float) and math.isnan(v)) else v


def lines_blob(season=2026):
    weeks = {}
    try:
        db = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
        db.row_factory = sqlite3.Row
        sched = {r["game_id"]: dict(r) for r in db.execute(
            "SELECT game_id, spread_line, total_line, home_moneyline, away_moneyline, "
            "home_score, away_score FROM schedules WHERE season=? AND game_type='REG'", (season,))}
        db.close()
    except sqlite3.Error:
        sched = {}
    wx = {}
    try:
        from .weather import load as wload
        for r in wload().itertuples():
            wx[r.game_id] = {
                "indoors": not bool(r.applies),
                "roof_unknown": bool(r.roof_unknown),
                "kind": None if isinstance(r.kind, float) or r.kind is None else r.kind,
                "pulled_at": None if isinstance(r.pulled_at, float) else r.pulled_at,
                "temp_f": _clean(r.temp_f), "wind_mph": _clean(r.wind_mph), "gust_mph": _clean(r.gust_mph),
                "precip_mm": _clean(r.precip_mm), "precip_prob": _clean(r.precip_prob)}
    except Exception:  # noqa: BLE001 -- the page works without weather
        wx = {}
    for f in sorted(FROZEN.glob(f"{season}_w*.json"), key=lambda p: int(p.stem.split("_w")[1])):
        wk = int(f.stem.split("_w")[1])
        d = json.loads(f.read_text())
        games = []
        for r in d["games"]:
            s = {k: _clean(v) for k, v in sched.get(r["game_id"], {}).items()}
            games.append({**r, "wx": wx.get(r["game_id"]), **{k: s.get(k) for k in (
                "spread_line", "total_line", "home_moneyline", "away_moneyline",
                "home_score", "away_score")}})
        games.sort(key=lambda g: (g["gameday"] or "", g["gametime"] or ""))
        weeks[wk] = {"games": games, "sd_margin": d["sd_margin"], "sd_total": d["sd_total"],
                     "windows": d.get("windows"), "windows50": d.get("windows50"),
                     "target_record": d.get("target_record"), "target_display": d.get("target_display"),
                     "train_through": d["train_through"], "reconstructed": d["reconstructed"],
                     "frozen_at": d.get("frozen_at")}
    try:
        bt = json.loads(BACKTEST.read_text())
    except (FileNotFoundError, ValueError):
        bt = None
    cur = max(weeks) if weeks else None
    return {"season": season, "currentWeek": cur, "weeks": weeks, "backtest": bt}


# ---------------------------------------------------------------------------
# Render helpers for the game cards and game pages. Display only: every number
# comes from lines_blob() (frozen predictions + live lines), nothing is rewritten.
from html import escape as _e


def index_by_game(blob):
    """{(season, week, away, home): (game, week_meta)} -- the join key is the team pair."""
    out = {}
    for wk, w in blob["weeks"].items():
        for g in w["games"]:
            out[(blob["season"], wk, g["away"], g["home"])] = (g, w)
    return out


def _sp(team, m):
    if abs(m) < 0.05:
        return f"{team} PK"
    return f"{team} {'-' if m > 0 else '+'}{abs(m):.1f}"


def _rnd(x):
    return int(x + 0.5)


def _res(diff, lean):
    """+1 hit, -1 miss, 0 push for a result diff against the line and the model's lean."""
    return 0 if diff == 0 else (1 if (diff > 0) == (lean > 0) else -1)


def grade(g):
    """Lean, line and (after the game) results. Missing pieces stay None."""
    line, tl = g.get("spread_line"), g.get("total_line")
    side = None if line is None else g["pred_margin"] - line
    ts = None if tl is None else g["pred_total"] - tl
    played = g.get("home_score") is not None and g.get("away_score") is not None
    margin = g["home_score"] - g["away_score"] if played else None
    total = g["home_score"] + g["away_score"] if played else None
    sres = _res(margin - line, side) if played and side not in (None, 0) else None
    tres = _res(total - tl, ts) if played and ts not in (None, 0) else None
    mres = None
    if played and margin != 0 and g["p_home"] != 0.5:
        mres = 1 if (margin > 0) == (g["p_home"] > 0.5) else -1
    return {"line": line, "tl": tl, "side": side, "ts": ts, "played": played,
            "sres": sres, "tres": tres, "mres": mres}


def _pill(r, cls="score", na_text="N/A"):
    if r is None:
        return f'<span class="{cls} na">{na_text}</span>'
    if r == 0:
        return f'<span class="{cls} na">PUSH</span>'
    return f'<span class="{cls} {"good" if r > 0 else "bad"}">{"HIT" if r > 0 else "MISS"}</span>'


def card_html(g):
    """Card parts: (final-score line, GAME LINES block). Rows show the bet the pill grades."""
    r = grade(g)
    home, away = g["home"], g["away"]
    mdl_sp = _sp(home, g["pred_margin"])

    def row(label, value, rid, pill="", title="", dim=False):
        t = f' title="{_e(title, quote=True)}"' if title else ""
        c = " dim" if dim else ""
        return (f'<span class="hits-label">{label}</span>'
                f'<span class="gl-v{c}" data-inspect-id="{rid}"{t}>{value}{pill}</span>')

    def pill(res, show):
        return _pill(res) if show and res is not None else ""
    # Winner: the favoured team and its chance.
    fav, pf = (home, g["p_home"]) if g["p_home"] >= 0.5 else (away, 1 - g["p_home"])
    win = row("Winner", f"{fav} {pf * 100:.0f}%", "index-gl-win", pill(r["mres"], r["played"] and r["mres"] is not None),
              f"Model chance that {fav} wins.")
    # Spread: the line on the lean side, which is the bet the pill grades.
    if r["line"] is None:
        spread = row("Spread", "no line", "index-gl-spread", "", f"This game has no spread line. Model {mdl_sp}.", True)
    elif r["side"] == 0:
        spread = row("Spread", "no lean", "index-gl-spread", "", f"Model {mdl_sp}. Line {_sp(home, r['line'])}. The model does not lean.", True)
    else:
        if r["side"] > 0:
            bet, mdl = _sp(home, r["line"]), mdl_sp
        else:
            bet, mdl = _sp(away, -r["line"]), _sp(away, -g["pred_margin"])
        spread = row("Spread", bet, "index-gl-spread", pill(r["sres"], r["played"]),
                     f"Model {mdl}. Line {bet}.")
    if r["tl"] is None:
        total = row("Total", "no line", "index-gl-total", "", f"This game has no total line. Model {g['pred_total']:.1f}.", True)
    elif r["ts"] == 0:
        total = row("Total", "no lean", "index-gl-total", "", f"Model {g['pred_total']:.1f}. Line {r['tl']:.1f}. The model does not lean.", True)
    else:
        total = row("Total", ("Over " if r["ts"] > 0 else "Under ") + f"{r['tl']:.1f}", "index-gl-total",
                    pill(r["tres"], r["played"]), f"Model total {g['pred_total']:.1f}. Line {r['tl']:.1f}.")
    final = ""
    score = ""
    if r["played"]:
        final = (f'<span class="final" data-inspect-id="index-gl-score">Final '
                 f'{away} {int(g["away_score"])} {home} {int(g["home_score"])}</span>')
    else:
        score = row("Model score", f"{_rnd(g['mu_away'])}-{_rnd(g['mu_home'])}", "index-gl-score",
                    "", f"Model score: {away} {_rnd(g['mu_away'])}, {home} {_rnd(g['mu_home'])}.")
    block = ('<div class="sec" data-inspect-id="index-gl"><span>Game lines</span></div>'
             f'<div class="gl-rows">{win}{spread}{total}{score}</div>')
    return final, block


PAGE_CSS = """<style>
  .glp{margin:6px 0 10px}
  .glp .picks-note{margin-bottom:12px}
  .glp-grid{display:grid; grid-template-columns:repeat(auto-fill,minmax(220px,1fr)); gap:10px 28px}
  .glp-grid > div{display:flex; flex-direction:column; gap:2px}
  .glp-grid .sub2{font-family:"IBM Plex Mono",monospace; font-size:11px; color:var(--ink-3)}
  .glp .pf-v{display:flex; align-items:center; gap:8px}
  .glp .flag{font-family:"IBM Plex Mono",monospace; font-size:10px; font-weight:600; color:var(--ink-2);
    letter-spacing:.04em; margin-right:12px}
  .glp .pill{display:inline-flex; align-items:center; height:18px; padding:0 6px; border-radius:2px;
    font-family:"IBM Plex Mono",monospace; font-size:10px; font-weight:600; letter-spacing:.04em; color:var(--surface)}
  .glp .pill.good{background:var(--good)} .glp .pill.bad{background:var(--bad)}
  .glp .pill.push{background:none; color:var(--ink-3); border:1px dashed var(--line)}
</style>"""


def page_html(g, w):
    """Full-detail Game lines section for a game page."""
    r = grade(g)
    home, away = g["home"], g["away"]
    h = w.get("windows50") or None

    def pm(k):
        return f" &plusmn;{h[k]['half']:.1f}" if h else ""

    def pill(x):
        if x is None:
            return ""
        return '<span class="pill push">PUSH</span>' if x == 0 else \
            f'<span class="pill {"good" if x > 0 else "bad"}">{"HIT" if x > 0 else "MISS"}</span>'

    def cell(k, v, sub="", p=""):
        s = f'<span class="sub2">{sub}</span>' if sub else ""
        return f'<div><span class="pf-k">{k}</span><span class="pf-v">{v}{p}</span>{s}</div>'

    cells = [cell("Model score", f"{away} {_rnd(g['mu_away'])} &ndash; {home} {_rnd(g['mu_home'])}",
                  f"&plusmn;{h['away_score']['half']:.1f} / &plusmn;{h['home_score']['half']:.1f}" if h else "")]
    cells.append(cell("Model spread", _sp(home, g["pred_margin"]) + pm("margin"),
                      f"line {_sp(home, r['line'])}" if r["line"] is not None else "no line"))
    if r["line"] is None:
        pass
    elif r["side"] == 0:
        cells.append(cell("Spread lean", "no lean"))
    else:
        cells.append(cell("Spread lean", _sp(home, r["line"]) if r["side"] > 0 else _sp(away, -r["line"]),
                          "", pill(r["sres"])))
    cells.append(cell("Model total", f"{g['pred_total']:.1f}" + pm("total"),
                      f"line {r['tl']:.1f}" if r["tl"] is not None else "no line"))
    if r["tl"] is None:
        pass
    elif r["ts"] == 0:
        cells.append(cell("Total lean", "no lean"))
    else:
        cells.append(cell("Total lean", ("Over " if r["ts"] > 0 else "Under ") + f"{r['tl']:.1f}", "", pill(r["tres"])))
    hm, am = g.get("home_moneyline"), g.get("away_moneyline")
    mkt = ""
    if hm is not None and am is not None:
        imp = lambda o: -o / (-o + 100) if o < 0 else 100 / (o + 100)  # noqa: E731
        mkt = f" &middot; {imp(hm) / (imp(hm) + imp(am)) * 100:.0f}% market"
    cells.append(cell("Home win chance", f"{home} {g['p_home'] * 100:.0f}% model{mkt}", "", pill(r["mres"])))
    if r["played"]:
        cells.append(cell("Final", f"{away} {int(g['away_score'])} &ndash; {home} {int(g['home_score'])}"))
    flags = []
    if g.get("home_qb_out"):
        flags.append(f"{home} QB out")
    if g.get("away_qb_out"):
        flags.append(f"{away} QB out")
    if (g.get("wx_used") or {}).get("heavy_rain_flag"):
        flags.append("Heavy rain")
    fl = ('<div class="pf-wide">' + "".join(f'<span class="flag">{_e(x)}</span>' for x in flags) + "</div>") if flags else ""
    note = ("The model is fit again each week on earlier games" +
            (", rebuilt after the game from earlier games only" if w.get("reconstructed") else
             f", frozen {_e(str(w['frozen_at'])[:10])}" if w.get("frozen_at") else "") +
            ". " + ("&plusmn; is the 50% window, which lands inside about half the time. " if h else "") +
            'Season record on <a href="/explore#explore-gamelines">Results</a>.')
    return (PAGE_CSS + '<section class="glp" data-inspect-id="report-gamelines">'
            '<h2 class="picks-h">Game lines<span class="unit">entertainment only</span></h2>'
            f'<div class="glp-grid">{"".join(cells)}</div>{fl}'
            f'<p class="picks-note" style="margin-top:10px">{note}</p></section>')
