"""Badges: trend arrows for players and defenses, and matchup flags for players.

Display only. A badge says a change or a matchup is unusual and the sample is
large enough to say so. It is not a pick and never a model input. Every rule
and threshold is fixed in projects/nfl-props/badges-plan-2026-10-04.md,
written before any count was computed. One function (compute_badges) writes
data/ratings/badges.json and every page reads that file.
"""
import html
import json
import math

import numpy as np
import pandas as pd

from .config import DATA_DIR

BADGES_PATH = DATA_DIR / "ratings" / "badges.json"

# ---- Thresholds (plan sections 5 and 6). Do not tune after seeing counts. ----
N_RECENT = 3
T_MIN = 2.0
SAME_SIDE = 2            # of the 3 recent games
PLAYER_BASE_MIN, PLAYER_BASE_MAX = 6, 12
PLAYER_REL_MIN = 0.25    # |recent - baseline| as a share of the baseline mean
PLAYER_SHRINK_K = 6      # games of position-typical spread mixed into the player's own
DEF_BASE_MIN, DEF_BASE_MAX = 8, 14
TIER_SHARE = 0.15        # top share of ranked players at a position
TIER_MIN_GAMES = 6
EXTREME_N = 5            # the 5 toughest or the 5 softest defenses vs a position
TONE_N = 10              # card line: ranks 1-10 read "tougher", 23-32 read "softer". Wider than EXTREME_N on purpose.
assert TONE_N >= EXTREME_N

POS_STAT = {"QB": "passing_yards", "RB": "rushing_yards", "WR": "receiving_yards", "TE": "receiving_yards"}
POS_LABEL = {"QB": "passing yds", "RB": "rushing yds", "WR": "receiving yds", "TE": "receiving yds"}
# defense category -> (stat key in defense_rank, label, unit, decimals)
DEF_CATS = {
    "total": ("epa_play", "Total", "EPA/play", 3), "run": ("epa_rush", "Run", "EPA/rush", 3),
    "pass": ("epa_db", "Pass", "EPA/dropback", 3),
    "qb": ("qb_ypg", "vs QB", "yds/g", 1), "rb": ("rb_ypg", "vs RB", "yds/g", 1),
    "wr": ("wr_ypg", "vs WR", "yds/g", 1), "te": ("te_ypg", "vs TE", "yds/g", 1),
}


# ---- Pure rules (unit tested) -----------------------------------------------
def trend_test(vals, season_flags, var_prior, k, base_min, base_max, rel_min=None):
    """vals: per-game values, most recent first. season_flags: True for current-season games.
    Returns None or {"dir": "up"|"down" (sign of the change), t, recent, base, n_base}.
    "up" here means the value rose. Callers map that to their own meaning."""
    if len(vals) < N_RECENT + base_min or not all(season_flags[:N_RECENT]):
        return None
    rec = np.asarray(vals[:N_RECENT], float)
    base = np.asarray(vals[N_RECENT:N_RECENT + base_max], float)
    if np.isnan(rec).any() or np.isnan(base).any():
        return None
    nb = len(base)
    s2 = (nb * base.var(ddof=1) + k * var_prior) / (nb + k)
    if not s2 > 0:
        return None
    delta = rec.mean() - base.mean()
    t = delta / math.sqrt(s2 * (1 / N_RECENT + 1 / nb))
    if abs(t) < T_MIN:
        return None
    if rel_min is not None and not (base.mean() > 0 and abs(delta) >= rel_min * base.mean()):
        return None
    side = (rec > base.mean()) if delta > 0 else (rec < base.mean())
    if side.sum() < SAME_SIDE:
        return None
    return {"dir": "up" if delta > 0 else "down", "t": float(t),
            "recent": float(rec.mean()), "base": float(base.mean()), "n_base": nb}


def matchup_kind(pos_rank, n_pos, n_games, def_rank, n_def=32):
    """"big", "caution" or None. Both sides must be extreme: a top-15% player at his
    position and one of the 5 softest (big) or toughest (caution) defenses vs that position."""
    if n_pos <= 0 or n_games < TIER_MIN_GAMES or pos_rank > math.ceil(TIER_SHARE * n_pos):
        return None
    if def_rank >= n_def - EXTREME_N + 1:
        return "big"
    if def_rank <= EXTREME_N:
        return "caution"
    return None


def matchup_tone(rank, n_def=32):
    """Word class for the card's matchup line: "tough", "soft" or "mid". Every rank that earns a
    Caution badge is "tough" and every rank that earns a Big day badge is "soft" (TONE_N >= EXTREME_N),
    so a card line can never point the opposite way from its badge."""
    if rank <= TONE_N:
        return "tough"
    return "soft" if rank > n_def - TONE_N else "mid"


def def_pos_table(dranks):
    """{team: {POS: {"rank": int, "ypg": float}}}: the adjusted whole-position defense rank."""
    out = {}
    for r in dranks.itertuples():
        out[r.team] = {pos: {"rank": int(getattr(r, f"{pos.lower()}_rank")),
                             "ypg": round(float(getattr(r, f"{pos.lower()}_ypg")), 1)} for pos in POS_STAT}
    return out


def card_line(data, opp, pos):
    """The card's matchup line: {"rk", "y", "t"} (rank of 32, adjusted yds/g allowed, tone) or None.
    Reads the same def_pos table the matchup badges were computed from."""
    e = (data or {}).get("def_pos", {}).get(opp, {}).get(pos)
    if not e:
        return None
    return {"rk": e["rank"], "y": e["ypg"], "t": matchup_tone(e["rank"])}


def check_consistency(data):
    """Raise if any matchup badge points opposite to its card line. Returns the number checked."""
    n = 0
    for key, e in data.get("players", {}).items():
        m = e.get("matchup")
        if not m:
            continue
        line = card_line(data, m["opp"], key.split("|")[1])
        want = {"big": "soft", "caution": "tough"}[m["kind"]]
        if line is None or line["t"] != want:
            raise AssertionError(f"matchup badge {m['kind']} for {key} vs {m['opp']} disagrees with card line {line}")
        n += 1
    return n


def ordinal(n):
    n = int(n)
    return f"{n}{'th' if 10 <= n % 100 <= 20 else {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')}"


# ---- Computation ------------------------------------------------------------
def _player_trends(season, week, df):
    """{(name_key|POS): badge dict} for the ranked players in df."""
    from .db import connect
    from .features import defense_vs_position
    from .odds import name_key
    with connect() as con:
        pg = pd.read_sql_query(
            "SELECT player_id, position, season, week, opponent_team, passing_yards, rushing_yards, "
            "receiving_yards FROM player_games WHERE season_type='REG' AND season >= ? "
            "AND (season < ? OR (season = ? AND week <= ?))", con, params=[season - 1, season, season, week])
    pg = pg[pg.player_id.isin(df.player_id)]
    d = defense_vs_position(season, week + 1)
    d = d.rename(columns={"defteam": "opponent_team"})
    pg = pg.merge(d, on=["opponent_team", "position"], how="left", suffixes=("", "_dm"))
    pg = pg.sort_values(["player_id", "season", "week"], ascending=[True, False, False])
    series = {}
    for pid, g in pg.groupby("player_id"):
        pos = g.position.iloc[0]
        stat = POS_STAT.get(pos)
        if stat is None:
            continue
        f = g[f"{stat}_dm"].fillna(1.0).clip(lower=.25) if f"{stat}_dm" in g else 1.0
        series[pid] = (g[stat].astype(float) / f).tolist(), (g.season == season).tolist()
    var_pos = {}
    for pos in POS_STAT:
        v = [np.var(series[p][0], ddof=1) for p in df[df.position == pos].player_id
             if p in series and len(series[p][0]) > 3]
        var_pos[pos] = float(np.mean(v)) if v else 0.0
    out = {}
    for r in df.itertuples():
        if r.player_id not in series:
            continue
        vals, flags = series[r.player_id]
        res = trend_test(vals, flags, var_pos[r.position], PLAYER_SHRINK_K,
                         PLAYER_BASE_MIN, PLAYER_BASE_MAX, PLAYER_REL_MIN)
        if res:
            word = "up" if res["dir"] == "up" else "down"
            res["title"] = (
                f"Trending {word}: {POS_LABEL[r.position]}, opponent removed, {res['recent']:.1f}/g over the last "
                f"{N_RECENT} games vs {res['base']:.1f}/g over the {res['n_base']} before (t={res['t']:+.1f}).")
            out[f"{name_key(r.name)}|{r.position}"] = {"trend": res}
    return out


def _defense_trends(season, week):
    from .db import connect
    from .defense_rank import STATS, adjust_games, team_games
    with connect() as con:
        g = team_games(con, season, week)
    g = adjust_games(g).sort_values(["defn", "season", "week"], ascending=[True, False, False])
    out = {}
    for cat, (st, label, unit, dp) in DEF_CATS.items():
        ok = g[g[st].notna()]
        var_pool = float((ok[st] - ok.groupby("defn")[st].transform("mean")).var())
        for team, t in ok.groupby("defn"):
            res = trend_test(t[st].tolist(), (t.season == season).tolist(), var_pool, 1e9,
                             DEF_BASE_MIN, DEF_BASE_MAX)
            if not res:
                continue
            tough = res["dir"] == "down"  # allows less than before
            res["dir"] = "up" if tough else "down"
            sign = "+" if unit.startswith("EPA") else ""
            res["title"] = (
                f"Trending {res['dir']} {label}: allowed {res['recent']:{sign}.{dp}f} {unit} over the last "
                f"{N_RECENT} games vs {res['base']:{sign}.{dp}f} over the {res['n_base']} before, opponents removed "
                f"(t={res['t']:+.1f}). Up = tougher, down = softer.")
            out.setdefault(team, {})[cat] = res
    return out


def _matchups(season, week, df, dranks):
    """{(name_key|POS): matchup dict} for the next unplayed week."""
    from .db import connect
    from .odds import name_key
    with connect() as con:
        sch = pd.read_sql_query("SELECT home_team, away_team FROM schedules WHERE season=? AND week=? AND game_type='REG'",
                                con, params=[season, week + 1])
    opp = {**dict(zip(sch.home_team, sch.away_team)), **dict(zip(sch.away_team, sch.home_team))}
    dr = dranks.set_index("team")
    out = {}
    for pos in POS_STAT:
        d = df[df.position == pos]
        n = len(d)
        k = pos.lower()
        for r in d.itertuples():
            o = opp.get(r.team)
            if o is None or o not in dr.index:
                continue
            rk = int(dr.loc[o, f"{k}_rank"])
            kind = matchup_kind(int(r.rank), n, int(r.n_games), rk)
            if not kind:
                continue
            lead = "Big day potential" if kind == "big" else "Caution"
            out[f"{name_key(r.name)}|{pos}"] = {"matchup": {
                "kind": kind, "opp": o, "week": week + 1,
                "title": (f"{lead}: #{int(r.rank)} {pos} by recent production vs {o}, {ordinal(rk)} of 32 vs {pos}, "
                          f"{dr.loc[o, f'{k}_ypg']:.1f} yds/g allowed (opponent-adjusted).")}}
    return out


def compute_badges(season=None, week=None, df=None):
    """Build the whole badge dict. df: the ranked-player frame from rating.current_ratings."""
    from .defense_rank import current_defense_ranks
    from .rating import current_ratings
    if df is None:
        season, week, df = current_ratings()
    _, _, dranks = current_defense_ranks()
    players = _player_trends(season, week, df)
    for k, v in _matchups(season, week, df, dranks).items():
        players.setdefault(k, {}).update(v)
    defs = _defense_trends(season, week)
    out = {"season": season, "through_week": week, "slate_week": week + 1,
           "ranked": {p: int((df.position == p).sum()) for p in POS_STAT},
           "players": players, "defenses": defs, "def_pos": def_pos_table(dranks)}
    check_consistency(out)
    return out


_CACHE = {}


def get_badges():
    """compute_badges() once per process (report builds call it for every row set)."""
    if "b" not in _CACHE:
        _CACHE["b"] = compute_badges()
    return _CACHE["b"]


def write_badges(season=None, week=None, df=None):
    b = compute_badges(season, week, df)
    _CACHE["b"] = b
    BADGES_PATH.write_text(json.dumps(b), encoding="utf-8")
    return b


# ---- Reading and markup (every page uses these) ------------------------------
_FILE = {"mtime": None, "data": {}}


def load_badges():
    """Read data/ratings/badges.json; re-read when the file changes. {} if absent."""
    try:
        m = BADGES_PATH.stat().st_mtime
        if m != _FILE["mtime"]:
            _FILE["data"], _FILE["mtime"] = json.loads(BADGES_PATH.read_text()), m
    except (OSError, ValueError):
        return {}
    return _FILE["data"]


def player_badges(key, opp=None, week=None, data=None):
    """List of [css class, text, title] for one player key 'namekey|POS'.
    The matchup badge shows only for the game it was computed for: opp and week must match.
    Pass opp=None to leave matchups out."""
    data = load_badges() if data is None else data
    e = data.get("players", {}).get(key)
    out = []
    if not e:
        return out
    t = e.get("trend")
    if t:
        out.append([f"bd bd-{t['dir']}", "▲" if t["dir"] == "up" else "▼", t["title"]])
    m = e.get("matchup")
    if m and opp == m["opp"] and week == m["week"]:
        out.append([f"bd bd-{m['kind']}", "Big day" if m["kind"] == "big" else "Caution", m["title"]])
    return out


def badge_html(items, inspect_id):
    """Markup for a list from player_badges (server-built pages)."""
    return "".join(
        f'<span class="{html.escape(c, quote=True)}" data-inspect-id="{inspect_id}" '
        f'title="{html.escape(t, quote=True)}">{html.escape(x)}</span>' for c, x, t in items)


def markers_html(depth_html="", chip=None, items=(), chip_id="", badge_id=""):
    """One name-marker group: depth, rank chip, trend, matchup, in that order.
    depth_html: markup for the depth number (plain <span class="dp"> or a pill), already safe.
    chip: (text, title) from chips.json, or None. items: player_badges() / defense_badge() output.
    Returns "" when there is nothing to show."""
    chip_html = (f'<span class="rk" data-inspect-id="{chip_id}" '
                 f'title="{html.escape(chip[1], quote=True)}">{html.escape(chip[0])}</span>') if chip else ""
    inner = depth_html + chip_html + badge_html(items, badge_id)
    return f'<span class="mk">{inner}</span>' if inner else ""


def defense_badge(team, cat, data=None):
    """[class, text, title] list (0 or 1 item) for a defense category."""
    data = load_badges() if data is None else data
    r = data.get("defenses", {}).get(team, {}).get(cat)
    if not r:
        return []
    return [[f"bd bd-d-{r['dir']}", "▲" if r["dir"] == "up" else "▼", r["title"]]]


# One stylesheet for every page. Palette tokens only, from each page's :root. Player arrows use good/bad. Defense arrows are neutral because tougher is not good or bad on its own.
BADGE_CSS = """
  /* Name markers. One system for every page: depth, rank chip, trend, matchup,
     in that order, in one .mk group. Spec: projects/nfl-props/design/name-markers-spec-2026-10-04.md.
     Depth is filled or plain. Rank is the only outline. Trend is a bare glyph. Matchup is tinted.
     position/bottom/margin resets override older copies baked into report files on disk. */
  .mk{display:inline-flex; align-items:center; gap:4px; margin-left:6px;
    vertical-align:middle; white-space:nowrap; flex:none}
  .mk .dp{font:inherit; color:inherit}
  .role,.rk,.bd,.pick-badge,.td-badge{box-sizing:border-box; display:inline-flex; align-items:center;
    justify-content:center; height:16px; padding:0 5px; border-radius:2px; margin:0;
    font-family:"IBM Plex Mono",monospace; font-size:10px; line-height:1; white-space:nowrap;
    position:static; bottom:auto; vertical-align:middle}
  .role{font-weight:600; letter-spacing:.06em; color:var(--surface)}
  .role.a{background:var(--den)} .role.b{background:var(--kc)}
  .rk{font-weight:500; color:var(--ink-2); background:transparent;
    border:1px solid var(--ink-3); box-shadow:none; cursor:help}
  .bd{font-weight:600; border:1px solid currentColor; cursor:help}
  .bd-up,.bd-down,.bd-d-up,.bd-d-down{border:0; background:none; padding:0 1px}
  .bd-up{color:var(--good)} .bd-down{color:var(--bad)}
  .bd-d-up{color:var(--ink)} .bd-d-down{color:var(--ink-3)}
  .bd-big{color:var(--den,var(--accent)); background:color-mix(in srgb, var(--den,var(--accent)) 12%, transparent)}
  .bd-caution{color:var(--ink); background:color-mix(in srgb, var(--ink) 10%, transparent); border-style:dashed}
"""
