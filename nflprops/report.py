"""The standard production-ranges report.

One code path builds the page whether it is rendered interactively or by a
scheduled job, so a report posted at 7am is the same artifact as one built by
hand. Everything visual lives in templates/report.html.
"""
import io, json, contextlib, pathlib, sys, urllib.request
from statistics import NormalDist

import pandas as pd
import numpy as np

from . import project as P, reliability as R
from .base import SPECS, DEF_WEIGHT
from . import engine_source as ES, spread as SP
from .features import defense_vs_position
from .config import require, ROOT, save_board
from .odds import name_key
from .db import connect
from .rating import chip_lookup as rating_chips
from .badges import card_line, get_badges, player_badges
from . import calibration as CAL

# scripts/vol_predict.py is not part of the nflprops package (it lives beside
# it, not inside it) -- same sys.path shim the scripts use.
_SCRIPTS = str(pathlib.Path(__file__).resolve().parent.parent / "scripts")
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)
from vol_predict import phase3_vol  # noqa: E402

# Errors ran ~10% larger than the model's own sd across 2023-2025, so the raw
# spread understates how wrong a projection can be.
SD_INFLATE = 1.10
# Situational screens. A leg failing any of these is shown but never shortlisted.
MIN_SNAP, MIN_GAMES = 0.55, 10
DEFAULT_TARGET = -300

# Rung choice (Sean, 2026-10-26; first used for week 8). For each player-stat
# the book prices a ladder of Over rungs. Instead of the rung nearest -300, take
# the rung priced -200 to -110 where the (shrunk) model probability beats the
# book's by the most, and only if that gap is above zero. Earlier weeks keep the
# nearest-to--300 rung exactly as bet.
RUNG_LO, RUNG_HI = -200, -110
RUNG_FROM = (2026, 8)


# One early game (Sean, 2026-10-26): week 7 MNF, DAL @ PHI. It had no report yet
# at the time. Every other week 7 report was already bet and must not change.
RUNG_EARLY = {(2026, 7, "DAL", "PHI")}


def rung_on(season, week, teams=()):
    if (season, week) >= RUNG_FROM:
        return True
    t = sorted(teams)
    return any((season, week) == e[:2] and t == sorted(e[2:]) for e in RUNG_EARLY)


def choose_rung(ladder, mu, sd, f):
    """Best rung in a ladder of (line, price) Overs, or None.

    mp is the same Normal(mu, sd) tail the report uses, shrunk toward bp by f
    (calibration.py). bp is the raw implied price, not de-vigged. Only prices
    from RUNG_LO to RUNG_HI inclusive count. The edge mp - bp must be above 0.
    """
    best = None
    for line, price in ladder or ():
        if price is None or not (RUNG_LO <= price <= RUNG_HI):
            continue
        bp = _imp(price)
        mp_raw = 1 - NormalDist(float(mu), float(sd)).cdf(float(line))
        mp = mp_raw + f * (bp - mp_raw)
        edge = mp - bp
        if edge > 0 and (best is None or edge > best["edge"]):
            best = {"line": float(line), "price": int(price), "bp": bp,
                    "mp": mp, "mp_raw": mp_raw, "edge": edge}
    return best

ALT = {"player_pass_yds_alternate": "passing_yards",
       "player_rush_yds_alternate": "rushing_yards",
       "player_reception_yds_alternate": "receiving_yards",
       "player_receptions_alternate": "receptions"}
BASE = "https://api.the-odds-api.com/v4/sports/americanfootball_nfl"


def upcoming_events():
    """Pre-game events on the board, with kickoff in ET. Free call."""
    k = require("ODDS_API_KEY")
    with urllib.request.urlopen(f"{BASE}/events?apiKey={k}", timeout=30) as r:
        ev = pd.DataFrame(json.load(r))
    if ev.empty:
        return ev
    ev["ct"] = pd.to_datetime(ev.commence_time, utc=True)
    ev = ev[ev.ct > pd.Timestamp.now(tz="UTC")].copy()
    ev["et"] = ev.ct.dt.tz_convert("America/New_York")
    return ev


def event_lines(event_id, target=DEFAULT_TARGET, season=None, week=None, teams=None):
    """The rung closest to `target` for each player-stat in one game.

    decision #37: when the caller passes season/week/teams, the raw response
    is saved to data/boards/ unmodified before any filtering -- zero extra
    API calls, since this response was already fetched.
    """
    k = require("ODDS_API_KEY")
    u = (f"{BASE}/events/{event_id}/odds?apiKey={k}&bookmakers=fanduel"
         f"&markets={','.join(ALT)}&oddsFormat=american")
    with urllib.request.urlopen(u, timeout=30) as r:
        d = json.load(r)
        rem = r.headers.get("x-requests-remaining")
    if season is not None and week is not None and teams is not None:
        save_board(season, week, teams, d)
    return parse_lines(d, target), rem


def parse_lines(d, target=DEFAULT_TARGET):
    """Closest-to-target rung per player-stat from one raw odds payload."""
    rows = []
    for bm in d.get("bookmakers", []):
        for m in bm.get("markets", []):
            stat = ALT.get(m["key"])
            if not stat:
                continue
            for o in m.get("outcomes", []):
                if o.get("name") == "Over":
                    rows.append({"stat": stat, "player": o.get("description"),
                                 "line": o.get("point"), "price": o.get("price")})
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["player_key"] = df.player.map(name_key)
    # Every rung, kept for the rung choice. line/price stay the rung nearest the target.
    ladders = {k: sorted({(float(a), int(b)) for a, b in zip(g.line, g.price)
                          if pd.notna(a) and pd.notna(b)})
               for k, g in df.groupby(["player_key", "stat"])}
    df["d"] = (df.price - target).abs()
    out = df.sort_values("d").drop_duplicates(["player_key", "stat"]).copy()
    out["ladder"] = [ladders[(k, s)] for k, s in zip(out.player_key, out.stat)]
    return out


def report_path(season, week, team_a, team_b):
    return ROOT / "reports" / f"report_{season}_w{week}_{team_a}-{team_b}.html"


def _imp(a):
    if pd.isna(a):
        return np.nan
    a = float(a)
    return (-a) / ((-a) + 100) if a < 0 else 100 / (a + 100)


# Decisions #42/#43: the 2026 holdout picked the new model for these stats.
# Games from the 2026 week 5 4pm kickoffs on use it as the card number; earlier games keep the old
# engine, as published.
NEW_PRIMARY_STATS = {"passing_yards", "rushing_yards"}
# Gate by kickoff (naive America/New_York, as stored in schedules): games that
# kick off at or after this moment use the new model. Weeks 1-5 games before it
# keep the old engine as published. Every later game is after it.
NEW_PRIMARY_FROM_KICKOFF = "2026-10-11 16:00"


# One-time override list. Empty: the Nix pin (DEN-LAC, 2026-10-11) was lifted after the
# serving rank fix (decision #44) gave him 214.4, a sane number. Keyed on game, player, stat.
PINNED_OLD = set()


def new_primary(kickoff, stat):
    return stat in NEW_PRIMARY_STATS and bool(kickoff) and kickoff >= NEW_PRIMARY_FROM_KICKOFF


def build(season, week, teams, lines, n_picks=6, rung_choice=None):
    """Return (rows, shortlist, qb_notes) for one game. `lines` comes from
    event_lines. `qb_notes` is the list of decision #36 QB1-override strings
    (empty when the sportsbook line agreed with the usage rank, or didn't
    narrow to exactly one priced QB). `rung_choice` None means on from week 8
    of 2026. True or False forces it (dry runs, tests)."""
    use_rung = rung_on(season, week, teams) if rung_choice is None else bool(rung_choice)
    with contextlib.redirect_stdout(io.StringIO()):
        pr = P.project_week(season, week)
        pool = R.star_pool(season, week)
        snaps = R.snap_share(season, week)
        cont = R.role_continuity(season, week)
    inj, _ = R.risk_flags(season, week)

    pr["player_key"] = pr["player"].map(name_key)

    # Decision #36: when FanDuel prices passing yards for exactly one QB on a
    # team, that QB is the featured QB1 -- replacing star_pool's usage-ranked
    # pick if it disagrees. This is the only copy; scripts/make_report.py
    # now calls build() instead of carrying its own.
    qb_notes = []
    if not lines.empty:
        qb_priced = set(lines.loc[lines.stat == "passing_yards", "player_key"])
        qb_universe = pr[(pr.position == "QB") & pr.team.isin(teams)][
            ["player_key", "team", "player"]].drop_duplicates("player_key")
        for _team in teams:
            priced = qb_universe[(qb_universe.team == _team) & qb_universe.player_key.isin(qb_priced)]
            if len(priced) != 1:
                continue
            new_key, new_name = priced.iloc[0].player_key, priced.iloc[0].player
            cur = pool[(pool.team == _team) & (pool.position == "QB")]
            if not cur.empty and cur.iloc[0].player_key == new_key:
                continue
            old_name = cur.iloc[0].full_name if not cur.empty else None
            pool = pool[~((pool.team == _team) & (pool.position == "QB"))]
            pool = pd.concat([pool, pd.DataFrame([{
                "player_key": new_key, "full_name": new_name, "team": _team,
                "position": "QB", "usage": np.nan, "rank": 1.0}])], ignore_index=True)
            qb_notes.append(f"QB set by sportsbook line: {new_name} ({_team})"
                             + (f", not {old_name}" if old_name else ""))

    pr = pr[pr.team.isin(teams) & pr.player_key.isin(set(pool.player_key))].copy()
    if pr.empty:
        return [], pr, qb_notes
    rank = {r.player_key: int(r["rank"]) for _, r in pool.iterrows()}

    # Decision #31 (extended by #38/#39/#40): Phase 3's volume/yardage models.
    # Since decision #42/#43 they are the primary number for passing_yards and
    # rushing_yards from that kickoff, and the secondary line for the other stats.
    # Never merged into one figure -- see #37.
    try:
        p3 = phase3_vol(season, week)
    except Exception as e:
        print(f"  ! phase3_vol failed, skipping: {e}")
        p3 = pd.DataFrame(columns=["player_id"])
    pr = pr.merge(p3, on="player_id", how="left") if "player_id" in pr.columns else pr
    # Decision #38 adds receptions (targets.pkl x catch_rate_career); decision
    # #39 adds passing_yards (features.db now carries QB passing features and
    # y_passing_yards).
    for _c in ("p3_receiving_yards", "p3_rushing_yards", "p3_receptions", "p3_passing_yards"):
        if _c not in pr.columns:
            pr[_c] = np.nan

    dvp = defense_vs_position(season, week)
    dvp_ix = dvp.set_index(["defteam", "position"]) if not dvp.empty else None

    def raw_mult(row):
        if dvp_ix is None or (row.opponent, row.position) not in dvp_ix.index:
            return np.nan
        d = dvp_ix.loc[(row.opponent, row.position)]
        col = SPECS[row.stat][2]
        if col not in d:
            return np.nan
        try:
            return float(d.get(col))
        except (TypeError, ValueError):
            return np.nan

    pr["raw_def"] = pr.apply(raw_mult, axis=1)
    pr["applied"] = pr.stat.map(lambda s: DEF_WEIGHT.get(s, 1.0) > 0)
    # Decisions #42/#43: per-stat primary engine. On the switched stats the new
    # model's number replaces the old engine's as the card's number (mu, range,
    # curve, mp, z, conf, shortlist all follow). The new model has no spread of
    # its own, so the old engine's sd_adj is kept around the new mean. The old
    # engine's number moves to the secondary comparison line.
    pr["old_proj"] = pr.proj
    pr["new_proj"] = [r[f"p3_{r['stat']}"] if f"p3_{r['stat']}" in r.index else np.nan
                      for _, r in pr.iterrows()]
    with connect() as con:
        _ko = pd.read_sql_query("SELECT game_id, gameday, gametime FROM schedules WHERE season=? AND week=?",
                                con, params=[season, week])
    ko = {r.game_id: f"{r.gameday} {r.gametime}" for r in _ko.itertuples() if r.gameday and r.gametime}
    # Per-stat source of mu (engine_source.py). Default reproduces the kickoff-gated rule.
    _src = ES.load()
    _slate = SP.slate_rows(season, week) if ES.needs_slate(_src) else None
    if _slate is not None:
        # Completions has no Phase 3 model: its "new" number is the attempts x rate baseline.
        _bm = _slate.set_index(["player_id", "stat"]).base_mu
        _cm = pr.stat == "completions"
        pr.loc[_cm, "new_proj"] = [_bm.get((i, "completions"), np.nan) for i in pr.loc[_cm, "player_id"]]

    def _use_new(g, pl, s, n):
        mode = _src["mu"].get(s, "old")
        if mode == "old" or pd.isna(n) or (g, pl, s) in PINNED_OLD:
            return False
        return True if mode == "new" else new_primary(ko.get(g), s)
    pr["use_new"] = [_use_new(g, pl, s, n) for g, pl, s, n in zip(pr.game_id, pr.player, pr.stat, pr.new_proj)]
    pr["proj"] = np.where(pr.use_new, pr.new_proj, pr.old_proj)
    # Weekly projection bias (calibration.json mu_bias): added to mu before ANY
    # probability, z, lo or hi. sd is not changed. Same gate as the rung rule, so
    # past reports re-render as they were. mu_raw keeps the number before it.
    pr["mu_raw"] = pr.proj
    if use_rung:
        pr["proj"] = pr.proj + pr.stat.map(CAL.bias_for)
    # The old engine's defense multiplier is not applied to a new-primary number.
    pr["applied"] = pr.applied & ~pr.use_new
    pr["sd_adj"] = pr.sd * SD_INFLATE
    if _slate is not None and any(v == "new" for v in _src["sd"].values()):
        # New sd: the same per-player sd (spread.py), no 1.10, scaled per stat on out-of-fold residuals.
        _sd = _slate.set_index(["player_id", "stat"]).sd
        _sc = ES.sd_scale()
        _new_sd = pd.Series([_sd.get((i, s), np.nan) * _sc[s] for i, s in zip(pr.player_id, pr.stat)], index=pr.index)
        _use_sd = pr.stat.map(lambda s: _src["sd"].get(s) == "new") & _new_sd.notna()
        pr["sd_adj"] = np.where(_use_sd, _new_sd, pr.sd_adj)
    pr["lo"] = (pr.proj - pr.sd_adj).clip(lower=0)
    pr["hi"] = pr.proj + pr.sd_adj

    if not lines.empty:
        cols = ["player_key", "stat", "line", "price"] + (["ladder"] if "ladder" in lines.columns else [])
        pr = pr.merge(lines[cols], on=["player_key", "stat"], how="left")
    else:
        pr["line"] = np.nan
        pr["price"] = np.nan
    if "ladder" not in pr.columns:
        pr["ladder"] = None

    pr["bp"] = pr.price.map(_imp)
    pr["mp"] = [np.nan if pd.isna(l) else 1 - NormalDist(float(m), float(s)).cdf(float(l))
                for l, m, s in zip(pr.line, pr.proj, pr.sd_adj)]
    # Decision #46: weekly self-adjustment. The one place the adjustment is
    # applied. Everything below (conf, shortlist, rows, Slack text) follows.
    pr["mp_raw"] = pr.mp
    _f = CAL.fraction()
    if _f:
        pr["mp"] = pr.mp + _f * (pr.bp - pr.mp)
    # The old rung (nearest the target price), kept beside whatever is chosen.
    pr["old_line"], pr["old_price"], pr["old_mp"], pr["old_bp"] = pr.line, pr.price, pr.mp, pr.bp
    pr["rung"] = None
    if use_rung:
        for ix, r in pr.iterrows():
            lad = r.ladder if isinstance(r.ladder, (list, tuple)) and len(r.ladder) else None
            if pd.isna(r.line):
                continue
            if lad is None:
                pr.at[ix, "rung"] = "fallback"
                continue
            c = choose_rung(lad, r.proj, r.sd_adj, _f)
            if c is None:
                pr.at[ix, "rung"] = "none"
                continue
            pr.at[ix, "rung"] = "chosen"
            for k in ("line", "price", "bp", "mp", "mp_raw"):
                pr.at[ix, k] = c[k]
    pr["z"] = (pr.line - pr.proj) / pr.sd_adj
    # Never claim more than the more pessimistic of model and market.
    pr["conf"] = pr[["mp", "bp"]].min(axis=1)

    pr = (pr.merge(snaps, on="player_key", how="left")
            .merge(cont[["player_key", "changed_team"]].drop_duplicates("player_key"),
                   on="player_key", how="left")
            .merge(inj[["player_key", "report_status"]].drop_duplicates("player_key"),
                   on="player_key", how="left"))
    with connect() as con:
        seen = pd.read_sql_query(
            "SELECT player_display_name, COUNT(*) n FROM player_games "
            "WHERE season_type='REG' GROUP BY 1", con)
    seen["player_key"] = seen.player_display_name.map(name_key)
    pr = pr.merge(seen[["player_key", "n"]], on="player_key", how="left")
    pr["snap_share"] = pr.snap_share.fillna(0)
    pr["changed_team"] = pr.changed_team.fillna(False)
    pr["clean"] = (~pr.changed_team
                   & ~pr.report_status.isin(["Questionable", "Doubtful", "Out"])
                   & (pr.snap_share >= MIN_SNAP)
                   & (pr.n.fillna(0) >= MIN_GAMES))

    # Shortlist: strictly by confidence among legs passing the screens, one per
    # player. Two legs on one player die together if he is hurt or benched,
    # which is the concentration a multi-leg slip exists to avoid.
    # With rung choice on, a leg that has a ladder but no qualifying rung shows
    # its old rung and is never shortlisted. The list gets shorter, not filled.
    short = (pr[pr.clean & pr.line.notna() & (pr.rung != "none")]
             .sort_values("conf", ascending=False)
             .drop_duplicates("player_key")
             .head(n_picks))
    pr["pick"] = None
    for i, ix in enumerate(short.index, 1):
        pr.loc[ix, "pick"] = i

    chips = rating_chips(name_key)
    bdata = get_badges()  # the data compute_badges() writes to data/ratings/badges.json
    rows = []
    for r in pr.itertuples():
        rk = chips.get((r.player_key, r.position))
        rows.append({
            "rk": rk[0] if rk else None, "rkt": rk[1] if rk else None,
            "bd": player_badges(f"{r.player_key}|{r.position}", r.opponent, week if season == bdata["season"] else None, bdata),
            "p": r.player, "role": r.position + str(rank.get(r.player_key, 1)),
            "tm": r.team, "vs": r.opponent, "s": r.stat,
            "mu": round(float(r.proj), 1),
            **({"mu_raw": round(float(r.mu_raw), 1)} if use_rung else {}), "sd": round(float(r.sd_adj), 1),
            "lo": round(float(r.lo), 1), "hi": round(float(r.hi), 1),
            "dr": (card_line(bdata, r.opponent, r.position, r.stat)
                   if season == bdata["season"] and week == bdata["slate_week"] else None),
            "def": None if pd.isna(r.raw_def) else round(float(r.raw_def), 2),
            "app": bool(r.applied),
            "line": None if pd.isna(r.line) else float(r.line),
            "price": None if pd.isna(r.price) else int(r.price),
            "z": None if pd.isna(r.z) else round(float(r.z), 2),
            "mp": None if pd.isna(r.mp) else round(float(r.mp), 3),
            "mp_raw": None if pd.isna(r.mp_raw) else round(float(r.mp_raw), 3),
            "bp": None if pd.isna(r.bp) else round(float(r.bp), 3),
            "pick": None if r.pick is None else int(r.pick),
            **({"rung": r.rung,
                "old_line": None if pd.isna(r.old_line) else float(r.old_line),
                "old_price": None if pd.isna(r.old_price) else int(r.old_price),
                "old_mp": None if pd.isna(r.old_mp) else round(float(r.old_mp), 3),
                "old_bp": None if pd.isna(r.old_bp) else round(float(r.old_bp), 3)}
               if use_rung else {}),
            # pe: the engine behind mu ("new" = Phase 3 model, "old" = project.py).
            # The other engine's number is no longer put on the card (2026-10-27).
            "pe": "new" if r.use_new else "old",
        })
    return rows, pr.loc[short.index], qb_notes


def render(rows, away, home, team_a, week, season, kick=None, same_game=True,
           target=DEFAULT_TARGET, qb_notes=None):
    corr = ("All of these are same-game legs, so they move together &mdash; a shootout "
            "carries them over, a grind carries them under. That makes the true joint "
            "chance <em>higher</em> than the naive figure, and it is also why FanDuel "
            "will pay less than the straight payout for them as one same-game parlay."
            if same_game else
            "These legs sit in different games, so they are close to independent &mdash; "
            "the naive joint chance is about right, and the straight payout is what you "
            "would be offered.")
    if qb_notes:
        corr = ("<strong>" + "</strong><br><strong>".join(qb_notes) + "</strong><br>" + corr)
    meta = {"teamA": team_a, "nameA": away, "nameB": home, "corrnote": corr}
    tpl = (ROOT / "templates" / "report.html").read_text()
    head = f'<h1>{away} <span class="at">at</span> {home}</h1>'
    # Kickoff time, not just the weekday. A Sunday slate has games at 1:00,
    # 4:05, 4:25 and 8:20, and "Sunday" cannot tell you which one you have
    # open. The listing sorts by this same time, read from the schedules
    # table (see scripts/serve_reports.py).
    eyebrow = (f"{kick:%A %-I:%M %p} ET · Week {week} · {season}" if kick is not None
               else f"Week {week} · {season}")
    title = f"{away.split()[-1]}–{home.split()[-1]} Production Ranges"
    return (tpl.replace("__DATA__", json.dumps(rows))
               .replace("__META__", json.dumps(meta))
               .replace("__TITLE__", title)
               .replace("__EYEBROW__", eyebrow)
               .replace("__HEADLINE__", head)
               .replace("__TARGET__", json.dumps(target))
               .replace("__TARGET_ABS__", str(abs(target))))
