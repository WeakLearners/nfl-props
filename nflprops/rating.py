"""Player rating, 0-100, inside one position. Display and test only.

A rating says how much a player produces per game in his role, opponent
removed. It is a pure function of the database: for week W it reads games
strictly before W (the features.py rule). No network. It describes past
production and is not a pick. See projects/nfl-props/rating-system-plan-2026-10-04.md.
"""
import json

import numpy as np
import pandas as pd
from scipy.optimize import lsq_linear
from scipy.stats import norm

from .config import DATA_DIR
from .features import defense_vs_position
from .project import DECAY, PRIOR_SEASON_W, MIN_GAMES, K_VOL_GAMES
from .shares import player_game_shares

RATINGS_DIR = DATA_DIR / "ratings"
WEIGHTS_PATH = RATINGS_DIR / "weights.json"
LOOKBACK_SEASONS = 2

# component -> (numerator col, denominator col or None = per-game mean,
#               defense key for numerator, defense key for denominator)
# A per-game volume has numerator = the volume column itself.
C = {
    "att_pg":     ("attempts", None, None, "attempts"),
    "ypa":        ("passing_yards", "attempts", "passing_yards", "attempts"),
    "cpoe":       ("cpoe_x_att", "attempts", None, None),
    "epa_db":     ("passing_epa", "dropbacks", None, None),
    "sack_rate":  ("sacks_suffered", "dropbacks", None, None),
    "rush_share": ("rush_share", None, None, None),
    "car_pg":     ("carries", None, None, "carries"),
    "tgt_pg":     ("targets", None, None, "targets"),
    "ypc":        ("rushing_yards", "carries", "rushing_yards", "carries"),
    "rush_epa":   ("rushing_epa", "carries", None, None),
    "snap_share": ("snap_share", None, None, None),
    "rz10_carry": ("rz10_carry_share", None, None, None),
    "ayshare":    ("air_yards_share", None, None, None),
    "ypt":        ("receiving_yards", "targets", "receiving_yards", "targets"),
    "rec_epa":    ("receiving_epa", "targets", None, None),
    "tgt_share":  ("target_share", None, None, None),
    "rz10_tgt":   ("rz10_target_share", None, None, None),
}
# First-guess weights from the plan (section 3). Phase 2 fit replaces them.
PLAN_WEIGHTS = {
    "QB": {"att_pg": .20, "ypa": .20, "cpoe": .20, "epa_db": .20, "sack_rate": -.10, "rush_share": .10},
    "RB": {"car_pg": .30, "tgt_pg": .15, "ypc": .15, "rush_epa": .10, "snap_share": .15, "rz10_carry": .15},
    "WR": {"tgt_pg": .30, "ayshare": .15, "ypt": .20, "rec_epa": .10, "snap_share": .15, "tgt_share": .10},
    "TE": {"tgt_pg": .30, "ypt": .20, "rec_epa": .10, "snap_share": .20, "rz10_tgt": .10, "ayshare": .10},
}
# Primary stat per position (Test A and the fit) and the volume floor that
# keeps the test population to players a market could price.
PRIMARY = {"QB": "passing_yards", "RB": "rushing_yards", "WR": "receiving_yards", "TE": "receiving_yards"}
VOLUME = {"QB": ("attempts", 10), "RB": ("carries", 5), "WR": ("targets", 3), "TE": ("targets", 3)}
Z_CLIP = 3.0


def load_weights():
    if WEIGHTS_PATH.exists():
        return json.loads(WEIGHTS_PATH.read_text())["weights"]
    return PLAN_WEIGHTS


def _prep(pg):
    pg = pg.copy()
    pg["dropbacks"] = pg.attempts.fillna(0) + pg.sacks_suffered.fillna(0)
    pg["cpoe_x_att"] = pg.passing_cpoe * pg.attempts
    return pg


def _w(h):
    """Decay by games back, prior seasons discounted (same as project._weights).
    h must be sorted by player, then most recent first."""
    rank = h.groupby("player_id").cumcount().values
    prior = h.season.values < h.groupby("player_id").season.transform("max").values
    return DECAY ** rank * np.where(prior, PRIOR_SEASON_W, 1.0)


def components(pg, season, week, dvp=defense_vs_position):
    """Raw (unweighted) component value per player, from games before (season, week)."""
    h = pg[((pg.season < season) | ((pg.season == season) & (pg.week < week)))
           & (pg.season >= season - LOOKBACK_SEASONS) & pg.position.isin(PLAN_WEIGHTS)]
    h = h.sort_values(["player_id", "season", "week"], ascending=[True, False, False]).copy()
    d = dvp(season, week)
    if len(d):
        h = h.merge(d.rename(columns={"defteam": "opponent_team"}), on=["opponent_team", "position"],
                    how="left", suffixes=("", "_dm"))
    h["_w"] = _w(h)
    pos = h.groupby("player_id").position.first()
    out = {}
    for name, (num, den, kn, kd) in C.items():
        n = h[num].astype(float)
        if kn and f"{kn}_dm" in h:
            n = n / h[f"{kn}_dm"].fillna(1.0).clip(lower=.25)
        if den is None:
            if kd and f"{kd}_dm" in h:
                n = n / h[f"{kd}_dm"].fillna(1.0).clip(lower=.25)
            dd = n.notna().astype(float)
        else:
            dd = h[den].astype(float)
            if kd and f"{kd}_dm" in h:
                dd = dd / h[f"{kd}_dm"].fillna(1.0).clip(lower=.25)
            ok = n.notna() & dd.notna()
            n, dd = n.where(ok, 0.0), dd.where(ok, 0.0)
        n = n.fillna(0.0)
        t = pd.DataFrame({"player_id": h.player_id, "n": n * h._w, "d": dd * h._w, "w": h._w})
        t["position"] = h.position
        out[name] = t.groupby("player_id")[["n", "d", "w"]].sum()
    games = h.groupby("player_id").size()
    res = pd.DataFrame({"position": pos, "n_games": games,
                        "name": h.groupby("player_id").player_display_name.first(),
                        "team": h.groupby("player_id").team.first()})
    res = res[res.n_games >= MIN_GAMES]
    if res.empty:
        return res.reset_index()
    for name, (num, den, _, _) in C.items():
        t = out[name].loc[res.index]
        p = res.position
        if den is None:  # per-game mean, shrunk on games
            mean_pos = (t.n / t.w).groupby(p).mean()
            res[name] = (t.n + K_VOL_GAMES * p.map(mean_pos)) / (t.w + K_VOL_GAMES)
        else:  # rate, shrunk on the denominator (K games of the position's typical volume)
            rate_pos = t.n.groupby(p).sum() / t.d.groupby(p).sum()
            den_pg = (t.d / t.w).groupby(p).mean()
            k = K_VOL_GAMES * p.map(den_pg)
            res[name] = (t.n + k * p.map(rate_pos)) / (t.d + k)
    # Baselines the tests compare against: last-5 average, and its volume.
    last5 = h.groupby("player_id").head(5)
    g5 = last5.groupby("player_id")
    res["last5_passing_yards"] = g5.passing_yards.mean()
    res["last5_rushing_yards"] = g5.rushing_yards.mean()
    res["last5_receiving_yards"] = g5.receiving_yards.mean()
    for c in ("attempts", "carries", "targets"):
        res[f"last5_{c}"] = g5[c].mean()
    return res.reset_index()


def _in_pool(res):
    ok = pd.Series(False, index=res.index)
    for pos_, (col, floor) in VOLUME.items():
        ok |= (res.position == pos_) & (res[f"last5_{col}"] >= floor)
    return ok


def zscores(res):
    """Z-score each component inside its position, scale set by the pool of
    players with market-sized volume. Clipped at +-3."""
    res = res.copy()
    pool = _in_pool(res)
    for name in C:
        col = res[name]
        for pos_ in PLAN_WEIGHTS:
            m = res.position == pos_
            ref = col[m & pool]
            sd = ref.std()
            res.loc[m, f"z_{name}"] = ((col[m] - ref.mean()) / sd).clip(-Z_CLIP, Z_CLIP) if sd and sd > 0 else 0.0
    return res


def rate(res, weights=None):
    """Add composite, rating_z, rating (0-100) and the top two drivers."""
    if res.empty:
        return res
    weights = weights or load_weights()
    res = zscores(res)
    res["composite"] = np.nan
    res["rating_z"] = np.nan
    res["drivers"] = ""
    pool = _in_pool(res)
    for pos_, wts in weights.items():
        m = res.position == pos_
        if not m.any():
            continue
        contrib = pd.DataFrame({k: res.loc[m, f"z_{k}"] * v for k, v in wts.items()})
        comp = contrib.sum(axis=1)
        res.loc[m, "composite"] = comp
        res.loc[m, "drivers"] = contrib.abs().apply(lambda r: ", ".join(r.nlargest(2).index), axis=1)
        ref = comp[pool[m]]
        sd = ref.std()
        res.loc[m, "rating_z"] = (comp - ref.mean()) / sd if sd and sd > 0 else 0.0
    res["rating"] = (100 * norm.cdf(res.rating_z)).round(1)
    return res


def rating_frame(season, week, pg=None, weights=None, dvp=defense_vs_position):
    pg = _prep(pg if pg is not None else player_game_shares())
    return rate(components(pg, season, week, dvp), weights)


def fit_weights(train_frames, actual):
    """Non-negative least squares per position of the primary stat (z-scored
    inside position and week) on component z-scores. Sack rate is forced <= 0.
    train_frames: list of zscores() frames with season/week columns; actual:
    DataFrame player_id, season, week + stat columns."""
    df = pd.concat(train_frames, ignore_index=True).merge(actual, on=["player_id", "season", "week"])
    df = df[_in_pool(df)]
    out = {}
    for pos_, wts in PLAN_WEIGHTS.items():
        d = df[df.position == pos_]
        stat = PRIMARY[pos_]
        y = d.groupby(["season", "week"])[stat].transform(lambda s: (s - s.mean()) / s.std()).values
        cols = list(wts)
        X = d[[f"z_{c}" for c in cols]].values
        lo = [-np.inf if c == "sack_rate" else 0.0 for c in cols]
        hi = [0.0 if c == "sack_rate" else np.inf for c in cols]
        sol = lsq_linear(X, y, bounds=(lo, hi)).x
        sol = sol / np.abs(sol).sum()
        out[pos_] = {c: round(float(v), 4) for c, v in zip(cols, sol)}
    return out


# ---- Current rankings (display only) -------------------------------------
# Sean, 2026-10-04: the rating ranks recent production. It is not a
# prediction and does not go into the model. It uses every game through the
# latest completed week, so the 2026 holdout rule does not apply here. No
# engine evaluation may use this output.
RANK_MIN_GAMES = 4   # games counted (two-season window); MIN_GAMES is 3 and too thin to rank
RANK_VOL = {"QB": ("attempts", "Att/g"), "RB": ("carries", "Car/g"),
            "WR": ("targets", "Tgt/g"), "TE": ("targets", "Tgt/g")}


def latest_completed_week():
    """Latest (season, week) in which every regular-season game has a score."""
    from .db import connect
    with connect() as con:
        s = pd.read_sql_query(
            "SELECT season, week, COUNT(*) n, SUM(home_score IS NOT NULL) done FROM schedules "
            "WHERE game_type='REG' GROUP BY season, week", con)
    s = s[s.n == s.done].sort_values(["season", "week"])
    return int(s.season.iloc[-1]), int(s.week.iloc[-1])


def current_ratings(pg=None, season=None, week=None, weights=None, dvp=defense_vs_position):
    """Ranked players through (season, week) inclusive. Returns (season, week, df).
    Ranked = counted games >= RANK_MIN_GAMES, last-5 volume at the market-sized
    floor (VOLUME), and at least one game in `season`. Others are dropped."""
    if season is None:
        season, week = latest_completed_week()
    pg = _prep(pg if pg is not None else player_game_shares())
    res = rating_frame(season, week + 1, pg, weights, dvp)
    if res.empty:
        return season, week, res
    played = pg[(pg.season == season) & (pg.week <= week)].player_id.unique()
    res = res[(res.n_games >= RANK_MIN_GAMES) & _in_pool(res) & res.player_id.isin(played)].copy()
    res["vol"] = [r[f"last5_{RANK_VOL[r.position][0]}"] for _, r in res.iterrows()]
    res = res.sort_values(["position", "rating_z"], ascending=[True, False])
    res["rank"] = res.groupby("position").cumcount() + 1
    return season, week, res.reset_index(drop=True)


_CACHE = {}


def chip_lookup(name_key):
    """{(name_key, position): (chip text, hover title)} for the current ranking.
    Computed once per process. Players below the floor are absent."""
    if "chips" not in _CACHE:
        s, w, df = current_ratings()
        _CACHE["chips"] = {
            (name_key(r["name"]), r.position): (
                f"#{r['rank']} {r.position}",
                f"Position rank by recent production (not the depth chart). "
                f"Rating {r.rating:.1f}, {int(r.n_games)} games.")
            for _, r in df.iterrows()}
    return _CACHE["chips"]


CHIPS_PATH = DATA_DIR / "ratings" / "chips.json"


def write_chips():
    """Dump chip_lookup() to data/ratings/chips.json so the slate and explore pages
    (a long-running server) read the current chips without recomputing a rating."""
    from .odds import name_key
    _CACHE.pop("chips", None)
    d = {f"{k[0]}|{k[1]}": list(v) for k, v in chip_lookup(name_key).items()}
    CHIPS_PATH.write_text(json.dumps(d), encoding="utf-8")
    return CHIPS_PATH


def render_rankings(season, week, df):
    """The full rankings page: one table per position. Plain HTML from templates/rankings.html."""
    import html
    from .config import ROOT
    out = []
    for pos_, label in (("QB", "Quarterbacks"), ("RB", "Running backs"),
                        ("WR", "Wide receivers"), ("TE", "Tight ends")):
        d = df[df.position == pos_]
        rows = "".join(
            f'<tr data-inspect-id="rankings-row"><td class="num">{r["rank"]}</td><td>{html.escape(r["name"])}</td>'
            f'<td>{html.escape(str(r.team))}</td><td class="num">{r.rating:.1f}</td>'
            f'<td class="num">{int(r.n_games)}</td><td class="num">{r.vol:.1f}</td></tr>'
            for _, r in d.iterrows())
        out.append(
            f'<h2 data-inspect-id="rankings-heading-{pos_.lower()}">{label}</h2>'
            f'<div class="tablewrap" data-inspect-id="rankings-table-{pos_.lower()}"><table>'
            f'<thead><tr><th class="num">Rank</th><th>Player</th><th>Team</th><th class="num">Rating</th>'
            f'<th class="num">Games</th><th class="num">{RANK_VOL[pos_][1]} (last 5)</th></tr></thead>'
            f'<tbody>{rows}</tbody></table></div>')
    tpl = (ROOT / "templates" / "rankings.html").read_text()
    return (tpl.replace("__WEEK__", f"{season} Week {week}")
               .replace("__MINGAMES__", str(RANK_MIN_GAMES))
               .replace("__FLOORS__", "QB 10 attempts, RB 5 carries, WR 3 targets, TE 3 targets")
               .replace("__TABLES__", "\n".join(out)))


def write_rankings():
    """Write reports/rankings.html (and reports/trends.html). Called by every report build, so the pages
    refresh whenever a weekly report is built."""
    from .config import ROOT
    season, week, df = current_ratings()
    path = ROOT / "reports" / "rankings.html"
    path.write_text(render_rankings(season, week, df), encoding="utf-8")
    write_chips()
    try:  # the Trends page refreshes on the same build; a failure must not stop a report
        from .trends import write_trends
        write_trends()
    except Exception as e:  # noqa: BLE001
        import sys
        print(f"trends page not rebuilt: {e}", file=sys.stderr)
    return path
