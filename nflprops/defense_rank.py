"""Defense rankings: total, run, pass, and by position. Display only, never a model input.

Rank 1 = toughest defense (allows the least). Built from what each defense
allowed, per game, from player_games and schedules. Each game is adjusted for
the offense faced and weighted by recency. See
projects/nfl-props/defense-rankings-2026-10-04.md.
"""
import numpy as np
import pandas as pd

from .project import DECAY, PRIOR_SEASON_W, K_VOL_GAMES

LOOKBACK_SEASONS = 1  # two seasons: the current one and the one before
# Each rank is the order of ONE stat, the same number the page shows.
RANK_STAT = {"total": "epa_play", "run": "epa_rush", "pass": "epa_db",
             "qb": "qb_ypg", "rb": "rb_ypg", "wr": "wr_ypg", "te": "te_ypg"}
# stat -> (numerator, denominator or None for a per-game value)
STATS = {
    "epa_db": ("pass_epa", "dropbacks"), "pass_ypg": ("pass_yds", None),
    "epa_rush": ("rush_epa", "carries"), "rush_ypg": ("rush_yds", None),
    "ypc": ("rush_yds", "carries"), "epa_play": ("epa", "plays"),
    "qb_ypg": ("qb_yds", None), "rb_ypg": ("rb_yds", None),
    "wr_ypg": ("wr_yds", None), "te_ypg": ("te_yds", None),
}
POSITIONS = (("qb", "QB"), ("rb", "RB"), ("wr", "WR"), ("te", "TE"))


def team_games(con, season, week):
    """One row per (game, defense): what the offense did, plus points allowed."""
    q = """SELECT game_id, season, week, team AS off, opponent_team AS defn,
                  SUM(COALESCE(passing_epa,0)) pass_epa, SUM(COALESCE(rushing_epa,0)) rush_epa,
                  SUM(COALESCE(attempts,0)+COALESCE(sacks_suffered,0)) dropbacks,
                  SUM(COALESCE(passing_yards,0)) pass_yds, SUM(COALESCE(carries,0)) carries,
                  SUM(COALESCE(rushing_yards,0)) rush_yds,
                  SUM(CASE WHEN position='QB' THEN COALESCE(passing_yards,0) ELSE 0 END) qb_yds,
                  SUM(CASE WHEN position='RB' THEN COALESCE(rushing_yards,0)+COALESCE(receiving_yards,0) ELSE 0 END) rb_yds,
                  SUM(CASE WHEN position='WR' THEN COALESCE(receiving_yards,0) ELSE 0 END) wr_yds,
                  SUM(CASE WHEN position='TE' THEN COALESCE(receiving_yards,0) ELSE 0 END) te_yds
           FROM player_games WHERE season_type='REG' AND season >= ?
             AND (season < ? OR (season = ? AND week <= ?))
           GROUP BY game_id, season, week, team, opponent_team"""
    g = pd.read_sql_query(q, con, params=[season - LOOKBACK_SEASONS, season, season, week])
    s = pd.read_sql_query("SELECT game_id, home_team, home_score, away_team, away_score FROM schedules "
                          "WHERE game_type='REG' AND home_score IS NOT NULL", con)
    pts = pd.concat([s.rename(columns={"home_team": "defn", "away_score": "pa"})[["game_id", "defn", "pa"]],
                     s.rename(columns={"away_team": "defn", "home_score": "pa"})[["game_id", "defn", "pa"]]])
    g = g.merge(pts, on=["game_id", "defn"], how="left")
    g["epa"], g["plays"] = g.pass_epa + g.rush_epa, g.dropbacks + g.carries
    return g


def _val(g, stat):
    num, den = STATS[stat]
    return g[num] / g[den].where(g[den] > 0) if den else g[num].astype(float)


def rank_defenses(g, season):
    """32 rows: team, adjusted stats, and one rank per RANK_STAT key (1 = allows the least of that stat)."""
    g = g.copy()
    for st in STATS:
        g[st] = _val(g, st)
        # Opponent adjustment: remove how far the offense faced sits from the league
        # (the offense's own average over the window minus the league average).
        off = g.groupby("off")[st].transform("mean")
        g[st] = g[st] - (off - g[st].mean())
    g = g.sort_values(["defn", "season", "week"], ascending=[True, False, False])
    rk = g.groupby("defn").cumcount().values
    prior = g.season.values < season
    g["w"] = DECAY ** rk * np.where(prior, PRIOR_SEASON_W, 1.0)
    out = pd.DataFrame(index=sorted(g.defn.unique()))
    for st, (num, den) in STATS.items():
        w = g.w * (g[den] if den else 1.0)
        ok = g[st].notna()
        mean = np.average(g.loc[ok, st], weights=w[ok])
        k = K_VOL_GAMES * (w[ok].groupby(g.defn[ok]).sum() / g.w[ok].groupby(g.defn[ok]).sum())
        a = (g.loc[ok, st] * w[ok]).groupby(g.defn[ok]).sum()
        out[st] = (a + k * mean) / (w[ok].groupby(g.defn[ok]).sum() + k)
    cur = g[g.season == season]
    out["pa_pg"] = cur.groupby("defn").pa.mean()
    out["games"] = cur.groupby("defn").size()
    for r, st in RANK_STAT.items():
        out[f"{r}_rank"] = out[st].rank(method="first").astype(int)  # lowest allowed = 1
    return out.reset_index(names="team")


# key, button label, rank stat, header unit, extra display columns (stat, header, decimals)
VIEWS = (("total", "Total", "epa_play", "EPA/play", ()),
         ("run", "Run", "epa_rush", "EPA/rush", (("rush_ypg", "Rush yds/g", 1), ("ypc", "Yds/carry", 2))),
         ("pass", "Pass", "epa_db", "EPA/dropback", (("pass_ypg", "Pass yds/g", 1),)))


def current_defense_ranks():
    """(season, week, df) through the latest completed week."""
    from .db import connect
    from .rating import latest_completed_week
    season, week = latest_completed_week()
    with connect() as con:
        g = team_games(con, season, week)
    return season, week, rank_defenses(g, season)


def render_defenses(season, week, df):
    """The Defenses view embedded in reports/rankings.html: Total, Run, Pass tables and a By position table, switched by buttons."""
    import html
    from .config import ROOT
    btns, tables = [], []

    def button(key, label):
        btns.append(f'<button type="button" aria-pressed="{str(key == "total").lower()}" data-v="{key}" '
                    f'data-inspect-id="defenses-btn-{key}">{label}</button>')

    def wrap(key, head, rows):
        tables.append(
            f'<div class="tablewrap" data-v="{key}" data-inspect-id="defenses-table-{key}"{"" if key == "total" else " hidden"}>'
            f'<table><thead><tr>{head}</tr></thead><tbody>{rows}</tbody></table></div>')

    for key, label, col, unit, extra in VIEWS:
        d = df.sort_values(f"{key}_rank")
        rows = "".join(
            f'<tr data-inspect-id="defenses-row-{key}"><td class="num">{r[f"{key}_rank"]}</td>'
            f'<td>{html.escape(r.team)}</td><td class="num">{r[col]:+.3f}</td>'
            + "".join(f'<td class="num">{r[c]:.{dp}f}</td>' for c, _, dp in extra)
            + f'<td class="num">{r.pa_pg:.1f}</td></tr>' for _, r in d.iterrows())
        head = (f'<th class="num">Rank</th><th>Team</th><th class="num">{unit} allowed</th>'
                + "".join(f'<th class="num">{n}</th>' for _, n, _ in extra) + '<th class="num">Pts/g allowed</th>')
        button(key, label)
        wrap(key, head, rows)
    d = df.sort_values("team")
    rows = "".join(
        '<tr data-inspect-id="defenses-row-pos"><td>' + html.escape(r.team) + "</td>"
        + "".join(f'<td class="num">{r[f"{k}_rank"]} <span class="yds">{r[f"{k}_ypg"]:.1f}</span></td>'
                  for k, _ in POSITIONS) + "</tr>" for _, r in d.iterrows())
    head = ('<th>Team</th>' + "".join(
        f'<th class="num" data-inspect-id="defenses-col-{k}">vs {n} rank / yds</th>' for k, n in POSITIONS))
    button("pos", "By position")
    wrap("pos", head, rows)
    tpl = (ROOT / "templates" / "defenses.html").read_text()
    return (tpl.replace("__WEEK__", f"{season} Week {week}").replace("__BUTTONS__", "".join(btns))
               .replace("__TABLES__", "\n".join(tables)))
