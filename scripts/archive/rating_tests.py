#!/usr/bin/env python
"""Rating Phase 3: Tests A, B, C on 2024-2025 (weights fit on 2021-2023).

Bars are fixed in projects/nfl-props/rating-system-plan-2026-10-04.md. Needs
data/ratings/ from scripts/rating_fit.py. The 2026 holdout is never read.
Run: .venv/bin/python scripts/rating_tests.py
"""
import sys, pathlib, io, contextlib, json
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from nflprops import rating as R
from nflprops.project import project_week, POS_STATS
from nflprops.shares import player_game_shares

TRAIN, TEST = (2021, 2022, 2023), (2024, 2025)
N_BOOT, N_SHUF, SEED = 4000, 20, 7
BAR_A, BAR_B = 0.02, 0.01
STATS = ["passing_yards", "completions", "rushing_yards", "receptions", "receiving_yards"]
CACHE = R.RATINGS_DIR / "proj_cache.parquet"


def load():
    pg = player_game_shares()
    pg = pg[pg.season <= 2025]
    act = pg[["player_id", "season", "week"] + STATS]
    rt = pd.concat([pd.read_parquet(p) for p in sorted(R.RATINGS_DIR.glob("20*_w*.parquet"))],
                   ignore_index=True)
    rt = rt[R._in_pool(rt)]
    return pg, act, rt


def projections(act):
    if CACHE.exists():
        return pd.read_parquet(CACHE)
    frames = []
    for s in TRAIN + TEST:
        for w in sorted(int(x) for x in act[act.season == s].week.unique()):
            with contextlib.redirect_stdout(io.StringIO()):
                p = project_week(s, w)
            if p.empty:
                print(f"  no projections {s} W{w}", file=sys.stderr)
                continue
            p["season"], p["week"] = s, w
            frames.append(p[["player_id", "position", "stat", "proj", "season", "week"]])
    out = pd.concat(frames, ignore_index=True)
    out.to_parquet(CACHE, index=False)
    return out


def boot_ci(per_week, rng):
    """95% interval of the mean over weeks, resampling weeks."""
    v = np.asarray(per_week, float)
    idx = rng.integers(0, len(v), (N_BOOT, len(v)))
    m = v[idx].mean(axis=1)
    return np.percentile(m, [2.5, 97.5]), m


def spearman_by_week(df, a, b, y):
    rows = []
    for (s, w), g in df.groupby(["season", "week"]):
        if len(g) >= 8:
            rows.append((s, w, spearmanr(g[a], g[y])[0], spearmanr(g[b], g[y])[0]))
    return pd.DataFrame(rows, columns=["season", "week", "rho_a", "rho_b"])


def test_a(rt, act, proj, rng):
    out = []
    for pos, stat in R.PRIMARY.items():
        d = rt[(rt.position == pos) & rt.season.isin(TEST)].merge(
            act[["player_id", "season", "week", stat]].rename(columns={stat: "y"}),
            on=["player_id", "season", "week"])
        d["l5"] = d[f"last5_{stat}"]
        d = d.merge(proj[(proj.stat == stat)][["player_id", "season", "week", "proj"]],
                    on=["player_id", "season", "week"], how="left")
        d = d.dropna(subset=["y", "l5", "rating"])
        wk = spearman_by_week(d, "rating", "l5", "y")
        wk["gain"] = wk.rho_a - wk.rho_b
        ci, _ = boot_ci(wk.gain, rng)
        dp = d.dropna(subset=["proj"])
        wp = spearman_by_week(dp, "rating", "proj", "y")
        rand = np.mean([spearmanr(rng.permutation(g.rating.values), g.y)[0]
                        for _, g in d.groupby(["season", "week"]) if len(g) >= 8])
        row = dict(position=pos, stat=stat, n_weeks=len(wk), n=len(d),
                   rho_rating=wk.rho_a.mean(), rho_last5=wk.rho_b.mean(),
                   rho_proj=wp.rho_b.mean(), rho_rating_on_proj_rows=wp.rho_a.mean(),
                   rho_random=rand, gain=wk.gain.mean(), ci_lo=ci[0], ci_hi=ci[1])
        for s in TEST:
            row[f"gain_{s}"] = wk[wk.season == s].gain.mean()
        row["pass"] = bool(row["gain"] >= BAR_A and ci[0] > 0)
        out.append(row)
    return pd.DataFrame(out)


def b_frame(rt, act, proj, shuffle=None):
    d = proj.merge(act.melt(id_vars=["player_id", "season", "week"], var_name="stat", value_name="y"),
                   on=["player_id", "season", "week", "stat"])
    d = d.merge(rt[["player_id", "season", "week", "rating_z"]], on=["player_id", "season", "week"])
    d = d.dropna(subset=["y", "proj", "rating_z"]).reset_index(drop=True)
    if shuffle is not None:
        d["rating_z"] = d.groupby(["position", "season", "week"]).rating_z.transform(
            lambda s: shuffle.permutation(s.values))
    return d


def ols(X, y):
    return np.linalg.lstsq(np.column_stack([np.ones(len(X)), X]), y, rcond=None)[0]


def pred(c, X):
    return np.column_stack([np.ones(len(X)), X]) @ c


def b_result(d, rng, boot=True):
    rows = []
    for pos, stats in POS_STATS.items():
        for stat in stats:
            g = d[(d.position == pos) & (d.stat == stat)]
            tr, te = g[g.season.isin(TRAIN)], g[g.season.isin(TEST)]
            c_new = ols(tr[["proj", "rating_z"]].values, tr.y.values)
            c_ctl = ols(tr[["proj"]].values, tr.y.values)
            te = te.assign(p_new=pred(c_new, te[["proj", "rating_z"]].values),
                           p_ctl=pred(c_ctl, te[["proj"]].values))
            row = dict(position=pos, stat=stat, coef_rating=c_new[2])
            for s in TEST:
                t = te[te.season == s]
                e0 = (t.proj - t.y).abs(); en = (t.p_new - t.y).abs(); ec = (t.p_ctl - t.y).abs()
                per = pd.DataFrame({"w": t.week, "e0": e0, "en": en, "ec": ec}).groupby("w").mean()
                row[f"mae_{s}"] = e0.mean()
                row[f"chg_{s}"] = 1 - en.mean() / e0.mean()
                row[f"ctl_chg_{s}"] = 1 - ec.mean() / e0.mean()
                row[f"vs_ctl_{s}"] = 1 - en.mean() / ec.mean()
                if boot:
                    idx = rng.integers(0, len(per), (N_BOOT, len(per)))
                    m0, mn = per.e0.values[idx].mean(1), per.en.values[idx].mean(1)
                    rel = 1 - mn / m0
                    row[f"lo_{s}"], row[f"hi_{s}"] = np.percentile(rel, [2.5, 97.5])
                    row[f"p_{s}"] = min(1.0, 2 * min((rel <= 0).mean(), (rel >= 0).mean()))
            rows.append(row)
    return pd.DataFrame(rows)


def bh(p):
    p = np.asarray(p); n = len(p); o = np.argsort(p)
    adj = np.empty(n); run = 1.0
    for rank, i in zip(range(n, 0, -1), o[::-1]):
        run = min(run, p[i] * n / rank); adj[i] = run
    return adj


def test_c(rt, act, proj, rng):
    a_rho, b_gain = [], []
    for k in range(N_SHUF):
        sh = np.random.default_rng(100 + k)
        sub = rt[rt.season.isin(TEST)].copy()
        sub["rating"] = sub.groupby(["position", "season", "week"]).rating.transform(
            lambda s: sh.permutation(s.values))
        for pos, stat in R.PRIMARY.items():
            d = sub[sub.position == pos].merge(
                act[["player_id", "season", "week", stat]].rename(columns={stat: "y"}),
                on=["player_id", "season", "week"]).dropna(subset=["y", "rating"])
            a_rho.append((pos, np.mean([spearmanr(g.rating, g.y)[0]
                          for _, g in d.groupby(["season", "week"]) if len(g) >= 8])))
        r = b_result(b_frame(rt, act, proj, shuffle=sh), rng, boot=False)
        b_gain.append(r.assign(gain=(r.vs_ctl_2024 + r.vs_ctl_2025) / 2)[["position", "stat", "gain"]])
    a = pd.DataFrame(a_rho, columns=["position", "rho"]).groupby("position").rho.agg(["mean", "std"])
    b = pd.concat(b_gain).groupby(["position", "stat"]).gain.agg(["mean", "std", "min", "max"])
    return a, b


def spot_check(pg, rt, act, n=20):
    """Re-derive last-5 average and the actual from raw SQL for random rows."""
    import sqlite3
    from nflprops.config import DB_PATH
    con = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    rows = rt[rt.season.isin(TEST)].sample(n, random_state=3)
    bad = 0
    for _, r in rows.iterrows():
        stat = R.PRIMARY[r.position]
        h = con.execute(f"SELECT {stat} FROM player_games WHERE player_id=? AND season_type='REG' "
                        "AND (season<? OR (season=? AND week<?)) ORDER BY season DESC, week DESC LIMIT 5",
                        (r.player_id, r.season, r.season, r.week)).fetchall()
        l5 = np.mean([x[0] or 0 for x in h])
        a = con.execute(f"SELECT {stat} FROM player_games WHERE player_id=? AND season=? AND week=? "
                        "AND season_type='REG'", (r.player_id, r.season, r.week)).fetchone()
        have = act[(act.player_id == r.player_id) & (act.season == r.season) & (act.week == r.week)][stat]
        ok = abs(l5 - r[f"last5_{stat}"]) < 1e-6 and (
            (a is None and have.empty) or (a is not None and not have.empty and a[0] == have.iloc[0]))
        bad += not ok
    con.close()
    return n, bad


def main():
    rng = np.random.default_rng(SEED)
    pg, act, rt = load()
    proj = projections(act)
    A = test_a(rt, act, proj, rng)
    d = b_frame(rt, act, proj)
    B = b_result(d, rng)
    pw = B[[f"p_{s}" for s in TEST]].max(axis=1)
    B["p_max"], B["p_bh"] = pw, bh(pw.values)
    B["pass"] = [bool(all(r[f"chg_{s}"] >= BAR_B and r[f"lo_{s}"] > 0 for s in TEST) and r.p_bh < 0.05)
                 for _, r in B.iterrows()]
    Ca, Cb = test_c(rt, act, proj, rng)
    n, bad = spot_check(pg, rt, act)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
    print("TEST A\n", A.round(4).to_string(index=False))
    print("\nTEST B\n", B.round(4).to_string(index=False))
    print("\nTEST C (A, shuffled Spearman by position)\n", Ca.round(4))
    print("\nTEST C (B, shuffled gain vs control)\n", Cb.round(5))
    print(f"\nSPOT CHECK: {n} rows, {bad} mismatches")
    print(f"rows: B train {int(d.season.isin(TRAIN).sum())}, test {int(d.season.isin(TEST).sum())}")
    R.RATINGS_DIR.mkdir(exist_ok=True)
    (R.RATINGS_DIR / "test_results.json").write_text(json.dumps(
        {"A": A.to_dict("records"), "B": B.to_dict("records"),
         "C_A": Ca.reset_index().to_dict("records"), "C_B": Cb.reset_index().to_dict("records"),
         "spot_check": {"n": n, "mismatch": bad}}, indent=1, default=float))


if __name__ == "__main__":
    main()
