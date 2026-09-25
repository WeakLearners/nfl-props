"""
Phase 3, Rung 2 -- ridge regression. Is the signal in these features linear?

THE LADDER (phase3-model.md). Rung 1 is the naive baseline, already on disk in
baselines.json. This is rung 2. Rung 3 (LightGBM) runs only if this one clears
its bar: beat the per-stat baseline by >= 2% MAE on a majority of the six
stats, in BOTH validation seasons.

WHY RIDGE AND NOT PLAIN REGRESSION. Many features here measure nearly the same
thing -- team_snap_share_l4 and team_snap_share_l8 move together almost
perfectly. When two columns say the same thing, plain least squares cannot
decide which deserves credit, so it lurches: +40 on one, -39 on the other.
Those weights do not survive a new season. Ridge makes large offsetting weights
expensive, so the credit gets split instead.

PROTOCOL, from the spec and from decisions #10, #14, #17:

- Train seasons 2019-2023, FIXED for both validation years. 2024 and 2025 are
  scored separately and must each clear the bar. Holding the training set
  constant is what isolates the season effect -- the whole reason decision #17
  added a second validation year. Giving 2025 an extra year of training data
  would confound "a better season" with "more data".
- Walk-forward inside the validation season: predicting week n+1 trains on
  2019-2023 plus that season's own weeks 1..n. A model that has seen week 14
  while predicting week 9 is being flattered, not tested.
- Per-stat populations (decision #14). The carries model is scored on players
  who carry the ball. Screens are leak-gated -- a player's qualifying average
  uses games strictly before the one being scored.
- The model TRAINS on the same screened population it is scored on. Training on
  all 43,672 rows would fill the fit with fourth-string players at zero targets
  and drag every prediction toward zero.
- MAE decides. RMSE is reported beside it, never optimized. If the two ever
  disagree about a winner, that disagreement is itself the finding.

COMPOSED VS DIRECT. Two routes to receiving yards: predict targets, predict
yards-per-target, multiply -- or predict yards in one shot. Both are built.
This is the measurement that tests decision #1's usage/conversion split.

COUNTS. Targets and carries cannot be negative and their spread grows with
their size. Plain least squares knows neither fact. Each usage model is
therefore fit twice: on the raw count, and on log(1+y) with the prediction
mapped back. Both are reported. If the gap is negligible we stop worrying
about it and say so in the decision record.

Read-only against features.db. Writes nothing to any database.
"""
import json, sqlite3, warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, RidgeCV

warnings.filterwarnings("ignore")

ROOT = Path.home() / "Code/nfl-props"
DB = ROOT / "data/features.db"
BASELINES = ROOT / "data/baselines.json"
OUT = ROOT / "data/rung2_ridge.json"

TRAIN_SEASONS = list(range(2019, 2024))     # 2019-2023
VALIDATION_SEASONS = (2024, 2025)
MIN_PER_GAME = 3.0
ALPHAS = np.logspace(-2, 4, 25)

# Columns that are never features: identifiers, outcomes, and the raw timestamp.
ID_COLS = {"player_id", "player_display_name", "game_id", "team", "opponent_team",
           "season", "week", "kickoff_utc"}
CATEGORICAL = ["position", "position_group", "season_type", "roof_state", "surface",
               "injury_report_status", "history_level"]

# Decision #39 (2026-09-24): the QB passing features added to training_rows
# alongside y_passing_yards. build_design() excludes these by default -- this
# is what stops every EXISTING model (targets, carries, receiving_yards,
# rushing_yards, receptions) from silently changing when a new column lands
# in the table. build_design() has no allowlist; every non-ID/non-y column in
# training_rows becomes a feature for every stat by construction, so a new
# additive column is not free -- it changes every other model's design
# matrix and refit unless explicitly excluded. Caught by rerunning
# vol_head_to_head.py right after this migration: pooled MAE on
# targets/carries/receiving_yards/rushing_yards moved by hundredths, purely
# from these 15 new columns entering rung2_ridge's ridge fit uninvited.
# include_passing=True (only the passing-yards model script passes this)
# is the one place they are meant to be used.
PASSING_COLS = {
    f"pass_attempts_l{w}" for w in (4, 8)
} | {"pass_attempts_career"} | {
    f"pass_completions_l{w}" for w in (4, 8)
} | {"pass_completions_career"} | {
    f"{n}_l{w}" for n in ("yards_per_attempt", "completion_rate", "pass_adot") for w in (4, 8)
} | {
    f"{n}_career" for n in ("yards_per_attempt", "completion_rate", "pass_adot")
}


def load():
    con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True, timeout=60)
    df = pd.read_sql("SELECT * FROM training_rows ORDER BY player_id, season, week", con)
    con.close()
    return df


def add_derived(df):
    """Conversion targets, and the leak-gated qualifying screens."""
    df["y_yards_per_target"] = np.where(df.y_targets > 0, df.y_receiving_yards / df.y_targets, np.nan)
    df["y_yards_per_carry"] = np.where(df.y_carries > 0, df.y_rushing_yards / df.y_carries, np.nan)
    g = df.groupby("player_id", sort=False)
    # expanding mean of games strictly before this one -- the same screen the
    # baselines used, so model and baseline are scored on identical rows.
    df["qual_targets"] = g["y_targets"].transform(lambda s: s.shift(1).expanding().mean()) >= MIN_PER_GAME
    df["qual_carries"] = g["y_carries"].transform(lambda s: s.shift(1).expanding().mean()) >= MIN_PER_GAME
    # Decision #39: passing_yards qualifying screen. pass_attempts_career is
    # already a shift(1)-before-expanding-mean feature (built the same way
    # as qual_targets/qual_carries use their own y_ column), so it can be
    # read directly rather than re-derived from a y_attempts label that
    # doesn't exist.
    df["qual_attempts"] = df["pass_attempts_career"].fillna(0) >= MIN_PER_GAME
    return df


def build_design(df, include_passing=False):
    """One-hot the categoricals, keep the numerics, add a missingness flag per
    numeric column that has any. Standing rule #10: ignorance gets a column.

    include_passing=False (the default, and every existing call site) excludes
    PASSING_COLS -- see that constant's comment. Only the passing_yards model
    should ever pass True."""
    y_cols = [c for c in df.columns if c.startswith("y_")]
    drop = ID_COLS | set(y_cols) | {"qual_targets", "qual_carries", "qual_attempts"}
    if not include_passing:
        drop = drop | PASSING_COLS
    feat = [c for c in df.columns if c not in drop]

    num = [c for c in feat if c not in CATEGORICAL]
    X = df[num].apply(pd.to_numeric, errors="coerce")

    miss = X.columns[X.isna().any()].tolist()
    for c in miss:
        X[c + "__isna"] = X[c].isna().astype(float)

    cat = pd.get_dummies(df[CATEGORICAL].astype("string").fillna("__unknown__"),
                         prefix=CATEGORICAL, dummy_na=False, dtype=float)
    return pd.concat([X, cat], axis=1), miss


SPECS = [
    # (layer, stat, target column, screen, count?)
    ("usage",      "targets",          "y_targets",          "qual_targets", True),
    ("usage",      "carries",          "y_carries",          "qual_carries", True),
    ("conversion", "yards_per_target", "y_yards_per_target", "qual_targets", False),
    ("conversion", "yards_per_carry",  "y_yards_per_carry",  "qual_carries", False),
    ("final",      "receiving_yards",  "y_receiving_yards",  "qual_targets", False),
    ("final",      "rushing_yards",    "y_rushing_yards",    "qual_carries", False),
    ("final",      "receiving_tds",    "y_receiving_tds",    "qual_targets", False),
    ("final",      "rushing_tds",      "y_rushing_tds",      "qual_carries", False),
    # Decision #39 (2026-09-24): passing_yards, added once training_rows
    # carried the QB passing features + y_passing_yards label. Appended, not
    # inserted -- every existing SPECS consumer keys by name, not position.
    ("final",      "passing_yards",    "y_passing_yards",    "qual_attempts", False),
]


def fit_predict(Xtr, ytr, Xte, log_space=False, alpha=None):
    """Median-impute and standardize using TRAIN ONLY, then ridge.

    Fitting the scaler on all rows would leak the validation season's
    distribution into training -- a quiet leak that no row-level audit catches.
    """
    med = Xtr.median()
    a = Xtr.fillna(med)
    b = Xte.fillna(med)
    # A column can be entirely missing inside one fold -- an early-season week
    # where no game had measured weather, say. Its median is then NaN and the
    # fill above does nothing. Send those to 0; the __isna flag already carries
    # the fact that the value was unknown, so no information is lost.
    a = a.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    b = b.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    mu, sd = a.mean(), a.std().replace(0, 1.0)
    a = ((a - mu) / sd).to_numpy(float)
    b = ((b - mu) / sd).to_numpy(float)

    t = np.log1p(np.clip(ytr, 0, None)) if log_space else ytr
    if alpha is None:
        m = RidgeCV(alphas=ALPHAS).fit(a, t)
        alpha = float(m.alpha_)
    else:
        m = Ridge(alpha=alpha).fit(a, t)
    p = m.predict(b)
    if log_space:
        p = np.expm1(p)
    return p, alpha, m


def main():
    df = add_derived(load())
    X_all, miss_cols = build_design(df)
    print(f"design matrix: {X_all.shape[0]:,} rows x {X_all.shape[1]} columns "
          f"({len(miss_cols)} numeric columns carry a missingness flag)")

    in_train = df.season.isin(TRAIN_SEASONS).to_numpy()
    preds = {}          # (season, stat) -> DataFrame(row_idx, y, yhat)
    alphas_used = {}
    coef_report = {}

    for season in VALIDATION_SEASONS:
        weeks = sorted(df.loc[df.season == season, "week"].unique())
        for layer, stat, tcol, screen, is_count in SPECS:
            variants = ["raw"] + (["log1p"] if is_count else [])
            for variant in variants:
                key = (season, stat, variant)
                rows_out = []
                alpha = None
                for i in range(1, len(weeks)):
                    pw = weeks[i]
                    prior = df.season.eq(season) & df.week.isin(weeks[:i])
                    tr_mask = (in_train | prior.to_numpy()) & df[screen].to_numpy() & df[tcol].notna().to_numpy()
                    te_mask = (df.season.eq(season) & df.week.eq(pw)).to_numpy() & df[screen].to_numpy() & df[tcol].notna().to_numpy()
                    if te_mask.sum() == 0 or tr_mask.sum() < 200:
                        continue
                    yhat, alpha, model = fit_predict(
                        X_all[tr_mask], df.loc[tr_mask, tcol].to_numpy(float),
                        X_all[te_mask], log_space=(variant == "log1p"),
                        alpha=alpha if i > 1 else None)   # tune once on step 1, reuse
                    rows_out.append(pd.DataFrame({
                        "idx": np.flatnonzero(te_mask),
                        "y": df.loc[te_mask, tcol].to_numpy(float),
                        "yhat": yhat}))
                    if i == len(weeks) - 1 and variant == "raw":
                        coef_report[(season, stat)] = pd.Series(
                            model.coef_, index=X_all.columns).sort_values(key=abs, ascending=False)
                if rows_out:
                    preds[key] = pd.concat(rows_out, ignore_index=True)
                    alphas_used[key] = alpha
        print(f"  {season}: fitted")

    # ---- composed final stats: usage x conversion -------------------------
    for season in VALIDATION_SEASONS:
        for stat, ucol, ccol, ycol in [
            ("receiving_yards", "targets", "yards_per_target", "y_receiving_yards"),
            ("rushing_yards",   "carries", "yards_per_carry",  "y_rushing_yards")]:
            u = preds.get((season, ucol, "raw"))
            c = preds.get((season, ccol, "raw"))
            if u is None or c is None:
                continue
            m = u.merge(c, on="idx", suffixes=("_u", "_c"))
            m["yhat"] = np.clip(m.yhat_u, 0, None) * np.clip(m.yhat_c, 0, None)
            m["y"] = df.loc[m.idx, ycol].to_numpy(float)
            preds[(season, stat, "composed")] = m[["idx", "y", "yhat"]]

    # ---- score -------------------------------------------------------------
    base = json.loads(BASELINES.read_text())["baselines"]
    layer_of = {s[1]: s[0] for s in SPECS}
    out, table = {}, []
    for (season, stat, variant), p in sorted(preds.items()):
        err = p.y - p.yhat
        mae, rmse = float(err.abs().mean()), float(np.sqrt((err ** 2).mean()))
        b = base[str(season)][layer_of[stat]][stat]
        bmae = b[b["winner"]]["mae"]
        lift = (bmae - mae) / bmae * 100
        out.setdefault(str(season), {}).setdefault(stat, {})[variant] = dict(
            mae=round(mae, 4), rmse=round(rmse, 4), n=int(len(p)),
            baseline_mae=bmae, baseline=b["winner"],
            lift_pct=round(lift, 2), alpha=alphas_used.get((season, stat, variant)))
        table.append((season, layer_of[stat], stat, variant, mae, bmae, lift, len(p)))

    OUT.write_text(json.dumps(dict(
        rung=2, model="ridge",
        protocol="train 2019-2023 fixed + walk-forward within validation season",
        bar="beat baseline by >= 2% MAE on a majority of the six stats, both seasons",
        results=out), indent=2))

    print(f"\n{'season':<8}{'layer':<11}{'stat':<17}{'variant':<10}{'modelMAE':>10}{'baseMAE':>10}{'lift%':>8}{'n':>7}")
    for r in sorted(table, key=lambda r: (r[1], r[2], r[3], r[0])):
        print(f"{r[0]:<8}{r[1]:<11}{r[2]:<17}{r[3]:<10}{r[4]:>10.4f}{r[5]:>10.4f}{r[6]:>+8.2f}{r[7]:>7}")

    print("\n--- top 12 ridge weights, 2025, raw variant ---")
    for stat in ("targets", "carries", "yards_per_target", "receiving_yards"):
        c = coef_report.get((2025, stat))
        if c is None:
            continue
        print(f"\n{stat}:")
        for name, v in c.head(12).items():
            print(f"   {v:>+9.4f}  {name}")

    print(f"\nwritten -> {OUT}")


if __name__ == "__main__":
    main()
