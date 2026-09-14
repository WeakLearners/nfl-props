"""Shared ranking logic for alternate-line legs.

Both the live Slack report and the historical retrospective call this, so a
backtested ranking and a posted ranking can never drift apart. Weights come
from scripts/validate_reliability.py (777 legs, 2025 W6/W10/W14): empirical
clear rate ranks outcomes at AUC 0.628 vs the price's 0.534, so history carries
the larger weight.
"""
import pandas as pd
import numpy as np

from . import reliability as R
from .odds import name_key

W_HIST, W_PRICE = 0.65, 0.35
MIN_HISTORY_GAMES = 6
PRICE_LO, PRICE_HI = -320, -180


def rank(alt, season, week):
    """alt: DataFrame with stat, player_book, line, price, game. Returns ranked."""
    alt = alt[(alt.price >= PRICE_LO) & (alt.price <= PRICE_HI)].copy()
    if alt.empty:
        return alt
    alt["player_key"] = alt.player_book.map(name_key)

    hist = R._history(season, week, R.DEFAULT_LOOKBACK)
    inj, sp = R.risk_flags(season, week)
    cont = R.role_continuity(season, week)
    snaps = R.snap_share(season, week)

    tm = hist[["player_key", "team", "position"]].drop_duplicates("player_key")
    alt = (alt.merge(tm, on="player_key", how="left")
              .merge(sp.drop_duplicates("team"), on="team", how="left")
              .merge(inj[["player_key", "report_status"]].drop_duplicates("player_key"),
                     on="player_key", how="left")
              .merge(cont[["player_key", "changed_team"]].drop_duplicates("player_key"),
                     on="player_key", how="left")
              .merge(snaps, on="player_key", how="left"))

    res = alt.apply(lambda r: R.clear_rate(hist, r.player_key, r.stat, r.line),
                    axis=1, result_type="expand")
    alt[["hit_rate", "n_games", "dnp_rate"]] = res
    alt["mkt_fair"] = alt.price.map(R.fair)

    alt["blend"] = np.where(alt.hit_rate.notna(),
                            W_PRICE * alt.mkt_fair + W_HIST * alt.hit_rate,
                            alt.mkt_fair)
    alt["penalty"] = (
        alt.report_status.isin(["Questionable"]).astype(float) * 0.08
        + alt.report_status.isin(["Doubtful", "Out"]).astype(float) * 0.60
        + (alt.dnp_rate.fillna(0) > 0.25).astype(float) * 0.04
        + (alt.abs_spread.fillna(0) > 9).astype(float) * 0.02
        + (alt.n_games.fillna(0) < MIN_HISTORY_GAMES).astype(float) * 0.10
        # A clear rate earned on another team describes a role he may not have.
        + alt.changed_team.fillna(False).astype(float) * 0.06
    )
    alt["score"] = alt.blend - alt.penalty
    return alt.sort_values("score", ascending=False)


def starters_only(r, floor=None):
    """Drop anyone who is not reliably on the field. A player with no snap
    history at all is dropped too -- absence of evidence is not a starter."""
    floor = R.STARTER_SNAP_PCT if floor is None else floor
    return r[r.snap_share.fillna(0) >= floor]


def stars_only(r, season, week):
    """Keep only fantasy-relevant players on lines worth watching."""
    pool = R.star_pool(season, week)
    r = r[r.player_key.isin(set(pool.player_key))]
    return r[r.apply(lambda x: x.line >= R.STAR_MIN_LINE.get(x.stat, 0), axis=1)]


def best_legs(alt, season, week, n=8, one_per_game=False, stars=False,
              allow_questionable=False):
    """Top legs, one per player. Optionally one per game, which is what a
    multi-game parlay wants -- two legs on the same offense are correlated, and
    that concentrates exactly the risk the ranking is trying to avoid."""
    r = rank(alt, season, week)
    if r.empty:
        return r
    r = r[r.n_games >= MIN_HISTORY_GAMES]
    r = starters_only(r)
    if not allow_questionable:
        r = r[~r.report_status.isin(["Questionable", "Doubtful", "Out"])]
    if stars:
        r = stars_only(r, season, week)
    r = r.drop_duplicates("player_key")
    if one_per_game and "game" in r:
        r = r.drop_duplicates("game")
    return r.head(n)


# Calibration of the blend score against real outcomes, fitted on 539
# starter-filtered validation legs. Close to identity, but the fit is only
# well-sampled below a ~0.78 blend (n=480); above that there are 59 legs and
# above 0.85 there are 9. Extrapolating the high end into a 12-way product is
# how a parlay estimate turns into nonsense, so survival() reports the market's
# view alongside the model's and never claims the difference is edge.
CAL_SLOPE, CAL_INTERCEPT = 1.078, -0.077
BLEND_TRUSTED_MAX = 0.78


def calibrated_p(blend):
    """Blend score -> probability, CLAMPED at the top of the well-sampled range.

    Above a 0.78 blend the validation set holds 59 legs and the fit is not
    trustworthy; extrapolating there produced a per-leg probability ~13 points
    above the market when the measured top-tercile edge is ~3 points. Clamping
    caps the estimate at 76.4%, which is what top-tercile starter legs actually
    hit (74.9%-77%). Better to understate a good leg than to compound an
    invented edge twelve times.
    """
    b = min(float(blend), BLEND_TRUSTED_MAX)
    p = CAL_SLOPE * b + CAL_INTERCEPT
    return float(min(max(p, 0.01), 0.97))


def survival(legs):
    """Return (model_estimate, market_implied, payout_multiple).

    The two probabilities will disagree, and the gap is NOT a measured edge --
    it is mostly the model extrapolating past where it was validated. Show
    both; let the reader see the disagreement rather than trusting one number.
    """
    import numpy as np
    blend = W_PRICE * legs.mkt_fair + W_HIST * legs.hit_rate
    model = float(np.prod([calibrated_p(b) for b in blend]))
    raw_imp = legs.price.apply(lambda a: (-a) / ((-a) + 100) if a < 0 else 100 / (a + 100))
    market = float(raw_imp.prod())
    dec = legs.price.apply(lambda a: 1 + (100 / (-a) if a < 0 else a / 100))
    return model, market, float(dec.prod())


def notes(row):
    out = []
    if row.report_status in ("Questionable", "Doubtful", "Out"):
        out.append(row.report_status.upper())
    if pd.notna(row.dnp_rate) and row.dnp_rate > 0.25:
        out.append(f"misses {row.dnp_rate*100:.0f}%")
    if pd.notna(row.abs_spread) and row.abs_spread > 9:
        out.append(f"spread {row.abs_spread:.0f}")
    if getattr(row, "changed_team", False) is True:
        out.append("NEW TEAM")
    if pd.notna(getattr(row, "snap_share", None)):
        out.append(f"{row.snap_share*100:.0f}% snaps")
    return ", ".join(out)
