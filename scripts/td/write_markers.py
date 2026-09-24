#!/usr/bin/env python
"""Write one touchdown marker per game, for the report listing to read.

Output is reports/td_markers_{season}_w{week}.json, keyed by the AWAY-HOME
abbreviation pair the report filenames already use, so the listing can join
on a name it already has.

Confidence is the lower of the model's number and FanDuel's, the same rule
every other leg on the page follows. A player the book has not priced keeps
the model's number alone and is labelled model_only, so an unchecked figure
can never be mistaken for a checked one.

The book half costs nothing extra. player_anytime_td joined the market list
in 2ee8b88, so the pull the report run already makes returns it.

PHASE3 COMBINED NUMBER, added 2026-09-24 (decision #30). Alongside the
established marker above, each entry now also carries the Phase 3 rush+rec
touchdown models' combined P(any TD) for the SAME marked player: p_rush and
p_rec from data/shipping/{rushing_tds,receiving_tds}.pkl (decision #29,
trained through 2025), combined as 1-(1-p_rush)(1-p_rec). That formula
assumes the two events are independent for one player in one game; measured
against 2019-2025 training_rows, they are not quite -- actual P(rush AND
rec) runs about 44% BELOW what independence predicts (weak negative
association, a player who scores on the ground is slightly less likely to
also score through the air in the same game), while the practical quantity
here, P(at least one), is off by only +0.4 percentage points, because both
events are individually rare. This is a second, independent read on the
same game, not a correction to the first -- reported side by side on
purpose, never averaged together. It is scoped to the one player the
existing marker already names; a per-card number for every player would be
a larger, riskier change to make_report.py's own DATA payload and is not
part of this pass.

Requires data/shipping/{rushing_tds,receiving_tds}.pkl to exist
(scripts/retrain_td_models.py). If they don't, phase3 fields are omitted
rather than the whole run failing -- a report with only the established
marker is the normal case this script has always produced.
"""
import argparse
import json
import pathlib
import sys

import joblib
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from nflprops.db import team_abbr_map  # noqa: E402
from nflprops.odds import devig_yesno, event_props, name_key  # noqa: E402
from nflprops.report import upcoming_events  # noqa: E402
from nflprops.tdmodel import design_matrix, features  # noqa: E402
from predict import train_through  # noqa: E402
from build_training_table import pregame_features  # noqa: E402
from rung2_ridge import build_design  # noqa: E402

SHIPPING = ROOT / "data" / "shipping"


def phase3_combined(season, week):
    """P(rush TD), P(rec TD), combined P(any TD) per rostered player, from
    the shipped Phase 3 models -- the SAME pregame_features() used for
    training, per decision #30's hard rule. Returns an empty frame (not an
    error) if the shipped artifacts are missing."""
    rush_path, rec_path = SHIPPING / "rushing_tds.pkl", SHIPPING / "receiving_tds.pkl"
    if not (rush_path.exists() and rec_path.exists()):
        print("  ! data/shipping/*.pkl not found -- skipping phase3 combined numbers")
        return pd.DataFrame(columns=["player_id", "p_rush", "p_rec", "p_combined"])

    pf = pregame_features(season, week)
    if pf.empty:
        return pd.DataFrame(columns=["player_id", "p_rush", "p_rec", "p_combined"])
    X, _ = build_design(pf)
    X = X.astype(float)

    out = pf[["player_id"]].copy()
    for stat, path, col in [("rushing_tds", rush_path, "p_rush"), ("receiving_tds", rec_path, "p_rec")]:
        art = joblib.load(path)
        Xs = X.reindex(columns=art["columns"], fill_value=0.0)
        out[col] = art["model"].predict_proba(Xs)[:, 1]
    out["p_combined"] = 1 - (1 - out.p_rush) * (1 - out.p_rec)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("season", type=int)
    ap.add_argument("week", type=int)
    ap.add_argument("--no-book", action="store_true")
    args = ap.parse_args()

    model, n_train = train_through(args.season, args.week)
    f = features(args.season, args.week)
    if f.empty:
        sys.exit(f"no roster rows for {args.season} W{args.week}")
    f["model_p"] = model.predict_proba(design_matrix(f))[:, 1]
    f["player_key"] = f.full_name.map(name_key)

    book = pd.DataFrame(columns=["player_key", "book_p", "devigged"])
    if not args.no_book:
        ev = upcoming_events()
        abbr = team_abbr_map()
        frames = []
        for e in ev.itertuples():
            a, h = abbr.get(e.away_team), abbr.get(e.home_team)
            if not a or not h:
                continue
            raw, _ = event_props(e.id)
            td = devig_yesno(raw)
            if td.empty:
                print(f"  ! no anytime-TD prices yet: {a}@{h}")
                continue
            td["player_key"] = td.player_book.map(name_key)
            frames.append(td[["player_key", "book_p", "devigged"]])
        if frames:
            book = pd.concat(frames, ignore_index=True).drop_duplicates("player_key")

    f = f.merge(book, on="player_key", how="left")
    # min(), never mean: the standing rule is to take the less optimistic of
    # the two whenever both exist.
    f["conf"] = f[["model_p", "book_p"]].min(axis=1)
    # book_vig says the book number still carries the vig, which it does
    # whenever FanDuel quotes anytime-TD one-sided. It is an overestimate,
    # and min() therefore hands the decision to the model more often.
    f["basis"] = f.book_p.notna().map({True: "model+book", False: "model_only"})
    f.loc[f.book_p.notna() & ~f.get("devigged", False).fillna(False), "basis"] = "model+book_vig"

    f = f.dropna(subset=["game_id", "conf"])
    top = f.loc[f.groupby("game_id")["conf"].idxmax()]

    p3 = phase3_combined(args.season, args.week)
    top = top.merge(p3, on="player_id", how="left") if "player_id" in top.columns else top.assign(
        p_rush=pd.NA, p_rec=pd.NA, p_combined=pd.NA)

    out = {}
    for r in top.itertuples():
        # game_id is "2026_02_MIN_CHI" -> away MIN, home CHI.
        parts = str(r.game_id).split("_")
        if len(parts) < 4:
            continue
        key = f"{parts[2]}-{parts[3]}"
        entry = {
            "player": r.full_name, "position": r.position, "team": r.team,
            "conf": round(float(r.conf), 3),
            "model_p": round(float(r.model_p), 3),
            "book_p": None if pd.isna(r.book_p) else round(float(r.book_p), 3),
            "basis": r.basis,
        }
        p_combined = getattr(r, "p_combined", None)
        if p_combined is not None and pd.notna(p_combined):
            entry["phase3"] = {
                "p_rush": round(float(r.p_rush), 3),
                "p_rec": round(float(r.p_rec), 3),
                "p_combined": round(float(p_combined), 3),
            }
        out[key] = entry

    path = ROOT / "reports" / f"td_markers_{args.season}_w{args.week}.json"
    path.write_text(json.dumps(out, indent=1, sort_keys=True))
    print(f"trained on {n_train:,} player-weeks; {len(out)} markers -> {path.name}")
    for k, v in sorted(out.items(), key=lambda kv: -kv[1]["conf"]):
        b = "" if v["book_p"] is None else f"  book {v['book_p']:.3f}"
        p3 = v.get("phase3")
        p3s = "" if not p3 else f"  phase3 {p3['p_combined']:.3f} (rush {p3['p_rush']:.3f} rec {p3['p_rec']:.3f})"
        print(f"  {k:<9} {v['player']:<22} {v['conf']:.3f}  "
              f"(model {v['model_p']:.3f}{b})  {v['basis']}{p3s}")


if __name__ == "__main__":
    main()
