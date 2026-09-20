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
"""
import argparse
import json
import pathlib
import sys

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from nflprops.odds import devig_yesno, event_props, name_key  # noqa: E402
from nflprops.report import upcoming_events  # noqa: E402
from nflprops.tdmodel import design_matrix, features  # noqa: E402
from predict import train_through  # noqa: E402


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
        import nflreadpy as nfl
        ev = upcoming_events()
        teams = nfl.load_teams().to_pandas()
        abbr = dict(zip(teams.team_name, teams.team_abbr))
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

    out = {}
    for r in top.itertuples():
        # game_id is "2026_02_MIN_CHI" -> away MIN, home CHI.
        parts = str(r.game_id).split("_")
        if len(parts) < 4:
            continue
        key = f"{parts[2]}-{parts[3]}"
        out[key] = {
            "player": r.full_name, "position": r.position, "team": r.team,
            "conf": round(float(r.conf), 3),
            "model_p": round(float(r.model_p), 3),
            "book_p": None if pd.isna(r.book_p) else round(float(r.book_p), 3),
            "basis": r.basis,
        }

    path = ROOT / "reports" / f"td_markers_{args.season}_w{args.week}.json"
    path.write_text(json.dumps(out, indent=1, sort_keys=True))
    print(f"trained on {n_train:,} player-weeks; {len(out)} markers -> {path.name}")
    for k, v in sorted(out.items(), key=lambda kv: -kv[1]["conf"]):
        b = "" if v["book_p"] is None else f"  book {v['book_p']:.3f}"
        print(f"  {k:<9} {v['player']:<22} {v['conf']:.3f}  "
              f"(model {v['model_p']:.3f}{b})  {v['basis']}")


if __name__ == "__main__":
    main()
