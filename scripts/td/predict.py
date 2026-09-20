#!/usr/bin/env python
"""Score an upcoming slate: who is most likely to score, and how confident.

Retrains from data/td_train.parquet on every complete week strictly before
the target, then calls nflprops.tdmodel.features() for the target week --
the same function that built every training row. No saved model artifact:
fitting 52k rows takes a couple of seconds, and a pickle on disk is one more
thing that can silently disagree with the code that made it.

Confidence follows the rule the rest of the app already uses: the LOWER of
the model's number and FanDuel's own, never the more optimistic of the two.
With --no-book the book half is skipped and the output says model_only, so a
number produced without a market check can never be mistaken for one with.
"""
import argparse
import pathlib
import sys

import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from nflprops.tdmodel import features, design_matrix  # noqa: E402

TRAIN = ROOT / "data" / "td_train.parquet"


def train_through(season, week):
    df = pd.read_parquet(TRAIN)
    prior = df[(df.season < season) | ((df.season == season) & (df.week < week))]
    if prior.empty:
        sys.exit(f"no completed weeks before {season} W{week}")
    m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06,
                                       max_leaf_nodes=31, l2_regularization=1.0,
                                       random_state=0)
    m.fit(design_matrix(prior), prior["td"])
    return m, len(prior)


def book_probs(season, week):
    """FanDuel's de-vigged anytime-TD probability per player. Costs credits."""
    from nflprops.odds import events, event_props, devig_yesno, name_key
    ev = events()
    ev = ev[(ev.season == season) & (ev.week == week)] if "season" in ev.columns else ev
    frames = []
    for eid in ev.event_id if hasattr(ev, "event_id") else []:
        raw, _ = event_props(eid)
        td = devig_yesno(raw)
        if not td.empty:
            frames.append(td)
    if not frames:
        return pd.DataFrame(columns=["player_key", "book_p", "price_yes"])
    b = pd.concat(frames, ignore_index=True)
    b["player_key"] = b.player_book.map(name_key)
    return b[["player_key", "book_p", "price_yes"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("season", type=int)
    ap.add_argument("week", type=int)
    ap.add_argument("--no-book", action="store_true",
                    help="skip the FanDuel pull; output is labelled model_only")
    args = ap.parse_args()

    model, n_train = train_through(args.season, args.week)
    f = features(args.season, args.week)
    if f.empty:
        sys.exit(f"no roster rows for {args.season} W{args.week}")
    f["model_p"] = model.predict_proba(design_matrix(f))[:, 1]

    if args.no_book:
        f["book_p"] = pd.NA
        f["conf"] = f["model_p"]
        f["basis"] = "model_only"
    else:
        from nflprops.odds import name_key
        b = book_probs(args.season, args.week)
        f["player_key"] = f.full_name.map(name_key)
        f = f.merge(b, on="player_key", how="left")
        # min(), not mean: the standing rule is never the more optimistic of
        # the two. Where the book has no price the model stands alone, and
        # the row says so rather than quietly looking like a checked number.
        f["conf"] = f[["model_p", "book_p"]].min(axis=1)
        f["basis"] = f.book_p.notna().map({True: "model+book", False: "model_only"})

    f = f.dropna(subset=["game_id"])
    top = f.loc[f.groupby("game_id")["conf"].idxmax()].sort_values("conf", ascending=False)
    cols = ["game_id", "full_name", "position", "team", "opponent",
            "model_p", "book_p", "conf", "basis"]
    print(f"trained on {n_train:,} player-weeks before {args.season} W{args.week}\n")
    print(top[cols].to_string(index=False,
          formatters={"model_p": "{:.3f}".format, "conf": "{:.3f}".format}))
    return top


if __name__ == "__main__":
    main()
