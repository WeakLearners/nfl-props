"""
Is the opponent already in the model, arriving through the betting line?

THE OBSERVATION THAT PROMPTED THIS. Removing the two betting-line columns hurts
four of four stats. Removing both defense columns hurts none. Phase 4 then
rebuilt the defense feature properly -- shrunk denominators, EPA, both -- and
still got nothing: 0 of 36 cells cleared their bar.

THE STORY THAT FITS. The book knows who is playing whom. A defense that stops
the run is priced into the spread and the total before the model sees anything.
So a defense column is not missing information. It is a second, noisier copy of
information the line already carried.

WHY THAT STORY NEEDS TESTING. It explains the result, and so would "the opponent
simply does not matter at this granularity." Those two are different claims with
different consequences, and Phase 4 could not tell them apart. A story that fits
the evidence is not the same as a story the evidence chose.

THE EXPERIMENT. Take the line away and see whether the defense feature suddenly
starts earning its keep.

                      line present      line removed
  defense absent          cell 1           cell 3
  defense present         cell 2           cell 4

  helps_with_line    = cell 1 - cell 2      (Phase 4 already says: about zero)
  helps_without_line = cell 3 - cell 4

If helps_without_line is clearly larger, the line was carrying the opponent, and
the defense column is redundant rather than useless. If both are about zero, the
opponent genuinely does not predict player props here once the player's own form
is in the model -- a different and more interesting finding.

WHICH DEFENSE FEATURE. Arm C from Phase 4 -- shrunk rates plus EPA, the best
instrument built. Using the old broken columns would confound the test: their
failure without the line could be their own noise rather than redundancy.

Both validation seasons are scored, because a result that appears in one and not
the other is the thing this project keeps catching. The 2026 holdout is not
touched, per decision #24.

Read-only against features.db and nflprops.db. Writes one JSON.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from phase4_opponent import (ROOT, load_pg, load_training_rows, fit_shrinkage_k,
                             make_feature_sets, score_mae_stats, score_td_stats)
from rung2_ridge import add_derived

OUT = ROOT / "data/line_vs_defense.json"

LINE_COLS = ["total_line", "spread_line_team"]


def strip_lines(df):
    """Remove the betting-line columns. Any __isna companion goes too -- leaving
    a flag that says 'the line was missing' would smuggle the line's presence
    back in through the side door."""
    drop = [c for c in df.columns
            if c in LINE_COLS or any(c.startswith(l + "__") for l in LINE_COLS)]
    return df.drop(columns=drop)


def main():
    pg = load_pg()
    k_table = fit_shrinkage_k(pg)
    tr = add_derived(load_training_rows())
    sets = make_feature_sets(tr, pg, k_table)

    cells = {
        "1_line_nodef":   sets["no_defense"],
        "2_line_def":     sets["arm_c_both"],
        "3_noline_nodef": strip_lines(sets["no_defense"]),
        "4_noline_def":   strip_lines(sets["arm_c_both"]),
    }

    results = {}
    for label, df in cells.items():
        print(f"=== {label}  ({df.shape[1]} cols) ===", flush=True)
        results[label] = {"mae": score_mae_stats(df, label),
                          "td": score_td_stats(df, label)}

    OUT.write_text(json.dumps(dict(
        question="does the defense feature earn its keep once the betting line is removed",
        defense_feature="arm C -- shrunk rates plus EPA, the best instrument Phase 4 built",
        line_cols=LINE_COLS, cells=results), indent=2))

    c1, c2 = results["1_line_nodef"], results["2_line_def"]
    c3, c4 = results["3_noline_nodef"], results["4_noline_def"]

    print(f"\n{'stat':<22}{'w/ line':>10}{'w/o line':>10}{'diff':>10}")
    print("(positive = the defense feature helped)")
    for k in c1["mae"]:
        a = (c1["mae"][k]["mae"] - c2["mae"][k]["mae"]) / c1["mae"][k]["mae"] * 100
        b = (c3["mae"][k]["mae"] - c4["mae"][k]["mae"]) / c3["mae"][k]["mae"] * 100
        print(f"{k:<22}{a:>+10.2f}{b:>+10.2f}{b - a:>+10.2f}")
    for k in c1["td"]:
        a = (c1["td"][k]["brier"] - c2["td"][k]["brier"]) / c1["td"][k]["brier"] * 100
        b = (c3["td"][k]["brier"] - c4["td"][k]["brier"]) / c3["td"][k]["brier"] * 100
        print(f"{k:<22}{a:>+10.2f}{b:>+10.2f}{b - a:>+10.2f}")

    # How much does the line itself buy? If this is near zero the whole
    # experiment is void -- there would be nothing for the line to be carrying.
    print(f"\n--- what the line itself is worth (nodef, line vs no line) ---")
    for k in c1["mae"]:
        d = (c3["mae"][k]["mae"] - c1["mae"][k]["mae"]) / c3["mae"][k]["mae"] * 100
        print(f"{k:<22}{d:>+10.2f}")
    for k in c1["td"]:
        d = (c3["td"][k]["brier"] - c1["td"][k]["brier"]) / c3["td"][k]["brier"] * 100
        print(f"{k:<22}{d:>+10.2f}")

    print(f"\nwritten -> {OUT}")


if __name__ == "__main__":
    main()
