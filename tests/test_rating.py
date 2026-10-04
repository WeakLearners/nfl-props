"""rating.py must be a function of games strictly before week W."""
import pathlib
import sys
import unittest

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from nflprops import rating as R

STAT_COLS = ["attempts", "completions", "passing_yards", "carries", "rushing_yards",
             "targets", "receptions", "receiving_yards", "receiving_epa", "passing_epa",
             "rushing_epa", "snap_share", "target_share", "air_yards_share"]


def synthetic():
    rng = np.random.default_rng(0)
    rows = []
    for i in range(60):
        pos = ["QB", "RB", "WR", "TE"][i % 4]
        for wk in range(1, 11):
            r = {c: float(rng.integers(0, 40)) for c in STAT_COLS}
            r.update(player_id=f"p{i}", player_display_name=f"P{i}", position=pos,
                     season=2024, week=wk, game_id=f"g{wk}", team="AAA",
                     opponent_team="BBB", sacks_suffered=1.0, passing_cpoe=0.0,
                     rz10_carry_share=0.0, rz10_target_share=0.0, rush_share=0.1)
            rows.append(r)
    return pd.DataFrame(rows)


class NoLookahead(unittest.TestCase):
    def test_future_rows_do_not_change_rating(self):
        pg = R._prep(synthetic())
        no_dvp = lambda s, w: pd.DataFrame()
        base = R.rating_frame(2024, 6, pg, dvp=no_dvp).set_index("player_id")
        bad = pg.copy()
        future = (bad.season > 2024) | ((bad.season == 2024) & (bad.week >= 6))
        for c in STAT_COLS:
            bad.loc[future, c] = bad.loc[future, c] * 100 + 999
        again = R.rating_frame(2024, 6, bad, dvp=no_dvp).set_index("player_id")
        pd.testing.assert_series_equal(base.rating, again.rating.loc[base.index])

    def test_week_one_of_data_is_not_rated(self):
        pg = R._prep(synthetic())
        out = R.rating_frame(2024, 3, pg, dvp=lambda s, w: pd.DataFrame())
        self.assertEqual(len(out), 0)  # 2 prior games < MIN_GAMES


if __name__ == "__main__":
    unittest.main()
