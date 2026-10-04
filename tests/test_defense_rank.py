"""Defense rankings (nflprops/defense_rank.py): 32 teams, ranks 1-32, rank 1 = least allowed."""
import pathlib
import sys
import unittest

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from nflprops import defense_rank as D


def synthetic():
    """32 defenses, 4 games each. Defense i allows more as i grows. Every offense plays every week."""
    teams = [f"T{i:02d}" for i in range(32)]
    rows = []
    for i, d in enumerate(teams):
        for wk in range(1, 5):
            rows.append(dict(game_id=f"{wk}-{d}", season=2026, week=wk, off=teams[(i + wk) % 32], defn=d,
                             pass_epa=0.1 * i * 30, rush_epa=0.05 * i * 20, dropbacks=30, carries=20,
                             pass_yds=200 + 3 * i, rush_yds=80 + 2 * i, pa=10 + i,
                             qb_yds=200 + 3 * i, rb_yds=90 + 2 * i, wr_yds=150 + i, te_yds=40 + i))
    g = pd.DataFrame(rows)
    g["epa"], g["plays"] = g.pass_epa + g.rush_epa, g.dropbacks + g.carries
    return g


class DefenseRank(unittest.TestCase):
    def setUp(self):
        self.out = D.rank_defenses(synthetic(), 2026)

    def test_32_teams_ranks_1_to_32_in_each_column(self):
        self.assertEqual(len(self.out), 32)
        for k in D.RANK_STAT:
            self.assertEqual(sorted(self.out[f"{k}_rank"]), list(range(1, 33)), k)

    def test_rank_order_equals_sort_order_of_the_shown_number(self):
        for k, st in D.RANK_STAT.items():
            by_rank = self.out.sort_values(f"{k}_rank")[st].tolist()
            self.assertEqual(by_rank, sorted(self.out[st].tolist()), k)  # rank 1 = least allowed

    def test_rank_1_allows_the_least(self):
        for k in D.RANK_STAT:
            self.assertEqual(self.out.sort_values(f"{k}_rank").team.iloc[0], "T00", k)
            self.assertEqual(self.out.sort_values(f"{k}_rank").team.iloc[-1], "T31", k)

    def test_no_missing_values(self):
        self.assertFalse(self.out[list(D.RANK_STAT.values()) + ["pa_pg"]].isna().any().any())


if __name__ == "__main__":
    unittest.main()
