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
                             pass_yds=200 + 3 * i, rush_yds=80 + 2 * i, pa=10 + i))
    g = pd.DataFrame(rows)
    g["epa"], g["plays"] = g.pass_epa + g.rush_epa, g.dropbacks + g.carries
    return g


class DefenseRank(unittest.TestCase):
    def setUp(self):
        self.out = D.rank_defenses(synthetic(), 2026)

    def test_32_teams_ranks_1_to_32_in_each_column(self):
        self.assertEqual(len(self.out), 32)
        for c in ("total_rank", "run_rank", "pass_rank"):
            self.assertEqual(sorted(self.out[c]), list(range(1, 33)))

    def test_rank_1_allows_the_least(self):
        for c in ("total_rank", "run_rank", "pass_rank"):
            self.assertEqual(self.out.sort_values(c).team.iloc[0], "T00")
            self.assertEqual(self.out.sort_values(c).team.iloc[-1], "T31")

    def test_no_missing_values(self):
        self.assertFalse(self.out[["epa_play", "epa_rush", "epa_db", "pa_pg"]].isna().any().any())


if __name__ == "__main__":
    unittest.main()
