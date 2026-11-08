"""Team record as of kickoff (nflprops/records.py)."""
import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from nflprops.records import fmt_record, record_before


def g(day, home, away, hs, as_, typ="REG", season=2026, t="13:00"):
    return dict(season=season, game_type=typ, gameday=day, gametime=t, home_team=home, away_team=away,
                home_score=hs, away_score=as_)


class Records(unittest.TestCase):
    def test_tie_counts_and_formats(self):
        games = [g("2026-09-10", "AAA", "BBB", 20, 10), g("2026-09-17", "CCC", "AAA", 17, 17),
                 g("2026-09-24", "AAA", "DDD", 3, 9)]
        self.assertEqual(record_before(games, "AAA", 2026, "2026-10-01 13:00"), (1, 1, 1))
        self.assertEqual(fmt_record(1, 1, 1), "1-1-1")
        self.assertEqual(fmt_record(5, 2, 0), "5-2")

    def test_later_game_and_other_cases_excluded(self):
        games = [g("2026-09-10", "AAA", "BBB", 20, 10),
                 g("2026-10-01", "AAA", "CCC", 30, 0, t="20:15"),             # kicks off at/after: excluded
                 g("2026-10-08", "DDD", "AAA", 0, 7),                         # later: excluded
                 g("2026-09-17", "AAA", "EEE", 9, 3, typ="PRE"),              # not regular season
                 g("2025-12-28", "AAA", "FFF", 9, 3, season=2025),            # other season
                 g("2026-09-24", "AAA", "GGG", None, None)]                   # no score
        self.assertEqual(record_before(games, "AAA", 2026, "2026-10-01 20:15"), (1, 0, 0))


if __name__ == "__main__":
    unittest.main()
