"""Upcoming legs on Results (scripts/serve_reports.py _upcoming): waiting in, final and void out."""
import pathlib
import sys
import unittest
from unittest import mock

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import serve_reports as S


def leg(p, game):
    return {"p": p, "tm": "AAA", "vs": "BBB", "role": "WR1", "s": "receiving_yards", "line": 50.5,
            "price": -120, "mp": 0.7, "bp": 0.55, "game": game, "file": "x.html", "pick": None}


class Upcoming(unittest.TestCase):
    def run_up(self, td_waiting):
        legs = [leg("Waiting Guy", "AAA @ BBB"), leg("Final Guy", "CCC @ DDD"), leg("Void Guy", "CCC @ DDD")]

        def fake_td(season, week, waiting_rows=None):
            waiting_rows.extend(td_waiting)
            return [], len(td_waiting), 0
        with mock.patch.object(S, "lined", return_value=legs), \
                mock.patch.object(S, "outcomes", return_value=({}, {"CCC@DDD"})), \
                mock.patch.object(S, "td_legs", side_effect=fake_td), \
                mock.patch.object(S, "chip_for", return_value=None), \
                mock.patch.object(S, "badges_for", return_value=[]):
            return S._upcoming(2026, 9)

    def test_only_waiting_games(self):
        td = [{"p": "Td Guy", "s": "anytime_td", "game": "AAA @ BBB", "role": "RB1", "vs": "BBB"}]
        out = self.run_up(td)
        self.assertEqual([r["p"] for r in out], ["Waiting Guy", "Td Guy"])
        self.assertTrue(all(r["hit"] is None and r["season"] == 2026 and r["week"] == 9 for r in out))

    def test_newest_week_only(self):
        with mock.patch.object(S.os, "listdir", return_value=[
                "report_2026_w8_AAA-BBB.html", "report_2026_w9_CCC-DDD.html", "report_2025_w18_EEE-FFF.html"]), \
                mock.patch.object(S, "_upcoming", return_value=[]) as up:
            self.assertEqual(S.upcoming_legs(), ([], (2026, 9)))
            up.assert_called_once_with(2026, 9)


if __name__ == "__main__":
    unittest.main()
