"""--dry on run_reports.py must write nothing: no report HTML, no saved board."""
import pathlib
import sys
import tempfile
import unittest
from unittest import mock

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import run_reports
from nflprops import report as RP


class DryRunTests(unittest.TestCase):
    def _run(self, argv):
        tmp = pathlib.Path(tempfile.mkdtemp())
        ev = pd.DataFrame([{
            "id": "e1", "away_team": "Away Team", "home_team": "Home Team",
            "et": pd.Timestamp("2026-10-04 13:00", tz="America/New_York")}])
        lines = pd.DataFrame([{"player_key": "x", "stat": "receptions", "line": 3.5, "price": -300}])
        short = pd.DataFrame([{"player": "P", "stat": "receptions", "line": 3.5,
                               "price": -300, "conf": 0.8}])
        calls = []

        def fake_lines(event_id, **kw):
            calls.append(kw)
            return lines, "100"

        with mock.patch.object(sys, "argv", ["run_reports.py"] + argv), \
             mock.patch.object(run_reports, "current_week", return_value=(2026, 4)), \
             mock.patch.object(run_reports, "pick_games", return_value=ev), \
             mock.patch("nflprops.db.team_abbr_map", return_value={"Away Team": "AW", "Home Team": "HM"}), \
             mock.patch.object(RP, "upcoming_events", return_value=ev), \
             mock.patch.object(RP, "event_lines", side_effect=fake_lines), \
             mock.patch.object(RP, "build", return_value=([{"x": 1}], short, [])), \
             mock.patch.object(RP, "render", return_value="<html></html>"), \
             mock.patch.object(RP, "report_path", return_value=tmp / "r.html"), \
             mock.patch("nflprops.slack.post", return_value="ok") as post:
            run_reports.main()
        return tmp, calls, post

    def test_dry_writes_nothing(self):
        tmp, calls, post = self._run(["sunday", "--dry"])
        self.assertEqual(list(tmp.iterdir()), [])
        self.assertEqual(calls, [{}])  # no season/week/teams -> no board saved
        post.assert_not_called()

    def test_live_writes_report_and_board(self):
        tmp, calls, post = self._run(["sunday"])
        self.assertTrue((tmp / "r.html").exists())
        self.assertEqual(calls[0]["teams"], ["AW", "HM"])
        post.assert_called_once()


if __name__ == "__main__":
    unittest.main()
