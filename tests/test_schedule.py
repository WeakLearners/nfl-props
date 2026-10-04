"""Game selection for run_reports.py auto: which sittings are due, what is skipped."""
import pathlib
import sys
import tempfile
import unittest
from unittest import mock

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from nflprops import schedule as SC
import run_reports
from nflprops import report as RP

ET = "America/New_York"


def events(*kicks):
    """kicks: 'YYYY-MM-DD HH:MM' in ET. Ids g0, g1, ..."""
    ev = pd.DataFrame({"id": [f"g{i}" for i in range(len(kicks))],
                       "et": [pd.Timestamp(k, tz=ET) for k in kicks]})
    ev["ct"] = ev.et.dt.tz_convert("UTC")
    return ev


def at(ts):
    return pd.Timestamp(ts, tz=ET).tz_convert("UTC")


def due_ids(ev, now, posted=(), handled=()):
    return [list(todo.id) for _, _, todo in SC.plan(ev, at(now), set(posted), set(handled))]


class SelectionTests(unittest.TestCase):
    def test_london_930_sunday_due_at_0630(self):
        ev = events("2026-10-04 09:30", "2026-10-04 13:00")
        self.assertEqual(due_ids(ev, "2026-10-04 06:35"), [["g0"]])  # 1 PM not yet

    def test_nothing_due_early(self):
        ev = events("2026-10-04 09:30")
        self.assertEqual(due_ids(ev, "2026-10-03 20:00"), [])

    def test_thanksgiving_three_games_three_sittings(self):
        ev = events("2026-11-26 12:30", "2026-11-26 16:30", "2026-11-26 20:20")
        self.assertEqual(len(SC.sittings(ev)), 3)
        self.assertEqual(due_ids(ev, "2026-11-26 09:35"), [["g0"]])
        self.assertEqual(due_ids(ev, "2026-11-26 13:35"), [["g0"], ["g1"]])  # g2 is 6.7 h out

    def test_saturday_and_friday_games(self):
        fri = events("2026-11-27 15:00")
        self.assertEqual(due_ids(fri, "2026-11-27 12:05"), [["g0"]])
        sat = events("2026-12-19 16:30")
        self.assertEqual(due_ids(sat, "2026-12-19 13:35"), [["g0"]])

    def test_mnf_doubleheader_is_one_sitting(self):
        ev = events("2026-10-12 19:15", "2026-10-12 20:15")
        self.assertEqual(due_ids(ev, "2026-10-12 16:20"), [["g0", "g1"]])

    def test_sunday_late_window_one_sitting(self):
        ev = events("2026-10-04 13:00", "2026-10-04 16:05", "2026-10-04 16:25")
        s = SC.sittings(ev)
        self.assertEqual([len(x) for x in s], [1, 2])

    def test_already_reported_game_skipped(self):
        ev = events("2026-10-04 16:05", "2026-10-04 16:25")
        plan = SC.plan(ev, at("2026-10-04 13:10"), set(), {"g0"})
        self.assertEqual(list(plan[0][2].id), ["g1"])  # only g1 is fetched
        self.assertEqual(len(plan[0][1]), 2)

    def test_posted_sitting_not_planned_again(self):
        ev = events("2026-10-04 09:30")
        key = SC.sitting_key(ev)
        self.assertEqual(SC.plan(ev, at("2026-10-04 07:00"), {key}, set()), [])

    def test_should_post_rules(self):
        s, n = at("2026-10-04 13:00"), at("2026-10-04 11:00")
        self.assertTrue(SC.should_post(2, 2, s, n))
        self.assertFalse(SC.should_post(2, 1, s, n))                       # wait for the other
        self.assertTrue(SC.should_post(2, 1, s, at("2026-10-04 12:30")))   # kickoff near
        self.assertFalse(SC.should_post(2, 0, s, at("2026-10-04 12:30")))


class AutoRunTests(unittest.TestCase):
    """Two passes over the same sitting: lines fetched once, Slack posted once."""

    def test_second_pass_does_not_refetch_or_repost(self):
        tmp = pathlib.Path(tempfile.mkdtemp())
        kick = pd.Timestamp.now(tz="UTC") + pd.Timedelta(hours=2)
        ev = pd.DataFrame([{"id": "e1", "away_team": "Away Team", "home_team": "Home Team", "ct": kick}])
        ev["et"] = ev.ct.dt.tz_convert(ET)
        lines = pd.DataFrame([{"player_key": "x", "stat": "receptions", "line": 3.5, "price": -300}])
        short = pd.DataFrame([{"player": "P", "stat": "receptions", "line": 3.5, "price": -300, "conf": 0.8}])
        report_file = tmp / "r.html"
        with mock.patch.object(run_reports, "STATE", tmp / "state.json"), \
             mock.patch.object(sys, "argv", ["run_reports.py", "auto"]), \
             mock.patch("nflprops.db.team_abbr_map", return_value={"Away Team": "AW", "Home Team": "HM"}), \
             mock.patch.object(run_reports, "game_week", return_value=(2026, 4)), \
             mock.patch.object(RP, "upcoming_events", return_value=ev), \
             mock.patch.object(RP, "event_lines", return_value=(lines, "100")) as fetch, \
             mock.patch.object(RP, "build", return_value=([{"x": 1}], short, [])), \
             mock.patch.object(RP, "render", return_value="<html></html>"), \
             mock.patch.object(RP, "report_path", return_value=report_file), \
             mock.patch("nflprops.rating.write_rankings", return_value=tmp / "rk.html"), \
             mock.patch("nflprops.slack.post", return_value="ok") as post:
            run_reports.main()
            run_reports.main()
            self.assertEqual(fetch.call_count, 1)
            self.assertEqual(post.call_count, 1)
            report_file.unlink()          # even with the file gone, the posted sitting stays closed
            run_reports.main()
            self.assertEqual(post.call_count, 1)


if __name__ == "__main__":
    unittest.main()
