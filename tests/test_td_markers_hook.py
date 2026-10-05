"""The scheduled TD-marker hook must be loud, never fatal, and backfill must never grade."""
import io
import json
import pathlib
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from types import SimpleNamespace

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "td"))

import run_reports
import write_markers


class HookTests(unittest.TestCase):
    def test_failure_is_loud_and_does_not_raise(self):
        bad = lambda *a, **k: SimpleNamespace(returncode=1, stdout="", stderr="no roster rows")
        err, out = io.StringIO(), io.StringIO()
        with redirect_stderr(err), redirect_stdout(out):
            ok = run_reports.write_td_markers(2026, 5, run=bad)
        self.assertFalse(ok)
        self.assertIn("TD MARKERS FAILED", err.getvalue())

    def test_crash_in_runner_does_not_raise(self):
        def boom(*a, **k):
            raise OSError("spawn failed")
        with redirect_stderr(io.StringIO()), redirect_stdout(io.StringIO()):
            self.assertFalse(run_reports.write_td_markers(2026, 5, run=boom))

    def test_success(self):
        good = lambda *a, **k: SimpleNamespace(returncode=0, stdout="trained; 16 markers\n", stderr="")
        with redirect_stdout(io.StringIO()):
            self.assertTrue(run_reports.write_td_markers(2026, 5, run=good))

    def test_rerun_keeps_started_game_drops_old_backfill(self):
        p = pathlib.Path(tempfile.mkdtemp()) / "m.json"
        p.write_text(json.dumps({"A-B": {"player": "old", "basis": "model+book"},
                                 "C-D": {"player": "x", "backfill": True}}))
        new = {"A-B": {"player": "new"}, "C-D": {"player": "y"}, "E-F": {"player": "z"}}
        m = write_markers.merge_started(new, p, upcoming={"C-D", "E-F"})
        self.assertEqual(m["A-B"]["player"], "old")   # started: real entry kept
        self.assertEqual(m["C-D"]["player"], "y")     # upcoming: overwritten
        self.assertEqual(write_markers.merge_started(new, p, upcoming=set()), new)
        # started game with no earlier entry must come out flagged, never live
        m2 = write_markers.merge_started({"G-H": {"player": "q", "basis": "model_only"}}, p, upcoming={"E-F"})
        self.assertTrue(m2["G-H"]["backfill"])
        self.assertEqual(m2["G-H"]["odds"], "missing")


if __name__ == "__main__":
    unittest.main()


class BadgeTests(unittest.TestCase):
    """The report page is built in the browser, so these check the template source."""
    def setUp(self):
        self.t = (ROOT / "templates" / "report.html").read_text()

    def test_badge_text_guard_and_tooltip(self):
        self.assertIn(">TD Alert</span>", self.t)
        self.assertIn("Backfill: model only, no pre-game odds", self.t)
        # one badge per game, top player only: name match AND a once-only latch
        self.assertIn("isScorer && !tdShown", self.t)
        self.assertEqual(self.t.count("tdShown = true"), 1)

    def test_backfill_marker_is_not_graded(self):
        import serve_reports
        tmp = pathlib.Path(tempfile.mkdtemp())
        (tmp / "td_markers_2026_w9.json").write_text(json.dumps(
            {"A-B": {"player": "p", "backfill": True, "basis": "model_only"}}))
        old = serve_reports.ROOT
        serve_reports.ROOT = str(tmp)
        try:
            self.assertEqual(serve_reports.td_legs(2026, 9), ([], 0, 0))
        finally:
            serve_reports.ROOT = old
