"""A report is written for EVERY game (Sean, 2026-11-13). No lines, no prices, an
empty shortlist or no qualifying rung never stop the file. Only a real projection
failure does, and that one is loud and sends a Slack alert."""
import pathlib
import sys
import tempfile
import types
import unittest
from unittest import mock

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import run_reports
from nflprops import report as RP

E = types.SimpleNamespace(id="e1", away_team="Away Team", home_team="Home Team",
                          et=pd.Timestamp("2026-11-12 20:15", tz="America/New_York"))
ROW = [{"p": "P"}]


def _empty_short():
    s = pd.DataFrame(columns=["player", "stat", "line", "price", "conf"])
    s.attrs["screens"] = {"legs": 30, "priced": 30, "reasons": [["fewer than 10 regular-season games", 5]]}
    return s


class EveryGameReport(unittest.TestCase):
    def _go(self, lines, build_ret=None, build_exc=None):
        tmp = pathlib.Path(tempfile.mkdtemp())
        out = tmp / "r.html"
        seen = {}
        bkw = dict(return_value=build_ret) if build_exc is None else dict(side_effect=build_exc)

        def fake_render(*a, **k):
            seen.update(k)
            return "<html></html>"

        with mock.patch.object(RP, "build", **bkw) as b, \
             mock.patch.object(RP, "render", side_effect=fake_render), \
             mock.patch.object(RP, "report_path", return_value=out), \
             mock.patch("nflprops.slack.post", return_value="ok") as post:
            item, _ = run_reports.build_game(RP, E, 2026, 10, "WAS", "NYG", 6, False,
                                             lines=lines, rem="1")
        return item, out, seen, b, post

    def test_empty_shortlist_writes_report_and_is_ready(self):
        item, out, seen, _, post = self._go(pd.DataFrame([{"x": 1}]), ([{"x": 1}], _empty_short(), []))
        self.assertTrue(out.exists())
        self.assertIsNotNone(item)
        self.assertEqual(item["n"], 0)
        self.assertIsNone(item["pay"])
        self.assertEqual(seen["screens"]["legs"], 30)
        post.assert_not_called()
        blocks = run_reports.digest_blocks([item], "T", "1")
        self.assertIn("Slip is empty", str(blocks))
        self.assertNotIn("straight", str(blocks[1:-2]))

    def test_empty_lines_writes_projection_only_report(self):
        item, out, _, b, _ = self._go(pd.DataFrame(), ([{"x": 1}], _empty_short(), []))
        self.assertTrue(out.exists())
        self.assertTrue(item["nolines"])
        self.assertTrue(b.call_args[0][3].empty)  # build() got the empty lines frame

    def test_real_failure_is_loud_and_alerts(self):
        for kw in (dict(build_exc=RuntimeError("boom")), dict(build_ret=([], pd.DataFrame(), []))):
            item, out, _, _, post = self._go(pd.DataFrame(), **kw)
            self.assertIsNone(item)
            self.assertFalse(out.exists())
            post.assert_called_once()
            self.assertIn("STEP FAILED", post.call_args.kwargs["text"])
            self.assertIn("WAS@NYG", post.call_args.kwargs["text"])


if __name__ == "__main__":
    unittest.main()
