"""Credit guard: no call below the floor, no crash at 0 credits, log rows, retry gate.
No test here makes a network call: urlopen is always mocked."""
import io
import json
import pathlib
import tempfile
import unittest
import urllib.error
from unittest import mock

import pandas as pd

from nflprops import credits as CR


class FakeResp(io.BytesIO):
    def __init__(self, body, headers):
        super().__init__(json.dumps(body).encode())
        self.headers = headers

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def http_error(code, body):
    return urllib.error.HTTPError("u", code, "x", {}, io.BytesIO(body.encode()))


class Base(unittest.TestCase):
    def setUp(self):
        d = pathlib.Path(tempfile.mkdtemp())
        self.p = [mock.patch.object(CR, "STATE", d / "credits.json"),
                  mock.patch.object(CR, "LOG", d / "credit_log.csv"),
                  mock.patch.object(CR, "EVENTS_CACHE", d / "events.json")]
        for p in self.p:
            p.start()
        CR.context = "S1"

    def tearDown(self):
        for p in self.p:
            p.stop()
        CR.context = ""


class GuardTests(Base):
    def test_call_records_remaining_and_log(self):
        r = FakeResp({"ok": 1}, {"x-requests-last": "4", "x-requests-remaining": "300"})
        with mock.patch("urllib.request.urlopen", return_value=r):
            d, rem = CR.fetch("http://x", "lines e1", 4)
        self.assertEqual((d, rem), ({"ok": 1}, 300))
        self.assertEqual(CR.remaining(), 300)
        total, by = CR.totals(since=CR.cycle_start() - pd.Timedelta(days=1))
        self.assertEqual((total, by), (4, {"S1": 4}))

    def test_below_floor_makes_no_call(self):
        CR._save_remaining(CR.FLOOR - 1)
        with mock.patch("urllib.request.urlopen") as u:
            d, rem = CR.fetch("http://x", "lines e1", 4)
        u.assert_not_called()
        self.assertIsNone(d)
        self.assertIn("skip", CR.LOG.read_text())

    def test_at_floor_still_allowed(self):
        CR._save_remaining(CR.FLOOR)
        self.assertTrue(CR.can_spend())

    def test_zero_credits_no_call_no_raise(self):
        CR._save_remaining(0)
        with mock.patch("urllib.request.urlopen") as u:
            d, rem = CR.fetch("http://x", "lines e1", 4)
        u.assert_not_called()
        self.assertEqual((d, rem), (None, 0))

    def test_api_says_out_of_credits_does_not_raise(self):
        with mock.patch("urllib.request.urlopen",
                        side_effect=http_error(401, '{"error_code":"OUT_OF_USAGE_CREDITS"}')):
            d, rem = CR.fetch("http://x", "lines e1", 4)
        self.assertEqual((d, rem), (None, 0))
        self.assertEqual(CR.remaining(), 0)

    def test_other_http_error_still_raises(self):
        with mock.patch("urllib.request.urlopen",
                        side_effect=http_error(401, '{"error_code":"INVALID_KEY"}')):
            with self.assertRaises(urllib.error.HTTPError):
                CR.fetch("http://x", "lines e1", 4)

    def test_free_events_call_clears_old_low_value(self):
        CR._save_remaining(5)
        r = FakeResp([{"id": "e1"}], {"x-requests-remaining": "500"})
        with mock.patch("urllib.request.urlopen", return_value=r):
            CR.events_json("http://x")
        self.assertTrue(CR.can_spend())

    def test_events_falls_back_to_saved_list_when_out(self):
        CR.EVENTS_CACHE.write_text(json.dumps([{"id": "e1"}]))
        with mock.patch("urllib.request.urlopen",
                        side_effect=http_error(401, "OUT_OF_USAGE_CREDITS")):
            self.assertEqual(CR.events_json("http://x"), [{"id": "e1"}])

    def test_event_lines_returns_empty_frame_when_guarded(self):
        from nflprops import report as RP
        CR._save_remaining(0)
        with mock.patch.object(RP, "require", return_value="k"), \
             mock.patch("urllib.request.urlopen") as u:
            lines, rem = RP.event_lines("e1")
        u.assert_not_called()
        self.assertTrue(lines.empty)


class RetryTests(unittest.TestCase):
    now = pd.Timestamp("2026-11-15 15:00", tz="UTC")

    def test_far_kickoff_not_due(self):
        self.assertFalse(CR.retry_due(None, self.now + pd.Timedelta(hours=5), self.now))

    def test_within_3h_never_tried_is_due(self):
        self.assertTrue(CR.retry_due(None, self.now + pd.Timedelta(hours=2), self.now))

    def test_recent_try_not_due_old_try_due(self):
        k = self.now + pd.Timedelta(hours=2)
        self.assertFalse(CR.retry_due((self.now - pd.Timedelta(minutes=30)).isoformat(), k, self.now))
        self.assertTrue(CR.retry_due((self.now - pd.Timedelta(minutes=121)).isoformat(), k, self.now))

    def test_cycle_starts_on_the_13th(self):
        n = pd.Timestamp("2026-11-12", tz="UTC").to_pydatetime()
        self.assertEqual((CR.cycle_start(n).month, CR.cycle_start(n).day), (10, 13))
        n = pd.Timestamp("2026-11-15", tz="UTC").to_pydatetime()
        self.assertEqual((CR.cycle_start(n).month, CR.cycle_start(n).day), (11, 13))


if __name__ == "__main__":
    unittest.main()
