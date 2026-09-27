"""Tests for the Slack Block Kit guard.

Reproduces the 2026-09-20 crash: a real 13-game Sunday slate produces 55
blocks (one header, four per game, two trailers), one more than Slack's
50-block-per-message ceiling, and the webhook 400s with "invalid_blocks".

tests/fixtures/w4_sunday_blocks.json is that exact payload, captured from
run_reports.py against the live DB for the current Sunday slate (13 games
with lines posted) without ever calling slack.post().
"""
import json
import pathlib
import sys
import unittest
from unittest import mock

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from nflprops import slack

FIXTURE = pathlib.Path(__file__).parent / "fixtures" / "w4_sunday_blocks.json"


def load_oversized_payload():
    return json.loads(FIXTURE.read_text())


class ValidateBlocksTests(unittest.TestCase):
    def test_real_slate_blocks_pass_content_validation(self):
        # The 55-block overflow is a block-COUNT problem, not a content
        # problem -- every individual block here is well-formed.
        blocks = load_oversized_payload()
        self.assertEqual(slack.validate_blocks(blocks), [])

    def test_flags_oversized_header(self):
        blocks = [slack.header("x" * (slack.MAX_HEADER_CHARS + 1))]
        problems = slack.validate_blocks(blocks)
        self.assertEqual(len(problems), 1)
        self.assertIn("header", problems[0])

    def test_flags_oversized_section(self):
        blocks = [slack.section("x" * (slack.MAX_SECTION_CHARS + 1))]
        problems = slack.validate_blocks(blocks)
        self.assertEqual(len(problems), 1)
        self.assertIn("section", problems[0])

    def test_flags_empty_section_text(self):
        blocks = [slack.section("   ")]
        problems = slack.validate_blocks(blocks)
        self.assertEqual(len(problems), 1)
        self.assertIn("empty", problems[0])

    def test_clean_blocks_have_no_problems(self):
        blocks = [slack.header("W4 · SUNDAY"), slack.section("hi"), slack.divider()]
        self.assertEqual(slack.validate_blocks(blocks), [])


class ChunkTests(unittest.TestCase):
    def test_real_slate_splits_under_max_blocks(self):
        blocks = load_oversized_payload()
        self.assertGreater(len(blocks), slack.MAX_BLOCKS)
        parts = slack._chunk(blocks)
        self.assertGreater(len(parts), 1)
        for part in parts:
            self.assertLessEqual(len(part), slack.MAX_BLOCKS)
        # No block dropped or duplicated across the split.
        self.assertEqual(sum(len(p) for p in parts), len(blocks))

    def test_splits_land_on_dividers_not_mid_game(self):
        blocks = load_oversized_payload()
        parts = slack._chunk(blocks)
        # Every part but the last should end on a divider -- that is what
        # keeps a game's header and legs from being split across messages.
        for part in parts[:-1]:
            self.assertEqual(part[-1].get("type"), "divider")


class PostTests(unittest.TestCase):
    def _mock_response(self, body=b"ok"):
        resp = mock.MagicMock()
        resp.read.return_value = body
        resp.__enter__.return_value = resp
        return resp

    @mock.patch("nflprops.slack.urllib.request.urlopen")
    def test_oversized_real_slate_posts_in_multiple_calls_without_crashing(self, mock_urlopen):
        mock_urlopen.return_value = self._mock_response()
        blocks = load_oversized_payload()

        result = slack.post(blocks=blocks, text="Production ranges — W4 sunday",
                             webhook="https://hooks.slack.test/fake")

        self.assertGreaterEqual(mock_urlopen.call_count, 2)
        # Each call's outgoing payload must itself respect the block cap.
        for call in mock_urlopen.call_args_list:
            req = call.args[0]
            payload = json.loads(req.data.decode())
            self.assertLessEqual(len(payload.get("blocks", [])), slack.MAX_BLOCKS)
        self.assertIn("ok", result)

    @mock.patch("nflprops.slack.urllib.request.urlopen")
    def test_content_violation_falls_back_to_plain_text(self, mock_urlopen):
        mock_urlopen.return_value = self._mock_response()
        bad_blocks = [slack.header("W4"), slack.section("x" * (slack.MAX_SECTION_CHARS + 1))]

        slack.post(blocks=bad_blocks, text="Production ranges — W4 sunday",
                   webhook="https://hooks.slack.test/fake")

        self.assertEqual(mock_urlopen.call_count, 1)
        req = mock_urlopen.call_args.args[0]
        payload = json.loads(req.data.decode())
        self.assertNotIn("blocks", payload)
        self.assertEqual(payload["text"], "Production ranges — W4 sunday")


if __name__ == "__main__":
    unittest.main()
