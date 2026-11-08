"""Serve-time report header: logos, gh-side spans and score, with and without the record span."""
import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import serve_reports as S

REC = '<span class="rec" data-inspect-id="header-record" style="font-weight:400;white-space:nowrap">(%s)</span>'
LA, LB = '<img class="tlogo" src="a">', '<img class="tlogo" src="b">'
WITH = '<h1>Jax ' + REC % "5-2" + ' <span class="at">at</span> Bal ' + REC % "4-4" + '</h1>'
WITHOUT = '<h1>Jax <span class="at">at</span> Bal</h1>'


class Header(unittest.TestCase):
    def check(self, h, rec):
        self.assertEqual(h.count('class="tlogo"'), 2)
        self.assertIn('class="gh-side ga"', h)
        self.assertIn('class="gh-side gb"', h)
        self.assertEqual(h.count('class="rec"'), 2 if rec else 0)

    def test_played_with_record(self):
        h = S.header_logos(S.header_scores(WITH, 33, 10), LA, LB)
        self.check(h, True)
        self.assertIn('<span class="gh-sc w">33</span>', h)
        self.assertIn('<span class="gh-sc">10</span>', h)
        # away: logo, name, record, score. home: score, logo, name, record.
        self.assertRegex(h, r'ga">' + LA + r'Jax <span class="rec"[^>]*>\(5-2\)</span> <span class="gh-sc w">33</span>')
        self.assertRegex(h, r'gb"><span class="gh-sc">10</span> ' + LB + r'Bal <span class="rec"[^>]*>\(4-4\)</span></span>')

    def test_played_without_record(self):
        h = S.header_logos(S.header_scores(WITHOUT, 33, 10), LA, LB)
        self.check(h, False)
        self.assertIn('<span class="gh-sc w">33</span>', h)

    def test_unplayed_with_and_without_record(self):
        h = S.header_logos(WITH, LA, LB)
        self.check(h, True)
        self.assertRegex(h, r'ga">' + LA + r'Jax <span class="rec"[^>]*>\(5-2\)</span></span>')
        self.assertRegex(h, r'gb">' + LB + r'Bal <span class="rec"[^>]*>\(4-4\)</span></span>')
        self.check(S.header_logos(WITHOUT, LA, LB), False)


if __name__ == "__main__":
    unittest.main()
