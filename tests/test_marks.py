"""Marks (nflprops/marks.py): key format and the POST validation. Uses a scratch file only."""
import json
import os
import pathlib
import sys
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from nflprops import marks as M

KEY = "2026|7|Jaxon Smith-Njigba|receiving_yards"


def post(path, **req):
    return M.apply_post(json.dumps(req).encode(), path)


class Marks(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.dir.name, "marks.json")

    def tearDown(self):
        self.dir.cleanup()

    def test_key_format(self):
        self.assertEqual(M.mark_key(2026, 7, "Jaxon Smith-Njigba", "receiving_yards"), KEY)
        self.assertTrue(M.valid_key(KEY))
        self.assertTrue(M.valid_key("2026|7|A.J. Brown|anytime_td"))

    def test_bad_keys(self):
        for k in ("", "x", "2026|7|a|", "2026|7||s", "2026|7|a|S", "2026|7|a|b|c",
                  "26|7|a|s", "2026|7|a\nb|s", "2026|7|" + "p" * 101 + "|s", 5, None):
            self.assertFalse(M.valid_key(k), k)

    def test_on_then_off(self):
        st, out = post(self.path, key=KEY, on=True)
        self.assertEqual((st, out), (200, {"marks": [KEY]}))
        self.assertEqual(M.keys(self.path), [KEY])
        st, out = post(self.path, key=KEY, on=False)
        self.assertEqual((st, out), (200, {"marks": []}))

    def test_rejects(self):
        self.assertEqual(post(self.path, key="bad", on=True)[0], 400)
        self.assertEqual(post(self.path, key=KEY, on="yes")[0], 400)
        self.assertEqual(post(self.path, key=KEY)[0], 400)
        self.assertEqual(post(self.path, key=KEY, on=True, extra=1)[0], 400)
        self.assertEqual(M.apply_post(b"not json", self.path)[0], 400)
        self.assertEqual(M.apply_post(b"[1]", self.path)[0], 400)
        self.assertEqual(M.apply_post(b" " * 3000 + b"{}", self.path)[0], 400)
        self.assertFalse(os.path.exists(self.path))


if __name__ == "__main__":
    unittest.main()
