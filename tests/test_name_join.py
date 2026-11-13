import unittest
from unittest import mock

import pandas as pd

from nflprops import rescored as RS
from nflprops.odds import name_key


class NameKeyTests(unittest.TestCase):
    def test_accent_is_ignored(self):
        self.assertEqual(name_key("Audric Estim\u00e9"), name_key("Audric Estime"))

    def test_periods_and_suffix_are_ignored(self):
        self.assertEqual(name_key("D.J. Moore"), name_key("DJ Moore"))
        self.assertEqual(name_key("Chris Godwin Jr."), name_key("Chris Godwin"))

    def test_apostrophe_is_ignored(self):
        self.assertEqual(name_key("Ja\u2019Marr Chase"), name_key("Ja'Marr Chase"))

    def test_different_players_do_not_match(self):
        self.assertNotEqual(name_key("Amon-Ra St. Brown"), name_key("Equanimeous St. Brown"))
        self.assertNotEqual(name_key("DJ Moore"), name_key("DJ Chark"))
        self.assertNotEqual(name_key("Audric Estime"), name_key("Audric Estes"))


class JoinOnNameTests(unittest.TestCase):
    def test_accent_row_joins_and_counts_unmatched(self):
        new = pd.DataFrame([{"week": 6, "player": "Audric Estime", "stat": "rushing_yards", "mu": 20.0},
                            {"week": 6, "player": "Nobody New", "stat": "rushing_yards", "mu": 5.0}])
        pub = pd.DataFrame([{"week": 6, "player": "Audric Estim\u00e9", "stat": "rushing_yards", "mu_pub": 18.0},
                            {"week": 6, "player": "Nobody Old", "stat": "rushing_yards", "mu_pub": 9.0}])
        with mock.patch("builtins.print") as pr:
            out = RS.join_on_name(new, pub, "left", "test")
        self.assertEqual(len(out), 2)
        self.assertEqual(out.loc[out.player == "Audric Estime", "mu_pub"].iloc[0], 18.0)
        self.assertTrue(out.loc[out.player == "Nobody New", "mu_pub"].isna().all())
        msg = pr.call_args[0][0]
        self.assertIn("1 keys matched", msg)
        self.assertIn("1 left rows with no match", msg)
        self.assertIn("1 right rows with no match", msg)

    def test_inner_join_drops_unmatched_but_counts_them(self):
        left = pd.DataFrame([{"week": 1, "player": "A B", "stat": "s", "x": 1}, {"week": 1, "player": "C D", "stat": "s", "x": 2}])
        right = pd.DataFrame([{"week": 1, "player": "A B", "stat": "s"}])
        with mock.patch("builtins.print") as pr:
            out = RS.join_on_name(left, right, "inner", "test")
        self.assertEqual(len(out), 1)
        self.assertIn("1 left rows with no match", pr.call_args[0][0])

    def test_key_finds_accent_row_in_index(self):
        d = pd.DataFrame([{"season": 2026, "week": 6, "player": "Audric Estim\u00e9", "stat": "rushing_yards", "mu": 1.0}])
        ix = d.assign(pk=d.player.map(name_key)).set_index(["week", "pk", "stat"])
        self.assertIn(RS.key(6, "Audric Estime", "rushing_yards"), ix.index)


if __name__ == "__main__":
    unittest.main()


class UnmatchedCountTests(unittest.TestCase):
    def test_refit_legs_prints_count_of_unmatched(self):
        import contextlib
        import io
        import pathlib
        import sys
        root = pathlib.Path(__file__).resolve().parent.parent
        sys.path.insert(0, str(root))
        sys.path.insert(0, str(root / "scripts"))
        import refit_new_path as R
        df = pd.DataFrame([{"week": 1, "player": "Audric Estimé", "stat": "rushing_yards",
                            "mu_new": 20.0, "sd_new": 10.0}])
        legs = [{"week": 1, "p": "Audric Estime", "s": "rushing_yards", "line": 15.5},
                {"week": 1, "p": "Nobody Else", "s": "rushing_yards", "line": 15.5}]
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            out = R.rescored_legs(legs, df)
        self.assertEqual(len(out), 1)
        self.assertIn("1 with no match", buf.getvalue())
