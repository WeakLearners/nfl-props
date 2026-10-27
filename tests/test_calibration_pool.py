"""The fraction fit leaves out edge-chosen rungs (selection feedback)."""
import pathlib
import sys
import unittest
from unittest import mock

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from nflprops import calibration as CAL


def leg(rung, week=8):
    d = {"s": "receptions", "mp": 0.6, "bp": 0.55, "hit": True, "week": week}
    if rung is not None:
        d["rung"] = rung
    return d


class FitPoolTests(unittest.TestCase):
    def test_chosen_excluded_others_kept(self):
        legs = [leg("chosen"), leg("fallback"), leg("none"), leg(None, week=7)]
        pool = CAL._fit_pool(legs)
        self.assertEqual([l.get("rung") for l in pool], ["fallback", "none", None])

    def test_fit_reports_excluded_count(self):
        legs = [leg("chosen"), leg("chosen"), leg("fallback"), leg(None, week=7)]
        with mock.patch.object(CAL, "_all_graded", return_value=legs):
            r = CAL.fit(write=False)
        self.assertEqual(r["legs"], 2)
        self.assertEqual(r["chosen_excluded"], 2)


class RescoreTests(unittest.TestCase):
    def test_rescored_leg_takes_new_mu_sd_and_probability(self):
        import pandas as pd
        from nflprops import rescored as RS
        d = pd.DataFrame([{"season": 2026, "week": 3, "player": "A B", "stat": "receptions", "mu": 5.0, "sd": 1.0, "actual": 6}])
        ix = d.set_index(["week", "player", "stat"])
        legs = [{"p": "A B", "s": "receptions", "week": 3, "season": 2026, "line": 5.0, "mp": 0.9, "mp_raw": 0.9, "bp": 0.6, "hit": True},
                {"p": "C D", "s": "receptions", "week": 3, "season": 2026, "line": 5.0, "mp": 0.9, "bp": 0.6, "hit": True},
                {"p": "E F", "s": "receptions", "week": 9, "season": 2026, "line": 5.0, "mp": 0.7, "bp": 0.6, "hit": True}]
        with mock.patch.object(RS, "max_week", return_value=7), mock.patch.object(RS, "index", return_value=ix):
            out = CAL._rescore(legs)
        self.assertEqual(len(out), 2)  # the unmatched week 3 leg is dropped, the week 9 leg is kept
        self.assertAlmostEqual(out[0]["mp_raw"], 0.5, places=6)  # Normal(5, 1) over a line of 5
        self.assertEqual(out[0]["mu"], 5.0)
        self.assertEqual(out[1]["mp"], 0.7)


if __name__ == "__main__":
    unittest.main()
