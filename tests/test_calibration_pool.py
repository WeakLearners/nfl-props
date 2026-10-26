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


if __name__ == "__main__":
    unittest.main()
