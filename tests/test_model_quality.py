"""Model quality numbers (nflprops/model_quality.py): small hand-checked frames."""
import pathlib
import sys
import unittest

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from nflprops import model_quality as M


def frame():
    return pd.DataFrame({
        "mu": [10.0, 20.0, 30.0, 40.0], "sd": [5.0, 5.0, 5.0, 5.0],
        "actual": [12.0, 20.0, 40.0, 38.0], "naive": [15.0, 20.0, 20.0, 40.0],
        "line": [11.5, 19.5, 29.5, 39.5], "mp": [0.6, 0.6, 0.6, 0.6], "mp_raw": [0.7, 0.5, 0.5, 0.5],
        "hit": [1.0, 1.0, 1.0, 0.0]})


class Summary(unittest.TestCase):
    def test_a1_a2(self):
        r = M.summarize(frame())
        self.assertAlmostEqual(r["mae"], 3.5)          # (2+0+10+2)/4
        self.assertAlmostEqual(r["naive_mae"], 6.25)   # (3+0+20+2)/4
        self.assertAlmostEqual(r["bias"], 2.5)         # (2+0+10-2)/4
        self.assertAlmostEqual(r["cov68"], 0.75)       # misses of 2,0,10,2 vs sd 5
        self.assertAlmostEqual(r["rel_mae"], 3.5 / r["naive_mae"])

    def test_brier_and_bins(self):
        r = M.summarize(frame())
        self.assertAlmostEqual(r["brier_raw"], ((0.3) ** 2 + 0.5 ** 2 + 0.5 ** 2 + 0.5 ** 2) / 4)
        self.assertEqual(r["b4_n"], 1)   # 0.7 falls in 60-80%
        self.assertEqual(r["b3_n"], 3)   # 0.5 falls in 40-60%
        self.assertAlmostEqual(r["b3_real"], 2 / 3)

    def test_auc(self):
        self.assertEqual(M._auc([0.9, 0.8, 0.2], [1, 1, 0]), 1.0)
        self.assertAlmostEqual(M._auc([0.5, 0.5], [1, 0]), 0.5)
        self.assertTrue(np.isnan(M._auc([0.5, 0.6], [1, 1])))


class RawProb(unittest.TestCase):
    def test_raw_falls_back_to_mp_only_when_missing(self):
        self.assertEqual(M.raw_prob({"mp": 0.7, "mp_raw": 0.6}), 0.6)
        self.assertEqual(M.raw_prob({"mp": 0.7}), 0.7)
        self.assertEqual(M.raw_prob({"mp": 0.7, "mp_raw": None}), 0.7)
        self.assertIsNone(M.raw_prob({}))


if __name__ == "__main__":
    unittest.main()
