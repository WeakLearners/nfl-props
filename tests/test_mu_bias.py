"""Weekly projection bias: shrink math, missing-file default, effect on mp."""
import json
import pathlib
import sys
import tempfile
import unittest
from unittest import mock

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from nflprops import calibration as CAL
from nflprops import report as RP


class BiasTests(unittest.TestCase):
    def test_shrink_math(self):
        self.assertAlmostEqual(CAL.shrink_bias(-20.0, 50), -10.0)
        self.assertAlmostEqual(CAL.shrink_bias(18.0, 150), 13.5)
        self.assertEqual(CAL.shrink_bias(5.0, 0), 0.0)

    def test_fit_bias_is_actual_minus_mu_raw_shrunk(self):
        rows = pd.DataFrame({"stat": ["passing_yards"] * 4 + ["receptions"] * 2,
                             "mu_raw": [200, 210, 220, 230, 4.0, 5.0],
                             "actual": [220, 230, 240, 250, 4.0, 5.0]})
        b, n = CAL.fit_bias(rows)
        self.assertEqual(n, {"passing_yards": 4, "receptions": 2})
        self.assertAlmostEqual(b["passing_yards"], round(20 * 4 / 54, 2))
        self.assertEqual(b["receptions"], 0.0)

    def test_missing_file_and_stat_default_to_zero(self):
        with mock.patch.object(CAL, "PATH", pathlib.Path(tempfile.mkdtemp()) / "none.json"):
            self.assertEqual(CAL.mu_bias(), {})
            self.assertEqual(CAL.bias_for("passing_yards"), 0.0)
        p = pathlib.Path(tempfile.mkdtemp()) / "c.json"
        p.write_text(json.dumps({"fraction": 0.7, "mu_bias": {"passing_yards": 12.5}}))
        with mock.patch.object(CAL, "PATH", p):
            self.assertEqual(CAL.bias_for("passing_yards"), 12.5)
            self.assertEqual(CAL.bias_for("rushing_yards"), 0.0)

    def test_bias_raises_mp_for_an_over(self):
        leg = {"s": "passing_yards", "mu": 250.0, "sd": 50.0, "line": 250.0,
               "mp_raw": 0.5, "mp": 0.5, "bp": 0.6, "hit": True}
        out = CAL.with_bias([leg, dict(leg, s="rushing_yards")], {"passing_yards": 20.0})
        self.assertGreater(out[0]["mp_raw"], 0.6)  # mu moved 0.4 sd up
        self.assertEqual(out[1]["mp_raw"], 0.5)    # no bias for this stat: untouched
        self.assertEqual(leg["mp_raw"], 0.5)       # input not changed

    def test_keep_only_where_brier_drops(self):
        def mk(st, hit):  # model says 50%, line at mu: a +bias raises mp
            return {"s": st, "mu": 100.0, "sd": 10.0, "line": 100.0, "mp_raw": 0.5,
                    "mp": 0.5, "bp": 0.5, "hit": hit}
        legs = [mk("up_stat", True)] * 8 + [mk("up_stat", False)] * 2 \
             + [mk("bad_stat", False)] * 8 + [mk("bad_stat", True)] * 2
        kept, info = CAL.select_bias(legs, {"up_stat": 5.0, "bad_stat": 5.0, "no_legs": 2.0}, 0.0)
        self.assertEqual(kept, {"up_stat": 5.0, "bad_stat": 0.0, "no_legs": 0.0})
        self.assertTrue(info["up_stat"]["kept"])
        self.assertLess(info["up_stat"]["brier_after"], info["up_stat"]["brier_before"])
        self.assertFalse(info["bad_stat"]["kept"])
        self.assertFalse(info["no_legs"]["kept"])
        self.assertIsNone(info["no_legs"]["brier_before"])

    def test_bias_uses_mu_raw_when_present(self):
        leg = {"s": "passing_yards", "mu": 270.0, "mu_raw": 250.0, "sd": 50.0,
               "line": 250.0, "mp_raw": 0.65, "bp": 0.6}
        out = CAL.with_bias([leg], {"passing_yards": 20.0})
        self.assertAlmostEqual(out[0]["mp_raw"], 1 - 0.5 * (1 + __import__("math").erf(-0.4 / 2 ** 0.5)) , places=3)

    def test_build_applies_bias_only_when_gate_on(self):
        import glob
        fs = [f for f in sorted(glob.glob(str(ROOT / "data" / "boards" / "2026_w7_*.json")))]
        for f in fs:
            with open(f) as fh:
                d = json.load(fh)
            if isinstance(d, dict) and d.get("bookmakers"):
                teams = pathlib.Path(f).name.split("_")[2].split("-")
                break
        else:
            self.skipTest("no board")
        lines = RP.parse_lines(d)
        with mock.patch.object(CAL, "bias_for", lambda s: 25.0 if s == "passing_yards" else 0.0):
            on = RP.build(2026, 7, teams, lines, rung_choice=True)[0]
            off = RP.build(2026, 7, teams, lines)[0]
        self.assertTrue(all("mu_raw" not in r for r in off))
        qb = [(a, b) for a in on for b in off if a["p"] == b["p"] and a["s"] == b["s"] == "passing_yards"]
        if not qb:
            self.skipTest("no QB row on this board")
        for a, b in qb:
            self.assertAlmostEqual(a["mu"] - a["mu_raw"], 25.0, delta=0.11)
            self.assertAlmostEqual(a["mu_raw"], b["mu"], delta=0.11)
        other = [(a, b) for a in on for b in off if a["p"] == b["p"] and a["s"] == b["s"] != "passing_yards"]
        for a, b in other:
            self.assertEqual(a["mu"], b["mu"])


if __name__ == "__main__":
    unittest.main()
