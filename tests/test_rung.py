"""Rung choice: pick the -200..-110 rung with the best model edge (week 8 on)."""
import glob
import json
import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from nflprops import report as RP


class ChooseRungTests(unittest.TestCase):
    MU, SD = 50.0, 10.0

    def test_bounds_are_inclusive(self):
        # Lines far below mu give a big edge, so only the price bound decides.
        for price in (-200, -110):
            c = RP.choose_rung([(30.5, price)], self.MU, self.SD, 0.0)
            self.assertIsNotNone(c, price)
        for price in (-201, -109, -300, 120):
            self.assertIsNone(RP.choose_rung([(30.5, price)], self.MU, self.SD, 0.0), price)

    def test_edge_must_be_above_zero(self):
        # Line far above mu: mp ~ 0, bp ~ 0.6, edge < 0 -> no rung.
        self.assertIsNone(RP.choose_rung([(90.5, -150)], self.MU, self.SD, 0.0))
        # Shrink f=1 puts mp on bp, so the edge is exactly 0 -> not above zero.
        self.assertIsNone(RP.choose_rung([(30.5, -150)], self.MU, self.SD, 1.0))
        self.assertIsNone(RP.choose_rung([], self.MU, self.SD, 0.0))
        self.assertIsNone(RP.choose_rung(None, self.MU, self.SD, 0.0))

    def test_largest_edge_wins_and_shrink_is_applied(self):
        ladder = [(45.5, -150), (40.5, -190), (35.5, -400), (30.5, -120)]
        c = RP.choose_rung(ladder, self.MU, self.SD, 0.0)
        edges = {l: RP.choose_rung([(l, p)], self.MU, self.SD, 0.0)["edge"]
                 for l, p in ladder if -200 <= p <= -110}
        self.assertEqual(c["line"], max(edges, key=edges.get))
        shrunk = RP.choose_rung([(45.5, -150)], self.MU, self.SD, 0.7)
        raw = RP.choose_rung([(45.5, -150)], self.MU, self.SD, 0.0)
        self.assertAlmostEqual(shrunk["mp"], raw["mp_raw"] + 0.7 * (raw["bp"] - raw["mp_raw"]))
        self.assertAlmostEqual(shrunk["bp"], RP._imp(-150))  # raw implied, not de-vigged

    def test_gate_is_week_8(self):
        self.assertFalse(RP.rung_on(2026, 7))
        self.assertTrue(RP.rung_on(2026, 8))
        self.assertTrue(RP.rung_on(2027, 1))


class BuildTests(unittest.TestCase):
    """One real saved week-7 board, read-only. Skipped when no board is on disk."""

    @classmethod
    def setUpClass(cls):
        fs = sorted(glob.glob(str(ROOT / "data" / "boards" / "2026_w7_*.json")))
        cls.board = None
        for f in fs:
            with open(f) as fh:
                d = json.load(fh)
            if isinstance(d, dict) and d.get("bookmakers"):
                cls.board, cls.teams = d, pathlib.Path(f).name.split("_")[2].split("-")
                break
        if cls.board is None:
            raise unittest.SkipTest("no week-7 board")
        cls.lines = RP.parse_lines(cls.board)
        cls.on = RP.build(2026, 7, cls.teams, cls.lines, rung_choice=True)
        cls.off = RP.build(2026, 7, cls.teams, cls.lines)  # week 7: gate is off

    def test_chosen_rung_in_bounds_with_old_rung_stored(self):
        chosen = [r for r in self.on[0] if r.get("rung") == "chosen"]
        self.assertTrue(chosen)
        for r in chosen:
            self.assertTrue(-200 <= r["price"] <= -110)
            self.assertGreater(r["mp"], r["bp"])
            self.assertIsNotNone(r["old_line"])
            self.assertIsNotNone(r["old_price"])
            self.assertIsNotNone(r["old_mp"])
            self.assertIsNotNone(r["old_bp"])

    def test_shortlist_has_only_chosen_legs(self):
        for r in self.on[0]:
            if r["pick"]:
                self.assertEqual(r["rung"], "chosen")

    def test_no_ladder_falls_back_and_is_marked(self):
        rows, short, _ = RP.build(2026, 7, self.teams, self.lines.drop(columns=["ladder"]),
                                  rung_choice=True)
        priced = [r for r in rows if r["line"] is not None]
        self.assertTrue(priced)
        self.assertTrue(all(r["rung"] == "fallback" for r in priced))
        old = {(r["p"], r["s"]): (r["line"], r["price"]) for r in self.off[0]}
        for r in priced:  # same line and price as the old behaviour
            self.assertEqual((r["line"], r["price"]), old[(r["p"], r["s"])])

    def test_gate_off_leaves_rows_as_before(self):
        self.assertTrue(all("rung" not in r for r in self.off[0]))


if __name__ == "__main__":
    unittest.main()
