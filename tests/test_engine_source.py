"""The build uses the new model for every stat (since 2026-10-27).
Golden rows were captured from the build on 2026-10-27 (tests/fixtures/w7_new_rows.json): week 7,
two games, empty lines, rung choice off."""
import json
import math
import pathlib
import sys
import unittest

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from nflprops import engine_source as ES, report as R, spread as SP

GOLD = json.loads((ROOT / "tests" / "fixtures" / "w7_new_rows.json").read_text())
KEYS = ("mu", "sd", "lo", "hi", "pe", "sp", "role", "tm", "vs", "def", "app")


class NewPathRows(unittest.TestCase):
    def test_week7_rows_match_the_golden_rows(self):
        for game, gold in GOLD.items():
            rows, _, _ = R.build(2026, 7, game.split("-"), pd.DataFrame(), rung_choice=False)
            got = {(r["p"], r["s"]): r for r in rows}
            self.assertEqual(len(got), len(gold), game)
            for g in gold:
                r = got[(g["p"], g["s"])]
                for k in KEYS:
                    self.assertEqual(r[k], g[k], (game, g["p"], g["s"], k))

    def test_every_row_has_a_finite_mu_and_sd(self):
        rows, _, _ = R.build(2026, 7, ["CLE", "TEN"], pd.DataFrame(), rung_choice=False)
        self.assertTrue(rows)
        for r in rows:
            self.assertTrue(math.isfinite(r["mu"]) and math.isfinite(r["sd"]) and r["sd"] > 0, (r["p"], r["s"]))

    def test_sd_scale_has_every_stat(self):
        sc = ES.sd_scale()
        self.assertEqual(set(sc), set(ES.STATS))
        self.assertTrue(all(0.5 < v < 1.5 for v in sc.values()))


class SpreadMatchesOldSd(unittest.TestCase):
    """spread.py is the sd that sat in project_week, moved unchanged (project.py is kept for this check)."""

    def test_slate_rows_sd_equals_project_week_sd(self):
        import contextlib
        import io
        from nflprops import project as P
        a = SP.slate_rows(2026, 7)
        with contextlib.redirect_stdout(io.StringIO()):
            b = P.project_week(2026, 7)
        m = a.merge(b[["player_id", "stat", "sd", "n_games"]], on=["player_id", "stat"], how="outer",
                    suffixes=("", "_old"), indicator=True)
        self.assertTrue((m._merge == "both").all())
        self.assertEqual(float((m.sd - m.sd_old).abs().max()), 0.0)
        self.assertEqual(int((m.n_games - m.n_games_old).abs().max()), 0)


if __name__ == "__main__":
    unittest.main()
