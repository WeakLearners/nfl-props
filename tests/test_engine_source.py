"""Phase 1: with the default engine_source config, week 7 rows are what they were before the switch existed.
Golden rows were captured from the pre-switch code on 2026-10-27 (tests/fixtures/w7_golden_rows.json)."""
import json
import pathlib
import sys
import unittest
from unittest import mock

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from nflprops import engine_source as ES, report as R, spread as SP

GOLD = json.loads((ROOT / "tests" / "fixtures" / "w7_golden_rows.json").read_text())
KEYS = ("mu", "sd", "lo", "hi", "pe", "role", "tm", "vs", "def", "app")


class Defaults(unittest.TestCase):
    def test_default_config_is_the_new_path_for_every_stat(self):
        cfg = ES.load()
        self.assertEqual(cfg["sd"], {s: "new" for s in ES.STATS})
        self.assertEqual(cfg["mu"], {s: "new" for s in ES.STATS})

    def test_week7_rows_unchanged_with_the_legacy_config(self):
        for game, gold in GOLD.items():
            with mock.patch.object(ES, "load", lambda: ES.LEGACY):
                rows, _, _ = R.build(2026, 7, game.split("-"), pd.DataFrame(), rung_choice=False)
            got = {(r["p"], r["s"]): r for r in rows}
            self.assertEqual(len(got), len(gold), game)
            for g in gold:
                r = got[(g["p"], g["s"])]
                for k in KEYS:
                    self.assertEqual(r[k], g[k], (game, g["p"], g["s"], k))


class NewDefaultRows(unittest.TestCase):
    def test_every_row_is_new_path_with_a_finite_mu_and_sd(self):
        import math
        rows, _, _ = R.build(2026, 7, ["CLE", "TEN"], pd.DataFrame(), rung_choice=False)
        self.assertTrue(rows)
        for r in rows:
            self.assertEqual(r["pe"], "new", (r["p"], r["s"]))
            self.assertTrue(math.isfinite(r["mu"]) and math.isfinite(r["sd"]) and r["sd"] > 0, (r["p"], r["s"]))


class SpreadMatchesOldSd(unittest.TestCase):
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
