"""Hold-up rates and the outlook label rule (nflprops/trends.py)."""
import pathlib
import sys
import unittest

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from nflprops import trends as T


def frame(ent, vals, season=2023, pos="WR", vol=5.0):
    return pd.DataFrame([dict(entity=ent, name=ent, team="AAA", position=pos, season=season,
                              week=i + 1, value=v, vol=vol, targets=vol, carries=0)
                         for i, v in enumerate(vals)])


class HoldupRates(unittest.TestCase):
    def test_held_and_faded_cases(self):
        held = frame("a", [10] * 3 + [20] * 3 + [20] * 3)      # +10, next 3 stay at 20: carry 1.0
        faded = frame("b", [10] * 3 + [20] * 3 + [10] * 3)     # +10, next 3 back to 10: carry 0.0
        r = T.holdup_rates({"target_share": pd.concat([held, faded])})
        k = "target_share|WR|B|rise|10+"
        self.assertEqual(r[k]["n"], 2)
        self.assertEqual(r[k]["held"], 1)
        self.assertEqual(r[k]["hold_share"], 0.5)

    def test_small_change_is_not_a_trend(self):
        r = T.holdup_rates({"target_share": frame("a", [10] * 3 + [11] * 3 + [11] * 3)})
        self.assertEqual(r, {})

    def test_below_volume_floor_is_dropped(self):
        r = T.holdup_rates({"target_share": frame("a", [10] * 3 + [20] * 6, vol=1.0)})
        self.assertEqual(r, {})

    def test_2026_is_never_used(self):
        r = T.holdup_rates({"target_share": frame("a", [10] * 3 + [20] * 6, season=2026)})
        self.assertEqual(r, {})

    def test_fall_is_kept_apart_from_rise(self):
        r = T.holdup_rates({"target_share": frame("a", [20] * 3 + [10] * 3 + [10] * 3)})
        self.assertIn("target_share|WR|B|fall|10+", r)
        self.assertNotIn("target_share|WR|B|rise|10+", r)


class LabelRule(unittest.TestCase):
    def test_thresholds(self):
        self.assertEqual(T.label({"n": 50, "hold_share": .60})[0], "Likely to hold")
        self.assertEqual(T.label({"n": 50, "hold_share": .40})[0], "Likely to fade")
        self.assertEqual(T.label({"n": 50, "hold_share": .50})[0], "Watch")

    def test_thin_or_missing_is_watch(self):
        self.assertEqual(T.label({"n": 49, "hold_share": .90})[0], "Watch")
        self.assertEqual(T.label(None)[0], "Watch")

    def test_text_shows_rate_and_n(self):
        self.assertEqual(T.label({"n": 212, "hold_share": .70})[1], "Likely to hold (70.0%, n=212)")


class PageData(unittest.TestCase):
    """The page script draws from page_data(). Guard the numbers it shows."""
    def data(self):
        row = dict(entity="e", name="A B", team="AAA", pos="WR", last3=24.0, base=6.0, baseline="2025", prior=6.0,
                   delta=18.0, games=3, vol=21, label="Likely to hold",
                   outlook="Likely to hold (62.9%, n=62, merged: rise 5-10 + 10+)", next="vs BBB",
                   dvp={"rk": 17, "y": 100.0, "t": "mid"})
        thin = dict(row, name="C D", delta=-5.0, label="Watch", outlook="Watch (history too thin, n=12)", dvp=None)
        empty = []
        sec = dict(target_share=[row, thin], carry_share=empty, snap_share=empty, rz10=empty, pass_rate=empty,
                   rank=[dict(name="E F", team="AAA", pos="QB", rank=21, prev=31, move=10, games=36, next="vs BBB")],
                   dvp={"QB": [], "RB": [], "WR": [], "TE": []}, scoring={})
        return dict(season=2026, week=3, sections=sec)

    def test_outlook_numbers_are_kept(self):
        rows = {r["nm"]: r for r in T.page_data(self.data())["rows"]}
        a = rows["A B"]
        self.assertEqual((a["cls"], a["pct"], a["n"], a["merged"], a["opp"]), ("hold", 62.9, 62, "rise 5-10 + 10+", 17))
        self.assertEqual(a["dl"], "+18.0 pts")

    def test_thin_history_has_no_rate(self):
        c = {r["nm"]: r for r in T.page_data(self.data())["rows"]}["C D"]
        self.assertEqual((c["cls"], c["pct"], c["n"], c["thin"]), ("watch", None, 12, True))

    def test_rank_rows_have_no_rate(self):
        e = {r["nm"]: r for r in T.page_data(self.data())["rows"]}["E F"]
        self.assertEqual((e["sig"], e["from"], e["to"], e["pct"], e["dl"]), ("rank", 31, 21, None, "+10"))

    def test_page_renders_data_and_tray_css(self):
        out = T.render(self.data())
        self.assertIn('"nm":"A B"', out.replace('": "', '":"'))
        self.assertIn(".ptray-bg", out)
        self.assertNotIn("__DATA__", out)


if __name__ == "__main__":
    unittest.main()
