"""Badge rules (nflprops/badges.py). Pure functions only: no database, no network, no Slack."""
import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from nflprops import badges as B

CUR3 = [True] * 3
STEADY = [50, 52, 48, 51, 49, 50, 53, 47, 50, 52, 48, 51]   # 12 baseline games, sd about 2
VAR_POS = 400.0                                              # a typical within-player variance (sd 20)


def trend(recent, base=STEADY, flags=None, var=VAR_POS, rel=B.PLAYER_REL_MIN, k=B.PLAYER_SHRINK_K,
          base_min=B.PLAYER_BASE_MIN):
    vals = list(recent) + list(base)
    return B.trend_test(vals, flags or CUR3 + [False] * len(base), var, k, base_min, B.PLAYER_BASE_MAX, rel)


class TrendRules(unittest.TestCase):
    def test_large_sustained_rise_gets_up(self):
        r = trend([110, 120, 105])
        self.assertEqual(r["dir"], "up")
        self.assertGreaterEqual(r["t"], B.T_MIN)

    def test_large_sustained_fall_gets_down(self):
        self.assertEqual(trend([5, 8, 2])["dir"], "down")

    def test_one_big_game_is_not_a_trend(self):
        self.assertIsNone(trend([150, 50, 50]))

    def test_small_change_gets_nothing(self):
        self.assertIsNone(trend([55, 56, 54]))

    def test_thin_baseline_gets_nothing(self):
        self.assertIsNone(trend([110, 120, 105], base=STEADY[:B.PLAYER_BASE_MIN - 1]))

    def test_recent_games_must_be_current_season(self):
        self.assertIsNone(trend([110, 120, 105], flags=[True, True, False] + [False] * 12))

    def test_thin_player_cannot_hide_behind_a_tiny_spread(self):
        # Six identical games have zero own variance. Shrinkage to the position spread keeps t modest.
        r = trend([62, 62, 62], base=[50] * 6)
        self.assertIsNone(r)

    def test_relative_floor(self):
        # Large t on a tiny spread, but under 25% of the baseline mean: no badge.
        flat = [100.0, 100.5, 99.5, 100, 100.2, 99.8, 100, 100.1]
        self.assertIsNone(B.trend_test([110, 112, 111] + flat, CUR3 + [False] * 8, 1.0, 6, 6, 12, B.PLAYER_REL_MIN))

    def test_direction_rule_two_of_three(self):
        # Mean rises through one huge game while two recent games sit below the baseline mean.
        self.assertIsNone(trend([300, 40, 40], var=1.0, k=1))

    def test_defense_rule_uses_pooled_spread(self):
        base = [100, 102, 98, 101, 99, 100, 103, 97, 100]
        r = B.trend_test([80, 82, 78] + base, CUR3 + [False] * 9, 25.0, 1e9, B.DEF_BASE_MIN, B.DEF_BASE_MAX)
        self.assertEqual(r["dir"], "down")
        self.assertIsNone(B.trend_test([80, 82, 78] + base[:7], CUR3 + [False] * 7, 25.0, 1e9,
                                       B.DEF_BASE_MIN, B.DEF_BASE_MAX))


class MatchupRules(unittest.TestCase):
    N = 40   # ranked players at the position: top tier is ceil(0.15 * 40) = 6

    def test_top_player_vs_softest_defense_is_big(self):
        self.assertEqual(B.matchup_kind(3, self.N, 12, 32), "big")
        self.assertEqual(B.matchup_kind(3, self.N, 12, 28), "big")

    def test_top_player_vs_toughest_defense_is_caution(self):
        self.assertEqual(B.matchup_kind(3, self.N, 12, 1), "caution")
        self.assertEqual(B.matchup_kind(6, self.N, 12, 5), "caution")

    def test_average_player_vs_mid_or_good_defense_gets_nothing(self):
        for d in (6, 10, 16, 20, 27):
            self.assertIsNone(B.matchup_kind(20, self.N, 12, d))
        for d in (1, 3, 5):
            self.assertIsNone(B.matchup_kind(20, self.N, 12, d))  # average player, even vs the best defense

    def test_top_player_vs_mid_defense_gets_nothing(self):
        for d in (6, 16, 27):
            self.assertIsNone(B.matchup_kind(1, self.N, 12, d))

    def test_weak_player_vs_softest_defense_gets_nothing(self):
        self.assertIsNone(B.matchup_kind(39, self.N, 12, 32))

    def test_tier_edge(self):
        self.assertEqual(B.matchup_kind(6, self.N, 12, 32), "big")
        self.assertIsNone(B.matchup_kind(7, self.N, 12, 32))

    def test_thin_player_gets_nothing(self):
        self.assertIsNone(B.matchup_kind(1, self.N, B.TIER_MIN_GAMES - 1, 32))


class Markup(unittest.TestCase):
    DATA = {"players": {"a b|WR": {
        "trend": {"dir": "up", "title": "T"},
        "matchup": {"kind": "big", "opp": "DEN", "week": 4, "title": "M"}}},
        "defenses": {"DEN": {"wr": {"dir": "down", "title": "D"}}}}

    def test_matchup_only_for_its_own_game(self):
        self.assertEqual(len(B.player_badges("a b|WR", "DEN", 4, self.DATA)), 2)
        self.assertEqual(len(B.player_badges("a b|WR", "KC", 4, self.DATA)), 1)
        self.assertEqual(len(B.player_badges("a b|WR", "DEN", 3, self.DATA)), 1)
        self.assertEqual(len(B.player_badges("a b|WR", None, None, self.DATA)), 1)

    def test_unknown_player_has_no_badge(self):
        self.assertEqual(B.player_badges("x y|WR", "DEN", 4, self.DATA), [])

    def test_defense_badge(self):
        self.assertEqual(B.defense_badge("DEN", "wr", self.DATA)[0][1], "▼")
        self.assertEqual(B.defense_badge("DEN", "te", self.DATA), [])

    def test_ordinal(self):
        self.assertEqual([B.ordinal(n) for n in (1, 2, 3, 11, 21, 28, 31, 32)],
                         ["1st", "2nd", "3rd", "11th", "21st", "28th", "31st", "32nd"])


class CardLineMatchesBadge(unittest.TestCase):
    """The card's matchup line and its matchup badge read one rank, so they cannot point opposite ways."""

    def test_every_caution_rank_reads_tough_and_every_big_rank_reads_soft(self):
        for rank in range(1, 33):
            kind = B.matchup_kind(1, 20, 10, rank)  # a top-ranked player, so only the defense rank decides
            tone = B.matchup_tone(rank)
            if kind == "caution":
                self.assertEqual(tone, "tough", rank)
            if kind == "big":
                self.assertEqual(tone, "soft", rank)
        self.assertEqual(B.matchup_kind(1, 20, 10, 4), "caution")
        self.assertEqual(B.matchup_kind(1, 20, 10, 29), "big")

    def test_caution_card_never_shows_a_soft_line_and_big_day_never_a_tough_one(self):
        # Chase Brown rec yds (JAX 4th vs RB) and Zay Flowers rec yds (TEN 28th vs WR), the 2026 week-4 cases.
        data = {"players": {"c b|RB": {"matchup": {"kind": "caution", "opp": "JAX", "week": 4}},
                            "z f|WR": {"matchup": {"kind": "big", "opp": "TEN", "week": 4}}},
                "def_pos": {"JAX": {"RB": {"rank": 4, "ypg": 61.0}}, "TEN": {"WR": {"rank": 28, "ypg": 130.0}}}}
        self.assertEqual(B.check_consistency(data), 2)
        self.assertEqual(B.card_line(data, "JAX", "RB")["t"], "tough")
        self.assertEqual(B.card_line(data, "TEN", "WR")["t"], "soft")

    def test_disagreement_is_caught(self):
        data = {"players": {"c b|RB": {"matchup": {"kind": "caution", "opp": "JAX", "week": 4}}},
                "def_pos": {"JAX": {"RB": {"rank": 20, "ypg": 90.0}}}}
        with self.assertRaises(AssertionError):
            B.check_consistency(data)

    def test_no_table_no_line(self):
        self.assertIsNone(B.card_line({}, "JAX", "RB"))

    def test_written_badges_file_is_consistent(self):
        import json
        if not B.BADGES_PATH.exists():
            self.skipTest("no badges.json")
        data = json.loads(B.BADGES_PATH.read_text())
        if "def_pos" in data:
            B.check_consistency(data)


if __name__ == "__main__":
    unittest.main()
