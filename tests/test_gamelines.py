import unittest

import numpy as np
import pandas as pd

from nflprops import gamelines as G


class GameLines(unittest.TestCase):
    def test_devig_removes_margin(self):
        p = G.devig(-110, -110)
        assert abs(float(p) - 0.5) < 1e-9
        assert float(G.devig(-200, 170)) > 0.6


    def test_win_prob_symmetry(self):
        assert abs(float(G.win_prob(0, 13)) - 0.5) < 1e-9
        assert abs(float(G.win_prob(7, 13)) + float(G.win_prob(-7, 13)) - 1) < 1e-9


    def test_fit_predict_uses_only_train_and_home_field_positive(self):
        rng = np.random.default_rng(0)
        teams = ["AAA", "BBB", "CCC", "DDD"]
        rows = []
        for t in range(60):
            h, a = rng.choice(teams, 2, replace=False)
            rows.append(dict(season=2024, week=t % 18 + 1, t=t, home_team=h, away_team=a,
                             home_score=24 + rng.normal(0, 7) + (6 if h == "AAA" else 0),
                             away_score=21 + rng.normal(0, 7), home_qb_out=0, away_qb_out=0,
                             home_rest=7, away_rest=7))
        tr = pd.DataFrame(rows)
        tgt = pd.DataFrame([dict(season=2024, week=19, t=61, home_team="AAA", away_team="BBB",
                                 home_qb_out=0, away_qb_out=0, home_rest=7, away_rest=7)])
        out = G.fit_predict(tr, tgt, {"alpha": 5.0, "half_life": 1e9, "wx": ()})
        assert out["pred_margin"].iloc[0] > 3
        assert abs(out["pred_total"].iloc[0] - (out["mu_home"] + out["mu_away"]).iloc[0]) < 1e-9

