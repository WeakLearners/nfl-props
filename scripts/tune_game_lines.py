#!/usr/bin/env python3
"""Pick the game-lines hyperparameters on 2024-2025 out-of-fold games. Never reads 2026 scores."""
import itertools, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from nflprops import gamelines as G

g = G.load_games()
g = g[g["season"] <= 2025]
rows = []
for a, h, qb, rest in itertools.product([5, 10, 20, 40, 60], [12, 20, 30, 45, 60], [True], [False]):
    o = G.walk_forward(g, [2024, 2025], dict(alpha=a, half_life=h, qb=qb, rest=rest))
    o = o[o.played & (o.week >= 3)]
    rows.append((a, h, qb, rest, (o.margin - o.pred_margin).abs().mean(), (o.total - o.pred_total).abs().mean(), len(o)))
rows.sort(key=lambda r: r[4] + r[5])
print("alpha half_life qb rest  mae_margin mae_total n")
for r in rows[:12]: print(*r)
print("--- best without qb/rest:", [r for r in rows if not r[2] and not r[3]][0])
print("--- best qb only:", [r for r in rows if r[2] and not r[3]][0])
