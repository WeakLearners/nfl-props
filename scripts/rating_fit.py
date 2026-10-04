#!/usr/bin/env python
"""Rating Phase 2: fit weights on 2021-2023 only, save weights and ratings.

Writes data/ratings/weights.json and data/ratings/{season}_w{week}.parquet for
2021-2025. The 2026 season is never read. Re-run: .venv/bin/python scripts/rating_fit.py
"""
import sys, pathlib, json
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import pandas as pd
from nflprops import rating as R
from nflprops.shares import player_game_shares

FIT, ALL = (2021, 2022, 2023), (2021, 2022, 2023, 2024, 2025)


def weeks(pg, season):
    return sorted(pg[pg.season == season].week.unique())


def main():
    pg = player_game_shares()
    pg = pg[pg.season <= 2025]          # the 2026 holdout is never read here
    prep = R._prep(pg)
    stats = ["player_id", "season", "week"] + sorted(set(R.PRIMARY.values()))
    actual = pg[stats]
    zf = []
    for s in FIT:
        for w in weeks(pg, s):
            z = R.zscores(R.components(prep, s, w))
            z["season"], z["week"] = s, w
            zf.append(z)
    weights = R.fit_weights(zf, actual)
    R.RATINGS_DIR.mkdir(parents=True, exist_ok=True)
    R.WEIGHTS_PATH.write_text(json.dumps(
        {"fit_on": "2021-2023", "method": "NNLS of primary stat z on component z",
         "weights": weights}, indent=1))
    for s in ALL:
        for w in weeks(pg, s):
            out = R.rate(R.components(prep, s, w), weights)
            out["season"], out["week"] = s, w
            out.to_parquet(R.RATINGS_DIR / f"{s}_w{w}.parquet", index=False)
    print(json.dumps(weights, indent=1))


if __name__ == "__main__":
    main()
