#!/usr/bin/env python
"""Walk every past week through tdmodel.features() and attach what happened.

Deliberately slow and dumb: one call per (season, week), exactly the call
the live slate will make. A faster vectorised build would be a second
implementation of the same arithmetic, which is the thing this whole module
exists to avoid.

A rostered player with no box-score row gets td=0 rather than being dropped.
He was available, the book would have priced him, and he did not score.
"""
import pathlib
import sys
import time

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import pandas as _pd  # noqa: E402
from nflprops.db import connect  # noqa: E402
from nflprops.tdmodel import features, labels  # noqa: E402


def week_is_complete(season, week):
    """Every scheduled game for the week has a final score.

    Without this, a week in progress poisons the training set. 2026 W2 was
    caught on 2026-09-20 doing exactly that: Thursday's game was played and
    the other fifteen were not, so labels() found 8 scorers among 497
    rostered players and reported a 1.6% touchdown rate. Every one of those
    489 zeros is a game that has not happened. Nothing would have raised --
    the rows look identical to a genuinely low-scoring week, and the model
    would simply have learned that nobody scores.
    """
    with connect() as con:
        g = _pd.read_sql_query(
            "SELECT home_score FROM schedules WHERE season=? AND week=? AND game_type='REG'",
            con, params=[season, week])
    return (not g.empty) and g.home_score.notna().all()

OUT = ROOT / "data" / "td_train.parquet"
SEASONS = range(2019, 2027)
WEEKS = range(1, 19)


def main():
    frames, t0 = [], time.time()
    for season in SEASONS:
        for week in WEEKS:
            f = features(season, week)
            if f.empty:
                continue
            if not week_is_complete(season, week):
                print(f"  {season} W{week:<2} skipped — week not finished")
                continue
            lab = labels(season, week)
            if lab.empty:
                continue
            m = f.merge(lab, on="player_id", how="left")
            m["td"] = m["td"].fillna(0).astype(int)
            frames.append(m)
            print(f"  {season} W{week:<2} {len(m):>4} players  td {m.td.mean():.3f}")
    df = pd.concat(frames, ignore_index=True)
    df.to_parquet(OUT, index=False)
    print(f"\n{len(df):,} rows -> {OUT}  ({time.time()-t0:.0f}s)")
    print(f"overall td rate {df.td.mean():.4f}")
    print(df.groupby('position').td.agg(['mean', 'size']).round(4).to_string())


if __name__ == "__main__":
    main()
