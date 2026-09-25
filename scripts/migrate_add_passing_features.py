"""Decision #39 (2026-09-24): add QB passing features + y_passing_yards to the
LIVE training_rows table, additively.

Why this script exists instead of just re-running build_training_table.py's
main(): main() does `df.to_sql("training_rows", con, if_exists="replace", ...)`
-- a full-table replace. The build_db.py lesson (2026-09-22 postmortem) is that
a full rebuild once silently dropped 2019-2025 rows and every sanity check
passed because it only looked at the current season. This script never
replaces the table. It:

  1. Builds the FULL fresh frame via assemble_features() -- the exact same
     function pregame_features() calls (decision #30's one-code-path rule),
     now extended with build_qb_passing_rolling().
  2. Proves every PRE-EXISTING column, for every existing row, is byte-for-
     byte identical to what's on disk right now. If this check fails, the
     script raises and writes nothing.
  3. Only then: ALTER TABLE to add the new columns (all NULL by default) and
     UPDATE them in place, keyed by (player_id, game_id). Existing rows keep
     their rowid; no row is dropped, added, or reordered.

Read+write against features.db. Writes ONLY the 16 new columns
(y_passing_yards + 15 features) into the existing training_rows table.
"""
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path.home() / "Code/nfl-props"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from build_training_table import load, assemble_features  # noqa: E402

FEATURES_DB = ROOT / "data/features.db"

NEW_FEATURE_COLS = (
    [f"pass_attempts_l{w}" for w in (4, 8)] + ["pass_attempts_career"]
    + [f"pass_completions_l{w}" for w in (4, 8)] + ["pass_completions_career"]
    + [f"{n}_l{w}" for n in ("yards_per_attempt", "completion_rate", "pass_adot") for w in (4, 8)]
    + [f"{n}_career" for n in ("yards_per_attempt", "completion_rate", "pass_adot")]
)
NEW_LABEL_COL = "y_passing_yards"
ALL_NEW_COLS = NEW_FEATURE_COLS + [NEW_LABEL_COL]


def main():
    con = sqlite3.connect(FEATURES_DB, timeout=60)
    before = pd.read_sql_query("SELECT * FROM training_rows", con)
    print(f"training_rows on disk: {before.shape[0]:,} rows x {before.shape[1]} cols")

    pg, snaps, injuries, schedules, xwalk, depth, stadiums, weather, spine = load()
    fresh = assemble_features(pg, snaps, injuries, schedules, xwalk, depth, stadiums, weather, spine)

    labels = pg[["player_id", "game_id", "targets", "carries", "receptions",
                 "receiving_yards", "rushing_yards", "receiving_tds", "rushing_tds",
                 "target_share", "air_yards_share", "wopr", "passing_yards"]].rename(columns={
        "targets": "y_targets", "carries": "y_carries", "receptions": "y_receptions",
        "receiving_yards": "y_receiving_yards", "rushing_yards": "y_rushing_yards",
        "receiving_tds": "y_receiving_tds", "rushing_tds": "y_rushing_tds",
        "target_share": "y_target_share_row", "air_yards_share": "y_air_yards_share_row",
        "wopr": "y_wopr_row", "passing_yards": NEW_LABEL_COL,
    })
    fresh = fresh.merge(labels, on=["player_id", "game_id"], how="left")

    # ---- Step 2: prove every pre-existing column is unchanged -------------
    pre_existing_cols = [c for c in before.columns if c not in ALL_NEW_COLS]
    missing_in_fresh = [c for c in pre_existing_cols if c not in fresh.columns]
    if missing_in_fresh:
        raise RuntimeError(f"fresh build is missing pre-existing columns: {missing_in_fresh}")

    key = ["player_id", "game_id"]
    b = before.sort_values(key).reset_index(drop=True)
    f = fresh[fresh.set_index(key).index.isin(b.set_index(key).index)].copy()
    f = f.sort_values(key).reset_index(drop=True)

    if len(b) != len(f):
        raise RuntimeError(
            f"row count mismatch: before={len(b)} fresh-matched={len(f)} "
            f"(fresh total={len(fresh)}) -- ABORTING, nothing written"
        )
    if not (b[key].reset_index(drop=True) == f[key].reset_index(drop=True)).all().all():
        raise RuntimeError("row key alignment mismatch after sort -- ABORTING, nothing written")

    mismatches = {}
    for c in pre_existing_cols:
        bc, fc = b[c], f[c]
        if pd.api.types.is_numeric_dtype(bc) or pd.api.types.is_bool_dtype(bc):
            bc_n = pd.to_numeric(bc, errors="coerce")
            fc_n = pd.to_numeric(fc, errors="coerce")
            both_nan = bc_n.isna() & fc_n.isna()
            close = np.isclose(bc_n.where(~both_nan, 0), fc_n.where(~both_nan, 0),
                                rtol=1e-9, atol=1e-9, equal_nan=False)
            ok = (close | both_nan).all()
        else:
            ok = ((bc.astype("string") == fc.astype("string")) | (bc.isna() & fc.isna())).all()
        if not ok:
            n_bad = int((~((bc == fc) | (bc.isna() & fc.isna()))).sum()) if not pd.api.types.is_numeric_dtype(bc) else "see above"
            mismatches[c] = n_bad

    if mismatches:
        raise RuntimeError(f"pre-existing columns changed by this edit -- ABORTING, nothing "
                            f"written: {mismatches}")

    print(f"verified: all {len(pre_existing_cols)} pre-existing columns identical across "
          f"{len(b):,} rows")

    per_season_before = before.groupby("season").size().sort_index()
    per_season_after = f.groupby("season").size().sort_index()
    if not per_season_before.equals(per_season_after):
        raise RuntimeError("per-season row counts changed -- ABORTING, nothing written")
    print("per-season row counts unchanged:")
    print(per_season_before.to_string())

    # ---- Step 3: additive ALTER + UPDATE -----------------------------------
    existing_cols = set(pd.read_sql_query("PRAGMA table_info(training_rows)", con)["name"])
    for c in ALL_NEW_COLS:
        if c not in existing_cols:
            con.execute(f'ALTER TABLE training_rows ADD COLUMN "{c}" REAL')
    con.commit()

    update_df = fresh[key + ALL_NEW_COLS].copy()
    set_clause = ", ".join(f'"{c}" = ?' for c in ALL_NEW_COLS)
    sql = f'UPDATE training_rows SET {set_clause} WHERE player_id = ? AND game_id = ?'
    payload = [
        tuple(None if pd.isna(v) else float(v) for v in row[2:]) + (row[0], row[1])
        for row in update_df[key + ALL_NEW_COLS].itertuples(index=False, name=None)
    ]
    con.executemany(sql, payload)
    con.commit()

    n_qb_with_attempts = int(fresh[NEW_LABEL_COL].notna().sum())
    print(f"\nwrote {len(ALL_NEW_COLS)} new columns to training_rows for {len(payload):,} rows")
    print(f"y_passing_yards populated (i.e. a passing stat line exists) for {n_qb_with_attempts:,} rows")

    after = pd.read_sql_query("SELECT COUNT(*) n FROM training_rows", con)
    print(f"training_rows row count after: {after.n.iloc[0]:,} (was {before.shape[0]:,})")
    con.close()


if __name__ == "__main__":
    main()
