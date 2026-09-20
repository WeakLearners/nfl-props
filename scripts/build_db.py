#!/usr/bin/env python
"""Rebuild the local nflverse mirror. Safe to re-run; every table is replaced.

usage:
  build_db.py            # every season, 2019 onward
  build_db.py current    # this season only -- ~2s, what the polling job runs
  build_db.py 2026 2025  # named seasons only
"""
import sys, time
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))

from nflprops import ingest
from nflprops.config import DB_PATH, CURRENT_SEASON

if __name__ == "__main__":
    args = sys.argv[1:]
    # "current" rather than a literal year: the polling job runs unattended for
    # years, and a hardcoded 2026 in a plist would quietly stop refreshing the
    # live season the moment the calendar turned over.
    if args == ["current"]:
        yrs = [CURRENT_SEASON]
    else:
        yrs = [int(a) for a in args] or None
    t0 = time.time()
    counts = ingest.build(yrs)
    for name, n in counts.items():
        print(f"  {name:<14} {n:>8,} rows")
    print(f"-> {DB_PATH}  ({time.time() - t0:.0f}s)")
