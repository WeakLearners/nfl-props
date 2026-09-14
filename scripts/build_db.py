#!/usr/bin/env python
"""Rebuild the local nflverse mirror. Safe to re-run; every table is replaced."""
import sys, time
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))

from nflprops import ingest
from nflprops.config import DB_PATH

if __name__ == "__main__":
    yrs = [int(a) for a in sys.argv[1:]] or None
    t0 = time.time()
    counts = ingest.build(yrs)
    for name, n in counts.items():
        print(f"  {name:<14} {n:>8,} rows")
    print(f"-> {DB_PATH}  ({time.time() - t0:.0f}s)")
