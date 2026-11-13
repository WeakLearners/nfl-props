"""Odds API credit guard and credit log.

The free plan gives 500 credits a month. The cycle resets on the 13th.
A call to the event-odds endpoint costs 1 credit for each market it asks for.
The event list is free.

Rules in this file:
- Every paid call goes through `fetch`. It reads x-requests-remaining from the
  response and keeps the last value in data/credits.json.
- When the last known value is below FLOOR, `fetch` makes no call. It returns
  None and the caller builds the model-only report.
- When the API says the credits are used up (HTTP 401 OUT_OF_USAGE_CREDITS or
  429), `fetch` records 0 remaining and returns None. It never raises for that.
- The free event call also records x-requests-remaining, so a new cycle
  clears an old low value on the next pass.
- Every paid call, skip and credit-out writes one row to data/credit_log.csv.
"""
import csv
import datetime as dt
import json
import os
import pathlib
import urllib.error
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parent.parent
STATE = ROOT / "data" / "credits.json"
LOG = ROOT / "data" / "credit_log.csv"
EVENTS_CACHE = ROOT / "data" / "events_cache.json"
FLOOR = int(os.environ.get("ODDS_CREDIT_FLOOR", "60"))
CYCLE_DAY = 13          # the plan cycle resets on this day of the month
LOG_COLS = ["ts", "sitting", "kind", "label", "cost", "remaining", "note"]

context = ""            # the sitting key; run_reports sets it for each sitting


def _now():
    return dt.datetime.now(dt.timezone.utc)


def _num(v):
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return None


def remaining():
    """Last known remaining credits, or None when unknown."""
    try:
        return _num(json.loads(STATE.read_text()).get("remaining"))
    except Exception:  # noqa: BLE001
        return None


def _save_remaining(rem):
    if rem is None:
        return
    try:
        STATE.parent.mkdir(exist_ok=True)
        STATE.write_text(json.dumps({"remaining": rem, "ts": _now().isoformat()}))
    except OSError:
        pass


def log(kind, label, cost, rem, note=""):
    """Append one row to the credit log. Never raises."""
    try:
        LOG.parent.mkdir(exist_ok=True)
        new = not LOG.exists()
        with LOG.open("a", newline="") as f:
            w = csv.writer(f)
            if new:
                w.writerow(LOG_COLS)
            w.writerow([_now().isoformat(timespec="seconds"), context, kind, label,
                        cost, "" if rem is None else rem, note])
    except OSError:
        pass


def can_spend(floor=None):
    """True when a paid call is allowed. Unknown remaining counts as allowed."""
    rem = remaining()
    return rem is None or rem >= (FLOOR if floor is None else floor)


def _exhausted(err):
    if err.code == 429:
        return True
    try:
        return "OUT_OF_USAGE_CREDITS" in err.read().decode("utf-8", "replace")
    except Exception:  # noqa: BLE001
        return False


def fetch(url, label, n_markets, timeout=30):
    """One paid call. Returns (json, remaining). json is None when skipped."""
    if not can_spend():
        rem = remaining()
        log("skip", label, 0, rem, f"below floor {FLOOR}")
        print(f"  ! credit guard: {label} skipped, {rem} credits left (floor {FLOOR})")
        return None, rem
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            d = json.load(r)
            last = _num(r.headers.get("x-requests-last"))
            rem = _num(r.headers.get("x-requests-remaining"))
    except urllib.error.HTTPError as e:
        if not _exhausted(e):
            raise
        _save_remaining(0)
        log("out", label, 0, 0, f"HTTP {e.code}: credits used up")
        print(f"  ! credit guard: {label} skipped, the API reports no credits left")
        return None, 0
    _save_remaining(rem)
    log("call", label, n_markets if last is None else last, rem)
    return d, rem


def events_json(url, timeout=30):
    """The free event list. Records remaining. Falls back to the last good
    list when the key has no credits, so a sitting still gets its report."""
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            d = json.load(r)
            rem = _num(r.headers.get("x-requests-remaining"))
    except urllib.error.HTTPError as e:
        if _exhausted(e) and EVENTS_CACHE.exists():
            _save_remaining(0)
            print("  ! credit guard: event list refused, using the saved list")
            return json.loads(EVENTS_CACHE.read_text())
        raise
    _save_remaining(rem)
    try:
        EVENTS_CACHE.parent.mkdir(exist_ok=True)
        EVENTS_CACHE.write_text(json.dumps(d))
    except OSError:
        pass
    return d


def retry_due(tried_at, kickoff, now, within_h=3.0, gap_min=120):
    """May a game with no lines be fetched again now?
    Only when its kickoff is within `within_h` hours and the last try was at
    least `gap_min` minutes ago. A game with no recorded try is due."""
    import pandas as pd
    if kickoff - now > pd.Timedelta(hours=within_h):
        return False
    if not tried_at:
        return True
    return now - pd.Timestamp(tried_at) >= pd.Timedelta(minutes=gap_min)


def cycle_start(now=None):
    """Start (UTC date) of the plan cycle that holds `now`."""
    n = now or _now()
    y, m = n.year, n.month
    if n.day < CYCLE_DAY:
        y, m = (y - 1, 12) if m == 1 else (y, m - 1)
    return dt.datetime(y, m, CYCLE_DAY, tzinfo=dt.timezone.utc)


def totals(path=None, since=None):
    """(credits used since `since`, dict of credits used by sitting)."""
    path = path or LOG
    since = since or cycle_start()
    total, by = 0, {}
    if not pathlib.Path(path).exists():
        return 0, by
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            if row["kind"] != "call":
                continue
            if dt.datetime.fromisoformat(row["ts"]) < since:
                continue
            c = _num(row["cost"]) or 0
            total += c
            by[row["sitting"]] = by.get(row["sitting"], 0) + c
    return total, by
