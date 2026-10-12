"""Which games get a report now. Pure functions: no network, no clock, no disk.

The `auto` mode of scripts/run_reports.py runs every 30 minutes. Each run asks
this module which sittings are due, and builds only the games that have no
report yet. A "sitting" is a group of games with close kickoffs (Sunday 4:05
and 4:25 PM are one sitting). Slack gets one post per sitting.
"""
import pandas as pd

HORIZON_H = 3.0      # build when the first kickoff of a sitting is <= 3 h away
SITTING_GAP_MIN = 60  # kickoffs <= 60 min apart share one sitting
# Sunday 12:00-17:00 ET is one sitting: the 1 PM and the 4 PM games build and
# post together, three hours before the first of them. Kickoffs outside that
# window (London 9:30 AM, SNF 8:20 PM) and every other day keep the gap rule.
SUNDAY_PM_START_H, SUNDAY_PM_END_H = 12, 17
HOLD_MIN = 45        # post a partial sitting when kickoff is <= 45 min away


def _sunday_pm(t):
    """The ET calendar date if t falls in the Sunday afternoon window, else None."""
    et = t.tz_convert("America/New_York")
    if et.weekday() == 6 and SUNDAY_PM_START_H <= et.hour < SUNDAY_PM_END_H:
        return et.date()
    return None


def sittings(ev):
    """Split events into sittings by kickoff gap. Returns a list of DataFrames."""
    if ev.empty:
        return []
    ev = ev.sort_values("ct")
    out, cur, last = [], [], None
    for e in ev.itertuples():
        joined = last is not None and (e.ct - last) <= pd.Timedelta(minutes=SITTING_GAP_MIN)
        if last is not None and not joined:
            d = _sunday_pm(last)
            joined = d is not None and d == _sunday_pm(e.ct)
        if last is not None and not joined:
            out.append(cur)
            cur = []
        cur.append(e.Index)
        last = e.ct
    out.append(cur)
    return [ev.loc[ix] for ix in out]


def sitting_key(s):
    """Stable id of a sitting: its first kickoff in UTC."""
    return s.ct.min().strftime("%Y%m%dT%H%MZ")


def plan(ev, now, posted, handled_ids, horizon_h=HORIZON_H):
    """Sittings due now. Returns a list of (key, all_games, todo_games).

    ev           upcoming events (columns id, ct)
    now          pd.Timestamp, tz-aware
    posted       set of sitting keys already posted to Slack
    handled_ids  event ids that already have a report (never refetched)
    A sitting not in `posted` appears even with no todo games, so the caller can
    post a built-but-unposted sitting. A posted sitting appears only while a
    game of it still has no report.
    """
    due = []
    for s in sittings(ev):
        key = sitting_key(s)
        if s.ct.min() - now > pd.Timedelta(hours=horizon_h):
            continue
        todo = s[~s.id.isin(handled_ids)]
        # A posted sitting is done, unless a game still has no report (its
        # lines were not up at post time). That game keeps being tried.
        if key in posted and todo.empty:
            continue
        due.append((key, s, todo))
    return due


def should_post(n_games, n_ready, start, now, hold_min=HOLD_MIN):
    """Post when every game is ready, or when the wait must end (kickoff near)."""
    if n_ready == 0:
        return False
    return n_ready >= n_games or (start - now) <= pd.Timedelta(minutes=hold_min)
