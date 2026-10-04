"""Which games get a report now. Pure functions: no network, no clock, no disk.

The `auto` mode of scripts/run_reports.py runs every 30 minutes. Each run asks
this module which sittings are due, and builds only the games that have no
report yet. A "sitting" is a group of games with close kickoffs (Sunday 4:05
and 4:25 PM are one sitting). Slack gets one post per sitting.
"""
import pandas as pd

HORIZON_H = 3.0      # build when the first kickoff of a sitting is <= 3 h away
SITTING_GAP_MIN = 60  # kickoffs <= 60 min apart share one sitting
HOLD_MIN = 45        # post a partial sitting when kickoff is <= 45 min away


def sittings(ev):
    """Split events into sittings by kickoff gap. Returns a list of DataFrames."""
    if ev.empty:
        return []
    ev = ev.sort_values("ct")
    out, cur, last = [], [], None
    for e in ev.itertuples():
        if last is not None and (e.ct - last) > pd.Timedelta(minutes=SITTING_GAP_MIN):
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
    A sitting with no todo games and no post pending still appears only if
    it is not in `posted`, so the caller can post a built-but-unposted sitting.
    """
    due = []
    for s in sittings(ev):
        key = sitting_key(s)
        if key in posted:
            continue
        if s.ct.min() - now > pd.Timedelta(hours=horizon_h):
            continue
        todo = s[~s.id.isin(handled_ids)]
        due.append((key, s, todo))
    return due


def should_post(n_games, n_ready, start, now, hold_min=HOLD_MIN):
    """Post when every game is ready, or when the wait must end (kickoff near)."""
    if n_ready == 0:
        return False
    return n_ready >= n_games or (start - now) <= pd.Timedelta(minutes=hold_min)
