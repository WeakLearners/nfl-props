#!/usr/bin/env python
"""Build a production-ranges report for every game on a slate, post to Slack.

Usage: run_reports.py [thursday|sunday|snf|mnf|auto] [--picks 6] [--dry]
       run_reports.py backfill EVENT_ID ISO_UTC_SNAPSHOT   (no Slack post)

`auto` is the scheduled mode (launchd, every 30 minutes). It builds a report
for every game whose sitting kicks off within 3 hours and has no report yet,
whatever the weekday or hour, and posts one Slack message per sitting.
The fixed slate names stay for manual runs.

One HTML report per game lands in reports/ (--dry writes nothing: no HTML, no
saved board, no Slack post). Slack gets a single digest with
each game's shortlist -- one message per slate, not one per game, because a
13-game Sunday would otherwise bury the channel.

Note: a scheduled job cannot publish to claude.ai. The HTML is written to disk;
publishing it as an artifact is a manual step.
"""
import sys, pathlib, json
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import pandas as pd


def current_week(connect):
    with connect() as con:
        s = pd.read_sql_query(
            "SELECT season, week FROM schedules WHERE gameday >= date('now') "
            "ORDER BY gameday LIMIT 1", con)
    if s.empty:
        sys.exit("no upcoming games -- run build_db.py")
    return int(s.season[0]), int(s.week[0])


def pick_games(ev, slate):
    """Which games this slate covers, by weekday and kickoff hour."""
    if slate == "thursday":
        return ev[ev.et.dt.dayofweek == 3].sort_values("ct").head(1)
    if slate == "mnf":
        return ev[ev.et.dt.dayofweek == 0].sort_values("ct").head(1)
    if slate == "snf":
        return ev[(ev.et.dt.dayofweek == 6) & (ev.et.dt.hour >= 19)].sort_values("ct").head(1)
    return ev[(ev.et.dt.dayofweek == 6) & (ev.et.dt.hour < 19)].sort_values("ct")


def game_week(con_fn, e, a, h):
    """(season, week) of one event from the schedules table, by ET date and teams."""
    with con_fn() as con:
        r = pd.read_sql_query(
            "SELECT season, week FROM schedules WHERE gameday=? AND away_team=? AND home_team=?",
            con, params=(f"{e.et:%Y-%m-%d}", a, h))
    return (int(r.season[0]), int(r.week[0])) if not r.empty else None


def dec(a):
    return 1 + (100 / (-a) if a < 0 else a / 100)


def build_game(RP, e, season, week, a, h, n_picks, dry, lines=None, rem=None, banner=None):
    """Build and write one game's report. Returns (item, rem) or (None, rem).
    item holds the Slack text so a later run can post it without refetching."""
    if lines is None:
        save = {} if dry else dict(season=season, week=week, teams=[a, h])
        lines, rem = RP.event_lines(e.id, **save)
    if lines.empty:
        print(f"  ! no alternate lines yet: {a}@{h}"); return None, rem
    rows, short, qb_notes = RP.build(season, week, [a, h], lines, n_picks=n_picks)
    if not rows or short.empty:
        print(f"  ! nothing cleared the screens: {a}@{h}"); return None, rem
    html = RP.render(rows, e.away_team, e.home_team, a, week, season,
                     kick=e.et, same_game=True, qb_notes=qb_notes)
    if banner:
        marker = '<div class="wrap"'
        html = html.replace(marker, banner + marker, 1) if marker in html else banner + html
    out = RP.report_path(season, week, a, h)
    if not dry:
        out.write_text(html)
    pay = short.price.map(dec).prod()
    joint = short.conf.prod()
    print(f"  {a} @ {h}: {len(short)} legs, pays {pay-1:.2f}x -> {out.name}")
    lines_txt = [
        f"`{i}.` *{r.player}* — {r.stat.replace('_',' ')} over {r.line:g}  "
        f"`{int(r.price)}`  *{r.conf*100:.0f}%*"
        for i, r in enumerate(short.itertuples(), 1)]
    return dict(file=out.name, away=e.away_team, home=e.home_team,
                when=f"{e.et:%a %-I:%M %p ET}", n=len(short), pay=pay, joint=joint,
                lines="\n".join(lines_txt)), rem


def digest_blocks(items, title, rem):
    """The Slack digest for a set of built games (same layout as always)."""
    from nflprops.slack import section, header, divider
    blocks = [header(title)]
    for it in items:
        blocks += [
            section(f"*{it['away']} at {it['home']}* · {it['when']}"),
            section(it["lines"]),
            section(f"_All {it['n']} straight: {it['pay']-1:.2f}× · naive joint chance "
                    f"{it['joint']*100:.0f}%. Same-game legs move together, so the true joint "
                    f"chance is higher and FanDuel will pay less than {it['pay']-1:.2f}× as an SGP._"),
            divider(),
        ]
    blocks.append(section(
        "_Confidence is the lower of the model's and FanDuel's own number, so it is never "
        "the more optimistic of the two. It measures how likely a leg is to *happen* — not "
        "that it is mispriced. Tested against real lines, this model was the less accurate "
        "of the two._"))
    blocks.append(section(f"_{len(items)} report(s) written to `reports/`. "
                          f"Credits remaining: {rem}._"))
    return blocks


STATE = pathlib.Path(__file__).resolve().parent.parent / "data" / "report_state.json"


def load_state():
    try:
        return json.loads(STATE.read_text())
    except Exception:
        return {"posted": {}, "games": {}}


def save_state(st):
    STATE.parent.mkdir(exist_ok=True)
    tmp = STATE.with_suffix(".tmp")
    tmp.write_text(json.dumps(st, indent=1))
    tmp.replace(STATE)


def auto(N_PICKS, DRY):
    """One scheduled pass. Quiet (and free) when nothing is due."""
    from nflprops import report as RP, schedule as SC
    from nflprops.db import connect, team_abbr_map
    from nflprops.slack import post

    ev = RP.upcoming_events()
    if ev.empty:
        print("auto: no pre-game events"); return
    now = pd.Timestamp.now(tz="UTC")
    abbr = team_abbr_map()
    st = load_state()

    # Resolve season/week/abbr per event once. A game with a report file, or
    # one built by an earlier auto pass, is handled and is never refetched.
    meta, handled = {}, set()
    for e in ev.itertuples():
        a, h = abbr.get(e.away_team), abbr.get(e.home_team)
        sw = game_week(connect, e, a, h) if a and h else None
        if not sw:
            print(f"  ! cannot place {e.away_team} @ {e.home_team}"); continue
        meta[e.id] = (a, h, *sw)
        if RP.report_path(sw[0], sw[1], a, h).exists() or e.id in st["games"]:
            handled.add(e.id)

    due = SC.plan(ev, now, set(st["posted"]), handled)
    if not due:
        print("auto: nothing due"); return
    rem, wrote = None, False
    for key, sit, todo in due:
        for e in todo.itertuples():
            if e.id not in meta:
                continue
            a, h, season, week = meta[e.id]
            item, rem = build_game(RP, e, season, week, a, h, N_PICKS, DRY)
            if item:
                st["games"][e.id] = item
                wrote = True
        ready = [st["games"][i] for i in sit.id if i in st["games"]]
        if SC.should_post(len(sit), len(ready), sit.ct.min(), now):
            season, week = next((meta[i][2:] for i in sit.id if i in meta), (None, None))
            title = f"Production ranges — {season} W{week} · {sit.et.min():%a %-I:%M %p} ET".upper()
            if DRY:
                print(f"[dry] would post sitting {key}: {len(ready)} game(s)")
            else:
                print(post(blocks=digest_blocks(ready, title, rem),
                           text=f"Production ranges — W{week} {key}"))
                st["posted"][key] = now.isoformat()
        else:
            print(f"sitting {key}: {len(ready)}/{len(sit)} ready, waiting")
        if not DRY:
            save_state(st)
    if wrote and not DRY:
        from nflprops.rating import write_rankings
        print(f"rankings -> {write_rankings().name}")


def backfill(event_id, asof, N_PICKS):
    """Report from a historical pre-kickoff odds snapshot. Never posts to Slack.
    The page carries a banner saying it was built after the game."""
    import types, urllib.request
    from nflprops import report as RP
    from nflprops.config import require, save_board
    from nflprops.db import connect, team_abbr_map
    k = require("ODDS_API_KEY")
    u = (f"https://api.the-odds-api.com/v4/historical/sports/americanfootball_nfl/events/{event_id}/odds"
         f"?apiKey={k}&date={asof}&bookmakers=fanduel&markets={','.join(RP.ALT)}&oddsFormat=american")
    with urllib.request.urlopen(u, timeout=30) as r:
        snap = json.load(r)
        rem, last = r.headers.get("x-requests-remaining"), r.headers.get("x-requests-last")
    d = snap["data"]
    print(f"snapshot {snap.get('timestamp')} (asked {asof}); cost {last}, remaining {rem}")
    ct = pd.Timestamp(d["commence_time"])
    if pd.Timestamp(snap["timestamp"]) >= ct:
        sys.exit("snapshot is not pre-kickoff; no report")
    e = types.SimpleNamespace(id=event_id, away_team=d["away_team"], home_team=d["home_team"],
                              et=ct.tz_convert("America/New_York"))
    abbr = team_abbr_map()
    a, h = abbr[d["away_team"]], abbr[d["home_team"]]
    season, week = game_week(connect, e, a, h)
    save_board(season, week, [a, h], d)
    banner = ('<div style="background:#fff3cd;color:#664d03;padding:10px 16px;font:14px sans-serif">'
              f'Built after the game from FanDuel lines taken at {snap["timestamp"]}, before kickoff. '
              'Not posted to Slack. Kept for complete data.</div>')
    item, _ = build_game(RP, e, season, week, a, h, N_PICKS, False,
                         lines=RP.parse_lines(d), rem=rem, banner=banner)
    print("backfill done" if item else "backfill: no report")


def main():
    from nflprops import report as RP
    from nflprops.db import connect, team_abbr_map
    from nflprops.slack import post

    SLATE = (sys.argv[1] if len(sys.argv) > 1 else "sunday").lower()
    def arg(f, d):
        return type(d)(sys.argv[sys.argv.index(f) + 1]) if f in sys.argv else d
    N_PICKS, DRY = arg("--picks", 6), "--dry" in sys.argv
    if SLATE == "auto":
        return auto(N_PICKS, DRY)
    if SLATE == "backfill":
        return backfill(sys.argv[2], sys.argv[3], N_PICKS)

    season, week = current_week(connect)
    ev = RP.upcoming_events()
    if ev.empty:
        sys.exit("no pre-game events on the board")
    games = pick_games(ev, SLATE)
    if games.empty:
        sys.exit(f"no {SLATE.upper()} game found")

    abbr = team_abbr_map()

    print(f"{SLATE.upper()}: {len(games)} game(s), {season} W{week}")
    if not DRY:
        from nflprops.rating import write_rankings
        print(f"rankings -> {write_rankings().name}")  # refreshed with every report build
    items, rem = [], None
    for e in games.itertuples():
        a, h = abbr.get(e.away_team), abbr.get(e.home_team)
        if not a or not h:
            print(f"  ! unknown team abbr for {e.away_team} @ {e.home_team}"); continue
        item, rem = build_game(RP, e, season, week, a, h, N_PICKS, DRY)
        if item:
            items.append(item)

    if not items:
        sys.exit("nothing to post")
    blocks = digest_blocks(items, f"Production ranges — {season} W{week} · {SLATE.upper()}", rem)

    if DRY:
        print(f"\n[dry] would post {len(blocks)} blocks and write {len(items)} reports; nothing written")
    else:
        print(post(blocks=blocks, text=f"Production ranges — W{week} {SLATE}"))
        print(f"posted; credits remaining {rem}")


if __name__ == "__main__":
    main()
