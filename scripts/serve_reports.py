#!/usr/bin/env python3
"""Static server for the generated reports, newest first.

Bound to 127.0.0.1 only. Tailscale Serve puts it on the tailnet, so the
listening socket itself never faces the network. Tailnet-only, not Funnel:
these pages name real players and real prices under Sean's own account, and
nothing about them needs the public internet.
"""
import json
import os
import re
import sqlite3
import sys
from datetime import datetime
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

# The book writes "Chris Godwin Jr." where nflverse has "Chris Godwin", so
# joining a pick to what the player actually did needs the project's own
# name rule. Importing it beats copying it: a second copy of that rule would
# drift, and a name that fails to join is a pick silently graded as unplayed.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from nflprops.odds import name_key
except Exception as _e:  # noqa: BLE001 — the pages still work ungraded
    sys.stderr.write(f"grading unavailable, name_key did not import: {_e}\n")
    name_key = None

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.join(BASE, "reports")
DB = os.path.join(BASE, "data", "nflprops.db")
MAP = os.path.join(BASE, "docs", "nfl-props-frontend-map.json")
INSPECTOR = os.path.join(BASE, "scripts", "inspector.js")
PORT = int(os.environ.get("NFLPROPS_WEB_PORT", "8792"))

# report_2026_w2_DET-BUF.html -> ("2026", "2", "DET @ BUF")
NAME = re.compile(r"report_(\d{4})_w(\d+)_([A-Z]+)-([A-Z]+)\.html$")

HEAD = """<!doctype html><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>NFL Props \u2014 reports</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@500;600;700&family=IBM+Plex+Mono:wght@400;500;600&family=Public+Sans:wght@400;500;600&display=swap">
<style>
  /* Same tokens, same type, same rules as templates/report.html. A listing
     that looked like a different product made the reports feel like someone
     else's page. Keep these in step with the template if that palette moves. */
  :root{{
    color-scheme: light;
    --bg:#f7f6f3; --surface:#fcfcfb; --line:#e2e0d9; --line-soft:#eceae3;
    --ink:#14140f; --ink-2:#52514e; --ink-3:#84827a;
    --accent:#eb6834;
    /* Results get their own two colours rather than the accent. The accent is
       orange and already means "this is the interesting one"; a result needs
       to read as landed or not, which is a different question. The red is a
       deep crimson rather than a red-orange, so it cannot be mistaken for the
       accent sitting a few pixels away. */
    --good:#1c7a4d; --bad:#b3261e;
  }}
  @media (prefers-color-scheme: dark){{
    :root:not([data-theme="light"]){{
      color-scheme: dark;
      --bg:#121211; --surface:#1a1a19; --line:#33322e; --line-soft:#262521;
      --ink:#f6f5ef; --ink-2:#c3c2b7; --ink-3:#8b897f;
      --accent:#d95926; --good:#4fbd85; --bad:#f2685c;
    }}
  }}
  :root[data-theme="dark"]{{
    color-scheme: dark;
    --bg:#121211; --surface:#1a1a19; --line:#33322e; --line-soft:#262521;
    --ink:#f6f5ef; --ink-2:#c3c2b7; --ink-3:#8b897f;
    --accent:#d95926; --good:#4fbd85; --bad:#f2685c;
  }}
  *{{box-sizing:border-box}}
  body{{
    margin:0; background:var(--bg); color:var(--ink);
    font-family:"Public Sans",ui-sans-serif,system-ui,sans-serif;
    font-size:15px; line-height:1.55; -webkit-font-smoothing:antialiased;
  }}
  .wrap{{max-width:720px; margin:0 auto; padding:40px 24px 72px}}
  header{{border-bottom:2px solid var(--ink); padding-bottom:18px; margin-bottom:8px}}
  .eyebrow{{
    font-family:"IBM Plex Mono",ui-monospace,monospace; font-size:11px;
    letter-spacing:.14em; text-transform:uppercase; color:var(--ink-3); margin:0 0 10px;
  }}
  h1{{
    font-family:"Barlow Condensed",Impact,sans-serif; font-weight:700;
    font-size:clamp(34px,6vw,56px); line-height:.95; letter-spacing:-.01em;
    margin:0; text-transform:uppercase;
  }}
  .sub{{color:var(--ink-2); margin:14px 0 0; max-width:64ch; font-size:14.5px}}
"""

# The listing's own furniture: week tabs, kickoff blocks, game cards. The
# slate page shares HEAD above and brings its own rules instead of these.
PAGE = HEAD + """
  /* The tab strip borrows the report's h2 voice -- condensed uppercase label,
     quiet monospace count -- and the header's 2px ink rule for the active one,
     so it reads as the same furniture rather than a widget bolted on. */
  .tabs{{
    display:flex; flex-wrap:wrap; gap:2px; margin:30px 0 0;
    border-bottom:1px solid var(--line);
  }}
  .tab{{
    appearance:none; background:none; border:0; cursor:pointer;
    border-bottom:2px solid transparent; padding:9px 14px 7px; margin-bottom:-1px;
    color:var(--ink-3); font-family:"Barlow Condensed",Impact,sans-serif;
    font-weight:600; font-size:20px; text-transform:uppercase; letter-spacing:.04em;
    line-height:1; display:flex; align-items:baseline; gap:7px;
  }}
  .tab:hover{{color:var(--ink-2)}}
  .tab[aria-selected="true"]{{color:var(--ink); border-bottom-color:var(--ink)}}
  .tab:focus-visible{{outline:2px solid var(--accent); outline-offset:-2px}}
  .tab .n{{
    font-family:"IBM Plex Mono",ui-monospace,monospace; font-size:10.5px;
    font-weight:400; letter-spacing:.1em; color:var(--ink-3);
    font-variant-numeric:tabular-nums;
  }}
  .panel[hidden]{{display:none}}
  /* A game card holds one thing: who is playing. The when lives in the
     heading above it, once for the whole block. Keep the frame in step with
     .card in templates/report.html — same border, same radius, same padding. */
  /* Four to a row. The content column is 672px wide, so a 148px minimum
     leaves room for four cards and their three gaps and not a fifth. The
     minimum is what does the work, not a fixed count: the same rule falls to
     two cards on a phone without a second rule to keep in step. */
  .rows{{
    display:grid; grid-template-columns:repeat(auto-fill,minmax(148px,1fr));
    gap:14px; margin-top:18px;
  }}
  /* One row, allowed to wrap. The matchup and its tally share the card's
     single line; the "No report" fine print takes a full basis below, so it
     drops to its own line without the card needing a second layout.
     Centred, not baseline-aligned. Baseline alignment lines up the text
     INSIDE the pill with the name's baseline, which leaves the pill's
     padding and background hanging below it. Both items are given a known
     height below so centring them is exact rather than approximate. */
  .card{{
    background:var(--surface); border:1px solid var(--line); border-radius:3px;
    padding:11px 12px 10px; display:flex; flex-wrap:wrap; align-items:center;
    gap:6px; text-decoration:none; color:inherit;
  }}
  a.card:hover{{border-color:var(--ink-3)}}
  a.card:focus-visible{{outline:2px solid var(--accent); outline-offset:2px}}
  /* line-height 1 makes the name's box its own cap height, near enough, so
     there is no invisible leading above and below to throw the centring off.
     20px is the row height both items are centred within. */
  .name{{
    font-family:"Barlow Condensed",sans-serif; font-weight:600; font-size:21px;
    letter-spacing:.01em; line-height:20px; min-width:0;
  }}
  /* A game with no report yet. Dimmed rather than hidden: the week is 16
     games whether or not this machine has got to all of them. No hover, no
     pointer, no focus ring — nothing about it should suggest a click. */
  .card.none{{
    background:none; border-style:dashed; border-color:var(--line-soft);
    cursor:default;
  }}
  .card.none .name{{color:var(--ink-3); font-weight:500}}
  /* "N/A" sits in the slot a tally will occupy once the game is reported and
     played. Same box, same place, so a week of cards keeps one column down
     its right edge whatever state each game is in. Hollow rather than filled:
     a coloured pill would read as a result, and this is the absence of one. */
  .score.na{{
    background:none; color:var(--ink-3);
    border:1px dashed var(--line); font-weight:400;
  }}
  /* How the picks for a finished game did, in the gap to the right of the
     matchup. Just the fraction: on a card that says DEN @ KC, "5/6" can only
     mean one thing, and the word "hit" was the part that stopped it fitting.
     margin-left:auto pins it to the right edge, so a column of cards lines
     its tallies up whatever the team names are. A filled pill rather than
     coloured text — at this size text alone read as a footnote. */
  .score{{
    margin-left:auto; display:inline-flex; align-items:center;
    height:18px; padding:0 6px; border-radius:2px;
    font-family:"IBM Plex Mono",monospace; font-size:10px; font-weight:600;
    letter-spacing:.04em; line-height:1; white-space:nowrap;
    font-variant-numeric:tabular-nums; color:var(--surface);
  }}
  .score.good{{background:var(--good)}}
  .score.bad{{background:var(--bad)}}
  /* Each kickoff gets its own labelled block, and the label says the whole
     when: day chip, clock time, date. Quieter than a week tab, louder than a
     card — a thin rule, condensed caps, the same monospace the tabs use. */
  .slot{{margin-top:26px}}
  .slot:first-child{{margin-top:18px}}
  /* Centred, not baseline-aligned, matching .win-h on the slate. The day
     chip is a box; the words beside it are text. Lining the box up by the
     text inside it leaves the chip sitting low, and the taller the heading
     the further it drifts. One row height here as a length, a height on the
     chip, and the row centres them exactly. */
  .slot-h{{
    display:flex; align-items:center; gap:9px;
    margin:0; padding-bottom:5px; border-bottom:1px solid var(--line-soft);
    font-family:"Barlow Condensed",Impact,sans-serif; font-weight:600;
    font-size:16px; line-height:18px; text-transform:uppercase;
    letter-spacing:.06em; color:var(--ink-2);
  }}
  .day{{
    display:inline-flex; align-items:center; height:16px; padding:0 5px;
    border-radius:2px; flex:none; line-height:1;
    font-family:"IBM Plex Mono",monospace; font-size:10px; font-weight:600;
    letter-spacing:.09em; color:var(--surface); background:var(--accent);
  }}
  .slot-time{{
    font-family:"IBM Plex Mono",monospace; font-variant-numeric:tabular-nums;
    font-size:15px; font-weight:500; letter-spacing:-.01em; white-space:nowrap;
    line-height:18px; text-transform:none; color:var(--ink);
  }}
  .slot-time.unknown{{font-size:13px; color:var(--ink-3); font-weight:400}}
  .slot-date{{
    font-family:"IBM Plex Mono",monospace; font-variant-numeric:tabular-nums;
    font-size:11.5px; font-weight:400; letter-spacing:0; line-height:18px;
    text-transform:none; color:var(--ink-2);
  }}
  .slot-h .n{{
    margin-left:auto; line-height:18px;
    font-family:"IBM Plex Mono",ui-monospace,monospace; font-size:10px;
    font-weight:400; letter-spacing:.1em; color:var(--ink-3);
    font-variant-numeric:tabular-nums; text-transform:none;
  }}
  .rows{{margin-top:12px}}
  /* One way through to the slate, sitting under the subtitle where a reader
     has just been told what the page is. Quiet on purpose: the listing's job
     is still the games. */
  a.slate-link{{
    display:inline-block; margin:16px 0 0; text-decoration:none;
    font-family:"IBM Plex Mono",ui-monospace,monospace; font-size:11px;
    letter-spacing:.12em; text-transform:uppercase; color:var(--accent);
    border-bottom:1px solid transparent;
  }}
  a.slate-link:hover{{border-bottom-color:var(--accent)}}
  .empty{{color:var(--ink-3); font-style:italic; margin-top:20px}}
  footer{{margin-top:40px; padding-top:16px; border-top:1px solid var(--line);
    color:var(--ink-3); font-size:12.5px; max-width:68ch}}
  @media (max-width:560px){{
    .wrap{{padding:24px 14px 48px}}
    a.row{{padding:11px 12px; gap:10px}}
    .g{{font-size:18px}}
  }}
</style>
<div class="wrap" data-inspect-id="index-wrap">
  <header>
    <p class="eyebrow" data-inspect-id="index-eyebrow">Production ranges</p>
    <h1 data-inspect-id="index-title">NFL Props</h1>
  </header>
  <p class="sub" data-inspect-id="index-subtitle">Pick a week. Games sit in kickoff order
    inside it. TD marks the likeliest scorer. Nothing here claims an edge.</p>
  <a class="slate-link" href="/slate" data-inspect-id="index-slate-link">Top picks by sitting &rarr;</a>
{tabs}
{rows}
  <footer data-inspect-id="index-footer">Projections, not edges. Tested against real
    sportsbook lines this model was the less accurate of the two.</footer>
</div>"""


TABS_JS = """
<script>
// The newest week's panel -- the last one, on the right -- ships visible and
// the rest ship with `hidden` set,
// so the page is correct before this file has run -- no flash of every week at
// once. The cost is that without JavaScript the older tabs cannot be opened.
// That is a smaller loss than it sounds: a report page is drawn entirely by its
// own script, so with JavaScript off a report is a blank page anyway.
(function () {
  var strip = document.querySelector('[role="tablist"]');
  if (!strip) return;
  var tabs = [].slice.call(strip.querySelectorAll('[role="tab"]'));

  function show(tab) {
    tabs.forEach(function (t) {
      var on = t === tab;
      t.setAttribute("aria-selected", on ? "true" : "false");
      t.tabIndex = on ? 0 : -1;
      document.getElementById(t.getAttribute("aria-controls")).hidden = !on;
    });
  }

  strip.addEventListener("click", function (e) {
    var t = e.target.closest('[role="tab"]');
    if (t) show(t);
  });

  // Left/right move between weeks, home/end jump to the newest or oldest.
  strip.addEventListener("keydown", function (e) {
    var i = tabs.indexOf(document.activeElement);
    if (i < 0) return;
    var to = null;
    if (e.key === "ArrowRight") to = tabs[(i + 1) % tabs.length];
    else if (e.key === "ArrowLeft") to = tabs[(i - 1 + tabs.length) % tabs.length];
    else if (e.key === "Home") to = tabs[0];
    else if (e.key === "End") to = tabs[tabs.length - 1];
    if (!to) return;
    e.preventDefault();
    to.focus();
    show(to);
  });
})();
</script>
"""


def kickoffs():
    """Kickoff time for every scheduled game, keyed season/week/away/home.

    Read from the local schedules table rather than from the report files.
    That costs no API credits, needs no change to a report's markup, and
    works for the reports already on disk — which matters, because the whole
    point is putting today's slate in order right now.

    Returns ET wall-clock strings exactly as nflverse stores them. Never
    raises: a listing that loses its ordering is a nuisance, a listing that
    500s is a broken page.
    """
    out = {}
    try:
        db = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
        for season, week, away, home, day, t in db.execute(
                "SELECT season, week, away_team, home_team, gameday, gametime FROM schedules"):
            if not day or not t:
                continue
            out[(int(season), int(week), away, home)] = f"{day} {t}"
        db.close()
    except Exception as e:  # noqa: BLE001 — see docstring
        sys.stderr.write(f"kickoff lookup unavailable: {e}\n")
    return out


def td_markers():
    """Likeliest touchdown scorer per game, keyed AWAY-HOME.

    Written by scripts/td/write_markers.py, read here rather than recomputed:
    the listing must render in milliseconds and the model takes seconds to
    fit. A missing file is normal, not an error -- it just means no markers
    have been written for that week yet.
    """
    out = {}
    try:
        for f in os.listdir(ROOT):
            if f.startswith("td_markers_") and f.endswith(".json"):
                out.update(json.loads(open(os.path.join(ROOT, f)).read()))
    except Exception as e:  # noqa: BLE001
        sys.stderr.write(f"td markers unavailable: {e}\n")
    return out


def index_html():
    try:
        files = [f for f in os.listdir(ROOT) if f.endswith(".html")]
    except FileNotFoundError:
        files = []
    kicks = kickoffs()
    records = {}

    # The week is the schedule's 16 games, not the 14 that happen to have a
    # report. A missing game used to be invisible, so a half-built week looked
    # like a finished one. Now every game gets a card and the ones with no
    # report yet say so, which turns "is that all of them?" into a question the
    # page answers by itself.
    reports, loose = {}, []
    for f in files:
        m = NAME.search(f)
        if m:
            reports[(int(m.group(1)), int(m.group(2)), m.group(3), m.group(4))] = f
        else:
            loose.append(f)

    # Only weeks that have at least one report. Without that, every future week
    # in the schedules table would show up as a wall of empty cards.
    live = {k[:2] for k in reports}

    entries = []          # (filename or None, label, season, week, kick)
    for (season, week, away, home), kick in kicks.items():
        if (season, week) in live:
            entries.append((reports.get((season, week, away, home)),
                            f"{away} @ {home}", season, week, kick))
    for (season, week, away, home), f in reports.items():
        if (season, week, away, home) not in kicks:
            entries.append((f, f"{away} @ {home}", season, week, None))
    for f in loose:
        entries.append((f, f[:-5], 0, 0, None))

    # Newest week first, and inside a week the games in the order they kick
    # off — the order you actually watch them in. Sorting the whole list by
    # kickoff alone would float last week's early games above this week's.
    # A game with no schedule row sorts last within its week rather than
    # vanishing or jumping to the top. The matchup breaks ties so a shared
    # kickoff lands in a stable order rather than whatever the table returned.
    entries.sort(key=lambda e: (-e[2], -e[3], e[4] or "9999", e[1]))

    groups = []          # [((season, week), [row html, ...]), ...]
    for f, label, season, week, kick in entries:
        # The card carries the matchup and nothing else.
        #
        # Every card in a block starts at the same moment, so a day, a time
        # and a date printed on each one repeats the same fact eight times
        # over. All of it moves up to the block's own heading, where it is
        # said once. What is left on the card is the only thing that differs
        # from its neighbours: who is playing.
        #
        # A game with no report is a card, not a link. Nothing to open, so it
        # is a <div> rather than a disabled <a> — there is no href to follow,
        # no focus stop, and nothing for the keyboard to land on. The fine
        # print under the matchup says why it is grey.
        if f:
            # How the picks for this game actually did, once it has finished.
            # A week is graded on first use and then remembered, so a listing
            # of fifteen games does the join twice rather than fifteen times.
            if (season, week) not in records:
                records[(season, week)] = game_record(season, week)
            got = records[(season, week)].get(label)
            score = ""
            if got:
                hits, n = got
                # Green when at least half landed, red when fewer did. Half
                # counts as green: three of six is the model doing what it
                # said it would, not a bad afternoon.
                tone = "good" if hits * 2 >= n else "bad"
                score = (f'<span class="score {tone}" data-inspect-id="index-hits"'
                         f' title="{hits} of {n} shortlisted picks hit">'
                         f'{hits}/{n}</span>')
            row = (f'<a class="card" data-inspect-id="index-row" href="/{f}">'
                   f'<span class="name" data-inspect-id="index-game-label">{label}</span>'
                   f'{score}</a>')
        else:
            row = (f'<div class="card none" data-inspect-id="index-row-pending"'
                   f' aria-disabled="true">'
                   f'<span class="name" data-inspect-id="index-game-label-pending">{label}</span>'
                   f'<span class="score na" data-inspect-id="index-no-report"'
                   f' title="No report generated for this game yet">N/A</span>'
                   f'</div>')

        # The heading for the run of games that kick off together, in three
        # pieces: a day chip, the clock time, and the calendar date. Entries
        # already arrive in kickoff order inside a week, so games that share a
        # kickoff are already adjacent and a run-length pass is enough.
        if kick:
            when = datetime.strptime(kick, "%Y-%m-%d %H:%M")
            slot = (f"{when:%a}".upper(), f"{when:%-I:%M %p}",
                    f"ET \u00b7 {when:%b %-d}")
        else:
            # No schedule row. Say so rather than showing the file's mtime,
            # which is when the report was written and not when anyone plays.
            slot = ("", "Kickoff unknown", "no schedule row")

        key = (season, week)
        if groups and groups[-1][0] == key:
            slots = groups[-1][1]
        else:
            slots = []
            groups.append((key, slots))
        if slots and slots[-1][0] == slot:
            slots[-1][1].append(row)
        else:
            slots.append((slot, [row]))

    # One tab per week, one panel of cards under it.
    #
    # Two different orders are at work here, on purpose. The strip reads left
    # to right in the order the weeks were played, the way a season reads. The
    # week that opens is the newest one, on the right, because that is the week
    # anyone loading this page came for. `groups` arrives newest-first from the
    # sort above, so reversing it gives the reading order and makes the last
    # tab the selected one.
    #
    # Season only appears in a label when more than one season is on disk.
    # Right now every report is 2026, and "2026 Week 2" next to "2026 Week 1"
    # spends horizontal room saying nothing.
    multi_season = len({g[0][0] for g in groups}) > 1

    def panel_body(slots):
        """One labelled block of cards per kickoff time.

        A Sunday runs 1:00, 4:05, 4:25 and 8:20. Those are four separate
        sittings, and a single undivided grid of fourteen cards hides that.
        The heading carries the whole when — day, time and date — because
        every card beneath it shares that answer exactly.
        """
        out = []
        for (day, time_s, date_s), rws in slots:
            n = len(rws)
            badge = (f'<span class="day" data-inspect-id="index-day-badge">{day}</span>'
                     if day else "")
            unknown = "" if day else " unknown"
            out.append(f'<div class="slot" data-inspect-id="index-slot">'
                       f'<h3 class="slot-h" data-inspect-id="index-slot-label">{badge}'
                       f'<span class="slot-time{unknown}" data-inspect-id="index-kickoff">{time_s}</span>'
                       f'<span class="slot-date" data-inspect-id="index-kickoff-date">{date_s}</span>'
                       f'<span class="n">{n} game{"" if n == 1 else "s"}</span></h3>'
                       f'<div class="rows">{"".join(rws)}</div></div>')
        return "".join(out)
    ordered = list(reversed(groups))
    latest = len(ordered) - 1
    counts = {k: sum(len(r) for _, r in slots) for k, slots in ordered}

    tabs, panels = [], []
    for n, ((season, week), slots) in enumerate(ordered):
        games = counts[(season, week)]
        if not season:
            label, slug = "Unscheduled", "none"
        else:
            label = f"{season} W{week}" if multi_season else f"Week {week}"
            slug = f"{season}-{week}"
        on = n == latest
        tabs.append(
            f'    <button class="tab" role="tab" id="tab-{slug}" type="button"'
            f' aria-controls="panel-{slug}" aria-selected="{"true" if on else "false"}"'
            f' tabindex="{"0" if on else "-1"}" data-inspect-id="index-week-tab">'
            f'{label}<span class="n">{games}</span></button>')
        panels.append(
            f'  <section class="panel" role="tabpanel" id="panel-{slug}"'
            f' aria-labelledby="tab-{slug}"{"" if on else " hidden"}'
            f' data-inspect-id="index-week-panel">'
            f'{panel_body(slots)}</section>')

    if not panels:
        tabs = []
        panels = ['  <p class="empty" data-inspect-id="index-empty">'
                  'No reports generated yet.</p>']

    tabstrip = ""
    if tabs:
        tabstrip = ('  <nav class="tabs" role="tablist" aria-label="Weeks"'
                    ' data-inspect-id="index-tablist">\n'
                    + "\n".join(tabs) + "\n  </nav>")

    return PAGE.format(tabs=tabstrip, rows="\n".join(panels)) + TABS_JS



# ---------------------------------------------------------------- slate page

# The legs a report actually shortlisted ride inside the report file, as the
# same JSON its own script draws from. Reading them back costs nothing: no
# Odds API credit, no re-projection, and no second copy of the picks that
# could drift away from the page it came from.
DATA_BLOB = re.compile(r"const DATA *= *(.*?);\n", re.S)

# Sunday is two sittings, not five kickoffs. 1:00 is the early slate; 4:05 and
# 4:25 are twenty minutes apart and nobody thinks of them as separate
# afternoons. Grouping by part of day says what a viewer means by "the late
# games" and folds Thursday and Monday night into their own sections without a
# special case.
def daypart(hhmm):
    h = int(hhmm[:2])
    if h < 15:
        return 0, "Early"
    if h < 19:
        return 1, "Afternoon"
    return 2, "Night"


_LEG_CACHE = {}


def shortlist(season, week):
    """Every shortlisted leg for one week, with its game and kickoff attached.

    Cached on the set of report files and their modification times, so a
    regenerated report is picked up on the next request and an unchanged week
    is parsed once. Fifteen reports is half a megabyte of HTML; parsing it on
    every page load would be wasteful rather than slow.
    """
    kicks = kickoffs()
    try:
        files = sorted(f for f in os.listdir(ROOT) if f.endswith(".html"))
    except FileNotFoundError:
        return []
    stamp = tuple((f, os.path.getmtime(os.path.join(ROOT, f))) for f in files)
    key = (season, week, stamp)
    if key in _LEG_CACHE:
        return _LEG_CACHE[key]

    legs = []
    for f in files:
        m = NAME.search(f)
        if not m or int(m.group(1)) != season or int(m.group(2)) != week:
            continue
        try:
            html = open(os.path.join(ROOT, f), encoding="utf-8").read()
            rows = json.loads(DATA_BLOB.search(html).group(1))
        except Exception as e:  # noqa: BLE001 — one unreadable report, not a 500
            sys.stderr.write(f"slate: cannot read {f}: {e}\n")
            continue
        away, home = m.group(3), m.group(4)
        kick = kicks.get((season, week, away, home))
        for r in rows:
            # A leg with no price was projected but never shortlisted. It has
            # no line to rank and no odds to compare, so it is not a pick.
            if not r.get("pick") or r.get("price") is None:
                continue
            r["game"] = f"{away} @ {home}"
            r["file"] = f
            r["kick"] = kick
            legs.append(r)

    _LEG_CACHE.clear()
    _LEG_CACHE[key] = legs
    return legs


# Only these four stats are ever shortlisted, and each is a column in
# player_games under the same name. Listing them keeps the query to the
# columns that exist rather than SELECT *.
# Every stat a report can carry a line on. scripts/grade_game.py grades the
# same five, and the two must agree -- a leg the write-up scores and the page
# leaves blank is the page quietly disagreeing with the record.
GRADED_STATS = ("passing_yards", "rushing_yards", "receiving_yards", "receptions",
                "completions")

_RESULT_CACHE = {}


def outcomes(season, week):
    """What each player actually did, for games that have finished.

    The gate is the final score, not the presence of a stat line. A game in
    progress already has rows in player_games, and a receiver with 12 yards at
    half time has not missed an over-29.5 — he is still playing. Grading him
    would print a loss that isn't real and then quietly correct itself later,
    which is worse than showing nothing.

    Returns ({name_key: {stat: value}}, {"AWAY@HOME"}) — the actuals, and the
    set of games that are genuinely over.
    """
    if name_key is None:
        return {}, set()
    try:
        db = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
        done = {f"{a}@{h}" for a, h in db.execute(
            "SELECT away_team, home_team FROM schedules "
            "WHERE season=? AND week=? AND away_score IS NOT NULL", (season, week))}
        acts = {}
        if done:
            cols = ", ".join(GRADED_STATS)
            for row in db.execute(
                    f"SELECT player_display_name, {cols} FROM player_games "
                    "WHERE season=? AND week=? AND season_type='REG'", (season, week)):
                k = name_key(row[0])
                if k:
                    acts[k] = dict(zip(GRADED_STATS, row[1:]))
        db.close()
        return acts, done
    except Exception as e:  # noqa: BLE001 — an ungraded page beats a 500
        sys.stderr.write(f"outcomes unavailable: {e}\n")
        return {}, set()


def grade_legs(season, week, legs):
    """Mark each leg hit or missed, in place, for games that have finished.

    A leg hits when the real number beats the line — the same rule
    scripts/grade_game.py writes into the vault, so the page and the graded
    write-up can never disagree. Legs from a game still being played, or from
    a player with no stat line, are left ungraded rather than guessed at.
    """
    acts, done = outcomes(season, week)
    for r in legs:
        r["hit"] = None
        if r["game"].replace(" @ ", "@") not in done:
            continue
        a = acts.get(name_key(r["p"]), {}).get(r["s"]) if name_key else None
        if a is not None and r.get("line") is not None:
            r["hit"] = bool(a > r["line"])
            r["actual"] = a
    return legs


def game_record(season, week):
    """Hits and graded legs per game, e.g. {"DEN @ KC": (1, 6)}.

    Cached alongside the parsed legs: the join is cheap, but doing it on every
    card of every page load is still work nobody asked for.
    """
    key = (season, week, len(shortlist(season, week)),
           tuple(sorted(outcomes(season, week)[1])))
    if key in _RESULT_CACHE:
        return _RESULT_CACHE[key]
    rec = {}
    for r in grade_legs(season, week, shortlist(season, week)):
        if r["hit"] is None:
            continue
        h, n = rec.get(r["game"], (0, 0))
        rec[r["game"]] = (h + (1 if r["hit"] else 0), n + 1)
    _RESULT_CACHE.clear()
    _RESULT_CACHE[key] = rec
    return rec


STAT_SHORT = {"receiving_yards": "rec yds", "rushing_yards": "rush yds",
              "passing_yards": "pass yds", "receptions": "rec",
              "completions": "comp"}


def payout(price):
    """What $100 wins at this American price, to the dollar.

    A price of -290 is not a quantity anyone has a feel for. "$34" is. The
    tables lead with the price because that is what the book shows, and carry
    the payout because that is what the price means.
    """
    return 100 * 100 // -price if price < 0 else price


def leg_rows(legs, graded):
    out = []
    for i, r in enumerate(legs, 1):
        stat = STAT_SHORT.get(r["s"], r["s"].replace("_", " "))
        gap = (r["mp"] - r["bp"]) * 100
        gap_cls = "gap up" if gap > 0 else "gap down"
        # Did this one land? Ungraded while the game is still being played,
        # so the column is simply empty rather than guessing at a result.
        hit = r.get("hit")
        mark = ("" if hit is None else
                f'<span class="mark {"hit" if hit else "miss"}"'
                f' data-inspect-id="slate-leg-result">'
                f'{r["actual"]:g}</span>')
        out.append(
            f'<tr data-inspect-id="slate-leg">'
            f'<td class="rank">{i}</td>'
            f'<td class="who"><a href="/{r["file"]}">{r["p"]}</a>'
            f'<span class="meta">{r["tm"]} {r["role"]} &middot; {r["game"]}</span></td>'
            f'<td class="stat">o{r["line"]:g} {stat}</td>'
            + (f'<td class="res">{mark}</td>' if graded else "")
            + (
            f'<td class="num price">{r["price"]}<span class="meta">${payout(r["price"])}</span></td>'
            f'<td class="num">{r["mp"] * 100:.1f}%</td>'
            f'<td class="num book">{r["bp"] * 100:.1f}%'
            f'<span class="{gap_cls}">{gap:+.1f}</span></td>'
            f'</tr>'))
    return "".join(out)


# Every table on the slate is cut to the same ruler. Three rankings stacked
# down a sitting are read as one list, and a browser left to its own devices
# sizes each table to its own longest name — so the HIT pills stepped left and
# right from table to table. These widths are the same for all of them, which
# is what makes the columns line up. Player takes whatever is left over.
SLATE_COLS = ('<col class="c-rank"><col><col class="c-stat">'
              '{res}<col class="c-price"><col class="c-pct"><col class="c-book">')


def slate_table(legs, title, note, graded):
    """One ranking, cut to the shared column ruler.

    Whether the Actual column exists is decided for the whole week, not for
    this table, so the three rankings under a sitting always have the same
    columns in the same places. A week nobody has played yet carries no Actual
    heading at all — five empty cells read as something broken rather than as
    something that has not happened yet.
    """
    head = ('<th></th><th>Player</th><th>Leg</th>'
            + ('<th class="res">Actual</th>' if graded else "")
            + '<th class="num">Price</th><th class="num">Model</th>'
              '<th class="num">Book</th>')
    cols = SLATE_COLS.format(res='<col class="c-res">' if graded else "")
    return (f'<div class="tbl" data-inspect-id="slate-table">'
            f'<h4 data-inspect-id="slate-table-title">{title}</h4>'
            f'<p class="note" data-inspect-id="slate-table-note">{note}</p>'
            f'<div class="scroll"><table>{cols}<thead><tr>{head}</tr></thead>'
            f'<tbody>{leg_rows(legs, graded)}</tbody></table></div></div>')


SLATE_PAGE = HEAD.replace("NFL Props — reports", "NFL Props — slate") + """
  /* The slate is the listing's other half: same tokens, same three typefaces,
     but rows of legs instead of a grid of games. The window heading reuses the
     listing's .slot-h anatomy so the two pages divide the day the same way. */
  /* The week strip is the listing's, rule for rule — same condensed caps,
     same quiet monospace count, same 2px ink underline on the open week. The
     two pages divide the season identically, so they look identical doing it.
     Keep these in step with .tabs in PAGE above. */
  .tabs{{
    display:flex; flex-wrap:wrap; gap:2px; margin:26px 0 0;
    border-bottom:1px solid var(--line);
  }}
  .tab{{
    appearance:none; background:none; border:0; cursor:pointer;
    border-bottom:2px solid transparent; padding:9px 14px 7px; margin-bottom:-1px;
    color:var(--ink-3); font-family:"Barlow Condensed",Impact,sans-serif;
    font-weight:600; font-size:20px; text-transform:uppercase; letter-spacing:.04em;
    line-height:1; display:flex; align-items:baseline; gap:7px;
  }}
  .tab:hover{{color:var(--ink-2)}}
  .tab[aria-selected="true"]{{color:var(--ink); border-bottom-color:var(--ink)}}
  .tab:focus-visible{{outline:2px solid var(--accent); outline-offset:-2px}}
  .tab .n{{
    font-family:"IBM Plex Mono",ui-monospace,monospace; font-size:10.5px;
    font-weight:400; letter-spacing:.1em; color:var(--ink-3);
    font-variant-numeric:tabular-nums;
  }}
  .panel[hidden]{{display:none}}
  a.back{{
    display:inline-block; margin:22px 0 0; text-decoration:none;
    font-family:"IBM Plex Mono",ui-monospace,monospace; font-size:11px;
    letter-spacing:.12em; text-transform:uppercase; color:var(--ink-3);
  }}
  a.back:hover{{color:var(--accent)}}
  .win{{margin-top:34px}}
  /* Five items of five different sizes on one line, two of them padded
     pills. Centred, not baseline-aligned — a baseline is a property of text,
     and a pill is a box. Lining a box up by the text inside it puts the box
     wherever the font's metrics happen to land, which is what left the day
     chip and the tally sitting low against the heading.
     One row height, set here as a length so no invisible leading creeps in,
     and every pill below carries a height of its own. */
  .win-h{{
    display:flex; align-items:center; gap:9px; flex-wrap:wrap;
    margin:0; padding-bottom:5px; border-bottom:2px solid var(--ink);
    font-family:"Barlow Condensed",Impact,sans-serif; font-weight:600;
    font-size:22px; line-height:22px; text-transform:uppercase;
    letter-spacing:.04em; color:var(--ink);
  }}
  .day{{
    display:inline-flex; align-items:center; height:18px; padding:0 6px;
    border-radius:2px; flex:none; line-height:1;
    font-family:"IBM Plex Mono",monospace; font-size:10px; font-weight:600;
    letter-spacing:.09em; color:var(--surface); background:var(--accent);
  }}
  .win-times{{
    font-family:"IBM Plex Mono",monospace; font-variant-numeric:tabular-nums;
    font-size:14px; font-weight:500; letter-spacing:-.01em; line-height:22px;
    text-transform:none; color:var(--ink-2);
  }}
  .win-h .n{{
    margin-left:auto; line-height:22px;
    font-family:"IBM Plex Mono",ui-monospace,monospace; font-size:10px;
    font-weight:400; letter-spacing:.1em; color:var(--ink-3);
    font-variant-numeric:tabular-nums; text-transform:none;
  }}
  .tbl{{margin-top:24px}}
  .tbl h4{{
    margin:0; font-family:"Barlow Condensed",Impact,sans-serif; font-weight:600;
    font-size:16px; text-transform:uppercase; letter-spacing:.06em; color:var(--ink-2);
  }}
  .note{{margin:2px 0 8px; font-size:12.5px; color:var(--ink-3); max-width:64ch}}
  .scroll{{overflow-x:auto}}
  /* Fixed layout means the widths below are obeyed instead of being treated
     as a suggestion the longest cell can overrule. Below 600px the sitting
     scrolls sideways as one piece rather than each table shrinking its own
     way. */
  table{{
    border-collapse:collapse; width:100%; min-width:600px;
    table-layout:fixed; font-size:14px;
  }}
  .c-rank{{width:30px}}
  .c-stat{{width:132px}}
  .c-res{{width:62px}}
  .c-price{{width:80px}}
  .c-pct{{width:64px}}
  .c-book{{width:76px}}
  th{{
    text-align:left; padding:0 8px 5px 0; border-bottom:1px solid var(--line);
    font-family:"IBM Plex Mono",ui-monospace,monospace; font-size:10px;
    font-weight:400; letter-spacing:.1em; text-transform:uppercase; color:var(--ink-3);
    white-space:nowrap;
  }}
  td{{padding:8px 8px 8px 0; border-bottom:1px solid var(--line-soft); vertical-align:top}}
  tr:last-child td{{border-bottom:0}}
  .rank{{
    font-family:"IBM Plex Mono",monospace; font-size:12px; color:var(--ink-3);
    width:1.6em; font-variant-numeric:tabular-nums;
  }}
  .who{{overflow:hidden}}
  .who a{{
    display:block; overflow:hidden; text-overflow:ellipsis; white-space:nowrap;
    font-family:"Barlow Condensed",sans-serif; font-weight:600; font-size:18px;
    line-height:1.1; color:inherit; text-decoration:none; letter-spacing:.01em;
  }}
  .who a:hover{{color:var(--accent)}}
  .meta{{
    display:block; overflow:hidden; text-overflow:ellipsis; white-space:nowrap;
    font-family:"IBM Plex Mono",monospace; font-size:10.5px;
    color:var(--ink-3); letter-spacing:.02em; margin-top:1px; font-weight:400;
  }}
  .stat{{
    font-family:"IBM Plex Mono",monospace; font-size:12.5px; color:var(--ink-2);
    white-space:nowrap;
  }}
  .num{{
    text-align:right; font-family:"IBM Plex Mono",monospace;
    font-variant-numeric:tabular-nums; white-space:nowrap;
  }}
  td.num{{font-size:14.5px; font-weight:500}}
  td.num .meta{{text-align:right}}
  .gap{{
    display:block; font-family:"IBM Plex Mono",monospace; font-size:10.5px;
    font-weight:400; margin-top:1px;
  }}
  .gap.up{{color:var(--accent)}}
  .gap.down{{color:var(--ink-3)}}
  /* Whether a leg landed, under its rank number. A filled pill in the same
     two grounds the listing uses, so one glance down the rank column reads as
     a column of results. HIT and MISS are spelled out rather than shown as a
     colour, so the meaning survives a print-out or a reader who cannot
     separate green from grey. */
  /* The result sits in its own column between the leg and its price: what
     was picked, then whether it landed, then what it paid. Under the rank
     number it was easy to miss — a column is a place the eye already goes. */
  .res{{white-space:nowrap; padding-right:14px}}
  .mark{{
    display:inline-block; width:38px; padding:3px 0; border-radius:2px;
    text-align:center; font-family:"IBM Plex Mono",monospace; font-size:11px;
    font-weight:600; font-variant-numeric:tabular-nums; color:var(--surface);
  }}
  .mark.hit{{background:var(--good)}}
  .mark.miss{{background:var(--bad)}}
  /* The sitting's own tally, on its heading. The same pill as a game card on
     the listing, one size up because the heading around it is bigger, and
     with a height of its own so the row can centre it exactly. */
  .score{{
    display:inline-flex; align-items:center; height:20px; padding:0 8px;
    border-radius:2px; line-height:1; white-space:nowrap;
    font-family:"IBM Plex Mono",monospace; font-size:11px; font-weight:600;
    letter-spacing:.08em; text-transform:uppercase;
    font-variant-numeric:tabular-nums; color:var(--surface);
  }}
  .score.good{{background:var(--good)}}
  .score.bad{{background:var(--bad)}}
  .empty{{color:var(--ink-3); font-style:italic; margin-top:20px}}
  footer{{margin-top:44px; padding-top:16px; border-top:1px solid var(--line);
    color:var(--ink-3); font-size:12.5px; max-width:68ch}}
  @media (max-width:560px){{
    .wrap{{padding:24px 14px 48px}}
    .win-h{{font-size:19px}}
  }}
</style>
<div class="wrap" data-inspect-id="slate-wrap">
  <header>
    <p class="eyebrow" data-inspect-id="slate-eyebrow">Consolidated picks</p>
    <h1 data-inspect-id="slate-title">The Slate</h1>
  </header>
  <p class="sub" data-inspect-id="slate-subtitle">Every shortlisted leg in the week,
    pooled by sitting and ranked three ways. Model is this project's own chance the
    leg lands. Book is what FanDuel's price implies. The small figure under Book is
    the difference, and it is the only column that says anything the book doesn't.</p>
  <a class="back" href="/" data-inspect-id="slate-back">&larr; All reports</a>
{tabs}
{windows}
  <footer data-inspect-id="slate-footer">Projections, not edges. Prices are whatever
    the report held when it was generated and move afterwards — check the live board.
    Tested against real sportsbook lines this model was the less accurate of the two.</footer>
</div>"""


def slate_windows(season, week):
    """One section per sitting for one week, or None if that week has none."""
    legs = grade_legs(season, week, [r for r in shortlist(season, week) if r["kick"]])

    # Pool by sitting: one section per day and part of day, in the order they
    # are played. Legs whose game has no schedule row have no sitting to join
    # and are left out rather than grouped under a heading that would be a lie.
    wins = {}
    for r in legs:
        date_s, time_s = r["kick"].split(" ")
        rank, part = daypart(time_s)
        wins.setdefault((date_s, rank, part), []).append(r)

    # One decision for the week, not one per table. Sunday afternoon being
    # graded while Monday night has not kicked off yet still leaves both
    # sittings with the same columns in the same places down the page.
    graded_week = any(r["hit"] is not None for r in legs)

    out = []
    for (date_s, _rank, part), rs in sorted(wins.items()):
        when = datetime.strptime(date_s, "%Y-%m-%d")
        # "1:00, 4:25 PM ET" rather than "1:00 PM, 4:25 PM ET" — the meridiem
        # is said once when every kickoff shares it. A London morning game in
        # the same sitting as an afternoon one would not, so that case keeps
        # AM and PM on each time instead of stamping the last one's onto all.
        times = sorted({r["kick"].split(" ")[1] for r in rs})
        clock = [datetime.strptime(t, "%H:%M") for t in times]
        if len({f"{t:%p}" for t in clock}) == 1:
            times_s = ", ".join(f"{t:%-I:%M}" for t in clock) + f" {clock[0]:%p} ET"
        else:
            times_s = ", ".join(f"{t:%-I:%M %p}" for t in clock) + " ET"
        games = len({r["game"] for r in rs})

        # How the sitting as a whole did, once its games have finished. A
        # single figure on the heading answers "was that a good afternoon"
        # without reading fifteen rows to work it out.
        graded = [r for r in rs if r["hit"] is not None]
        tally = ""
        if graded:
            hits = sum(1 for r in graded if r["hit"])
            tone = "good" if hits * 2 >= len(graded) else "bad"
            tally = (f'<span class="score {tone}" data-inspect-id="slate-window-hits">'
                     f'{hits}/{len(graded)} hit</span>')

        tables = (
            slate_table(sorted(rs, key=lambda r: -r["mp"])[:5],
                        "Most likely to happen",
                        "Ranked by the model's own chance the leg lands.", graded_week)
            + slate_table(sorted(rs, key=lambda r: r["price"])[:5],
                          "Shortest odds",
                          "The heaviest favourites, and so the smallest payouts.",
                          graded_week)
            + slate_table(sorted(rs, key=lambda r: -r["price"])[:5],
                          "Longest odds",
                          "The best payouts on the shortlist. Still favourites, all of them.",
                          graded_week))
        out.append(
            f'  <section class="win" data-inspect-id="slate-window">'
            f'<h3 class="win-h" data-inspect-id="slate-window-label">'
            f'<span class="day" data-inspect-id="slate-day-badge">{when:%a}</span>'
            f'{part}'
            f'<span class="win-times" data-inspect-id="slate-window-times">{times_s}</span>'
            f'{tally}'
            f'<span class="n">{games} game{"" if games == 1 else "s"}, '
            f'{len(rs)} legs</span></h3>{tables}</section>')
    return out


def slate_html():
    """Every week's shortlisted legs, pooled by sitting and ranked three ways.

    A report answers one game. Nobody watches one game. This answers the
    question a report cannot: out of everything on the board at four o'clock,
    which five are the likeliest, which five pay least, and which five pay most.

    One tab per week, newest selected, exactly as the listing does it — the
    two pages divide the season the same way and share the same tab script.
    """
    weeks = set()
    try:
        for f in os.listdir(ROOT):
            m = NAME.search(f)
            if m:
                weeks.add((int(m.group(1)), int(m.group(2))))
    except FileNotFoundError:
        pass
    if not weeks:
        return SLATE_PAGE.format(tabs="", windows='  <p class="empty" '
                                 'data-inspect-id="slate-empty">'
                                 'No reports generated yet.</p>')

    # Oldest first, so the strip reads left to right in the order the weeks
    # were played and the newest week is the last tab — the one that opens.
    ordered = sorted(weeks)
    multi_season = len({s for s, _ in ordered}) > 1

    tabs, panels = [], []
    for n, (season, week) in enumerate(ordered):
        wins = slate_windows(season, week)
        if not wins:
            wins = ['  <p class="empty" data-inspect-id="slate-empty-unscheduled">'
                    'No shortlisted legs with a scheduled kickoff.</p>']
        legs = len([x for x in shortlist(season, week) if x["kick"]])
        label = f"{season} W{week}" if multi_season else f"Week {week}"
        slug = f"{season}-{week}"
        on = n == len(ordered) - 1
        tabs.append(
            f'    <button class="tab" role="tab" id="tab-{slug}" type="button"'
            f' aria-controls="panel-{slug}" aria-selected="{"true" if on else "false"}"'
            f' tabindex="{"0" if on else "-1"}" data-inspect-id="slate-week-tab">'
            f'{label}<span class="n">{legs}</span></button>')
        panels.append(
            f'  <section class="panel" role="tabpanel" id="panel-{slug}"'
            f' aria-labelledby="tab-{slug}"{"" if on else " hidden"}'
            f' data-inspect-id="slate-week-panel">'
            f'{"".join(wins)}</section>')

    tabstrip = ('  <nav class="tabs" role="tablist" aria-label="Weeks"'
                ' data-inspect-id="slate-tablist">\n'
                + "\n".join(tabs) + "\n  </nav>")
    return SLATE_PAGE.format(tabs=tabstrip, windows="\n".join(panels))

# ------------------------------------------------------------- theme toggle

# Three states, cycled in this order, because "auto" is a real answer and not
# an absence of one: a laptop that goes dark at sunset should take the page
# with it unless someone has said otherwise.
#
# The first script runs before the browser has painted anything. That is the
# whole reason it is inlined at the top of the document rather than loaded as
# a file: a stored choice applied after first paint shows the wrong theme for
# a frame, and on a dark-mode phone that frame is a white flash.
#
# Every read and write of localStorage is wrapped. A private window, cleared
# site data, or a browser set to block storage makes the accessor itself
# throw, and a theme button is not worth a blank page.
THEME = """<script>
(function () {
  var r = document.documentElement;
  function stored() {
    try { return localStorage.getItem("nflprops-theme"); } catch (e) { return null; }
  }
  var t = stored();
  if (t === "light" || t === "dark") r.setAttribute("data-theme", t);

  function label() {
    return (r.getAttribute("data-theme") || "auto").toUpperCase();
  }
  function paint(b) {
    b.textContent = label();
    b.setAttribute("aria-label", "Theme: " + label().toLowerCase() +
                   ". Click to change.");
  }
  function build() {
    var b = document.createElement("button");
    b.id = "nflt-btn";
    b.type = "button";
    paint(b);
    b.addEventListener("click", function () {
      // auto -> light -> dark -> auto. Leaving the attribute off is what
      // hands the page back to the operating system's own setting.
      var now = r.getAttribute("data-theme");
      var next = now === "light" ? "dark" : (now === "dark" ? null : "light");
      if (next) r.setAttribute("data-theme", next);
      else r.removeAttribute("data-theme");
      try {
        if (next) localStorage.setItem("nflprops-theme", next);
        else localStorage.removeItem("nflprops-theme");
      } catch (e) { /* the page still works, the choice just will not stick */ }
      paint(b);
    });
    document.body.appendChild(b);
  }
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", build);
  } else {
    build();
  }
})();
</script>
<style>
  /* Bottom left, because the inspector's crosshair already owns bottom
     right. Same mono small-caps and sharp 3px corner as everything else, so
     it reads as part of the page rather than a browser control. */
  #nflt-btn{
    position:fixed; left:14px; bottom:14px; z-index:2147483000;
    appearance:none; cursor:pointer; padding:7px 11px; border-radius:3px;
    border:1px solid var(--line); background:var(--surface); color:var(--ink-2);
    font-family:"IBM Plex Mono",ui-monospace,monospace; font-size:10px;
    font-weight:600; letter-spacing:.12em;
  }
  #nflt-btn:hover{color:var(--ink); border-color:var(--ink-3)}
  /* A report page names its accent --den, after the away team, so --accent
     is undefined there. An undefined variable invalidates the whole
     declaration rather than part of it, which would have left the button
     with no focus ring at all on every report. The literal is the fallback. */
  #nflt-btn:focus-visible{
    outline:2px solid var(--accent, #eb6834); outline-offset:2px;
  }
</style>
"""

def data_version():
    """A short string that changes when there is something new to see.

    Deliberately NOT the database file's timestamp. The refresh job rewrites
    this season's tables every thirty minutes whether or not a game finished,
    so a timestamp would send every open page reloading all day for nothing.
    These three counts only move when a game goes final, when new stat lines
    land, or when a report is written.
    """
    done = lines = -1
    try:
        db = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
        done = db.execute("SELECT COUNT(*) FROM schedules "
                          "WHERE away_score IS NOT NULL").fetchone()[0]
        lines = db.execute("SELECT COUNT(*) FROM player_games").fetchone()[0]
        db.close()
    except Exception as e:  # noqa: BLE001 — a page that cannot poll still renders
        sys.stderr.write(f"data-version unavailable: {e}\n")
    # Counting the files is not enough. run_reports.py rewrites a report under
    # the same name when prices move, so the count stays put while the page a
    # reader is looking at goes out of date. The newest write time moves on
    # every rewrite, which is the event an open tab needs to hear about.
    reports, newest = -1, 0
    try:
        names = [f for f in os.listdir(ROOT) if NAME.search(f)]
        reports = len(names)
        newest = int(max((os.path.getmtime(os.path.join(ROOT, f))
                          for f in names), default=0))
    except OSError:
        pass
    return f"{done}.{lines}.{reports}.{newest}"


LIVE = """<script>
(function () {
  // Watch for new results and reload when they land.
  //
  // The page reads finished games straight out of the database on every
  // request, so a reload is all it takes to show a score that arrived a
  // minute ago. What the browser lacks is any way to know. So it asks.
  //
  // Nothing reloads unless the answer actually changed. A tab left open on
  // Sunday afternoon is quiet until a game ends, then catches up by itself.
  var seen = null, waiting = false;
  function look() {
    fetch("/_data-version", { cache: "no-store" })
      .then(function (r) { return r.text(); })
      .then(function (v) {
        v = v.trim();
        if (seen === null) { seen = v; return; }
        if (v === seen) return;
        // Never yank the page out from under someone reading it. A hidden tab
        // is told to catch up the moment it comes back to the front.
        if (document.hidden) { waiting = true; return; }
        location.reload();
      })
      .catch(function () { /* server restarting; ask again next time */ });
  }
  document.addEventListener("visibilitychange", function () {
    if (waiting && !document.hidden) location.reload();
  });
  look();
  setInterval(look, 45000);
})();
</script>
"""

# For a report old enough that the template cannot rebuild it. Carries its own
# rule, because that page's stylesheet was written before a.back existed.
BACK_LINK = ('<style>a.back{display:inline-block; margin:0 0 18px;'
             'text-decoration:none; font-family:"IBM Plex Mono",ui-monospace,'
             'monospace; font-size:11px; letter-spacing:.12em;'
             'text-transform:uppercase; color:var(--ink-3)}'
             'a.back:hover{color:var(--den)}</style>'
             '<a class="back" href="/" data-inspect-id="report-back">'
             '&larr; All reports</a>')

CHARSET = '<meta charset="utf-8">'
DOCTYPE = "<!doctype html>"


def report_actuals(fname, html):
    """What every player on this page actually did in this one game.

    Keyed "player|stat" so a card can look itself up. Read from the page's own
    row list, not from the shortlist, so a player we projected but never picked
    still gets graded — the projection was a claim either way, and hiding the
    ones we did not bet on would only ever flatter the model.

    Empty until the game is final: a receiver with 12 yards at half time has
    not missed an over-29.5, and printing a MISS that quietly corrects itself
    an hour later is worse than printing nothing. Same gate the slate uses,
    for the same reason.
    """
    m = NAME.search(fname)
    if not m or name_key is None:
        return {}
    season, week, away, home = int(m.group(1)), int(m.group(2)), m.group(3), m.group(4)
    acts, done = outcomes(season, week)
    if f"{away}@{home}" not in done:
        return {}
    try:
        rows = json.loads(DATA_BLOB.search(html).group(1))
    except Exception as e:  # noqa: BLE001 — one unreadable report, not a 500
        sys.stderr.write(f"actuals: cannot read {fname}: {e}\n")
        return {}
    # No line filter here. A stat nobody offered a number on -- completions,
    # every week so far -- still has a projection, and that projection was
    # either close or it wasn't. The page shows the real figure on the curve
    # either way; only the hit/miss verdict needs a line to exist.
    out = {}
    for r in rows:
        a = acts.get(name_key(r["p"]), {}).get(r["s"])
        if a is not None:
            out[f'{r["p"]}|{r["s"]}'] = a
    return out


def with_theme(html):
    """Put the doctype and theme switch at the top of any page this server serves.

    Spliced per request, never written to disk — the same bargain the
    inspector overlay makes. That is what lets the fifteen reports already
    generated get a working switch without regenerating any of them, which
    would cost four Odds API credits a game and change nothing else.

    The doctype is here for the same reason. A page without one is rendered
    by rules a browser kept for the sake of websites written in 1999. The
    listing and slate pages always declared one; reports never did, so the
    site was serving two page types under two sets of rules. The template
    now emits it, but the fifteen reports on disk predate that, and this
    splice covers them at no cost.

    Order is fixed: doctype, then charset, then everything else. The charset
    must still land inside the first 1024 bytes, which is the window a
    browser scans before it gives up and guesses the encoding. The doctype
    spends 15 of them.
    """
    head = THEME + LIVE
    if CHARSET not in html:
        html = head + html
    else:
        html = html.replace(CHARSET, CHARSET + "\n" + head, 1)
    if not html.lstrip().lower().startswith("<!doctype"):
        html = DOCTYPE + "\n" + html
    return html


class Handler(SimpleHTTPRequestHandler):
    # Tailscale Serve proxies in front of this. An HTTP/1.0 server behind a
    # 1.1 proxy is a known way to get a request that never returns, so speak
    # 1.1 -- every response here carries a Content-Length.
    protocol_version = "HTTP/1.1"

    def __init__(self, *a, **kw):
        super().__init__(*a, directory=ROOT, **kw)

    def guess_type(self, path):
        """Name the encoding on every text response, not just the index.

        The index below sets it by hand. Report files go through the base
        handler, whose mimetypes lookup returns a bare "text/html" — so the
        browser guesses, guesses latin-1, and renders every em dash and
        middle dot as mojibake. The files themselves were always valid UTF-8.
        """
        t = super().guess_type(path)
        if t.startswith("text/") and "charset=" not in t:
            return t + "; charset=utf-8"
        return t

    def _send(self, body, ctype):
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _inspect_report(self, name):
        """Serve a report with the inspector overlay spliced in.

        Every report gets it, with nothing to type. The overlay itself stays
        dormant until the crosshair button is pressed, and the button remembers
        its state, so an armed session survives a reload.

        The overlay is still never written into the files on disk. It is added
        here, per request -- so a report that is copied, mailed, or uploaded to
        Cloudflare Pages carries none of this code, and that holds without
        anyone remembering to strip it. The data-inspect-id attributes do live
        in the files, but they are inert markup that nothing reads unless this
        script is present.
        """
        path = os.path.join(ROOT, name)
        if not os.path.isfile(path) or not name.endswith(".html"):
            self.send_error(404)
            return
        html = open(path, encoding="utf-8").read()

        # The likeliest touchdown scorer rides in here rather than being baked
        # into the file. write_markers.py may run after a report was generated,
        # and a report already on disk should still show a marker written
        # later. The template treats an absent marker as the normal case.
        m = NAME.search(name)
        td = td_markers().get(f"{m.group(3)}-{m.group(4)}") if m else None
        if td:
            blob = json.dumps({"player": td["player"], "conf": td["conf"],
                               "basis": td["basis"], "team": td.get("team")})
            # Before the report's own script, which runs at the end of the body.
            html = html.replace("<meta charset=\"utf-8\">",
                                "<meta charset=\"utf-8\">\n"
                                f"<script>window.__TD__={blob};</script>", 1)

        # Results, spliced per request for the same reason the theme switch is:
        # a report is written before kickoff and never touched again, so the
        # only place the outcome can come from is the database, now.
        # Week 1's report predates the TARGET constant, so
        # rerender_reports.py cannot rebuild it against the current template
        # and it never picked up the back link. Splice one in rather than
        # leave one page on the site with no way out but the browser's own
        # button, which a phone in full screen does not show.
        if 'class="back"' not in html:
            html = html.replace(
                '<div class="wrap" data-inspect-id="page-wrap">',
                '<div class="wrap" data-inspect-id="page-wrap">\n' + BACK_LINK, 1)

        # Only for a page that can draw them. Week 1's report cannot be
        # rebuilt against the current template, so its script has no result()
        # and the data would sit in the page unread.
        acts = report_actuals(name, html) if "function result(" in html else {}
        if acts:
            html = html.replace("<meta charset=\"utf-8\">",
                                "<meta charset=\"utf-8\">\n"
                                f"<script>window.__ACTUALS__={json.dumps(acts)};"
                                "</script>", 1)

        body = (with_theme(html) +
                '\n<script src="/_inspector.js"></script>\n').encode()
        self._send(body, "text/html; charset=utf-8")

    def do_GET(self):
        # Query string first: everything below matches on the bare path.
        path, _, _query = self.path.partition("?")

        if path == "/_inspector.js":
            self._send(open(INSPECTOR, "rb").read(),
                       "application/javascript; charset=utf-8")
            return
        if path == "/_data-version":
            # Polled by every open page. Cheap on purpose: two COUNTs and a
            # directory listing, and never cached anywhere.
            # _send already sets Cache-Control: no-store on everything.
            self._send(data_version().encode(), "text/plain; charset=utf-8")
            return
        if path == "/_frontend-map.json":
            self._send(open(MAP, "rb").read(), "application/json; charset=utf-8")
            return
        if path.endswith(".html") and path != "/index.html":
            self._inspect_report(os.path.basename(path))
            return

        if path in ("/slate", "/slate/", "/slate.html"):
            # The slate gets the overlay on the same terms as the listing:
            # spliced in per request, never written to disk. Nothing is stored
            # for this page at all — it is read back out of the report files.
            body = (with_theme(slate_html() + TABS_JS) +
                    '\n<script src="/_inspector.js"></script>\n').encode()
            self._send(body, "text/html; charset=utf-8")
            return

        if path in ("/", "/index.html"):
            # The listing gets the overlay too, on the same terms as a report:
            # added here, never stored. Its own elements are mapped under the
            # report-index node, and they live in this file rather than in a
            # template -- the listing is built by index_html() above.
            body = (with_theme(index_html()) +
                    '\n<script src="/_inspector.js"></script>\n').encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            # The index is a directory listing; never let a phone cache a stale one.
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)
            return
        super().do_GET()

    def log_message(self, fmt, *a):
        # On by default. When a phone "just hangs", the first thing worth
        # knowing is whether the request arrived here at all.
        sys.stderr.write("%s %s\n" % (self.address_string(), fmt % a))
        sys.stderr.flush()


if __name__ == "__main__":
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
