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

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.join(BASE, "reports")
DB = os.path.join(BASE, "data", "nflprops.db")
MAP = os.path.join(BASE, "docs", "nfl-props-frontend-map.json")
INSPECTOR = os.path.join(BASE, "scripts", "inspector.js")
PORT = int(os.environ.get("NFLPROPS_WEB_PORT", "8792"))

# report_2026_w2_DET-BUF.html -> ("2026", "2", "DET @ BUF")
NAME = re.compile(r"report_(\d{4})_w(\d+)_([A-Z]+)-([A-Z]+)\.html$")

PAGE = """<!doctype html><meta charset="utf-8">
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
  }}
  @media (prefers-color-scheme: dark){{
    :root:not([data-theme="light"]){{
      color-scheme: dark;
      --bg:#121211; --surface:#1a1a19; --line:#33322e; --line-soft:#262521;
      --ink:#f6f5ef; --ink-2:#c3c2b7; --ink-3:#8b897f;
      --accent:#d95926;
    }}
  }}
  :root[data-theme="dark"]{{
    color-scheme: dark;
    --bg:#121211; --surface:#1a1a19; --line:#33322e; --line-soft:#262521;
    --ink:#f6f5ef; --ink-2:#c3c2b7; --ink-3:#8b897f;
    --accent:#d95926;
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
  /* A game card is the report's player card with different nouns in it. Same
     grid, same border, same card-top of label-plus-badge, same oversized
     monospace figure, same dotted comparison row underneath. Kickoff plays the
     part of the projection; the touchdown marker plays the part of model vs
     book. Keep these rules in step with .card in templates/report.html. */
  .rows{{
    display:grid; grid-template-columns:repeat(auto-fill,minmax(258px,1fr));
    gap:14px; margin-top:18px;
  }}
  a.card{{
    background:var(--surface); border:1px solid var(--line); border-radius:3px;
    padding:13px 14px 11px; display:flex; flex-direction:column; gap:2px;
    text-decoration:none; color:inherit;
  }}
  a.card:hover{{border-color:var(--ink-3)}}
  a.card:focus-visible{{outline:2px solid var(--accent); outline-offset:2px}}
  .card-top{{display:flex; justify-content:space-between; align-items:baseline; gap:8px}}
  .name{{
    font-family:"Barlow Condensed",sans-serif; font-weight:600; font-size:19px;
    letter-spacing:.01em; line-height:1.1; flex:1;
  }}
  .day{{
    font-family:"IBM Plex Mono",monospace; font-size:10px; font-weight:600;
    letter-spacing:.09em; padding:2px 5px; border-radius:2px;
    color:var(--surface); background:var(--accent); flex:none;
  }}
  .fig{{
    font-family:"IBM Plex Mono",monospace; font-variant-numeric:tabular-nums;
    font-size:21px; font-weight:500; letter-spacing:-.02em; white-space:nowrap;
  }}
  .fig.unknown{{font-size:15px; color:var(--ink-3); font-weight:400}}
  .rng{{
    font-family:"IBM Plex Mono",monospace; font-variant-numeric:tabular-nums;
    font-size:11.5px; color:var(--ink-2); margin-top:-1px;
  }}
  /* Each kickoff time gets its own labelled block. The label is quieter than
     a week tab and louder than a card: a thin rule, condensed caps, and the
     same monospace count the tabs use. */
  .slot{{margin-top:26px}}
  .slot:first-child{{margin-top:18px}}
  .slot-h{{
    display:flex; justify-content:space-between; align-items:baseline; gap:12px;
    margin:0; padding-bottom:5px; border-bottom:1px solid var(--line-soft);
    font-family:"Barlow Condensed",Impact,sans-serif; font-weight:600;
    font-size:16px; text-transform:uppercase; letter-spacing:.06em; color:var(--ink-2);
  }}
  .slot-h .n{{
    font-family:"IBM Plex Mono",ui-monospace,monospace; font-size:10px;
    font-weight:400; letter-spacing:.1em; color:var(--ink-3);
    font-variant-numeric:tabular-nums; text-transform:none;
  }}
  .rows{{margin-top:12px}}
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
    markers = td_markers()

    entries = []
    for f in files:
        m = NAME.search(f)
        season = int(m.group(1)) if m else 0
        week = int(m.group(2)) if m else 0
        kick = kicks.get((season, week, m.group(3), m.group(4))) if m else None
        entries.append((f, m, season, week, kick))

    # Newest week first, and inside a week the games in the order they kick
    # off — the order you actually watch them in. Sorting the whole list by
    # kickoff alone would float last week's early games above this week's.
    # A report with no schedule row sorts last within its week rather than
    # vanishing or jumping to the top.
    entries.sort(key=lambda e: (-e[2], -e[3], e[4] or "9999"))

    groups = []          # [((season, week), [row html, ...]), ...]
    for f, m, season, week, kick in entries:
        label = f"{m.group(3)} @ {m.group(4)}" if m else f[:-5]
        if kick:
            when = datetime.strptime(kick, "%Y-%m-%d %H:%M")
            day = f"{when:%a}".upper()
            time_s = f"{when:%-I:%M %p}"
            date_s = f"ET \u00b7 {when:%b %-d}"
            fig_cls, badge = "fig", f'<span class="day" data-inspect-id="index-day-badge">{day}</span>'
        else:
            # No schedule row. Say so rather than showing the file's mtime,
            # which is when the report was written and not when anyone plays.
            day, time_s, date_s = "", "kickoff unknown", "no schedule row"
            fig_cls, badge = "fig unknown", ""

        row = (f'<a class="card" data-inspect-id="index-row" href="/{f}">'
               f'<div class="card-top">'
               f'<span class="name" data-inspect-id="index-game-label">{label}</span>'
               f'{badge}</div>'
               f'<div class="{fig_cls}" data-inspect-id="index-kickoff">{time_s}</div>'
               f'<div class="rng" data-inspect-id="index-kickoff-date">{date_s}</div>'
               f'</a>')

        # Slot label for the run of games that kick off together: "Sun 1:00 PM".
        # Entries already arrive in kickoff order inside a week, so games that
        # share a kickoff are already adjacent and a run-length pass is enough.
        slot = f"{when:%a} {when:%-I:%M %p}" if kick else "Kickoff unknown"

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
        The label is the slot; the cards under it all start together.
        """
        out = []
        for slot, rws in slots:
            n = len(rws)
            out.append(f'<div class="slot" data-inspect-id="index-slot">'
                       f'<h3 class="slot-h" data-inspect-id="index-slot-label">{slot}'
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

        body = (html + '\n<script src="/_inspector.js"></script>\n').encode()
        self._send(body, "text/html; charset=utf-8")

    def do_GET(self):
        # Query string first: everything below matches on the bare path.
        path, _, _query = self.path.partition("?")

        if path == "/_inspector.js":
            self._send(open(INSPECTOR, "rb").read(),
                       "application/javascript; charset=utf-8")
            return
        if path == "/_frontend-map.json":
            self._send(open(MAP, "rb").read(), "application/json; charset=utf-8")
            return
        if path.endswith(".html") and path != "/index.html":
            self._inspect_report(os.path.basename(path))
            return

        if path in ("/", "/index.html"):
            # The listing gets the overlay too, on the same terms as a report:
            # added here, never stored. Its own elements are mapped under the
            # report-index node, and they live in this file rather than in a
            # template -- the listing is built by index_html() above.
            body = (index_html() +
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
