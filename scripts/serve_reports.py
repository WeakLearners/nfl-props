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
PORT = int(os.environ.get("NFLPROPS_WEB_PORT", "8792"))

# report_2026_w2_DET-BUF.html -> ("2026", "2", "DET @ BUF")
NAME = re.compile(r"report_(\d{4})_w(\d+)_([A-Z]+)-([A-Z]+)\.html$")

PAGE = """<!doctype html><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>NFL Props — reports</title>
<style>
 :root{{color-scheme:light dark;--bg:#f7f6f3;--fg:#14140f;--mut:#84827a;--line:#e2e0d9;--card:#fcfcfb}}
 @media (prefers-color-scheme:dark){{:root{{--bg:#14140f;--fg:#f2f1ec;--mut:#8d8b83;--line:#2b2a25;--card:#1b1a16}}}}
 *{{box-sizing:border-box}}
 body{{margin:0;background:var(--bg);color:var(--fg);
   font:16px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}}
 .wrap{{max-width:640px;margin:0 auto;padding:32px 16px 56px}}
 h1{{font-size:20px;margin:0 0 4px}}
 p.sub{{color:var(--mut);font-size:14px;margin:0 0 24px}}
 a.row{{display:flex;justify-content:space-between;align-items:flex-start;gap:12px;
   padding:14px 16px;margin-bottom:8px;background:var(--card);border:1px solid var(--line);
   border-radius:10px;text-decoration:none;color:inherit}}
 .g{{font-weight:600}} .m{{color:var(--mut);font-size:13px;white-space:nowrap;text-align:right}}
 .k{{display:block;font-weight:600;color:var(--fg)}}
 .w{{display:block;font-size:12px}}
 .td{{display:block;font-size:12px;color:var(--mut);margin-top:3px}}
 .td b{{color:var(--fg);font-weight:600}}
 .pc{{font-variant-numeric:tabular-nums}}
 .empty{{color:var(--mut)}}
</style>
<div class="wrap"><h1>NFL Props</h1>
<p class="sub">Production ranges, in kickoff order. TD = likeliest scorer. No edge claim.</p>
{rows}</div>"""


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

    rows = []
    for f, m, season, week, kick in entries:
        label = f"{m.group(3)} @ {m.group(4)}" if m else f[:-5]
        wk = f"{season} W{week}" if m else ""
        if kick:
            when = datetime.strptime(kick, "%Y-%m-%d %H:%M")
            top = f"{when:%a %-I:%M %p} ET"
        else:
            # No schedule row. Say so rather than showing the file's mtime,
            # which is when the report was written and not when anyone plays.
            top = "kickoff unknown"
        # The marker's confidence is the LOWER of the model's number and
        # FanDuel's, the same rule every leg inside a report follows. Where
        # FanDuel quotes anytime-TD one-sided there is no way to strip the
        # vig, so that book number is an overestimate and the label says so.
        td = markers.get(f"{m.group(3)}-{m.group(4)}") if m else None
        td_html = ""
        if td:
            note = {"model+book": "model &amp; book",
                    "model+book_vig": "model &amp; book (vig in)",
                    "model_only": "model only"}.get(td["basis"], td["basis"])
            td_html = (f'<span class="td">TD <b>{td["player"]}</b> '
                       f'<span class="pc">{td["conf"]*100:.0f}%</span> · {note}</span>')
        rows.append(f'<a class="row" href="/{f}"><span class="g">{label}{td_html}</span>'
                    f'<span class="m"><span class="k">{top}</span>'
                    f'<span class="w">{wk}</span></span></a>')
    if not rows:
        rows = ['<p class="empty">No reports generated yet.</p>']
    return PAGE.format(rows="\n".join(rows))


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

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            body = index_html().encode()
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
