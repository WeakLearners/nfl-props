#!/usr/bin/env python3
"""Static server for the generated reports, newest first.

Bound to 127.0.0.1 only. Tailscale Serve puts it on the tailnet, so the
listening socket itself never faces the network. Tailnet-only, not Funnel:
these pages name real players and real prices under Sean's own account, and
nothing about them needs the public internet.
"""
import os
import re
import sys
from datetime import datetime
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "reports")
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
 a.row{{display:flex;justify-content:space-between;align-items:baseline;gap:12px;
   padding:14px 16px;margin-bottom:8px;background:var(--card);border:1px solid var(--line);
   border-radius:10px;text-decoration:none;color:inherit}}
 .g{{font-weight:600}} .m{{color:var(--mut);font-size:13px;white-space:nowrap}}
 .empty{{color:var(--mut)}}
</style>
<div class="wrap"><h1>NFL Props</h1>
<p class="sub">Production ranges. Newest first. No edge claim.</p>
{rows}</div>"""


def index_html():
    try:
        files = [f for f in os.listdir(ROOT) if f.endswith(".html")]
    except FileNotFoundError:
        files = []
    files.sort(key=lambda f: os.path.getmtime(os.path.join(ROOT, f)), reverse=True)
    rows = []
    for f in files:
        m = NAME.search(f)
        label = f"{m.group(3)} @ {m.group(4)}" if m else f[:-5]
        wk = f"{m.group(1)} W{m.group(2)}" if m else ""
        when = datetime.fromtimestamp(os.path.getmtime(os.path.join(ROOT, f)))
        rows.append(f'<a class="row" href="/{f}"><span class="g">{label}</span>'
                    f'<span class="m">{wk} · {when:%b %-d, %-I:%M%p}</span></a>')
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
