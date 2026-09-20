#!/usr/bin/env python3
"""Publish generated NFL-props reports to Cloudflare Pages.

The problem this solves: the report is a file on Sean's Mac mini. A file on
a machine only exists on the internet while that machine is awake and its
tunnel is up -- and both of those failed once already (Tailscale Serve on
the mini, working from the mini itself, unreachable from a phone on
cellular). Cloudflare Pages is a CDN: once a file is pushed there, it is
served from Cloudflare's edge network, not from this Mac. The Mac can be
asleep, off, or unplugged and the link still works.

This script only does the pushing. It builds a small, throwaway folder
containing just the report HTML files plus a freshly generated index page
(the same index serve_reports.py renders, reused here so there's exactly
one place that knows the file-naming convention), and uploads that folder
with `wrangler pages deploy`. It does not know or care that Cloudflare
Access sits in front of the result -- that's configured once, by hand, in
the Cloudflare Zero Trust dashboard (see docs/hosting.md) and is invisible
to anything that talks to Pages over the API.

No CSVs, logs, or cached odds responses are ever included -- only
report_*.html and the generated index.

Usage:
    scripts/publish_report.py                              # publish every report
    scripts/publish_report.py report_2026_w2_DET-BUF.html   # publish just one
                                                             # (index still lists all)

Reads CLOUDFLARE_API_TOKEN, CLOUDFLARE_ACCOUNT_ID, and (optionally)
CF_PAGES_PROJECT from .env. Never prompts -- built to run unattended from
launchd. Any failure prints to stderr and exits non-zero; the scheduled
jobs route stderr to logs/launchd.log, so a broken publish shows up
there instead of vanishing silently.
"""
import os
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPORTS = ROOT / "reports"
ENV_FILE = ROOT / ".env"
WRANGLER_PKG = "wrangler@3"  # pinned: an unattended job should not pick up
                              # a new major version's breaking changes on its own.

NAME = re.compile(r"report_(\d{4})_w(\d+)_([A-Z]+)-([A-Z]+)\.html$")

# Same look as scripts/serve_reports.py's index -- kept in sync by eye since
# this is the only other place that renders it. If that page's markup
# changes, update this too.
INDEX_PAGE = """<!doctype html><meta charset="utf-8">
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


def load_env():
    """.env values, without ever printing or logging them."""
    env = dict(os.environ)
    if ENV_FILE.exists():
        for line in ENV_FILE.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            env.setdefault(k.strip(), v.strip())
    return env


def build_index(html_files):
    rows = []
    for f in html_files:
        m = NAME.search(f.name)
        label = f"{m.group(3)} @ {m.group(4)}" if m else f.name[:-5]
        wk = f"{m.group(1)} W{m.group(2)}" if m else ""
        when = datetime.fromtimestamp(f.stat().st_mtime)
        rows.append(
            f'<a class="row" href="/{f.name}"><span class="g">{label}</span>'
            f'<span class="m">{wk} · {when:%b %-d, %-I:%M%p}</span></a>'
        )
    if not rows:
        rows = ['<p class="empty">No reports generated yet.</p>']
    return INDEX_PAGE.format(rows="\n".join(rows))


def main():
    env = load_env()
    token = env.get("CLOUDFLARE_API_TOKEN")
    account = env.get("CLOUDFLARE_ACCOUNT_ID")
    project = env.get("CF_PAGES_PROJECT", "nfl-props")

    if not token or not account:
        print(
            "publish_report: missing CLOUDFLARE_API_TOKEN / CLOUDFLARE_ACCOUNT_ID "
            "in .env -- see docs/hosting.md to create them. Refusing to run "
            "wrangler without them rather than hanging on an interactive login.",
            file=sys.stderr,
        )
        sys.exit(1)

    requested = sys.argv[1:]
    if requested:
        missing = [n for n in requested if not (REPORTS / n).exists()]
        if missing:
            print(f"publish_report: not found in reports/: {', '.join(missing)}", file=sys.stderr)
            sys.exit(1)

    all_reports = sorted(REPORTS.glob("report_*.html"))
    if not all_reports:
        print("publish_report: no report_*.html files in reports/ -- nothing to publish", file=sys.stderr)
        sys.exit(1)

    with tempfile.TemporaryDirectory(prefix="nflprops-publish-") as tmp:
        site = Path(tmp)
        for f in all_reports:
            shutil.copy(f, site / f.name)
        (site / "index.html").write_text(build_index(all_reports))

        cmd = [
            "npx", "--yes", WRANGLER_PKG, "pages", "deploy", str(site),
            "--project-name", project,
            "--branch", "main",
            "--commit-dirty=true",
        ]
        proc_env = dict(os.environ)
        proc_env["CLOUDFLARE_API_TOKEN"] = token
        proc_env["CLOUDFLARE_ACCOUNT_ID"] = account
        proc_env["CI"] = "true"  # tells wrangler not to try anything interactive

        result = subprocess.run(cmd, env=proc_env, capture_output=True, text=True, timeout=120)
        sys.stdout.write(result.stdout)
        sys.stderr.write(result.stderr)
        if result.returncode != 0:
            print(f"publish_report: wrangler exited {result.returncode}", file=sys.stderr)
            sys.exit(result.returncode)

    print(f"publish_report: pushed {len(all_reports)} report(s) to Cloudflare Pages project '{project}'")


if __name__ == "__main__":
    main()
