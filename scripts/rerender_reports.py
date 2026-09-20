#!/usr/bin/env python3
"""Rebuild reports already on disk against the current template, for free.

A finished report is templates/report.html with seven values substituted in,
and every one of them is still sitting in the file: DATA, META and TARGET as
JavaScript constants, the title, the eyebrow and the headline as markup. Pull
those back out, run them through today's template, and the report picks up
every template change since it was written -- new markup, new styling, new
script -- without a single odds lookup or API credit.

This supersedes backfill_inspect_ids.py, which could only re-apply edits to
lines that carried an inspector anchor. Anything else the template gained --
the touchdown highlight, for instance -- was invisible to it.

    python3 scripts/rerender_reports.py           # check every report, write nothing
    python3 scripts/rerender_reports.py --write   # rewrite the ones that changed

Safety: a report is only rewritten when its DATA, META and TARGET survive the
round trip byte for byte. A file that will not re-render cleanly is reported
and left exactly as it was.
"""
import json
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
TPL = ROOT / "templates" / "report.html"
REPORTS = ROOT / "reports"

DATA_RE = re.compile(r"const DATA = (\[.*?\]);\n", re.S)
META_RE = re.compile(r"const META = (\{.*?\});\n", re.S)
TARGET_RE = re.compile(r"const TARGET = (-?\d+);\n")
TITLE_RE = re.compile(r"<title>(.*?)</title>", re.S)
EYEBROW_RE = re.compile(r'<p class="eyebrow"[^>]*>(.*?)</p>', re.S)
HEADLINE_RE = re.compile(r"(<h1.*?</h1>)", re.S)


def parts(html):
    """Recover the seven substituted values, or say which one is missing."""
    got = {}
    for key, rx in (("data", DATA_RE), ("meta", META_RE), ("target", TARGET_RE),
                    ("title", TITLE_RE), ("eyebrow", EYEBROW_RE),
                    ("headline", HEADLINE_RE)):
        m = rx.search(html)
        if not m:
            return None, key
        got[key] = m.group(1)
    return got, None


def render(tpl, got):
    return (tpl.replace("__DATA__", got["data"])
               .replace("__META__", got["meta"])
               .replace("__TITLE__", got["title"])
               .replace("__EYEBROW__", got["eyebrow"])
               .replace("__HEADLINE__", got["headline"])
               .replace("__TARGET__", got["target"])
               .replace("__TARGET_ABS__", str(abs(int(got["target"])))))


def main():
    write = "--write" in sys.argv
    tpl = TPL.read_text()
    files = sorted(REPORTS.glob("report_*.html"))
    if not files:
        print("no reports on disk")
        return 0

    changed = same = failed = 0
    for f in files:
        html = f.read_text(encoding="utf-8")
        got, missing = parts(html)
        if got is None:
            print(f"SKIP  {f.name}: could not find {missing} -- left untouched")
            failed += 1
            continue

        out = render(tpl, got)

        # The round trip has to be lossless on the three values that carry the
        # actual numbers. If any of them comes back different, something about
        # this file does not match the template's shape and rewriting it would
        # be a guess.
        back, _ = parts(out)
        if (back is None
                or json.loads(back["data"]) != json.loads(got["data"])
                or json.loads(back["meta"]) != json.loads(got["meta"])
                or back["target"] != got["target"]):
            print(f"SKIP  {f.name}: data did not survive the round trip -- left untouched")
            failed += 1
            continue

        if out == html:
            same += 1
            continue
        if write:
            f.write_text(out, encoding="utf-8")
        changed += 1
        print(f"{'rewrote' if write else 'would rewrite'} {f.name}")

    print(f"\n{changed} to rewrite, {same} already current, {failed} skipped")
    if not write and changed:
        print("re-run with --write to apply")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
