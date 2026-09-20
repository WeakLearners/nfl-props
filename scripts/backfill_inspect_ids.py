#!/usr/bin/env python3
"""Add inspector anchors to reports that were generated before they existed.

A finished report is the template with seven placeholders substituted. Nothing
else about the markup changes. So the same insertions that put data-inspect-id
attributes into templates/report.html apply, character for character, to a
report already sitting on disk -- no re-render, no odds lookup, no API credit.

Idempotent: a file that already carries the anchors is skipped. Run it once
after adding an anchor to the template, or never again if you do not care
about old reports.

    python3 scripts/backfill_inspect_ids.py            # report what it would do
    python3 scripts/backfill_inspect_ids.py --write    # actually edit the files
"""
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
TPL = ROOT / "templates" / "report.html"
REPORTS = ROOT / "reports"


TAG = re.compile(r'<[a-zA-Z][^<>]*data-inspect-id="[a-z0-9-]+"[^<>]*>')
STANDALONE = re.compile(r'^\s*\w+\.dataset\.inspectId="[a-z0-9-]+";\s*$')


def substitutions():
    """Derive (before, after) pairs by reading anchors back out of the template.

    Keeping a second hand-written copy of the edits would rot the moment the
    template gained another anchor, so the template stays the single source of
    truth. Three shapes have to be handled separately, and getting any of them
    wrong loses anchors silently:

    1. An anchored HTML tag. Match on the tag alone, never the whole line --
       the eyebrow's line holds __EYEBROW__, which a finished report replaced
       with a kickoff time long ago, so a line-level match would never hit.

    2. A `dataset.inspectId` set alongside other code on one line. The rest of
       the line is the thing to match on.

    3. A `dataset.inspectId` on a line of its own. There is nothing to match --
       it is a pure insertion. Anchor it to the line above and re-emit both.
       These four (player-card, pick-row, table-row, stat-heading) are the most
       clickable elements on the page, so skipping them is not an option.
    """
    lines = TPL.read_text().splitlines()
    pairs = []
    for i, line in enumerate(lines):
        placed = set()
        for tag in TAG.findall(line):                                  # shape 1
            bare = re.sub(r'\s*data-inspect-id="[a-z0-9-]+"', "", tag)
            if bare != tag:
                pairs.append((bare, tag))
                placed |= set(re.findall(r'data-inspect-id="([a-z0-9-]+)"', tag))

        # Shape 1b: an anchor the tag matcher could not reach. Two tags on this
        # page are assembled by JavaScript string concatenation -- the curve
        # <svg> runs across several lines, and the matchup <div> carries a
        # `d.def<0.97` ternary whose own angle bracket breaks any tag pattern.
        # Neither line holds a build-time placeholder, so matching the whole
        # line is safe here even though it is not safe for the eyebrow.
        rest = set(re.findall(r'data-inspect-id="([a-z0-9-]+)"', line)) - placed
        if rest:
            bare = re.sub(r'\s*data-inspect-id="[a-z0-9-]+"', "", line)
            if bare.strip() and bare != line:
                pairs.append((bare, line))

        if "dataset.inspectId=" not in line:
            continue
        if STANDALONE.match(line):                                     # shape 3
            prev = lines[i - 1]
            if prev.strip():
                pairs.append((prev, prev + "\n" + line))
        else:                                                          # shape 2
            bare = re.sub(r'\s*\w+\.dataset\.inspectId="[a-z0-9-]+";', "", line)
            if bare.strip() and bare != line:
                pairs.append((bare, line))
    return pairs


def main():
    write = "--write" in sys.argv
    pairs = substitutions()
    files = sorted(REPORTS.glob("report_*.html"))
    if not files:
        print("no reports on disk")
        return 0

    touched = skipped = 0
    for f in files:
        s = f.read_text(encoding="utf-8")
        if "data-inspect-id" in s:
            skipped += 1
            continue
        applied = 0
        for before, after in pairs:
            if s.count(before) >= 1:
                s = s.replace(before, after)
                applied += 1
        if write:
            f.write_text(s, encoding="utf-8")
        n = len(re.findall(r'data-inspect-id="[a-z0-9-]+"', s))
        print(f"{'wrote' if write else 'would write'} {f.name}: "
              f"{applied} edits, {n} static anchors")
        touched += 1

    print(f"\n{touched} report(s) {'updated' if write else 'pending'}, "
          f"{skipped} already anchored")
    if not write and touched:
        print("re-run with --write to apply")
    return 0


if __name__ == "__main__":
    sys.exit(main())
