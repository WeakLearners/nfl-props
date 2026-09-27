"""Slack delivery via incoming webhook.

stdlib only -- this runs from a launchd timer where a missing dependency would
fail silently at 6am on a Sunday.
"""
import json
import urllib.request
import urllib.error

from .config import require


# Slack refuses a message with more than 50 blocks, with a flat
# "invalid_blocks" and no hint about which rule was broken.
#
# That is what killed the 2026 W2 Sunday post on 2026-09-20. Every report
# generated correctly -- 13 games, all written to reports/ -- and then the
# single delivery step threw. One header, four blocks per game, two trailers:
# 55. The slate size decides whether this job works, and nothing in the code
# knew that. A 13-game Sunday failed where a 9-game one would have passed.
MAX_BLOCKS = 50

# Per-block content limits Slack also enforces. _chunk() fixes an over-length
# *message* by splitting it into several. None of these can be fixed that way
# -- an oversized header or section is oversized in every message it lands
# in -- so validate_blocks() catches them separately and post() falls back
# to a plain-text message rather than crash on a second, different
# "invalid_blocks" 400.
MAX_SECTION_CHARS = 3000
MAX_HEADER_CHARS = 150
MAX_FIELDS = 10
MAX_FIELD_CHARS = 2000


def validate_blocks(blocks):
    """Check blocks against Slack's per-block Block Kit limits.

    Returns a list of human-readable problem strings; empty means clean.
    Does not check the 50-block-per-message ceiling -- post() handles that
    by splitting, which is a real fix, not a validation failure.
    """
    problems = []
    for i, b in enumerate(blocks or []):
        t = b.get("type")
        if t in ("header", "section"):
            text = b.get("text", {}).get("text", "")
            if not text.strip():
                problems.append(f"block {i} ({t}): empty text")
                continue
            limit = MAX_HEADER_CHARS if t == "header" else MAX_SECTION_CHARS
            if len(text) > limit:
                problems.append(
                    f"block {i} ({t}): text is {len(text)} chars, limit {limit}")
        if t == "section":
            fields = b.get("fields") or []
            if len(fields) > MAX_FIELDS:
                problems.append(
                    f"block {i} (section): {len(fields)} fields, limit {MAX_FIELDS}")
            for j, f in enumerate(fields):
                ftext = f.get("text", "")
                if not ftext.strip():
                    problems.append(f"block {i} field {j}: empty text")
                elif len(ftext) > MAX_FIELD_CHARS:
                    problems.append(
                        f"block {i} field {j}: {len(ftext)} chars, limit {MAX_FIELD_CHARS}")
    return problems


def _chunk(blocks, size=MAX_BLOCKS):
    """Split blocks into postable runs, preferring to break after a divider.

    A blind split can land mid-game, putting a game's header in one message
    and its legs in the next. Backing up to the last divider keeps each game
    whole. If a single run has no divider to back up to -- possible only if
    one game somehow produced 50 blocks -- it falls back to a hard cut, since
    an awkward split still beats nothing arriving.
    """
    out = []
    i = 0
    while i < len(blocks):
        end = min(i + size, len(blocks))
        if end < len(blocks):
            cut = end
            while cut > i + 1 and blocks[cut - 1].get("type") != "divider":
                cut -= 1
            if cut > i + 1:
                end = cut
        out.append(blocks[i:end])
        i = end
    return out


def _post_one(url, blocks, text):
    payload = {"text": text}
    if blocks:
        payload["blocks"] = blocks
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            return r.read().decode()
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"slack {e.code}: {e.read().decode()}") from None


def post(blocks=None, text="", webhook=None):
    """Post to the webhook's bound channel. Returns Slack's response body.

    Splits automatically past MAX_BLOCKS and posts the parts in order, so a
    big slate arrives as several messages rather than as nothing at all.

    Before any of that, validate_blocks() checks the per-block limits that
    splitting can't fix. If those fail, this posts `text` alone (no blocks)
    instead of sending Slack something it will 400 on -- a plainer message
    beats a launchd job that dies at the delivery step.
    """
    url = webhook or require("SLACK_WEBHOOK_URL")
    if blocks:
        problems = validate_blocks(blocks)
        if problems:
            fallback = text or "(report generated; Slack formatting failed validation -- see logs)"
            return _post_one(url, None, fallback)
    if not blocks or len(blocks) <= MAX_BLOCKS:
        return _post_one(url, blocks, text)
    parts = _chunk(blocks)
    return " ".join(
        _post_one(url, part, text if n == 0 else f"{text} ({n + 1}/{len(parts)})")
        for n, part in enumerate(parts)
    )


def section(md):
    return {"type": "section", "text": {"type": "mrkdwn", "text": md}}


def header(t):
    return {"type": "header", "text": {"type": "plain_text", "text": t}}


def divider():
    return {"type": "divider"}
