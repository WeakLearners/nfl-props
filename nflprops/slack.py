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
    """
    url = webhook or require("SLACK_WEBHOOK_URL")
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
