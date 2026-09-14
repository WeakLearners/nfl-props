"""Slack delivery via incoming webhook.

stdlib only -- this runs from a launchd timer where a missing dependency would
fail silently at 6am on a Sunday.
"""
import json
import urllib.request
import urllib.error

from .config import require


def post(blocks=None, text="", webhook=None):
    """Post to the webhook's bound channel. Returns Slack's response body."""
    url = webhook or require("SLACK_WEBHOOK_URL")
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


def section(md):
    return {"type": "section", "text": {"type": "mrkdwn", "text": md}}


def header(t):
    return {"type": "header", "text": {"type": "plain_text", "text": t}}


def divider():
    return {"type": "divider"}
