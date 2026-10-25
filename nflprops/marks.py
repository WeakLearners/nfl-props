"""Site-wide marks on recommendation legs, stored on the server.

One key per leg: "season|week|player|stat", e.g. "2026|7|Jaxon Smith-Njigba|receiving_yards".
scripts/marks.js builds the same key in the browser (markKey). Keep the two in step.
The store is data/marks.json: {"marks": {"<key>": "<iso timestamp>"}}.
"""
import datetime
import json
import os
import re
import tempfile
import threading

PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "marks.json")
MAX_BODY = 2048
MAX_KEY = 200
KEY_RE = re.compile(r"^\d{4}\|\d{1,2}\|[^|\x00-\x1f\x7f]{1,100}\|[a-z_]{1,40}$")
_LOCK = threading.Lock()


def mark_key(season, week, player, stat):
    return f"{int(season)}|{int(week)}|{player}|{stat}"


def valid_key(key):
    return isinstance(key, str) and len(key) <= MAX_KEY and bool(KEY_RE.match(key))


def load(path=None):
    try:
        with open(path or PATH, encoding="utf-8") as f:
            m = json.load(f).get("marks", {})
        return m if isinstance(m, dict) else {}
    except (OSError, ValueError, AttributeError):
        return {}


def keys(path=None):
    return sorted(load(path))


def _save(m, path):
    d = os.path.dirname(path)
    os.makedirs(d, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=d, prefix=".marks-", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump({"marks": m}, f, ensure_ascii=False, indent=1)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def apply_post(body, path=None):
    """Handle a POST body. Returns (status, payload). The only write the site makes."""
    path = path or PATH
    if len(body) > MAX_BODY:
        return 400, {"error": "body too large"}
    try:
        req = json.loads(body.decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        return 400, {"error": "bad json"}
    if (not isinstance(req, dict) or set(req) != {"key", "on"}
            or not valid_key(req["key"]) or not isinstance(req["on"], bool)):
        return 400, {"error": "bad request"}
    with _LOCK:
        m = load(path)
        if req["on"]:
            m.setdefault(req["key"], datetime.datetime.now().isoformat(timespec="seconds"))
        else:
            m.pop(req["key"], None)
        _save(m, path)
    return 200, {"marks": sorted(m)}


def star_html(key, inspect_id, marked=()):
    """Server-side twin of markStar() in scripts/marks.js."""
    from html import escape
    on = key in marked
    return (f'<button type="button" class="mk-star{" on" if on else ""}" data-mark-key="{escape(key)}"'
            f' data-inspect-id="{escape(inspect_id)}" aria-pressed="{str(on).lower()}"'
            f' title="Mark this recommendation">{chr(0x2605) if on else chr(0x2606)}</button>')
