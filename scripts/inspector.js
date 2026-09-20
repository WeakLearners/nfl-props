/* Click-to-inspect overlay for the production-ranges report.
 *
 * Never lives in a generated report file. scripts/serve_reports.py injects a
 * tag pointing here on every report it serves, so there is nothing to type.
 * The HTML on disk -- the thing that might one day be hosted -- stays free of
 * it regardless, because the injection happens per request.
 *
 * Dormant until armed. The crosshair button, bottom right, switches it on and
 * remembers that across reloads.
 *
 * It resolves what you clicked through data-inspect-id, then looks that id up
 * in docs/nfl-props-frontend-map.json (served at /_frontend-map.json). The map
 * is the source of truth; this file is a pointing device for it. An id present
 * in the DOM but missing from the map simply will not highlight -- that is the
 * designed failure, and it means the map has drifted.
 */
(function () {
  "use strict";
  if (window.__nflInspector) return;
  window.__nflInspector = true;

  var MAP_URL = "/_frontend-map.json";
  var STORE = "nflprops.inspector.on";
  var index = null;      // id -> element entry
  var mapMeta = null;
  var on = false;
  var frame = 0;         // pending requestAnimationFrame
  var hovered = null;    // element under the cursor
  var pinned = null;     // element the open bubble belongs to
  var toastTimer = 0;

  /* ---------- the map ---------- */

  function loadMap() {
    return fetch(MAP_URL, { cache: "no-store" })
      .then(function (r) { return r.json(); })
      .then(function (doc) {
        index = {};
        mapMeta = doc.meta || {};
        (doc.nodes || []).forEach(function (n) {
          (n.uiElements || []).forEach(function (e) {
            index[e.id] = { entry: e, file: n.file, node: n.label };
          });
        });
        return index;
      });
  }

  // A data-inspect-id may carry several space-separated ids when one tag is
  // described by more than one map entry. First id that the map knows wins.
  function resolve(el) {
    if (!el || !index) return null;
    var raw = el.getAttribute("data-inspect-id") || "";
    var ids = raw.split(/\s+/);
    for (var i = 0; i < ids.length; i++) {
      if (index[ids[i]]) return index[ids[i]];
    }
    return null;
  }

  /* ---------- chrome ---------- */

  // Both pages already load Barlow Condensed, IBM Plex Mono and Public Sans,
  // so the overlay can sit on the same type for free. It stays deliberately
  // orange and slightly rounder than the page: this is tool chrome, and you
  // should never mistake it for part of the report.
  var css = [
    "#nfli-btn{position:fixed;right:14px;bottom:14px;z-index:2147483000;",
    "width:38px;height:38px;border-radius:50%;border:1px solid rgba(128,128,128,.45);",
    "background:#1a1a19;color:#f6f5ef;font:16px/38px 'IBM Plex Mono',ui-monospace,monospace;",
    "text-align:center;cursor:pointer;padding:0;box-shadow:0 2px 10px rgba(0,0,0,.28)}",
    "#nfli-btn[data-on='1']{background:#eb6834;color:#fff;border-color:#eb6834}",
    "#nfli-box{position:fixed;z-index:2147482000;pointer-events:none;display:none;",
    "border:2px solid #eb6834;background:rgba(235,104,52,.12);border-radius:2px}",
    "#nfli-lab{position:fixed;z-index:2147482500;pointer-events:none;display:none;",
    "background:#14140f;color:#f7f6f3;font:500 11px/1.4 'IBM Plex Mono',ui-monospace,monospace;",
    "letter-spacing:.02em;padding:3px 6px;border-radius:2px;white-space:nowrap;",
    "max-width:60vw;overflow:hidden}",
    "#nfli-bub{position:fixed;z-index:2147483100;display:none;width:340px;max-width:92vw;",
    "background:#fcfcfb;color:#14140f;border:1px solid #e2e0d9;border-radius:4px;",
    "box-shadow:0 8px 30px rgba(0,0,0,.22);padding:14px;",
    "font:15px/1.55 'Public Sans',ui-sans-serif,system-ui,sans-serif}",
    "@media (prefers-color-scheme:dark){#nfli-bub{background:#1a1a19;color:#f6f5ef;",
    "border-color:#33322e}}",
    "#nfli-bub h4{margin:0 0 2px;display:flex;justify-content:space-between;",
    "gap:12px;align-items:baseline;border-bottom:1px solid #e2e0d9;padding-bottom:6px;",
    "font:600 22px/1.05 'Barlow Condensed',Impact,sans-serif;text-transform:uppercase;",
    "letter-spacing:.04em}",
    "@media (prefers-color-scheme:dark){#nfli-bub h4{border-color:#33322e}}",
    "#nfli-bub .kind{font:400 11px/1 'IBM Plex Mono',ui-monospace,monospace;",
    "letter-spacing:.1em;text-transform:none;color:#84827a;flex:none}",
    "#nfli-bub .desc{margin:8px 0 0;font-size:13.5px;color:#52514e}",
    "@media (prefers-color-scheme:dark){#nfli-bub .desc{color:#c3c2b7}}",
    "#nfli-bub .path{margin:7px 0 0;color:#84827a;word-break:break-all;",
    "font:400 11px/1.45 'IBM Plex Mono',ui-monospace,monospace}",
    "#nfli-bub textarea{width:100%;margin-top:10px;min-height:56px;resize:vertical;",
    "font:14px/1.5 'Public Sans',ui-sans-serif,system-ui,sans-serif;padding:7px 8px;",
    "border-radius:3px;color:inherit;background:transparent;border:1px solid #e2e0d9;",
    "box-sizing:border-box}",
    "@media (prefers-color-scheme:dark){#nfli-bub textarea{border-color:#33322e}}",
    "#nfli-bub textarea:focus{outline:2px solid #eb6834;outline-offset:1px;border-color:transparent}",
    "#nfli-bub .row{display:flex;gap:8px;margin-top:10px}",
    "#nfli-bub button{flex:1;padding:8px;border-radius:3px;cursor:pointer;",
    "border:1px solid #84827a;background:transparent;color:inherit;text-transform:uppercase;",
    "font:600 11px/1.2 'IBM Plex Mono',ui-monospace,monospace;letter-spacing:.08em}",
    "#nfli-bub button.primary{background:#eb6834;border-color:#eb6834;color:#fff}",
    "#nfli-toast{position:fixed;left:50%;bottom:22px;transform:translateX(-50%);",
    "z-index:2147483200;background:#14140f;color:#f7f6f3;padding:9px 14px;border-radius:3px;",
    "display:none;box-shadow:0 4px 18px rgba(0,0,0,.3);",
    "font:500 12px/1.3 'IBM Plex Mono',ui-monospace,monospace;letter-spacing:.02em}"
  ].join("");

  var style = document.createElement("style");
  style.textContent = css;
  document.head.appendChild(style);

  function mk(id, tag) {
    var e = document.createElement(tag || "div");
    e.id = id;
    document.body.appendChild(e);
    return e;
  }
  var btn = mk("nfli-btn", "button");
  btn.textContent = "⌖";               // crosshair
  btn.title = "Inspect mode - hover to locate an element, click to compose a prompt";
  var box = mk("nfli-box");
  var lab = mk("nfli-lab");
  var bub = mk("nfli-bub");
  var toast = mk("nfli-toast");

  /* ---------- copying ---------- */

  // The tailnet serves these reports over plain http, and browsers switch off
  // navigator.clipboard outside a secure context. Without the fallback below,
  // "copy" would do nothing at all on http://your-mac:8080 and report
  // success. The hidden-textarea route still works there.
  function copy(text) {
    if (navigator.clipboard && window.isSecureContext) {
      return navigator.clipboard.writeText(text).catch(legacy);
    }
    return Promise.resolve(legacy());

    function legacy() {
      var ta = document.createElement("textarea");
      ta.value = text;
      ta.setAttribute("readonly", "");
      ta.style.cssText = "position:fixed;top:-1000px;opacity:0";
      document.body.appendChild(ta);
      ta.select();
      var ok = false;
      try { ok = document.execCommand("copy"); } catch (e) { ok = false; }
      document.body.removeChild(ta);
      if (!ok) throw new Error("clipboard unavailable");
    }
  }

  function say(msg) {
    toast.textContent = msg;
    toast.style.display = "block";
    clearTimeout(toastTimer);
    toastTimer = setTimeout(function () { toast.style.display = "none"; }, 2200);
  }

  function composePrompt(hit, note) {
    var e = hit.entry;
    var out = [
      "In " + (mapMeta.repo || "the nfl-props repo") + ", the report UI element \"" +
        e.label + "\" (" + e.kind + ").",
      "",
      "What it is: " + e.description,
      "Defined in: " + hit.file,
      "Find it with: " + e.locate.grep
    ];
    if (note && note.trim()) {
      out.push("", "Requested change: " + note.trim());
    }
    return out.join("\n");
  }

  /* ---------- hover ---------- */

  function paint() {
    frame = 0;
    if (!hovered) { box.style.display = "none"; lab.style.display = "none"; return; }
    var r = hovered.el.getBoundingClientRect();
    box.style.display = "block";
    box.style.left = r.left + "px";
    box.style.top = r.top + "px";
    box.style.width = r.width + "px";
    box.style.height = r.height + "px";
    lab.style.display = "block";
    lab.textContent = hovered.hit.entry.label + "  ·  " + hovered.hit.entry.id;
    var lr = lab.getBoundingClientRect();
    lab.style.left = Math.min(r.left, innerWidth - lr.width - 8) + "px";
    lab.style.top = Math.max(r.top - lr.height - 5, 4) + "px";
  }

  function onMove(ev) {
    if (pinned) return;                       // bubble open: freeze the highlight
    var el = ev.target.closest && ev.target.closest("[data-inspect-id]");
    if (el && (bub.contains(el) || btn === el)) el = null;
    var hit = resolve(el);
    hovered = hit ? { el: el, hit: hit } : null;
    if (!frame) frame = requestAnimationFrame(paint);
  }

  /* ---------- click ---------- */

  function openBubble(el, hit) {
    pinned = el;
    var e = hit.entry;
    bub.innerHTML = "";

    var h = document.createElement("h4");
    var nm = document.createElement("span");
    nm.textContent = e.label;
    var kd = document.createElement("span");
    kd.className = "kind";
    kd.textContent = e.kind;
    h.appendChild(nm); h.appendChild(kd);

    var d = document.createElement("p");
    d.className = "desc";
    d.textContent = e.description;

    var p = document.createElement("p");
    p.className = "path";
    p.textContent = e.locate.path;

    var ta = document.createElement("textarea");
    ta.placeholder = "What should Claude change here? (optional)";

    var row = document.createElement("div");
    row.className = "row";
    var bPrompt = document.createElement("button");
    bPrompt.className = "primary";
    bPrompt.textContent = "Copy prompt";
    var bGrep = document.createElement("button");
    bGrep.textContent = "Copy grep";
    row.appendChild(bPrompt); row.appendChild(bGrep);

    bPrompt.addEventListener("click", function () {
      try {
        copy(composePrompt(hit, ta.value));
        say("Prompt copied - paste it into Claude");
        closeBubble();
        // Copying a full prompt means you are done pointing; the next move is
        // pasting. Leaving the overlay armed would block using the report.
        enable(false);
      } catch (err) { say("Could not copy: " + err.message); }
    });
    bGrep.addEventListener("click", function () {
      try {
        copy(e.locate.grep);
        say("grep command copied");   // stays on - you may want several
      } catch (err) { say("Could not copy: " + err.message); }
    });

    [h, d, p, ta, row].forEach(function (n) { bub.appendChild(n); });

    bub.style.display = "block";
    var r = el.getBoundingClientRect();
    var br = bub.getBoundingClientRect();
    var left = Math.min(Math.max(8, r.left), innerWidth - br.width - 8);
    var top = r.bottom + 8;
    if (top + br.height > innerHeight - 8) top = Math.max(8, r.top - br.height - 8);
    bub.style.left = left + "px";
    bub.style.top = top + "px";
    ta.focus();
  }

  function closeBubble() {
    pinned = null;
    bub.style.display = "none";
  }

  function onClick(ev) {
    if (bub.contains(ev.target) || ev.target === btn) return;
    var el = ev.target.closest && ev.target.closest("[data-inspect-id]");
    var hit = resolve(el);
    // Inspecting must never fire the report's own handlers underneath.
    ev.preventDefault();
    ev.stopPropagation();
    if (!hit) { closeBubble(); return; }
    openBubble(el, hit);
  }

  function onKey(ev) {
    if (ev.key !== "Escape") return;
    if (pinned) { closeBubble(); return; }
    enable(false);
  }

  function onScroll() { if (!pinned && !frame) frame = requestAnimationFrame(paint); }

  /* ---------- on/off ---------- */

  function enable(next) {
    if (next === on) return;
    on = next;
    btn.setAttribute("data-on", on ? "1" : "0");
    try { localStorage.setItem(STORE, on ? "1" : "0"); } catch (e) {}
    if (on) {
      document.addEventListener("mousemove", onMove, true);
      document.addEventListener("click", onClick, true);
      document.addEventListener("keydown", onKey, true);
      window.addEventListener("scroll", onScroll, true);
      window.addEventListener("resize", onScroll, true);
      say("Inspect on - hover an element, click to compose");
    } else {
      document.removeEventListener("mousemove", onMove, true);
      document.removeEventListener("click", onClick, true);
      document.removeEventListener("keydown", onKey, true);
      window.removeEventListener("scroll", onScroll, true);
      window.removeEventListener("resize", onScroll, true);
      if (frame) { cancelAnimationFrame(frame); frame = 0; }
      hovered = null;
      closeBubble();
      box.style.display = "none";
      lab.style.display = "none";
    }
  }

  btn.addEventListener("click", function (ev) {
    ev.preventDefault();
    ev.stopPropagation();
    enable(!on);
  }, true);

  // The toast's hide timer is deliberately cleaned up here and nowhere else.
  // Put it in the enable/disable path and an auto-disable fired right after a
  // toast would cancel that toast's own timer, leaving it stuck on screen.
  window.addEventListener("pagehide", function () { clearTimeout(toastTimer); });

  loadMap().then(function () {
    var want = "0";
    try { want = localStorage.getItem(STORE) || "0"; } catch (e) {}
    if (want === "1") enable(true);
  }).catch(function (err) {
    btn.disabled = true;
    btn.title = "Inspect unavailable: frontend map did not load (" + err.message + ")";
    btn.style.opacity = ".45";
  });
})();
