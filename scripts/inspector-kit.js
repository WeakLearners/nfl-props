/* inspector-kit 1.1.0 -- click-to-inspect overlay over a frontend map, with tickets.
 *
 * Canonical copy: second-brain/.claude/skills/inspector-mode/assets/inspector-kit.js
 * Each app keeps a byte-identical copy. Do not edit an app's copy. Change this
 * file, bump the version above, and copy it to every app (see SKILL.md).
 *
 * Framework-free. Load it as a <script> or import it for its side effect. It
 * defines window.InspectorKit.create(config) and does nothing until called.
 *
 *   InspectorKit.create({
 *     mapUrl:     "/_frontend-map.json",   // or loadMap: () => Promise<mapDoc>
 *     storageKey: "myapp.inspector.on",    // remembers on/off across reloads
 *     repo:       "my-app repo",           // prompt text; default map meta.repo
 *     button:     "float",                 // "float" | ".css-selector" | Element | false
 *     lazy:       false,                   // true: load the map on first enable
 *     theme:      { accent, bg, fg, muted, border, font, mono }   // CSS strings
 *     onChange:   function (on) {}         // for an app-owned button
 *     tickets:    { url: "https://host:8793/api/tickets", app: "overwatch" }
 *                 // optional: adds "Add ticket", the main action. The request
 *                 // goes to the org ticket log, and a scheduled agent works it.
 *   })  ->  { enable(bool), toggle(), isOn(), destroy() }
 *
 * The map is the source of truth (built by the app-map skill). Elements resolve
 * through data-inspect-id. An id in the DOM but missing from the map does not
 * highlight: that is the designed failure, and it means the map has drifted.
 */
(function () {
  "use strict";
  if (window.InspectorKit) return;

  var VERSION = "1.1.0";
  var Z = 2147482000;

  function create(cfg) {
    cfg = cfg || {};
    var t = cfg.theme || {};
    var accent = t.accent || "#eb6834";
    var bg = t.bg || "#fcfcfb", fg = t.fg || "#14140f";
    var muted = t.muted || "#84827a", border = t.border || "#e2e0d9";
    var font = t.font || "ui-sans-serif,system-ui,sans-serif";
    var mono = t.mono || "ui-monospace,SFMono-Regular,Menlo,monospace";
    var dark = t.dark !== false && !t.bg;   // follow prefers-color-scheme only on default colours
    var store = cfg.storageKey || "inspector.on";

    var index = null, meta = {}, loading = null;
    var on = false, frame = 0, hovered = null, pinned = null, toastTimer = 0;
    var mx = 0, my = 0;

    /* ---------- the map ---------- */

    function fetchMap() {
      if (cfg.loadMap) return Promise.resolve(cfg.loadMap());
      return fetch(cfg.mapUrl || "/_frontend-map.json", { cache: "no-store" })
        .then(function (r) { if (!r.ok) throw new Error("HTTP " + r.status); return r.json(); });
    }
    function loadMap() {
      if (index) return Promise.resolve(index);
      if (!loading) {
        loading = fetchMap().then(function (doc) {
          doc = (doc && doc["default"]) || doc;
          index = {};
          meta = doc.meta || {};
          (doc.nodes || []).forEach(function (n) {
            (n.uiElements || []).forEach(function (e) {
              index[e.id] = { entry: e, file: (e.locate && e.locate.path) || n.file || "", node: n.label };
            });
          });
          return index;
        });
        loading.catch(function () { loading = null; });
      }
      return loading;
    }

    // A data-inspect-id may carry several space-separated ids when one tag is
    // described by more than one map entry. The first id the map knows wins.
    function resolve(el) {
      if (!el || !index) return null;
      var ids = (el.getAttribute("data-inspect-id") || "").split(/\s+/);
      for (var i = 0; i < ids.length; i++) if (index[ids[i]]) return index[ids[i]];
      return null;
    }

    /* ---------- chrome ---------- */

    var css = [
      ".insp-btn{position:fixed;right:14px;bottom:14px;z-index:" + (Z + 1000) + ";width:38px;height:38px;",
      "border-radius:50%;border:1px solid rgba(128,128,128,.45);background:#1a1a19;color:#f6f5ef;",
      "font:16px/38px " + mono + ";text-align:center;cursor:pointer;padding:0;box-shadow:0 2px 10px rgba(0,0,0,.28)}",
      ".insp-btn[data-on='1']{background:" + accent + ";color:#fff;border-color:" + accent + "}",
      ".insp-btn[data-slot]{position:static;width:auto;height:auto;border:0;background:transparent;",
      "color:inherit;box-shadow:none;font-size:16px;line-height:1;padding:0 6px}",
      ".insp-btn[data-slot][data-on='1']{background:transparent;color:" + accent + "}",
      ".insp-btn:disabled{opacity:.45;cursor:not-allowed}",
      ".insp-box{position:fixed;z-index:" + Z + ";pointer-events:none;display:none;",
      "border:2px solid " + accent + ";background:color-mix(in srgb," + accent + " 12%,transparent);border-radius:3px}",
      ".insp-lab{position:fixed;z-index:" + (Z + 500) + ";pointer-events:none;display:none;max-width:280px;",
      "background:#14140f;color:#f7f6f3;font:500 11px/1.4 " + mono + ";padding:4px 7px;border-radius:3px}",
      ".insp-lab small{display:block;opacity:.7;font-weight:400}",
      ".insp-bub{position:fixed;z-index:" + (Z + 1100) + ";display:none;width:340px;max-width:92vw;box-sizing:border-box;",
      "background:" + bg + ";color:" + fg + ";border:1px solid " + border + ";border-radius:6px;",
      "box-shadow:0 8px 30px rgba(0,0,0,.22);padding:14px;font:14px/1.5 " + font + "}",
      ".insp-bub h4{margin:0;display:flex;justify-content:space-between;gap:12px;align-items:baseline;",
      "font:600 15px/1.2 " + font + "}",
      ".insp-bub .kind{font:400 11px/1 " + mono + ";color:" + muted + ";flex:none}",
      ".insp-bub .desc{margin:6px 0 0;font-size:13px;opacity:.85}",
      ".insp-bub .path{margin:6px 0 0;color:" + muted + ";word-break:break-all;font:400 11px/1.45 " + mono + "}",
      ".insp-bub textarea{width:100%;margin-top:10px;min-height:56px;resize:vertical;box-sizing:border-box;",
      "font:13px/1.5 " + font + ";padding:7px 8px;border-radius:4px;color:inherit;background:transparent;",
      "border:1px solid " + border + "}",
      ".insp-bub textarea:focus{outline:2px solid " + accent + ";outline-offset:1px;border-color:transparent}",
      ".insp-bub .row{display:flex;gap:6px;margin-top:10px;justify-content:flex-end}",
      ".insp-bub button{padding:6px 10px;border-radius:4px;cursor:pointer;border:1px solid " + border + ";",
      "background:transparent;color:inherit;font:600 11px/1.2 " + mono + "}",
      ".insp-bub button.ghost{border-color:transparent;color:" + muted + "}",
      ".insp-bub button.primary{background:" + accent + ";border-color:" + accent + ";color:#fff}",
      ".insp-toast{position:fixed;left:50%;bottom:22px;transform:translateX(-50%);z-index:" + (Z + 1200) + ";",
      "background:#14140f;color:#f7f6f3;padding:9px 14px;border-radius:4px;display:none;",
      "box-shadow:0 4px 18px rgba(0,0,0,.3);font:500 12px/1.3 " + mono + "}"
    ];
    if (dark) css.push(
      "@media (prefers-color-scheme:dark){.insp-bub{background:#1a1a19;color:#f6f5ef;border-color:#33322e}",
      ".insp-bub textarea,.insp-bub button{border-color:#33322e}}");
    var style = document.createElement("style");
    style.textContent = css.join("");
    document.head.appendChild(style);

    var made = [style];
    function mk(cls, tag) {
      var e = document.createElement(tag || "div");
      e.className = cls;
      e.setAttribute("data-inspector-ui", "");
      document.body.appendChild(e);
      made.push(e);
      return e;
    }
    var box = mk("insp-box"), lab = mk("insp-lab"), bub = mk("insp-bub"), toast = mk("insp-toast");

    var btn = null;
    if (cfg.button !== false) {
      btn = mk("insp-btn", "button");
      btn.type = "button";
      btn.textContent = "⌖";   // crosshair
      btn.title = "Inspect mode: hover an element, click to compose a Claude prompt";
      btn.setAttribute("aria-label", "Toggle inspect mode");
      btn.setAttribute("data-inspector-toggle", "");
      var slot = typeof cfg.button === "string" && cfg.button !== "float" ? document.querySelector(cfg.button)
               : (cfg.button && cfg.button.nodeType === 1 ? cfg.button : null);
      if (slot) { slot.appendChild(btn); btn.setAttribute("data-slot", "1"); }
      btn.addEventListener("click", function (ev) { ev.preventDefault(); ev.stopPropagation(); toggle(); }, true);
    }

    /* ---------- copying ---------- */

    // Apps are often opened over plain http (tailnet, LAN IP). Browsers turn
    // off navigator.clipboard outside a secure context, so the hidden-textarea
    // route is the fallback. Returns a promise of true/false so the caller can
    // say out loud when a copy failed.
    function copy(text) {
      function legacy() {
        try {
          var ta = document.createElement("textarea");
          ta.value = text;
          ta.setAttribute("readonly", "");
          ta.style.cssText = "position:fixed;top:0;left:0;opacity:0;pointer-events:none";
          document.body.appendChild(ta);
          ta.select();
          ta.setSelectionRange(0, text.length);
          var ok = document.execCommand("copy");
          document.body.removeChild(ta);
          return ok;
        } catch (e) { return false; }
      }
      if (navigator.clipboard && window.isSecureContext) {
        return navigator.clipboard.writeText(text).then(function () { return true; }, legacy);
      }
      return Promise.resolve(legacy());
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
        "In " + (cfg.repo || meta.repo || "this repo") + ", the UI element \"" + e.label + "\" (" + e.kind + ").",
        "",
        "What it is: " + (e.description || "(no description in the map)"),
        "Defined in: " + hit.file,
        "Find it with: " + ((e.locate && e.locate.grep) || "(no grep in the map)")
      ];
      if (note && note.trim()) out.push("", "Requested change: " + note.trim());
      return out.join("\n");
    }

    // Tickets (1.1.0). One POST to the org ticket route. The custom header
    // forces a CORS preflight, so only origins the server allows can write.
    function sendTicket(hit, note) {
      var e = hit.entry, tk = cfg.tickets;
      var body = JSON.stringify({
        app: tk.app, element: e.id, label: e.label, kind: e.kind, file: hit.file,
        grep: (e.locate && e.locate.grep) || "", page: location.pathname, request: note.trim()
      });
      return fetch(tk.url, {
        method: "POST", mode: "cors", credentials: "omit",
        headers: { "Content-Type": "application/json", "X-Inspector": "1" }, body: body
      }).then(function (r) {
        return r.json().catch(function () { return {}; }).then(function (j) {
          if (!r.ok) throw new Error(j.error || ("HTTP " + r.status));
          return j.ticket || "ticket";
        });
      });
    }

    /* ---------- hover ---------- */

    function isOurs(el) { return !!(el && el.closest && el.closest("[data-inspector-ui],[data-inspector-toggle]")); }

    function paint() {
      frame = 0;
      if (!hovered || pinned) {
        if (!pinned) box.style.display = "none";
        lab.style.display = "none";
        return;
      }
      var r = hovered.el.getBoundingClientRect();
      box.style.display = "block";
      box.style.left = r.left + "px";
      box.style.top = r.top + "px";
      box.style.width = r.width + "px";
      box.style.height = r.height + "px";
      lab.innerHTML = "";
      lab.appendChild(document.createTextNode(hovered.hit.entry.label));
      var s = document.createElement("small");
      s.textContent = hovered.hit.entry.kind + " · click to ask Claude";
      lab.appendChild(s);
      lab.style.display = "block";
      var lr = lab.getBoundingClientRect();
      lab.style.left = Math.min(mx + 14, innerWidth - lr.width - 8) + "px";
      lab.style.top = Math.min(my + 14, innerHeight - lr.height - 8) + "px";
    }

    function onMove(ev) {
      mx = ev.clientX; my = ev.clientY;
      if (pinned) return;   // bubble open: freeze the highlight
      var el = ev.target.closest && ev.target.closest("[data-inspect-id]");
      if (isOurs(ev.target)) el = null;
      var hit = resolve(el);
      hovered = hit ? { el: el, hit: hit } : null;
      if (!frame) frame = requestAnimationFrame(paint);
    }

    /* ---------- click ---------- */

    function button(text, cls, fn) {
      var b = document.createElement("button");
      b.type = "button";
      b.textContent = text;
      if (cls) b.className = cls;
      b.addEventListener("click", fn);
      return b;
    }

    function openBubble(el, hit) {
      pinned = el;
      var e = hit.entry;
      bub.innerHTML = "";

      var h = document.createElement("h4");
      var nm = document.createElement("span"); nm.textContent = e.label;
      var kd = document.createElement("span"); kd.className = "kind"; kd.textContent = e.kind;
      h.appendChild(nm); h.appendChild(kd);
      var d = document.createElement("p"); d.className = "desc"; d.textContent = e.description || "";
      var p = document.createElement("p"); p.className = "path"; p.textContent = hit.file;
      var ta = document.createElement("textarea");
      ta.placeholder = cfg.tickets ? "What should change here? (needed for a ticket)" : "What should Claude change here? (optional)";

      // On a failed copy the bubble stays open with the text, so nothing is lost.
      function copyPrompt() {
        copy(composePrompt(hit, ta.value)).then(function (ok) {
          if (!ok) { say("Copy blocked - select the text and copy by hand"); return; }
          say("Prompt copied - paste it into Claude");
          closeBubble();
          // Copying a full prompt means you are done pointing. The next move
          // is pasting, and an armed overlay would block using the app.
          enable(false);
        });
      }
      var primary = copyPrompt;   // Enter runs the main action; Shift+Enter is a new line
      ta.addEventListener("keydown", function (ev) {
        if (ev.key === "Enter" && !ev.shiftKey) { ev.preventDefault(); primary(); }
      });

      var row = document.createElement("div");
      row.className = "row";
      row.appendChild(button("Cancel", "ghost", closeBubble));
      row.appendChild(button("Copy grep", "", function () {
        copy((e.locate && e.locate.grep) || "").then(function (ok) {
          say(ok ? "grep command copied" : "Copy blocked - select the text and copy by hand");
        });   // stays on: you may want several
      }));
      if (cfg.tickets && cfg.tickets.url) {
        // A ticket needs the request text. The bubble stays open on any failure.
        var addTicket = function () {
          if (!ta.value.trim()) { say("Write the change first, then add the ticket"); ta.focus(); return; }
          sendTicket(hit, ta.value).then(function (id) {
            say(id + " added - a scheduled agent picks it up");
            ta.value = "";
            closeBubble();   // stays on: you may log several in one pass
          }, function (err) { say("Ticket not saved: " + err.message); });
        };
        row.appendChild(button("Copy prompt", "", copyPrompt));
        row.appendChild(button("Add ticket", "primary", addTicket));
        primary = addTicket;
      } else {
        row.appendChild(button("Copy prompt", "primary", copyPrompt));
      }

      [h, d, p, ta, row].forEach(function (n) { bub.appendChild(n); });
      lab.style.display = "none";
      bub.style.display = "block";
      var r = el.getBoundingClientRect(), br = bub.getBoundingClientRect();
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
      if (!frame) frame = requestAnimationFrame(paint);
    }

    function onClick(ev) {
      if (isOurs(ev.target)) return;
      var el = ev.target.closest && ev.target.closest("[data-inspect-id]");
      var hit = resolve(el);
      // Inspecting must never fire the app's own handlers underneath.
      ev.preventDefault();
      ev.stopPropagation();
      if (!hit) { closeBubble(); return; }
      hovered = { el: el, hit: hit };
      paint();
      openBubble(el, hit);
    }

    function onKey(ev) {
      if (ev.key !== "Escape") return;
      if (pinned) { closeBubble(); return; }
      enable(false);
    }

    function onScroll() { if (!pinned && !frame) frame = requestAnimationFrame(paint); }

    /* ---------- on/off ---------- */

    function listen(add) {
      var f = add ? "addEventListener" : "removeEventListener";
      document[f]("mousemove", onMove, true);
      document[f]("click", onClick, true);
      document[f]("keydown", onKey, true);
      window[f]("scroll", onScroll, true);
      window[f]("resize", onScroll, true);
    }

    function enable(next) {
      next = !!next;
      if (next === on) return;
      if (next && !index) {
        loadMap().then(function () { enable(true); }, function (err) {
          say("Inspect unavailable: the frontend map did not load (" + err.message + ")");
        });
        return;
      }
      on = next;
      if (btn) btn.setAttribute("data-on", on ? "1" : "0");
      try { localStorage.setItem(store, on ? "1" : "0"); } catch (e) {}
      listen(on);
      if (on) {
        say("Inspect on - hover an element, click to compose");
      } else {
        if (frame) { cancelAnimationFrame(frame); frame = 0; }
        hovered = null;
        pinned = null;
        bub.style.display = "none";
        box.style.display = "none";
        lab.style.display = "none";
      }
      if (cfg.onChange) { try { cfg.onChange(on); } catch (e) {} }
    }

    function toggle() { enable(!on); }

    // The toast's hide timer is cleaned up here and nowhere else. In the
    // enable/disable path, an auto-disable right after a toast would cancel
    // that toast's own timer and leave it stuck on screen.
    function onHide() { clearTimeout(toastTimer); }
    window.addEventListener("pagehide", onHide);

    function destroy() {
      enable(false);
      onHide();
      window.removeEventListener("pagehide", onHide);
      made.forEach(function (n) { if (n.parentNode) n.parentNode.removeChild(n); });
    }

    var want = "0";
    try { want = localStorage.getItem(store) || "0"; } catch (e) {}
    if (!cfg.lazy || want === "1") {
      loadMap().then(function () { if (want === "1") enable(true); }, function (err) {
        if (btn) { btn.disabled = true; btn.title = "Inspect unavailable: frontend map did not load (" + err.message + ")"; }
      });
    }

    return { enable: enable, toggle: toggle, isOn: function () { return on; }, destroy: destroy };
  }

  window.InspectorKit = { version: VERSION, create: create };
})();
