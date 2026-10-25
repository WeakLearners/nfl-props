/* Site-wide marks on recommendation legs. Served at /_marks.js and spliced into
 * every page by with_chrome(). Marks live on the server (/api/marks), so every
 * origin and machine sees the same set.
 *
 * Key format: "season|week|player|stat". Keep markKey in step with
 * nflprops/marks.py mark_key.
 */
(function () {
  "use strict";
  if (window.markKey) return;
  var marked = {}, loaded = false;

  function esc(s) { return String(s).replace(/[&<>"]/g, function (c) { return {"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;"}[c]; }); }
  window.markKey = function (season, week, p, s) { return season + "|" + week + "|" + p + "|" + s; };
  // Star button for one leg. Every caller passes its own inspect id.
  window.markStar = function (key, id) {
    var on = !!marked[key];
    return '<button type="button" class="mk-star' + (on ? " on" : "") + '" data-mark-key="' + esc(key) +
      '" data-inspect-id="' + esc(id) + '" aria-pressed="' + on + '" title="Mark this recommendation">' +
      (on ? "★" : "☆") + '</button>';
  };
  // Season and week of a report page, read from its file name.
  window.markReportCtx = function () {
    var m = /report_(\d{4})_w(\d+)_/.exec(location.pathname);
    return m ? {season: +m[1], week: +m[2]} : null;
  };

  var css = document.createElement("style");
  css.textContent = ".mk-star{all:unset;box-sizing:border-box;cursor:pointer;display:inline-block;" +
    "min-width:20px;text-align:center;font-size:16px;line-height:1;color:var(--ink-3);vertical-align:middle}" +
    ".mk-star:hover{color:var(--accent)}.mk-star.on{color:var(--accent)}" +
    ".mk-star:focus-visible{outline:2px solid var(--accent);outline-offset:1px}";
  document.head.appendChild(css);

  function sync(root) {
    (root || document).querySelectorAll("[data-mark-key]").forEach(function (b) {
      var on = !!marked[b.getAttribute("data-mark-key")];
      b.classList.toggle("on", on);
      b.setAttribute("aria-pressed", on);
      b.textContent = on ? "★" : "☆";
    });
  }
  window.markHas = function (k) { return !!marked[k]; };
  // A change (a toggle, a revert, or the first load) also tells the page, so a view
  // filtered to marked legs can redraw. No second fetch: the page reads markHas().
  function changed() { sync(); document.dispatchEvent(new Event("marks-changed")); }
  function put(b, on) {
    var k = b.getAttribute("data-mark-key"), was = !!marked[k];
    if (on) marked[k] = 1; else delete marked[k];
    changed();
    fetch("/api/marks", {method: "POST", headers: {"Content-Type": "application/json"},
      body: JSON.stringify({key: k, on: on})})
      .then(function (r) { if (!r.ok) throw 0; return r.json(); })
      .then(function (j) { marked = {}; j.marks.forEach(function (x) { marked[x] = 1; }); changed(); })
      .catch(function () { if (was) marked[k] = 1; else delete marked[k]; changed(); });
  }
  // Capture phase: a star inside a clickable row must not also open the row.
  document.addEventListener("click", function (e) {
    var b = e.target.closest && e.target.closest(".mk-star");
    if (!b) return;
    e.preventDefault(); e.stopPropagation();
    put(b, !marked[b.getAttribute("data-mark-key")]);
  }, true);

  // Report pages are built in the browser from DATA, and old ones are on disk,
  // so the stars are added here to the cards and shortlist rows by their DOM.
  function decorateReport() {
    var c = window.markReportCtx(); if (!c) return;
    document.querySelectorAll(".card[data-stat]").forEach(function (card) {
      var id = card.querySelector(".card-id"), nm = card.querySelector(".card-id .name");
      if (!id || !nm || id.querySelector(".mk-star")) return;
      var t = Array.prototype.filter.call(nm.childNodes, function (n) { return n.nodeType === 3; }).pop();
      if (!t) return;
      id.insertAdjacentHTML("beforeend", window.markStar(window.markKey(c.season, c.week, t.textContent, card.dataset.stat), "mark-star-card"));
    });
    document.querySelectorAll("#picklist .pk-nm[data-p]").forEach(function (n) {
      var row = n.closest(".pk-name");
      if (!row || row.querySelector(".mk-star")) return;
      row.insertAdjacentHTML("beforeend", window.markStar(window.markKey(c.season, c.week, n.dataset.p, n.dataset.s), "mark-star-pick"));
    });
  }

  function start() {
    decorateReport();
    new MutationObserver(function (ms) {
      if (loaded && ms.some(function (m) { return m.addedNodes.length; })) sync();
    }).observe(document.body, {childList: true, subtree: true});
    fetch("/api/marks", {cache: "no-store"}).then(function (r) { return r.json(); })
      .then(function (j) { j.marks.forEach(function (x) { marked[x] = 1; }); loaded = true; changed(); })
      .catch(function () { loaded = true; });
  }
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", start); else start();
})();
