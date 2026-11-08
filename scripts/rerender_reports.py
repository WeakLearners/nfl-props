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


def headers_only(write):
    """--headers: rewrite only the <h1> of every report (team records as of kickoff). Nothing else is touched."""
    sys.path.insert(0, str(ROOT))
    from nflprops.records import headline
    changed = same = failed = 0
    for f in sorted(REPORTS.glob("report_*.html")):
        html = f.read_text(encoding="utf-8")
        fm = re.match(r"report_(\d{4})_w(\d+)_([A-Z]+)-([A-Z]+)\.html", f.name)
        m = HEADLINE_RE.search(html)
        if not fm or not m:
            print(f"SKIP  {f.name}: no header found"); failed += 1; continue
        plain = re.sub(r' <span class="rec"[^>]*>.*?</span>', "", m.group(1))
        hm = re.match(r'<h1>(.*?) <span class="at">at</span> (.*?)</h1>$', plain, re.S)
        if not hm:
            print(f"SKIP  {f.name}: header has an unexpected shape"); failed += 1; continue
        new = headline(hm.group(1), hm.group(2), int(fm.group(1)), int(fm.group(2)), fm.group(3), fm.group(4))
        out = html[:m.start()] + new + html[m.end():]
        # Drop the stylesheet rule an earlier build added; the span carries its own style now.
        out = re.sub(r"  /\* Team record as of kickoff.*?vertical-align:\.18em\}\n", "", out, flags=re.S)
        if out == html:
            same += 1; continue
        if write:
            f.write_text(out, encoding="utf-8")
        changed += 1
    print(f"{changed} headers {'rewritten' if write else 'to rewrite'}, {same} already current, {failed} skipped")
    return 1 if failed else 0


def main():
    if "--headers" in sys.argv:
        return headers_only("--write" in sys.argv)
    write = "--write" in sys.argv
    tpl = TPL.read_text()
    files = sorted(REPORTS.glob("report_*.html"))
    chips = None
    if "--week" in sys.argv:
        # --week 2026:4 limits the run to one week and adds rank chips to its rows.
        # Past weeks are never touched. The odds board in each file is kept as saved.
        sea, wk = sys.argv[sys.argv.index("--week") + 1].split(":")
        files = [f for f in files if f.name.startswith(f"report_{sea}_w{wk}_")]
        sys.path.insert(0, str(ROOT))
        from nflprops.rating import chip_lookup
        from nflprops.odds import name_key
        from nflprops.badges import card_line, get_badges, player_badges
        chips = chip_lookup(name_key), name_key, get_badges(), player_badges, (int(sea), int(wk))
    if "--only" in sys.argv:
        # --only 2026:8 limits the run to one week and changes nothing but the template and the window.
        sea, wk = sys.argv[sys.argv.index("--only") + 1].split(":")
        files = [f for f in files if f.name.startswith(f"report_{sea}_w{wk}_")]
    if not files:
        print("no reports on disk")
        return 0

    only = None
    if "--only" in sys.argv:
        sea, wk = sys.argv[sys.argv.index("--only") + 1].split(":")
        only = (int(sea), int(wk))
    wxd_cache = {}

    def wx_deltas(season, week):
        """(player key, stat) -> mu with weather minus mu without, for one week. Computed once."""
        if (season, week) in wxd_cache:
            return wxd_cache[(season, week)]
        sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
        from nflprops import spread as SP
        from nflprops.odds import name_key
        from vol_predict import phase3_vol
        sl = SP.slate_rows(season, week)
        ids = {name_key(r.player): r.player_id for r in sl.drop_duplicates("player_id").itertuples()}
        on = phase3_vol(season, week).set_index("player_id")
        off = phase3_vol(season, week, weather="off").set_index("player_id")
        out = {}
        for k, pid in ids.items():
            for st in ("passing_yards", "rushing_yards", "receiving_yards", "receptions"):
                c = f"p3_{st}"
                if pid in on.index and pid in off.index and c in on.columns:
                    out[(k, st)] = float(on.at[pid, c] - off.at[pid, c])
        wxd_cache[(season, week)] = out
        return out

    changed = same = failed = 0
    for f in files:
        html = f.read_text(encoding="utf-8")
        got, missing = parts(html)
        if got is None:
            print(f"SKIP  {f.name}: could not find {missing} -- left untouched")
            failed += 1
            continue

        # The template now draws the 50% window (WINDOW_DISPLAY). A saved report holds the 68% lo/hi,
        # so convert once, from the saved sd, and mark the rows (w = 50). sd and every probability stay.
        rows = json.loads(got["data"])
        if rows and not all(r.get("w") == 50 for r in rows):
            sys.path.insert(0, str(ROOT))
            from nflprops import engine_source as ES
            a, b = ES.sd_scale_display(), ES.sd_scale()
            for r in rows:
                half = r["sd"] * a[r["s"]] / b[r["s"]]
                r["lo"], r["hi"], r["w"] = round(max(0.0, r["mu"] - half), 1), round(r["mu"] + half, 1), 50
            got = dict(got, data=json.dumps(rows))
        # Kickoff weather line and the "wx" marks (only for the week named by --only; other files keep what they had).
        if only and f.name.startswith(f"report_{only[0]}_w{only[1]}_"):
            sys.path.insert(0, str(ROOT))
            from nflprops.weather import game_summary
            from nflprops.odds import name_key
            meta = json.loads(got["meta"])
            m = re.match(r"report_(\d{4})_w(\d+)_([A-Z]+)-([A-Z]+)\.html", f.name)
            meta["wx"] = game_summary(f"{m.group(1)}_{int(m.group(2)):02d}_{m.group(3)}_{m.group(4)}")
            got = dict(got, meta=json.dumps(meta))
            d = wx_deltas(only[0], only[1])
            rows = json.loads(got["data"])
            for r in rows:
                r["wxd"] = round(d.get((name_key(r["p"]), r["s"]), 0.0), 1)
            got = dict(got, data=json.dumps(rows))
        # Team records in the header, for the week named by --week or --only. Other weeks keep their header.
        wk_sel = chips[4] if chips else only
        fm = re.match(r"report_(\d{4})_w(\d+)_([A-Z]+)-([A-Z]+)\.html", f.name)
        if wk_sel and fm and (int(fm.group(1)), int(fm.group(2))) == tuple(wk_sel):
            sys.path.insert(0, str(ROOT))
            from nflprops.records import headline
            plain = re.sub(r' <span class="rec"[^>]*>.*?</span>', "", got["headline"])
            hm = re.match(r'<h1>(.*?) <span class="at">at</span> (.*?)</h1>$', plain, re.S)
            if hm:
                got = dict(got, headline=headline(hm.group(1), hm.group(2), int(fm.group(1)), int(fm.group(2)),
                                                  fm.group(3), fm.group(4)))
        if chips:
            rows = json.loads(got["data"])
            for r in rows:
                rk = chips[0].get((chips[1](r["p"]), r["role"][:2]))
                r["rk"], r["rkt"] = (rk if rk else (None, None))
                bdata, pb, (b_sea, b_wk) = chips[2], chips[3], chips[4]
                r["bd"] = pb(f"{chips[1](r['p'])}|{r['role'][:2]}", r.get("vs"),
                             b_wk if b_sea == bdata["season"] else None, bdata)
                r["dr"] = (card_line(bdata, r.get("vs"), r["role"][:2])
                           if b_sea == bdata["season"] and b_wk == bdata["slate_week"] else None)
            got = dict(got, data=json.dumps(rows))
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
