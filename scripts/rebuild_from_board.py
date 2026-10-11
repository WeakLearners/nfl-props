#!/usr/bin/env python
"""Rebuild one report's DATA from a saved odds board, with zero API calls.

usage: rebuild_from_board.py SEASON WEEK TEAM_A,TEAM_B BOARD_JSON
Keeps the report's page, meta and title; replaces only the DATA rows. Used when a
game has already kicked off, so make_report.py can no longer fetch its lines.
"""
import json, re, sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from nflprops import report as RP
parse_lines = RP.parse_lines

season, week, teams, board = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3].split(","), sys.argv[4]
lines = parse_lines(json.load(open(board)))
rows, short, _ = RP.build(season, week, teams, lines)
f = RP.report_path(season, week, *teams)
html = f.read_text()
new, n = re.subn(r"const DATA = \[.*?\];\n", lambda m: "const DATA = " + json.dumps(rows) + ";\n", html, count=1, flags=re.S)
assert n == 1
f.write_text(new)
print(f.name, len(rows), "rows,", int(short.shape[0]), "picks")
