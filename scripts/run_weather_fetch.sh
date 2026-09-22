#!/bin/zsh
# Hourly forecast capture for the prediction engine.
#
# Order matters. The spine is rebuilt FIRST, because fetch_weather.py reads
# kickoff times from it. A flexed or delayed game that never reaches the spine
# leaves the fetcher aiming at a stale hour, and a forecast stored for the wrong
# hour is worse than no forecast at all.
#
# The fetch itself only acts on games kicking off 2.5-3.5h out. Every other run
# is a no-op by design, which is why hourly is safe and cheap.

set -u
ROOT="$HOME/Code/nfl-props"
PY="$ROOT/.venv/bin/python"
LOG="$ROOT/logs/weather-fetch.log"

echo "=== $(date -u '+%Y-%m-%d %H:%M:%S') UTC ===" >> "$LOG"
"$PY" "$ROOT/scripts/build_game_spine.py" >> "$LOG" 2>&1 || echo "SPINE REBUILD FAILED" >> "$LOG"
"$PY" "$ROOT/scripts/fetch_weather.py" forecast >> "$LOG" 2>&1 || echo "FORECAST FETCH FAILED" >> "$LOG"

# keep the log from growing without bound
tail -n 2000 "$LOG" > "$LOG.tmp" && mv "$LOG.tmp" "$LOG"
