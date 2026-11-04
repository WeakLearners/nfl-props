#!/bin/zsh
# Tuesday 09:00 job (launchd com.nflprops.week). Results are final by now.
# Order matters:
#   1 grade_game      grade the finished games
#   2 fit_sd_scale    refit both window scales (50% shown, 68% basis) from the finished weeks
#   3 grade_week      week write-up and model_quality.csv, with the fresh scales
#   4 fetch_game_weather   forecast for the coming week
#   5 game_lines_week      freeze the coming week's game lines (before its first kickoff)
# A step that fails does not stop the ones after it; its error goes to the log.
ROOT="$HOME/Code/nfl-props"
PY="$ROOT/.venv/bin/python"
cd "$ROOT" || exit 1
for step in "scripts/grade_game.py --no-refresh" "scripts/fit_sd_scale.py" "scripts/grade_week.py" \
            "scripts/fetch_game_weather.py" "scripts/game_lines_week.py"; do
  echo "=== $(date -u '+%Y-%m-%d %H:%M:%S') UTC  $step"
  "$PY" ${=step} || echo "STEP FAILED: $step"
done
