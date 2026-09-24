"""Project-wide constants. Single source of truth for paths and scope."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
REPORT_DIR = ROOT / "reports"
DB_PATH = DATA_DIR / "nflprops.db"

# History depth. 2019+ keeps us in the modern passing era while still giving
# ~6 full seasons of efficiency priors.
FIRST_SEASON = 2019
CURRENT_SEASON = 2026

# Only offensive skill positions produce the props we bet.
SKILL_POSITIONS = ("QB", "RB", "WR", "TE", "FB")

# Columns kept from load_player_stats. The full table is 150 wide and mostly
# kicking/defense/special-teams noise we never price.
PLAYER_GAME_COLS = [
    "player_id", "player_display_name", "position", "position_group",
    "season", "week", "season_type", "game_id", "team", "opponent_team",
    # passing
    "completions", "attempts", "passing_yards", "passing_tds",
    "passing_interceptions", "sacks_suffered", "passing_air_yards",
    "passing_epa", "passing_cpoe",
    # rushing
    "carries", "rushing_yards", "rushing_tds", "rushing_first_downs",
    "rushing_epa",
    # receiving
    "receptions", "targets", "receiving_yards", "receiving_tds",
    "receiving_air_yards", "receiving_yards_after_catch", "receiving_epa",
    "racr", "target_share", "air_yards_share", "wopr",
    "fantasy_points_ppr",
]


# --- secrets -----------------------------------------------------------------
# Loaded from the project-root .env (gitignored, chmod 600). Keys are read
# through require() rather than at import time so that the data and modeling
# layers stay importable on a machine that has no credentials at all.
import os
from dotenv import load_dotenv

load_dotenv(ROOT / ".env")


def require(name: str) -> str:
    val = os.getenv(name, "").strip()
    if not val:
        raise RuntimeError(
            f"{name} is not set. Add it to {ROOT / '.env'} (see .env.example)."
        )
    return val


def get(name: str, default=None):
    return os.getenv(name, "").strip() or default


# --- raw board capture ---------------------------------------------------
# decision #37 (2026-09-24): every run that fetches the sportsbook's board
# saves what it got, unmodified, before any filtering -- so a past report's
# selection logic (e.g. decision #36's QB1 rule) can be replayed later
# against what was actually on the board, not reconstructed from the
# already-filtered rows a report kept. Zero extra API calls: this only
# writes down a response that was already fetched.
def save_board(season, week, teams, payload) -> None:
    """Write raw Odds API response(s) to data/boards/, timestamped so a call
    never overwrites an earlier one. Never raises -- a failed save prints a
    warning and lets the caller's report still render."""
    import json as _json
    from datetime import datetime, timezone
    try:
        boards_dir = DATA_DIR / "boards"
        boards_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        name = f"{season}_w{week}_{'-'.join(teams)}_{ts}.json"
        (boards_dir / name).write_text(_json.dumps(payload))
    except Exception as e:
        print(f"  ! could not save raw board ({season} W{week} {'-'.join(teams)}): {e}")
