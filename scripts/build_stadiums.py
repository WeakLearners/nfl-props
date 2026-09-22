"""
Job 3 (Phase 2A) — build the `stadiums` table in features.db.

Reads `schedules` from the live database (read-only) to find every stadium_id
that actually appears, then attaches static geo/roof/surface/timezone facts.

Roof/surface are not guessed from scratch: for each stadium_id we take the
MODAL non-null `roof` / `surface` value recorded in `schedules` itself, and
only fall back to real-world knowledge for stadium_ids that have zero non-null
records (MAD01, RIO00 — both one-off international venues as of 2026).

Decision #3 is non-negotiable: dome / closed-roof is a STATE ("no weather"),
never a NULL. Every row in this table gets a non-null roof_type.

Read-only against nflprops.db. Writes only the `stadiums` table in features.db.
"""

import collections
import sqlite3
from pathlib import Path

LIVE_DB = Path.home() / "Code/nfl-props/data/nflprops.db"
FEATURES_DB = Path.home() / "Code/nfl-props/data/features.db"

# Static facts per stadium_id. Coordinates to 3 decimals, IANA tz names.
# roof_type/surface here are FALLBACKS only, used when schedules has no
# non-null recorded value for that stadium_id (MAD01, RIO00 today).
STADIUM_FACTS = {
    "ATL97": dict(lat=33.755, lon=-84.401, tz="America/New_York"),   # Mercedes-Benz Stadium, Atlanta
    "BAL00": dict(lat=39.278, lon=-76.623, tz="America/New_York"),   # M&T Bank Stadium, Baltimore
    "BOS00": dict(lat=42.091, lon=-71.264, tz="America/New_York"),   # Gillette Stadium, Foxborough
    "BUF00": dict(lat=42.774, lon=-78.787, tz="America/New_York"),   # Highmark Stadium, Orchard Park
    "CAR00": dict(lat=35.226, lon=-80.853, tz="America/New_York"),   # Bank of America Stadium, Charlotte
    "CHI98": dict(lat=41.862, lon=-87.617, tz="America/Chicago"),    # Soldier Field, Chicago
    "CIN00": dict(lat=39.095, lon=-84.516, tz="America/New_York"),   # Paycor Stadium, Cincinnati
    "CLE00": dict(lat=41.506, lon=-81.700, tz="America/New_York"),   # Huntington Bank Field, Cleveland
    "DAL00": dict(lat=32.747, lon=-97.093, tz="America/Chicago"),    # AT&T Stadium, Arlington
    "DEN00": dict(lat=39.744, lon=-105.020, tz="America/Denver"),    # Empower Field at Mile High, Denver
    "DET00": dict(lat=42.340, lon=-83.046, tz="America/New_York"),   # Ford Field, Detroit (Eastern)
    "FRA00": dict(lat=50.069, lon=8.646, tz="Europe/Berlin"),        # Deutsche Bank Park, Frankfurt
    "GER00": dict(lat=48.219, lon=11.625, tz="Europe/Berlin", surface="grass"),  # Allianz Arena, Munich
    "GNB00": dict(lat=44.501, lon=-88.062, tz="America/Chicago"),    # Lambeau Field, Green Bay
    "HOU00": dict(lat=29.685, lon=-95.411, tz="America/Chicago"),    # NRG Stadium, Houston
    "IND00": dict(lat=39.760, lon=-86.164, tz="America/New_York"),   # Lucas Oil Stadium, Indianapolis (Eastern)
    "JAX00": dict(lat=30.324, lon=-81.637, tz="America/New_York"),   # EverBank Stadium, Jacksonville
    "KAN00": dict(lat=39.049, lon=-94.484, tz="America/Chicago"),    # Arrowhead Stadium, Kansas City
    "LAX01": dict(lat=33.953, lon=-118.339, tz="America/Los_Angeles"),  # SoFi Stadium, Inglewood
    "LAX97": dict(lat=33.864, lon=-118.261, tz="America/Los_Angeles"),  # Dignity Health Sports Park, Carson
    "LAX99": dict(lat=34.014, lon=-118.288, tz="America/Los_Angeles"),  # LA Memorial Coliseum
    "LON00": dict(lat=51.556, lon=-0.280, tz="Europe/London"),       # Wembley Stadium, London
    "LON02": dict(lat=51.604, lon=-0.066, tz="Europe/London"),       # Tottenham Hotspur Stadium, London
    "MAD01": dict(lat=40.453, lon=-3.688, tz="Europe/Madrid",
                  roof_type="retractable", surface="grass"),         # Santiago Bernabeu, Madrid (retractable roof since 2024)
    "MEL00": dict(lat=-37.820, lon=144.983, tz="Australia/Melbourne"),  # Melbourne Cricket Ground
    "MEX00": dict(lat=19.303, lon=-99.150, tz="America/Mexico_City"),   # Estadio Azteca, Mexico City
    "MIA00": dict(lat=25.958, lon=-80.239, tz="America/New_York"),   # Hard Rock Stadium, Miami Gardens
    "MIN01": dict(lat=44.974, lon=-93.258, tz="America/Chicago"),    # U.S. Bank Stadium, Minneapolis
    "MUN01": dict(lat=48.219, lon=11.625, tz="Europe/Berlin"),       # Allianz Arena, Munich (see GER00 note)
    "NAS00": dict(lat=36.166, lon=-86.771, tz="America/Chicago"),    # Nissan Stadium, Nashville
    "NOR00": dict(lat=29.951, lon=-90.081, tz="America/Chicago"),    # Caesars Superdome, New Orleans
    "NYC01": dict(lat=40.813, lon=-74.074, tz="America/New_York"),   # MetLife Stadium, East Rutherford
    "OAK00": dict(lat=37.752, lon=-122.201, tz="America/Los_Angeles"),  # RingCentral Coliseum, Oakland
    "PAR00": dict(lat=48.924, lon=2.360, tz="Europe/Paris"),         # Stade de France, Paris
    "PHI00": dict(lat=39.901, lon=-75.167, tz="America/New_York"),   # Lincoln Financial Field, Philadelphia
    "PHO00": dict(lat=33.528, lon=-112.263, tz="America/Phoenix"),   # State Farm Stadium, Glendale
    "PIT00": dict(lat=40.447, lon=-80.016, tz="America/New_York"),   # Acrisure Stadium, Pittsburgh
    "RIO00": dict(lat=-22.912, lon=-43.230, tz="America/Sao_Paulo",
                  roof_type="outdoor", surface="grass"),             # Maracana, Rio de Janeiro (open-air)
    "SAO00": dict(lat=-23.545, lon=-46.474, tz="America/Sao_Paulo", surface="grass"),  # Neo Quimica Arena, Sao Paulo
    "SEA00": dict(lat=47.595, lon=-122.332, tz="America/Los_Angeles"),  # Lumen Field, Seattle
    "SFO01": dict(lat=37.403, lon=-121.970, tz="America/Los_Angeles"),  # Levi's Stadium, Santa Clara
    "TAM00": dict(lat=27.976, lon=-82.503, tz="America/New_York"),   # Raymond James Stadium, Tampa
    "VEG00": dict(lat=36.091, lon=-115.184, tz="America/Los_Angeles"),  # Allegiant Stadium, Las Vegas
    "WAS00": dict(lat=38.908, lon=-76.864, tz="America/New_York"),   # Commanders Field, Landover
}

ROOF_MAP = {
    "outdoors": "outdoor",
    "open": "retractable",   # a retractable-roof venue recorded open on that date
    "closed": "retractable", # a retractable-roof venue recorded closed on that date
    "dome": "dome",
}


def modal_roof_and_surface(con):
    """Per stadium_id, the modal non-null roof/surface recorded in schedules."""
    cur = con.execute("SELECT stadium_id, roof, surface FROM schedules")
    roof_counts = collections.defaultdict(collections.Counter)
    surface_counts = collections.defaultdict(collections.Counter)
    for stadium_id, roof, surface in cur.fetchall():
        if roof:
            roof_counts[stadium_id][roof] += 1
        if surface and surface.strip():
            surface_counts[stadium_id][surface.strip().lower()] += 1
    return roof_counts, surface_counts


def resolve_roof_type(stadium_id, roof_counter):
    """
    A stadium that shows BOTH 'open' and 'closed' in the data is a retractable
    roof used both ways. A stadium showing only one of those is still
    retractable-capable (ATL97/DAL00/HOU00/IND00/PHO00 all qualify by real-world
    architecture too) so 'open' or 'closed' alone still maps to retractable.
    """
    if not roof_counter:
        return None
    top_value, _ = roof_counter.most_common(1)[0]
    return ROOF_MAP[top_value]


def main():
    con_live = sqlite3.connect(f"file:{LIVE_DB}?mode=ro", uri=True)

    # stadium_id -> (modal stadium name, modal "home_team" as recorded — this
    # is the game's home team, which rotates for international "home games"
    # and is not a fixed property of the venue; kept only for human reference).
    cur = con_live.execute(
        "SELECT stadium_id, stadium, home_team FROM schedules WHERE stadium_id IS NOT NULL"
    )
    name_counts = collections.defaultdict(collections.Counter)
    team_counts = collections.defaultdict(collections.Counter)
    for stadium_id, stadium_name, home_team in cur.fetchall():
        if stadium_name:
            name_counts[stadium_id][stadium_name] += 1
        if home_team:
            team_counts[stadium_id][home_team] += 1
    venues = [
        (sid, name_counts[sid].most_common(1)[0][0], team_counts[sid].most_common(1)[0][0])
        for sid in sorted(name_counts)
    ]

    roof_counts, surface_counts = modal_roof_and_surface(con_live)

    rows = []
    warnings = []
    for stadium_id, stadium_name, home_team in venues:
        facts = STADIUM_FACTS.get(stadium_id)
        if facts is None:
            warnings.append(f"NO STATIC FACTS for stadium_id={stadium_id} ({stadium_name})")
            continue

        roof_type = resolve_roof_type(stadium_id, roof_counts[stadium_id])
        if roof_type is None:
            roof_type = facts.get("roof_type")
        if roof_type is None:
            warnings.append(f"NO ROOF DATA at all for {stadium_id} ({stadium_name}) — check STADIUM_FACTS fallback")
            continue

        surface = None
        if surface_counts[stadium_id]:
            surface, _ = surface_counts[stadium_id].most_common(1)[0]
        if surface is None:
            surface = facts.get("surface")
        if surface is None:
            warnings.append(f"NO SURFACE DATA at all for {stadium_id} ({stadium_name})")
            surface = "unknown"

        rows.append((
            stadium_id,
            stadium_name,
            home_team,
            facts["lat"],
            facts["lon"],
            roof_type,
            surface,
            facts["tz"],
        ))

    con_feat = sqlite3.connect(FEATURES_DB)
    con_feat.execute("""
        CREATE TABLE IF NOT EXISTS stadiums (
            stadium_key   TEXT PRIMARY KEY,
            stadium_name  TEXT,
            home_team     TEXT,
            lat           REAL NOT NULL,
            lon           REAL NOT NULL,
            roof_type     TEXT NOT NULL CHECK (roof_type IN ('outdoor','dome','retractable')),
            surface       TEXT NOT NULL,
            tz            TEXT NOT NULL
        )
    """)
    con_feat.execute("DELETE FROM stadiums")
    con_feat.executemany(
        "INSERT INTO stadiums (stadium_key, stadium_name, home_team, lat, lon, roof_type, surface, tz) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        rows,
    )
    con_feat.commit()

    print(f"Wrote {len(rows)} stadium rows to {FEATURES_DB}")
    for w in warnings:
        print("WARNING:", w)

    # Acceptance check: every game in schedules resolves to a stadium_key with
    # a non-null roof_type, including the 2026 rows that arrive with roof=NULL.
    cur = con_live.execute("SELECT COUNT(*) FROM schedules")
    total_games = cur.fetchone()[0]
    cur = con_live.execute("SELECT DISTINCT stadium_id FROM schedules")
    all_stadium_ids = {r[0] for r in cur.fetchall()}
    resolved_ids = {r[0] for r in rows}
    unresolved = all_stadium_ids - resolved_ids
    print(f"Games in schedules: {total_games}")
    print(f"Distinct stadium_ids in schedules: {len(all_stadium_ids)}")
    print(f"Unresolved stadium_ids: {sorted(unresolved) if unresolved else 'none'}")

    cur = con_live.execute(
        "SELECT COUNT(*) FROM schedules WHERE season=2026 AND (roof IS NULL OR roof='')"
    )
    null_2026 = cur.fetchone()[0]
    print(f"2026 games with NULL/empty roof in schedules (resolved via stadiums table): {null_2026}")

    con_live.close()
    con_feat.close()


if __name__ == "__main__":
    main()
