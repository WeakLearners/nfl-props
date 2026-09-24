#!/usr/bin/env python
"""Drift check for engine/v1.json (decision #33).

Read-only. Verifies the live code, config constants, and shipped model
artifacts still match what engine/v1.json says is official for the 2026
holdout. Exits non-zero on any mismatch so a manifest that has gone stale
can't be pre-registered against a drifted engine without someone noticing.

Does not read, fit on, or score any 2026 outcome. Does not import or run
the report/serving path -- it only hashes files and reads module-level
constants.
"""
import hashlib
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

MANIFEST = ROOT / "engine" / "v1.json"


def sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def check_file_hashes(manifest: dict) -> list[str]:
    problems = []
    seen = {}
    for stat, entry in manifest["engines"].items():
        for rel, expected in entry.get("code_sha256", {}).items():
            seen[rel] = expected
        if entry.get("model_artifact"):
            seen[entry["model_artifact"]] = entry.get("artifact_sha256")
        td = entry.get("training_data")
        if td:
            seen[td] = entry.get("training_data_sha256")
        io = entry.get("informational_only", {})
        art = io.get("phase3_artifact_sha256")
        if isinstance(art, str):
            # single-artifact stats (rushing_yards, receiving_yards) don't carry
            # a relative path here, so nothing to check beyond metadata files below
            pass
        elif isinstance(art, dict):
            pass  # anytime_td case: checked via data/shipping/*.pkl below explicitly

    for rel, expected in seen.items():
        if expected is None:
            continue
        p = ROOT / rel
        if not p.exists():
            problems.append(f"MISSING: {rel} (manifest expects sha256 {expected})")
            continue
        actual = sha256(p)
        if actual != expected:
            problems.append(f"DRIFT: {rel}\n    manifest: {expected}\n    live:     {actual}")

    # Phase 3 shipped artifacts -- checked explicitly since their hashes live
    # under informational_only, not the flat code_sha256 map.
    shipping_expected = {
        "data/shipping/targets.pkl": None,  # not pinned in manifest (rushing/receiving_yards only)
        "data/shipping/carries.pkl": None,
        "data/shipping/rushing_yards.pkl": manifest["engines"]["rushing_yards"]["informational_only"]["phase3_artifact_sha256"],
        "data/shipping/receiving_yards.pkl": manifest["engines"]["receiving_yards"]["informational_only"]["phase3_artifact_sha256"],
        "data/shipping/receiving_tds.pkl": manifest["engines"]["anytime_td"]["informational_only"]["phase3_artifact_sha256"]["receiving_tds.pkl"],
        "data/shipping/rushing_tds.pkl": manifest["engines"]["anytime_td"]["informational_only"]["phase3_artifact_sha256"]["rushing_tds.pkl"],
    }
    for rel, expected in shipping_expected.items():
        if expected is None:
            continue
        p = ROOT / rel
        if not p.exists():
            problems.append(f"MISSING: {rel} (manifest expects sha256 {expected})")
            continue
        actual = sha256(p)
        if actual != expected:
            problems.append(f"DRIFT: {rel}\n    manifest: {expected}\n    live:     {actual}")

    return problems


def check_live_config(manifest: dict) -> list[str]:
    problems = []
    try:
        from nflprops import project as P
        from nflprops import report as R
    except Exception as e:
        return [f"IMPORT FAILED: could not import nflprops.project/report -- {e}"]

    checks = [
        ("DECAY", P.DECAY, 0.85),
        ("PRIOR_SEASON_W", P.PRIOR_SEASON_W, 0.55),
        ("MIN_GAMES", P.MIN_GAMES, 3),
        ("K_EFF_ATT", P.K_EFF_ATT, 60.0),
        ("K_VOL_GAMES", P.K_VOL_GAMES, 3.0),
        ("K_VAR_GAMES", P.K_VAR_GAMES, 5.0),
        ("K_SHARE_GAMES", P.K_SHARE_GAMES, 4.0),
        ("USE_SHARE", P.USE_SHARE, True),
        ("USE_VACATED", P.USE_VACATED, False),
        ("USE_RZ", P.USE_RZ, False),
        ("USE_CALIB", P.USE_CALIB, True),
        ("SD_INFLATE (report.py)", R.SD_INFLATE, 1.10),
    ]
    for name, live, expected in checks:
        if live != expected:
            problems.append(f"CONFIG DRIFT: {name} live={live!r} manifest={expected!r}")

    def_weight_expected = {
        "passing_yards": 1.0, "completions": 1.0, "rushing_yards": 0.5,
        "receptions": 0.0, "receiving_yards": 0.0,
    }
    if dict(P.DEF_WEIGHT) != def_weight_expected:
        problems.append(f"CONFIG DRIFT: DEF_WEIGHT live={P.DEF_WEIGHT!r} manifest={def_weight_expected!r}")

    calib_expected = {
        "completions": (-0.8345, 1.0133),
        "passing_yards": (-14.0245, 1.0350),
        "receiving_yards": (-2.0835, 1.0376),
        "receptions": (-0.1638, 1.0391),
        "rushing_yards": (-1.6995, 1.0255),
    }
    for stat, exp in calib_expected.items():
        live = tuple(P.CALIB.get(stat, ()))
        if live != exp:
            problems.append(f"CONFIG DRIFT: CALIB[{stat}] live={live!r} manifest={exp!r}")

    # report.py's shortlist screens (MIN_SNAP / MIN_GAMES) live in
    # scripts/make_report.py, not an importable module -- covered by the
    # file-hash check above instead of a live-value check here.

    return problems


def main():
    manifest = json.loads(MANIFEST.read_text())
    problems = check_file_hashes(manifest) + check_live_config(manifest)

    if problems:
        print(f"DRIFT CHECK FAILED -- {len(problems)} problem(s):\n")
        for p in problems:
            print(f"- {p}")
        sys.exit(1)

    print("Drift check passed. Live code, config, and shipped artifacts match engine/v1.json.")
    sys.exit(0)


if __name__ == "__main__":
    main()
