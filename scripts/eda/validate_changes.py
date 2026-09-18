#!/usr/bin/env python
"""Walk-forward check on the four 2026-09-17 changes, one at a time.

Each configuration re-projects every week of 2023-2025 using only data from
before that week, then scores against what actually happened. Calibration is
fit per test season on earlier seasons only, so a season is never corrected
using itself.
"""
import sys, pathlib, io, contextlib
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import numpy as np, pandas as pd
from nflprops import project as P
from project_week import actuals

WEEKS = [(s, w) for s in (2023, 2024, 2025) for w in range(1, 19)]


def collect(tag, **flags):
    saved = {k: getattr(P, k) for k in flags}
    for k, v in flags.items():
        setattr(P, k, v)
    frames = []
    try:
        for s, w in WEEKS:
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    pr = P.project_week(s, w)
                df = pr.merge(actuals(s, w), on=["player_id", "stat"])
                df["season"], df["week"] = s, w
                frames.append(df)
            except Exception as e:
                print(f"  skip {s} W{w}: {e}", file=sys.stderr)
    finally:
        for k, v in saved.items():
            setattr(P, k, v)
    out = pd.concat(frames, ignore_index=True)
    out["cfg"] = tag
    return out


def calibrate_walk_forward(df):
    """Fit actual ~ a + b*proj per stat on earlier seasons, apply to the later
    one. Never fits on the season it corrects."""
    done = []
    for test in (2024, 2025):
        tr, te = df[df.season < test], df[df.season == test].copy()
        for stat, g in te.groupby("stat"):
            t = tr[tr.stat == stat]
            if len(t) < 200:
                continue
            b, a = np.polyfit(t.proj, t.actual, 1)
            te.loc[g.index, "proj"] = np.clip(a + b * g.proj, 0, None)
        done.append(te)
    return pd.concat(done, ignore_index=True)


def mae_table(frames, key_pos=True):
    rows = []
    for tag, df in frames.items():
        d = df[df.season >= 2024]           # common window across configs
        grp = ["position", "stat"] if key_pos else ["stat"]
        for k, g in d.groupby(grp):
            rows.append({"cfg": tag, "key": " ".join(k) if isinstance(k, tuple) else k,
                         "n": len(g), "mae": (g.proj - g.actual).abs().mean()})
    t = pd.DataFrame(rows).pivot_table(index=["key"], columns="cfg", values="mae")
    n = pd.DataFrame(rows).groupby("key").n.max()
    return t.join(n)


if __name__ == "__main__":
    print("A: control (new features off)")
    A = collect("A_control", USE_RZ=False, USE_VACATED=False, USE_CALIB=False)
    print("B: vacated usage on")
    B = collect("B_vacated", USE_RZ=False, USE_VACATED=True, USE_CALIB=False)
    print("C: red zone on")
    C = collect("C_redzone", USE_RZ=True, USE_VACATED=False, USE_CALIB=False)
    print("D: both on")
    D = collect("D_both", USE_RZ=True, USE_VACATED=True, USE_CALIB=False)
    print("E: both + walk-forward calibration")
    E = calibrate_walk_forward(D.copy()); E["cfg"] = "E_calib"

    frames = {"A_control": A, "B_vacated": B, "C_redzone": C, "D_both": D, "E_calib": E}
    for f in frames.values():
        f.to_csv(f"reports/validate_{f.cfg.iloc[0]}.csv", index=False)

    t = mae_table(frames)
    cols = [c for c in ["A_control", "B_vacated", "C_redzone", "D_both", "E_calib"] if c in t]
    t = t[cols + ["n"]]
    for c in cols[1:]:
        t[c + "_%"] = (t[c] / t["A_control"] - 1) * 100
    pd.set_option("display.width", 250)
    print("\n=== MAE by position/stat, test seasons 2024-2025 (lower better) ===")
    print(t.round(3).to_string())

    # production coefficients: fit on everything available
    print("\n=== calibration coefficients (fit on 2023-2025, for CALIB) ===")
    print("CALIB_FIT_ON = \"2023-2025 walk-forward, fitted 2026-09-17\"")
    print("CALIB = {")
    for stat, g in D.groupby("stat"):
        b, a = np.polyfit(g.proj, g.actual, 1)
        print(f'    "{stat}": ({a:.4f}, {b:.4f}),')
    print("}")
