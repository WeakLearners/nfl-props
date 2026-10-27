#!/usr/bin/env python
"""Second pass: vacated usage with the roster fix, plus a red-zone weight sweep."""
import sys, pathlib, io, contextlib
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import numpy as np, pandas as pd
from nflprops import project as P
from project_week import actuals

WEEKS = [(s, w) for s in (2023, 2024, 2025) for w in range(1, 19)]

def collect(tag, **flags):
    saved = {k: getattr(P, k) for k in flags}
    for k, v in flags.items(): setattr(P, k, v)
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
        for k, v in saved.items(): setattr(P, k, v)
    out = pd.concat(frames, ignore_index=True); out["cfg"] = tag
    print(f"  {tag} done ({len(out)} rows)", flush=True)
    return out

def calib_wf(df):
    done = []
    for test in (2024, 2025):
        tr, te = df[df.season < test], df[df.season == test].copy()
        for stat, g in te.groupby("stat"):
            t = tr[tr.stat == stat]
            if len(t) < 200: continue
            b, a = np.polyfit(t.proj, t.actual, 1)
            te.loc[g.index, "proj"] = np.clip(a + b * g.proj, 0, None)
        done.append(te)
    return pd.concat(done, ignore_index=True)

OFF = dict(USE_RZ=False, USE_VACATED=False, USE_CALIB=False)
cfgs = {
    "control":   dict(OFF),
    "vac_.55":   dict(OFF, USE_VACATED=True, VACATED_ALPHA=0.55),
    "vac_.30":   dict(OFF, USE_VACATED=True, VACATED_ALPHA=0.30),
    "vac_1.0":   dict(OFF, USE_VACATED=True, VACATED_ALPHA=1.00),
    "rz_.05":    dict(OFF, USE_RZ=True, RZ_WEIGHT=0.05),
    "rz_.15":    dict(OFF, USE_RZ=True, RZ_WEIGHT=0.15),
}
frames = {k: collect(k, **v) for k, v in cfgs.items()}

rows = []
for tag, df in frames.items():
    d = df[df.season >= 2024]
    for k, g in d.groupby(["position", "stat"]):
        rows.append({"cfg": tag, "key": " ".join(k), "mae": (g.proj - g.actual).abs().mean()})
t = pd.DataFrame(rows).pivot_table(index="key", columns="cfg", values="mae")
order = list(cfgs)
t = t[order]
for c in order[1:]:
    t[c + "_%"] = (t[c] / t["control"] - 1) * 100
pd.set_option("display.width", 260)
print("\n=== MAE, test 2024-2025 (negative % = better) ===")
print(t.round(3).to_string())

best = min(order[1:4], key=lambda c: (t[c] / t["control"]).mean())
print(f"\nbest vacated alpha: {best}")
