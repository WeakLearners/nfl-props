#!/usr/bin/env python
"""Post the market-validation verdict to Slack."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from nflprops.slack import post, section, header, divider

Q1 = """```
stat                n   model    line   winner
completions        71    4.54    4.33   LINE
passing_yards      72   51.62   47.12   LINE
receiving_yards   254   26.19   24.53   LINE
receptions        262    1.64    1.58   LINE
rushing_yards     100   26.09   25.15   LINE
ALL               759   18.09   16.94   LINE
```"""

Q2 = """```
bets          599
hit rate     48.7%   (break-even ~52.4%)
ROI          -8.0%
total       -47.8 units
```"""

PARLAY = """```
legs    parlay EV
  4        -28%
  8        -49%
 12        -63%
```"""

blocks = [
    header("NFL Props — STOP. Validated against the real market."),
    section("*The model loses to FanDuel.* Phase 3 is done and it kills the "
            "project as designed. I've halted before phase 4."),
    divider(),
    section("*Method:* real FanDuel props from The Odds API historical endpoint, "
            "snapshotted 15 min before the 1pm kickoffs on three 2025 Sundays "
            "(W6, W10, W14). Joined to the projection the model would have made "
            "using only prior data, and to what actually happened.\n"
            "*759 props.* Cost 1,966 credits of 20,000."),
    section("*Q1 — whose number is closer to reality?* (MAE, lower is better)"),
    section(Q1),
    section("FanDuel's line beat the model on *every single stat.*"),
    divider(),
    section("*Q2 — what happens if you actually bet it?* Every claimed edge ≥3%, "
            "priced at the real offered odds:"),
    section(Q2),
    section("z = −1.79 vs break-even, and it agrees with Q1, which was measured "
            "independently.\n\n_Rushing yards came in at +6.1% ROI — *ignore it.* "
            "n=82, z=+0.67. That's noise, and chasing it is exactly the mistake "
            "this test was built to catch._"),
    divider(),
    section("*Why phase 2 looked good and wasn't.*\n"
            "The +6–11% edge over a career average was real and irrelevant. "
            "FanDuel is far better than a career average. My pseudo-line had "
            "Mahomes at 280.3 passing yards; FanDuel hung 223.5. I was beating a "
            "strawman."),
    section("*Why this is worst for the parlay design specifically.*\n"
            "Parlay EV is the product of per-leg EVs. At −8.0% a leg:"),
    section(PARLAY),
    section("A $1 twelve-leg ticket returns about *37 cents* in expectation. The "
            "leverage that made parlays attractive works just as hard in reverse.\n\n"
            "Against the stated bar — beat $30/month in profit — this would lose "
            "money at a steady clip."),
    divider(),
    section("*Three options:*\n"
            "1. *Kill it.* Clean answer to an interesting question for ~$30 and a Sunday.\n"
            "2. *Rebuild around the market as the prior* — only deviate where the model "
            "has info the line hasn't absorbed. Honest, but shrinks the bettable "
            "universe to near nothing, and no promise it clears zero.\n"
            "3. *Chase a real informational edge* — injury/role news before it's priced, "
            "or same-game-parlay correlation FanDuel underprices. A different project "
            "from better regression on public box-score data, which the market already "
            "does very well."),
    section("Full writeup: `projects/nfl-props/market-validation.md`"),
]

print(post(blocks=blocks, text="NFL Props — STOP. Model loses to FanDuel (-8.0% ROI)."))
