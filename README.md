# nfl-props

A reliability ranker for NFL player props, built and validated in a day.

It does **not** try to beat the sportsbook. It tried, failed, proved it failed,
and became something else.

---

## What it does

Ranks FanDuel alternate-line player props by how *reliably* they hit, then
posts the best legs to Slack an hour before kickoff on a `launchd` schedule
(Thursday, Sunday, SNF, MNF).

The premise: at a fixed price the market has already equalised its own view, so
two props at −250 are both ~70% by construction. The question worth answering
isn't "which is mispriced" — it's "which one actually holds up."

## The result

At a fixed price, a player's **historical clear rate** separates hits from
misses by **24.7 points**:

| clear rate | n | actually hit |
|---|---|---|
| 9–59% | 174 | 48.3% |
| 59–75% | 161 | 61.5% |
| 75–100% | 137 | **73.0%** |

`n=777` historical FanDuel alternate legs, 2025 W6/W10/W14, priced −180 to
−320, each joined to the real outcome. Clear rates computed from prior weeks
only, enforced in SQL.

Historical clear rate ranks outcomes at **AUC 0.628**; the price itself manages
**0.534**.

## What failed first

The original plan was to out-predict the market with a projection model. It
beat a naive career average by 6–11% MAE — and that turned out to be
irrelevant, because the market is far better than a career average.

Two independent tests killed it:

| test | result |
|---|---|
| ROI vs the closing line (n=599) | **−8.0%** |
| closing-line value (n=315) | **−0.006 pts net** |

FanDuel's line beat the model's point estimate on *every single stat*. That
code is still in the tree, unused, because the negative result is the most
useful thing in the repo.

## Five artifacts that looked like edge and weren't

Each of these produced an encouraging number that dissolved on inspection:

1. **Backup QBs** — career averages full of starts against recent DNPs, so the
   model screamed UNDER. Correct, and worthless: no book hangs a line on a QB
   who won't take a snap.
2. **Deep-bench receivers** — scoring the model on players projected for 8
   yards, where no market exists. This alone made receiving props look 12%
   *worse* than a career average when they were actually 6% better.
3. **Live in-game odds** — joining pre-game projections to in-play lines. A
   receiver sitting on 80 yards carries a 106.5 full-game line, which reads as
   a 48-point edge.
4. **Unexpected DNPs** — "UNDER hits 54.8%" collapsed to exactly 52.4%
   (break-even) once players below 50% snaps were excluded.
5. **An uncalibrated survival estimate** — treating a ranking score as a
   probability and compounding it over 12 parlay legs, producing a claimed
   5.39x EV. No one gets 5x their money from a sportsbook in expectation.

The common thread: **every one came from scoring situations the market would
never price.** Guarding against that is most of what this codebase does.

## Screens, each validated against outcomes

| screen | why |
|---|---|
| 55% snap floor | below it, top-tercile legs hit 70.1%; above, ~77% |
| Questionable excluded | they hit 8 points below their own price |
| new-team penalty | a clear rate earned elsewhere describes a role he may not have |
| 6-game history minimum | thin history is not evidence |
| stars mode (default) | costs ~nothing — high bars hit the same 74.6% as low bars |

## Known limits

- **The alternate-line band holds ~8 points** (prices imply 70.5%, reality is
  62.4%) — nearly double the standard prop market. These slider bets are where
  the book takes its margin.
- **Early season is the weak spot.** Week 1 clear rates describe *last*
  season's roles, and roughly a third of rostered players change teams. The
  validation ran on Weeks 6/10/14, where history is current.
- **The ranking is validated, profitability is not.** Top-tercile ROI came in
  at +4.9% with t=1.19 — not significant. Pick better legs with confidence;
  don't treat it as a money printer.
- Thresholds were chosen after looking at the validation weeks. Live results
  are the real out-of-sample test.

## Layout

```
nflprops/     ingest, leakage-safe features, projection (retired), odds,
              reliability ranking, Slack
scripts/      build_db, check_db, score, validate_*, run_report
launchd/      scheduled report agents
```

```bash
python scripts/build_db.py        # nflverse mirror, 2019-present, ~7s
python scripts/check_db.py        # sanity checks incl. history depth
python scripts/run_report.py snf  # post a slate report
```

Needs `ODDS_API_KEY` and `SLACK_WEBHOOK_URL` in `.env` (see `.env.example`).

## A note on the numbers

`scripts/backtest.py` is quarantined: it printed per-stat figures that could
not be reproduced from the frames its own function returned. `scripts/score.py`
replaced it. Any figure in this README comes from the latter.

Built for entertainment. It is not investment advice, and the expected value of
a 12-leg parlay is firmly negative no matter how good the legs look.
