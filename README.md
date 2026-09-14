# nfl-props

An NFL player-prop experiment that failed twice, and a record of how each
failure was found.

> **STATUS: the headline result was wrong.** An earlier version of this README
> claimed a validated edge from ranking props by a player's historical clear
> rate. That claim came from a test with a bug in it. When the test is fixed
> the effect disappears entirely. Scheduled reports are disabled. Details in
> [The second failure](#the-second-failure-and-how-it-was-found) below.

---

## What it does

Ranks FanDuel alternate-line player props by how *reliably* they hit, then
posts the best legs to Slack an hour before kickoff on a `launchd` schedule
(Thursday, Sunday, SNF, MNF).

The premise: at a fixed price the market has already equalised its own view, so
two props at −250 are both ~70% by construction. The question worth answering
isn't "which is mispriced" — it's "which one actually holds up."

## The second failure, and how it was found

The idea was simple: if a player has beaten a number in most of his recent
games, he should be more likely to beat it again. A test across 777 past bets
seemed to confirm it — strong track records hit about 3 in 4, weak ones about 1
in 2.

**The test was broken.** When it looked up "how often has this player beaten
this number," it wasn't using the number from the bet it was scoring. The
evidence is visible in the stored data: Justin Jefferson's *over 49.5 yards*
and *over 59.5 yards* were both recorded with the same track record. He clearly
beats 49.5 more often than 59.5, so the same figure for both is impossible.

87% of the checked rows carried a wrong value.

Recomputed correctly, across the same games:

| track record at that number | bets | actually won |
|---|---|---|
| weakest third | 252 | 61.9% |
| middle third | 233 | 61.4% |
| strongest third | 241 | 61.0% |

**No difference.** A strong track record at a number says nothing about whether
the next bet wins. The ranking does not rank.

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

## Screens — status uncertain

These were added on top of the ranking and each looked justified at the time:
a 55% snap floor, excluding players listed Questionable, a penalty for players
on new teams, a minimum amount of history, and a "stars only" mode.

**The snap-floor and stars findings were measured using the same broken data
and should be assumed wrong until re-tested.** The Questionable finding came
from a separate dataset and may survive, but has not been re-checked.

## What still holds

One finding survives, and it is the useful one:

**These slider bets are expensive.** Across 777 of them, the prices implied a
70.5% chance of winning and they actually won 62.4% of the time. That gap is
roughly double what the standard prop market charges. This was measured from
prices and outcomes only — it doesn't depend on the broken track-record
calculation.

Everything else in the "what works" column is withdrawn pending a rebuild of
the test harness.

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
