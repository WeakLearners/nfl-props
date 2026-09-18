# The Jefferson Row

*A postmortem on an NFL player-prop model, September 2026.*

I built a model that predicted NFL player performance, found an edge worth 24.7
points, shipped it, and then discovered the test that proved it was comparing
against the wrong number. This is what the wreckage looked like.

---

## What it was supposed to do

The idea was ordinary enough. FanDuel posts a number for each player — Justin
Jefferson over 49.5 receiving yards, say — and attaches a price. If I could
predict the player's range better than the book could, I could find the ones
priced wrong.

Early results were encouraging. Walking forward through 2023–2025 and
re-projecting every week using only prior data, the model beat a player's plain
career average by 6–11% across all five statistics, over 18,840 props. That part
held up. It still does.

Then came the finding that seemed to make it a business. Across 777 historical
bets, a player's track record *at that specific number* appeared to predict
whether the bet won. The strongest third hit around three in four. The weakest,
about one in two. A spread of 24.7 points.

That result drove everything downstream: the tool that picked which legs to bet,
the scheduled Slack reports, and the first version of the public README.

## The row that had been sitting there the whole time

It did not come apart in review. It came apart because I asked for a different
way of selecting legs, ran the new method against the old one, and got answers
that disagreed. Chasing the disagreement led back to the stored validation file.

There, two rows. Justin Jefferson, over 49.5 yards. Justin Jefferson, over 59.5
yards. Both recorded with the **same clear rate**.

> A player clears 49.5 yards more often than he clears 59.5. One figure cannot
> describe both. The number was impossible, it was printed in my own file, and
> every check I had run looked straight past it.

The validation was looking up each player's record against a number that was not
the one being scored. Recomputing 259 rows from scratch, **225 of them — 87% —
disagreed** with what the file said.

Recomputed correctly across the same games:

| Track record at that number | Bets | Won |
|---|---:|---:|
| Weakest third | 252 | 61.9% |
| Middle third | 233 | 61.4% |
| Strongest third | 241 | 61.0% |

The claimed 24.7-point spread is 0.9 points, and it runs the wrong way. The
ranking does not rank.

## Two more tests, arriving independently at the same place

With the headline claim dead, the honest question remained: does the model
out-predict FanDuel at all? I pulled real prop lines from three sampled 2025
Sundays, snapshotted at 12:45pm Eastern, just before the 1pm kickoffs. 759 props
matched to both a real line and an actual result.

**Mean absolute error — model vs. the posted line.** Lower is better.

| Statistic | n | Model | Line | Closer |
|---|---:|---:|---:|---|
| Completions | 71 | 4.54 | 4.33 | Line |
| Passing yards | 72 | 51.62 | 47.12 | Line |
| Receiving yards | 254 | 26.19 | 24.53 | Line |
| Receptions | 262 | 1.64 | 1.58 | Line |
| Rushing yards | 100 | 26.09 | 25.15 | Line |
| **All** | **759** | **18.09** | **16.94** | **Line** |

FanDuel is closer to reality on every single statistic.

Betting it confirmed the same thing from the other direction. Taking every prop
where the model claimed at least a 3% edge, priced at the real offered odds:
**599 bets, 48.7% won.** Break-even is about 52.4%. That is a return of
**−8.0%** per dollar, or −47.8 units.

Phase 2 had shown the model beating a career average by 6–11%. Both results are
true, and the second one explains why the first one didn't matter. FanDuel is
not using a career average. It employs people to do this full time, and it was
better than my model on every stat I measured.

### Why this is fatal for parlays specifically

The whole point of the project was small same-game parlays. A parlay multiplies
the edge of each leg together, which is wonderful when the edge is positive and
merciless when it isn't.

| Legs | Parlay EV |
|---:|---:|
| 4 | −28% |
| 8 | −49% |
| 12 | −63% |

A $1 twelve-leg ticket returns about 37 cents in expectation.

### The closing-line test, and a statistic that lies

Professionals judge a model by closing-line value — whether the market moves
toward your side after you bet. It converges much faster than win and loss. I
picked sides at the Thursday line and graded them against Sunday's pre-kickoff
line. 315 matched props.

The model beat the close **53.3%** of the time. That looks like a small edge. It
isn't, and the reason is worth sitting with.

| Outcome | n | Avg. line move |
|---|---:|---:|
| Beat the close | 168 | +1.88 |
| Lost the close | 147 | −2.16 |
| **Net movement** | **315** | **−0.006** |

Won the coin flip slightly more often. Lost by more when it lost. Weighted by
how far the line actually moved, the picks sit on the wrong side of the market —
and the net figure is six thousandths of a point from zero.

A win rate without magnitude is a misleading statistic. This is precisely the
case it misleads on.

One more check killed any remaining hope. If the model knew something, its most
confident picks should show the best closing-line value. Sorted by claimed edge,
the beat rates ran 57.4%, 46.6%, 50.0%, 56.1%. No order at all.

**Verdict.** Two independent tests, measured different ways on different
samples, agree: the model does not out-predict FanDuel. Better regression on
public box-score data is a dead end. The market already does that, with more
data and full-time staff.

## What actually survived

One result never touched the broken calculation, because it was measured from
prices and outcomes alone. Across those same 777 bets, the alternate-line band
carried prices implying a **70.5%** chance and won **62.4%** of the time —
roughly double the house margin of the standard prop market.

That is a negative finding, it cost real work to establish, and it is the only
thing here that survived a correct test.

The measurement rig also survived, which matters more than it sounds. Closing-line
value worked exactly as advertised as an early indicator — props that beat the
close went on to win 52.4% of the time, those that lost it won 41.1%. An
eleven-point spread. The instrument is sound. The model is what failed it.

## Six edges in one day

The track-record claim was not an isolated mistake. It was the sixth apparent
edge to dissolve on inspection in a single day of work.

| # | What looked like an edge | Outcome |
|---:|---|---|
| 01 | Backup quarterbacks. Career averages full of old starts, against players who would not take a snap. Kenny Pickett projected 7.0 completions, actually recorded 0, and the backtest scored it as a win. | Caught before shipping |
| 02 | Bench receivers. The model was being graded on 8,459 rows no book would ever price — deep reserves projected for single-digit yards. | Caught before shipping |
| 03 | Live in-game odds, which price information the pre-game model does not have. | Caught before shipping |
| 04 | Unexpected inactives, scored as correct low projections. | Caught before shipping |
| 05 | An uncalibrated survival estimate that read as confidence. | Caught before shipping |
| 06 | The track record at a price — the Jefferson row. | **Reached Slack and a public repo first** |

A seventh belongs on the list for a different reason. The backtest harness
itself, `scripts/backtest.py`, printed per-statistic figures that could not be
reproduced from the data its own collection step returned — including an early
and very pleasing "60% directional accuracy." Those numbers are discarded. A
second scorer, `scripts/score.py`, is the one I trust.

## What I did about it

The scheduled Slack jobs were stopped immediately. The public README was
rewritten with the correction at the top and pushed. Three scripts that depended
on the broken finding — `run_report.py`, `safest.py`, `report_week.py` — were
quarantined: they now exit on launch rather than run.

The screens derived from the same data are marked **unproven rather than
disproven**, which is a distinction I want to keep. The 55%-of-snaps floor may
well be sound. It has not been re-checked against a correct test, so it does not
get to claim anything yet.

The project still exists, with a much smaller promise. It reports the range a
player is likely to land in, alongside the book's nearest number. It does not
claim to know which prop is mispriced. It tried to answer that question twice
and could not do it honestly.

## The lesson, stated plainly

> Every check I ran asked whether the *answer* looked plausible. Not one asked
> whether the *test* was correct.

That is the whole postmortem in a sentence. A 24.7-point spread across 777 bets
looks exactly like a real finding. It has a sample size, a monotone trend, and a
mechanism you can tell a story about. Everything a result is supposed to have,
except a validation that compared the right two things.

The impossible row was in the file the entire time. What was missing was any
check that would have had to look at it — a test asking not "is this number
believable" but "does this number contradict another number I also believe."

I have kept the figure that started it all, and I would rather show it than a
working model. A model that appears to work is easy to produce and easy to fake.
Finding your own validation bug, quantifying it at 87% of rows, watching your
headline result flatten to nothing, and then pulling the code that depended on
it — that is harder, and it is the part I would want someone to know about
before they trusted anything else I built.

---

### Scope

Built September 2026. Entertainment project, never funded, never wagered.
Walk-forward backtest across 2023–2025; market validation and closing-line tests
on sampled 2025 weeks 6, 10 and 14, using real FanDuel prices via a historical
odds feed. No odds data is republished here — every figure is an aggregate
computed from that backtest.

### Correction history

The track-record finding was published to a public README and a scheduled report
before it was retracted. The retraction is dated the same day it was found, and
is recorded in full at `retraction.md` in the project vault.
