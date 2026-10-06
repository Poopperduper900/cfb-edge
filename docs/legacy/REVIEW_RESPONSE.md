# Response to the Gemini review

Seven suggestions. Four built, one rebuilt on a different mechanism, two
declined. Below is the reasoning for each, including where Gemini was right
about a problem but wrong about the fix.

One framing point first. Six of the seven items are restatements of limitations
I flagged in my own docstrings — "the model acknowledges this is a crude
metric," "the model requires a manual `p_play`." That's a reasonable thing for a
reviewer to surface, but it means the list is a TODO inventory, not an
independent audit. None of it questions the modelling choices themselves, and
none of it touches the thing that actually determines whether this is worth
running: whether the ratings beat the closing line at all. That's still unknown
until `validate` runs on real data, and every item below is precision work that
may be irrelevant if the answer is no.

---

## Built as suggested

### Weather ingestion — correct, with one important correction

Gemini was right: `project_game` priced wind, temperature and precipitation, and
`ingest.py` never fetched any of it. Now in `weather.py` via Open-Meteo — no API
key, free to 10k calls/day, archive back to 1940.

The correction Gemini missed matters more than the ingestion. The obvious build
is: pull ERA5 reanalysis for historical games, fit the wind coefficient, deploy
against forecasts. That silently overstates every weather adjustment you will
ever make, because the coefficient is estimated on perfect information and
applied to a noisy forecast. At kickoff-minus-48h, wind forecasts carry an RMSE
of 3–4 mph. At −0.34 points per mph, a 3.5 mph forecast error injects ~1.2
points of pure noise — comparable to the entire edge being hunted.

So the module fits on Open-Meteo's **Historical Forecast API** (archived
operational model runs, coverage from ~2021), not ERA5, and shrinks the whole
adjustment by forecast horizon:

```
24mph wind, Saturday morning   -6.67 pts   (confidence 0.96)
24mph wind, Thursday           -4.93 pts   (confidence 0.71)
24mph wind, Monday             -2.42 pts   (confidence 0.35)
```

This is also why weather is a Friday/Saturday play rather than a Monday one —
and why the market is slow to price it, since most of the money is down before
the forecast is trustworthy.

Also added: a dome/retractable-roof guard. Forgetting it is the most common
weather-model bug in football and it produces confidently wrong numbers on a
handful of games every year. Verified at zero.

### Positional portal value — right, but the star ratings were the bigger problem

Position multipliers added (`POSITION_VALUE`, QB at 4.6× down to long snapper at
0.1×). But position weighting is the easy half.

The half that matters more: **for portal players, recruiting rating is close to
the wrong variable entirely.** A three-star who just threw for 3,200 yards at a
Sun Belt school is worth far more than his 0.84 composite; a former five-star
transferring as a fourth-year backup is worth far less than his 0.98.
Recruiting rating measures what a high-school evaluator thought. Production
measures what happened. So `portal_score` now joins to prior-season PPA and
snaps, and credibility-weights production against stars by snap count.

Effect on a synthetic four-player transfer class:

```
star-only ranking:     LS, QB, CB, RB     ← a 5-star long snapper leads
position+production:   QB, CB, RB, LS
```

### QB dropoff — right target, unimplementable as stated

Gemini suggested computing the starter's EPA against "the historical or
projected EPA of their specific backup." The direction is correct. The problem
it doesn't account for: **the backup usually has no data.** A true freshman QB2
has zero college snaps. That's exactly why the original used a team-strength
proxy — not because it was better, but because the obvious approach has no
denominator in most cases.

`qb.py` implements the hierarchical version instead — three evidence tiers with
honest uncertainty attached:

```
QB2 has 400 dropbacks     4.60 +/- 2.4 pts   tier 1  backup has real sample
QB2 has 90 dropbacks      6.37 +/- 3.4 pts   tier 2  blended with prior
QB2 never played (4*)     6.40 +/- 4.6 pts   tier 3  prior, not measurement
QB2 never played (3*)     6.71 +/- 4.6 pts   tier 3  prior, not measurement
```

The `sd_points` column is the one that should govern behaviour. A dropoff of
6.4 ± 4.6 means the line became a coin flip with fatter tails, and the correct
action in most such spots is no action.

One trap worth naming: the garbage-time filter in `ratings.py` deletes exactly
the plays backups take. Computing backup EPA from filtered plays leaves almost
no sample, and what survives is biased — it's prevent defense against a losing
team. `backup_epa` therefore runs unfiltered with an explicit garbage-time
correction.

---

## Rebuilt on a different mechanism

### Dynamic volume shocks

Gemini's instinct is right — the volume shock shouldn't be a constant. The
proposed lever is wrong on two counts:

1. **Opposing defense variance drives efficiency, not volume uncertainty.**
   Folding it into the volume shock collapses the two channels the model exists
   to keep apart. That decomposition is the whole reason this beats a
   yards-based book line.
2. **Pace is already an input** to `team_play_estimate`. Adding it to the shock
   term double-counts it — and worse, inflates variance when what pace actually
   does is shift the mean.

What actually drives CFB volume uncertainty is game script and role stability.
So the fix went in two places.

**`script.py` — joint simulation.** Rather than haircut the mean by an expected
pull fraction, simulate the game margin and let volume fall out of it. Pass
rate, total plays, and benching are now all conditioned on the same margin draw,
so they correlate the way they do in reality. Result on a 24-point favourite's
lead back:

```
                        joint                        static (old)
even game        mean  94.4  sd 51.9            mean  98.4  sd 49.1
-24 favourite    mean  93.9  sd 51.8  skew +0.93 mean  87.7  sd 46.0

  starter | game stays close (<14):  106.5
  starter | blowout (>=28):           80.6
  backup  | blowout (>=28):           55.9   (vs 34.7 in close games)
  starter/backup correlation:        -0.106  (static forces this to ~0)
```

The old model shifted the mean down and kept the spread tight — a number that
never actually happens. The truth is bimodal: ~107 yards if it stays close, ~81
if it doesn't. And the same blowout that kills the starter's over makes the
backup's over live, which is a real correlated pair the static version couldn't
represent at all.

Receiving is where a favourite's script hurts unambiguously, because fewer pass
attempts and benching hit the same player twice: WR1 projects **53.9 → 38.6
yards (−28%)** going from neutral script to 24-point favourite.

Finding this rewrote a bug in my original code too. PROE was applied to the
*final* margin, but a team that wins by 24 wasn't up 24 for four quarters — the
time-averaged differential is roughly 45% of the final margin. Using the final
margin inflated the run-heavy script so badly that big favourites' backs
projected *above* their neutral baseline. Fixed.

**`role_instability()` — sigma from evidence, not a constant.** Derived from
games observed, share volatility, and role security:

```
bell cow, 9 games      sigma 0.176
committee, 4 games     sigma 0.485
true freshman, 2 games sigma 0.600
                       (shipped constant was 0.210 for all three)
```

A 0.21 constant was the average of situations that should never share a number.

---

## Declined

### NLP sentiment scraper for injury news

Declining this one on four grounds, and the last is the important one.

**Cost and terms.** X's free API tier has no search access. Basic is $200/month
with read caps too low for a 136-team watchlist; Pro is $5,000/month. Scraping
around it violates ToS and gets you IP-banned, usually mid-season.

**Sentiment is the wrong tool.** You don't want sentiment — you want structured
extraction of `{player, status, source credibility, timestamp}`. Scoring "coach
says he's day-to-day" for positive/negative valence produces a number that means
nothing. This is a named-entity and status-classification problem wearing a
sentiment-analysis costume.

**False positives are asymmetric and expensive.** A pipeline that wrongly sets
`p_play=0.3` on a healthy starter manufactures a large fake edge that you then
bet real money into. The failure mode isn't missing news; it's confidently
fabricating it.

**And the core one: the edge isn't in knowing, it's in knowing first.** Books
reprice CFB QB news within seconds to minutes. If you learn it from a public
tweet, so did the trader — and a batch NLP job running on a cron schedule is
strictly slower than a human with a column open. Automating this optimises the
wrong variable.

What I'd build instead: a curated list of ~150 beat writers (one or two per FBS
team), keyword filter rather than sentiment, push notification, human
confirmation, manual `p_play`. The bottleneck is judgment, not throughput.

### Ourlads depth chart scraper

**Terms.** Ourlads prohibits automated access and actively blocks it. That's a
maintenance treadmill on top of a licensing problem.

**Published CFB depth charts are a poor input regardless.** "OR" designations
everywhere, and staffs that misdirect on purpose during game weeks. Building
infrastructure to reliably acquire low-quality data is the expensive kind of
mistake.

**The manual step Gemini wants removed is doing work.** It's about twenty
minutes weekly, and it's the step where you actually look at the roster — which
is where data errors get caught before they become bets.

Built the better version instead: `script.derive_depth_chart()` reconstructs an
empirical depth chart from who actually touched the ball over the last four
games, with share, games active, share volatility, and group concentration. More
truthful than the published chart, already in the pipeline, no scraping.

### PFF data — agree in principle, with a caveat on where it pays

This is the strongest item on the list. PFF College does have snap counts and
route participation, and route participation was the single best volume
predictor in my NFL work. Public CFB data has no substitute.

Two caveats before you spend the money.

Check the licence terms for betting use specifically — PFF operates its own
betting products and their data licensing distinguishes between analysis and
wagering. Worth confirming before it becomes load-bearing.

More importantly: **think about where the edge concentrates.** If every sharp
CFB bettor has PFF, the P4 prop market already reflects route participation and
buying it gets you to par rather than ahead. The asymmetry is in G5 — where PFF
still charts the games but far fewer people are modelling them, and where the
prop limits are low enough that the market never gets corrected. If you buy it,
point it at the Sun Belt and the MAC, not the SEC.

And sequence it after `validate`. If the ratings turn out to carry no
information over the closing line, better usage data on the prop side is still
worth having — but you'd want to know that before the subscription, not after.
