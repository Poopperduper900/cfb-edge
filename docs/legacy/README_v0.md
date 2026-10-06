# cfbmodel

College football game and player-prop model. Ridge-adjusted EPA ratings,
market-implied power ratings, key-number-aware margin distributions, and a
volume/efficiency prop simulator — plus the walk-forward test that tells you
whether any of it is worth betting.

Built for the 2026 season. Week 0 is **Aug 29**, Week 1 runs **Sep 3–6**.

---

## Start here: what "beat Vegas" actually means

You asked for a model that beats Vegas. Here is the honest shape of that
problem before any code runs.

**Sides and totals.** The CFB closing spread has a mean absolute error around
10.5–11 points. The best public models — SP+, FPI, Massey — land in the same
neighbourhood, and none of them reliably beats the closer. At -110 you need
**52.38%** to break even. A genuinely good CFB model produces something like
53–54% against closing lines, which is a 1–3% ROI, which over a 15-week season
at four bets a week is a sample so small that a losing season is the *expected*
outcome roughly a third of the time. That is what winning looks like. Anyone
showing you 60% is showing you a backtest with lookahead in it.

**Your NFL result is the null hypothesis here.** You already found that
opponent-adjusted EPA added nothing on top of the NFL closing spread
(coefficient p = 0.63). CFB is a structurally different market — 136 FBS teams,
enormous talent dispersion, most games barely traded until Friday — so the same
test can genuinely come out differently. But it has to be *run*, not assumed.
`run_slate.py validate` is that test and it is the first command in the workflow
for a reason.

**Where CFB edge structurally exists**, ranked by how much I'd trust each:

1. **Openers, not closers.** Beating the close means beating everyone who bet
   the game. Beating the open means being early. The `opener_vs_closer` report
   measures exactly this.
2. **Weeks 0–3.** The market has no current-season data either. Everyone is
   running priors, so a better prior is a real edge — the only time of year
   that's true. This is why `priors.py` exists and gets 25% of the effort.
3. **G5 and weeknight games.** A Tuesday MAC total is priced with a fraction of
   the attention a Saturday SEC game gets. Lower limits, softer numbers.
4. **Player props.** Highest hold (6–9%) but by far the least efficient market
   in the sport, and the volume-vs-efficiency decomposition that worked for you
   in the NFL transfers directly.
5. **No injury report.** CFB has no mandated injury disclosure. This is the
   single largest information asymmetry in American sports betting and it is a
   *reporting* edge, not a modelling one. Beat writers, team accounts, and
   practice reports beat any model.

**Where edge does not exist:** Ohio State–Michigan on Saturday afternoon. If
your model disagrees by 4 points with a number that has taken seven figures of
sharp action, the model is wrong. `market_softness()` haircuts your bet size in
exactly those spots.

---

## Architecture

```
cfbmodel/
  config.py      constants — HFA, margin SD, key numbers, thresholds
  ingest.py      CFBD v2 API, disk-cached (free tier = 1,000 calls/month)
  ratings.py     ridge-adjusted EPA ratings + market-implied power ratings
  priors.py      preseason: recruiting, returning production, portal
  game_model.py  projection + key-number-aware discrete margin pmf
  props.py       volume/efficiency simulation with blowout-pull correction
  script.py      joint game-script simulation (margin -> volume, correlated)
  qb.py          hierarchical QB value + injury dropoff with backup fallback
  weather.py     Open-Meteo ingestion, dome guard, forecast-horizon shrinkage
  edge.py        devig (multiplicative / power / Shin), EV, Kelly, CLV
  backtest.py    walk-forward + the market-efficiency test
run_slate.py     CLI
tests/smoke_test.py    synthetic end-to-end validation, no API key needed
tests/test_upgrades.py validation for script/qb/weather/portal
REVIEW_RESPONSE.md     what was accepted, corrected, and declined from review
```

### Two rating systems, deliberately separate

`fit_epa_ratings` is your information: opponent-adjusted EPA per play from
play-by-play, via ridge regression with recency weights and a garbage-time
filter. Ridge rather than OLS because the L2 penalty is doing empirical-Bayes
shrinkage — which is exactly what you want in Week 3 when a team has played two
cupcakes.

`fit_market_ratings` is the market's information: power ratings backed out of
closing spreads by least squares. Same units, shared scale.

The whole model lives in the difference between them. A model that only
reproduces market ratings has no edge; one that ignores them has no discipline.

### The margin distribution is the actual bet

A spread bet at -3 is not a bet on E[margin], it's a bet on P(margin > 3).
CFB margins pile up on 3 and 7, so a continuous normal misprices every number
near a key number by 1–3 points of implied probability — larger than most edges
you will ever find. `margin_pmf` uses a Student-t base (real tail weight;
40-point results aren't rounding errors), SD scaled with expected total,
multiplicative key-number bumps, and zero mass on ties.

### Props: the three CFB-specific problems

1. **No snap counts, no route participation.** Route participation was your
   strongest NFL volume predictor and public CFB data simply doesn't have it.
   Box-score usage share is a noisier substitute, so shrinkage has to be heavier.
2. **Blowout pull risk is enormous.** Handled by joint simulation in
   `script.py`: the game margin is drawn first, and pass rate, total plays and
   benching are all conditioned on it. A 24-point favourite's back is bimodal —
   ~107 yards if the game stays close, ~81 if it doesn't — and the same blowout
   that kills his over makes the backup's over live. A static mean haircut
   produces a number that never happens.
3. **No injury report.** Every projection takes an explicit `p_play` you supply
   from reporting.

---

## Workflow

```bash
export CFBD_API_KEY=...          # free at collegefootballdata.com/key
pip install -r requirements.txt

python tests/smoke_test.py       # verifies the plumbing, no key needed

python run_slate.py pull --seasons 2019 2021 2022 2023 2024 2025
python run_slate.py validate --seasons 2022 2023 2024 2025    # ← decides everything
python run_slate.py preseason --season 2026
python run_slate.py slate --season 2026 --week 1
python run_slate.py props --season 2026 --week 5 --home Auburn --away Georgia
```

`validate` prints a coefficient table. Read it like this:

| result | meaning | action |
|---|---|---|
| model coef p > 0.10 | your ratings add nothing over the closing line | don't bet sides or totals — go to props |
| model coef p < 0.05, coef ≈ 0.15 | there's signal worth ~15% weight | set `--w-model 0.15`, not 1.0 |
| model coef > 0.5 | almost certainly lookahead in your pipeline | find the leak |

Then read the **segmented** table. Aggregate results hide everything. A model
can be worthless against the Saturday P4 closer and genuinely profitable against
a Tuesday MAC opener, and only the split shows you that.

---

## On your data sources

Of the five links you sent, one is an input and four aren't:

- **247 composite team rankings** — real input, and already available through
  CFBD's `/recruiting/teams` endpoint, so `priors.py` pulls it there.
- **Sports-Reference CFB** — same underlying data as CFBD, behind Cloudflare,
  and their terms prohibit scraping. Use CFBD.
- **TeamRankings odds history** — CFBD's `/lines` endpoint carries both opening
  and closing numbers, which is what you actually need for CLV.
- **ESPN FPI** — this is a competitor's *output*, not an input. Feeding another
  model's ratings into yours means inheriting its errors and calling the
  correlation a signal. Use it as a benchmark only.
- **Ourlads depth charts** — the one genuinely useful thing CFBD lacks, and they
  block automated access. `ingest.depth_charts()` reads a CSV you export by hand.

One free resource worth adding: the CFBD Model Pick'em at
`predictions.collegefootballdata.com` lets you submit weekly projected margins
and benchmark against a field of other modellers. It's the cheapest honest
feedback loop available and it costs nothing.

---

## Discipline

- **CLV is the KPI.** Over the sample you'll accumulate in one season, win rate
  tells you almost nothing and beat-the-close tells you almost everything. If
  median CLV is ≤ 0 after 100 bets, the model isn't working and any profit is
  variance. `edge.clv_report()`.
- **Quarter Kelly, 2% cap.** Every edge gets haircut by market softness, sample
  confidence, and a flat 0.75 on top — because the failure mode of every model
  in this lineage has been believing its own edges at face value.
- **More than eight plays on a slate is a bug, not a good week.** Almost always
  a mis-rated team or a stale line in the feed.
- **Run `bootstrap_roi` before believing any backtest.** Sixty bets at -110 has
  a 95% ROI confidence interval of roughly ±13%, which is wider than any real
  edge. Most backtest "results" are inside that band.
- **Books limit CFB prop winners fast.** If this works, account longevity
  becomes the binding constraint long before bankroll does.

## Calibration constants

Everything in `config.py` — HFA, margin SD, key-number weights, shrinkage
priors — are seeds, not ground truth. Re-fit them from your own data as soon as
you have the seasons pulled. The key-number weights in particular are the
highest-leverage numbers in the repo and deserve their own fit against
2015–2025 FBS margins.
