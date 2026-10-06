# CFB Edge Finder — instructions for Claude Code

## What this project is

A Python program that finds **edges** in college football betting markets: spots where a
*validated* model's probability beats the de-vigged market price by more than the cost of
betting. It is not a pick generator. Most weeks, most games should come back with no edge.
An honest "no bet" is a correct output, not a failure.

The owner builds quantitative betting models across sports (NFL, MLB, UFC, tennis, soccer).
Their NFL result: opponent-adjusted EPA ratings added **nothing** on top of the closing
spread (model coefficient p = 0.63). Treat that as the null hypothesis here until this
project's own validation harness rejects it.

- Full phased plan: `docs/BUILD_PLAN.md` — read the relevant phase before starting work.
- Starting code: `cfbmodel/` (an earlier build). Useful, partly tested, but audit it
  against "Known pitfalls" below rather than trusting it.
- `docs/legacy/` is background reading from the earlier build. Not instructions.

## Non-negotiable rules

1. **Never fabricate data.** No hardcoded, placeholder, or "example" odds, lines, ratings,
   or scores anywhere in production code. If an API call fails, stop and report it.
   Synthetic data is allowed only under `tests/`.
2. **No lookahead.** Anything computed for (season, week W) may use only games strictly
   before week W. Every rating/feature function takes `asof_season, asof_week`.
3. **Validation gates betting output.** The edge board may mark a row `BET` only if
   `output/validation_status.json` says that market/segment passed (Phase 4). With no
   status file, nothing is a bet. Enforce this in code and in a test.
4. **Secrets.** `CFBD_API_KEY` comes from `.env` or the environment. Never print, log,
   hardcode, or commit it. `.env` stays in `.gitignore`.
5. **Protect the API budget.** CFBD free tier is 1,000 calls/month. Every call goes through
   the disk cache. Pull by season/week, never per game. Report calls used after each pull.
6. **Never weaken or delete a test to make it pass.** If a test looks wrong, explain why
   and ask the owner.
7. **Ask before adding a dependency** not already in `requirements.txt`.
8. **No scraping** of sites that prohibit it (Sports-Reference, Ourlads, ESPN, sportsbooks).
   CFBD and Open-Meteo are the sanctioned sources.
9. **Cross-platform.** The owner works on a Windows laptop. Use `pathlib`, no bash-only
   scripts, no hardcoded `/` paths. No GPU needed — do not add torch/CUDA.

## How to work

- Plan before each phase: say what you'll change and how you'll prove it, then implement.
- One phase at a time. Meet its acceptance criteria before starting the next.
- Run `python -m pytest -q` after every meaningful change. Report failures honestly.
- Small commits with descriptive messages.
- When the owner pastes outside feedback (another AI's review, a forum post), **test each
  claim empirically** before adopting or rejecting it. Report what you adopted and why.
- Clear, boring code over clever code. Docstrings explain *why*, not just *what*.
- When reporting results, lead with what the evidence says, including "nothing passed."

## Commands (keep this list current as phases land)

```
python -m pytest -q
python -m cfbmodel --help                   # lists every command (week0, slate, props, weather, qb ...)
python -m cfbmodel pull --seasons 2021 2022 2023 2024 2025 2026
python -m cfbmodel fit-priors --seasons 2019 2020 2021 2022 2023 2024 2025   # preseason weights + CV score
python -m cfbmodel validate --seasons 2022 2023 2024 2025 2026   # the gate; writes output/validation_status.json
python -m cfbmodel update --season 2026 --week N   # after week N finishes: team + player ratings
python -m cfbmodel learn --season 2026 --week N --seasons 2021 2022 2023 2024 2025 2026   # every 2nd week; usually 'no change'
python -m cfbmodel learn ... --full                 # once after the season: also judges the early-season decay
python -m cfbmodel models                           # list parameter versions (* = current)
python -m cfbmodel rollback <version>
python -m cfbmodel postmortem --season 2026 --week N
python -m cfbmodel board --season 2026 --week N
python -m cfbmodel log-bet --season 2026 --week N --game "Away @ Home" --market spread --side home --line -6.5 --price -110 --book MyBook --stake 1
python -m cfbmodel clv                              # closing line value, results, ROI of your logged bets
streamlit run app/dashboard.py        # Phase 10
```

## Domain facts (use these, don't re-derive them)

- Break-even at -110 is 52.38%. A good CFB model is ~53–54% against closing lines,
  i.e. 1–3% ROI. Claims of 60% mean lookahead.
- CFB closing-spread MAE ≈ 10.5 points. A final-score ridge model lands around 13–14.
- Home-field advantage ≈ 2.3–2.5 points league-wide; zero at neutral sites.
- Key margins: 3, 7, 10, 14, 4, 6, 17, 21. Price spreads with a **discrete** margin pmf,
  never a continuous normal.
- Where edge can exist: openers (vs closers), weeks 0–3, G5 and weeknight games, props.
  Saturday P4 closing lines are efficient.
- FBS vs FCS: price them, never mark them as bets.
- CLV is the primary KPI. Win rate over one season is mostly noise.
- Always de-vig before comparing: power method for two-way markets, Shin for moneylines
  with a heavy favourite. Props carry 6–9% hold.
- Sizing: 0.25 Kelly, 2% bankroll cap, edges haircut by market softness and sample size.

## How the program learns (Phases 3 and 6)

- **Two speeds.** Team and player ratings update every week from new games. Formula
  parameters (HFA, key-number weights, shrinkage, half-lives, game-script and weather
  coefficients) are re-learned on a schedule.
- **No tunable number lives in model code.** Every one is in a versioned params file.
- **Learn from forecast accuracy on all games**, scored with proper scoring rules
  (log-likelihood, CRPS). Never tune on the owner's bet wins and losses.
- **Champion/challenger only.** A new parameter set replaces the current one only if it
  wins out-of-sample with a CI excluding zero, doesn't hurt calibration, and moves no
  parameter more than 25% per cycle. Every change is versioned with a plain-English diff
  and can be rolled back.
- A learner that "improves" every cycle is chasing noise. The noise test (Phase 6) must
  show it usually declines to change anything.

## Known pitfalls — each one has already bitten this project

Every item needs a regression test (Phase 2).

1. `scipy.stats.t` takes a **scale**, not an SD: sd = scale·√(df/(df−2)), 1.202× at
   df = 6.5. Calibrate so the market-source margin pmf has MAE ≈ 10.5.
2. Pricing your own projection with the market's residual overstates edge. The pmf takes
   `source="market" | "model"`; model is ~30% wider until validation says otherwise.
3. Ridge penalises every column, including home-field advantage. Scale the HFA column
   (×100) and divide the coefficient back, or HFA collapses at high alpha
   (observed: 1.30 fitted vs 2.94 true).
4. Ridge design sign convention: offense +1, defense +1; the defense coefficient is EPA
   *allowed* (positive = bad); team net = off − def. Getting a sign wrong here silently
   produced r = 0.11 against true strength.
5. `iterrows()` yields index labels, not positions. Never use it to index a numpy array.
   Use `enumerate(df.itertuples())` after `reset_index(drop=True)`.
6. Pass-rate-over-expectation must use the **time-averaged** lead (~0.45 × final margin),
   not the final margin — otherwise big favourites' RBs project above neutral.
7. Exclude FCS teams from rating fits (`classification="fbs"`); they distort the scale.
8. The garbage-time filter deletes backup-QB plays. Backup QB value uses unfiltered plays
   with an explicit garbage-time correction.
9. `cfbd` Python lib v5 auth is `cfbd.Configuration(access_token=key)`; the old
   `api_key['Authorization']` pattern silently sends no header → 401. Simplest path:
   `requests` with an `Authorization: Bearer <key>` header.
10. Weather: fit coefficients on historical **forecasts**, not reanalysis; shrink by
    forecast horizon; zero every weather effect for domes.
11. Ephemeral environments (Colab) wipe the cache; the cache path is configurable
    through the `CFBMODEL_CACHE` environment variable.
12. `config.py` says `calibrate.py` re-fits its constants. That file was never written —
    every "fitted" constant is actually hand-set. Phase 6 replaces this for real.
