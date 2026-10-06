# Audit of `cfbmodel/` against CLAUDE.md "Known pitfalls"

Phase 0, 2026-10-06. The audit itself changed no model code (only moved/wired:
`run_slate.py` → `cfbmodel/cli.py`, `run_week0.py` → `cfbmodel/week0.py`, key loading → `keys.py`). Bug A was then fixed in a separate commit, test first.
Every claim marked "measured" comes from `python docs/audit_experiments.py` (synthetic data, no API key).

**Headline: the earlier build has two bugs that would have misled you, neither of which is on the
12-item list.** (1) A lookahead leak in every rating fit, so any backtest run so far is untrustworthy
(**fixed 2026-10-06**, see bug A). (2) The CLV sign is backwards. Details under "Other bugs". Nothing below needs your decision yet; the
fixes are scheduled in the phases named.

Status key: **HANDLED** = correct in code · **PARTIAL** = some of it · **NOT HANDLED** = missing.
"Test today" = whether a test currently guards it (Phase 2 adds the missing ones).

| # | Pitfall | Status | Where | Test today |
|---|---------|--------|-------|-----------|
| 1 | t takes a *scale*, not an SD | **HANDLED** | `game_model.margin_pmf` (`game_model.py:128-155`); `config.margin_scale_market = 11.22` is a scale, `sd = scale·√(df/(df−2))`. Measured at exp. margin 0, total 52: market pmf MAE **10.25** (target 10.5±0.4), SD **13.29** (13.6±0.5). | `test_pitfalls.py::test_t_scale` |
| 2 | Model pmf must be wider than market pmf | **HANDLED** | `margin_pmf(source="market"\|"model")`, default `"model"` (`game_model.py:130,149`); `cover_prob` passes it through (`:168-173`). Measured model MAE 13.16 vs market 10.25 = **1.28×** (needs ≥1.25×). Loose ends: `total_probs` is a plain normal with no `source`; `moneyline_prob` (`:179`), `props.py:117`, `script.py:133` always use the default (safe direction). | `test_pitfalls.py::test_model_pmf_wider` |
| 3 | Ridge shrinks the HFA column | **HANDLED in Phase 2** (was latent) | Was: `ratings.py:121-122` (EPA fit, alpha 220) and `:227` (market fit, alpha 1.0) both feed an unscaled HFA column to `Ridge`. Harmless today: measured HFA 2.40 at alpha 1, but **2.18 at alpha 100 and 1.09 at alpha 1000** (true 2.4); ×100 scaling gives 2.46 / 2.49. It becomes live in Phase 6, when ridge alphas are learned. *Fix:* the HFA column is scaled by `ratings.HFA_SCALE` (100) in both fits and the coefficient scaled back; both fits take an `alpha` argument. | `test_hfa_not_shrunk_market_fit`, `test_hfa_not_shrunk_epa_fit` (fail if the scaling is switched off) |
| 4 | Ridge sign convention | **HANDLED** | `ratings.py:118` (offense +1, defense +1), `:138` `net = off − def`; same in `_sub_ratings` (`:165`). Evidence: synthetic recovery r = **0.929** (EPA), **0.999** (market). Margin on the EPA bar is thin (needs > 0.9). | `test_rating_recovery` (EPA r > 0.9, market r > 0.98; fails if the defense sign is flipped) |
| 5 | `iterrows()` labels used as positions | **HANDLED** (no violation found) | 9 `iterrows` sites reviewed: `priors.py:51`, `backtest.py:149`, `weather.py:145`, `cli.py` (slate, props, weather), `week0.py`. All read row values with `_` for the label; none index an array by it. | `test_no_positional_iterrows_static` (AST scan: every loop discards the label) and `_behavioural` (non-contiguous frames give identical results) |
| 6 | PROE must use time-averaged lead | **HANDLED in Phase 2** | Handled in `script.simulate_game_script` (`script.py:151`: `0.45 × margin`). **Not** handled in `props.pass_rate_over_expectation` (`props.py:96`), which applies the slope to the full expected margin. It had no callers but was a trap; *fixed*: it now applies the same 0.45 factor. | `test_proe_time_averaged` |
| 7 | Exclude FCS from rating fits | **HANDLED in Phase 2** | Was partial: `ratings.clean_plays` drops plays whose offense/defense conference is missing (`ratings.py:69-70`), but only if those columns exist and only in the EPA fit. No API call passes `classification="fbs"` (`ingest.py`), and `fit_market_ratings` / `fit_total_ratings` (`ratings.py:197-260`) use every line row, so FCS opponents enter the market fit if CFBD returns them. *Fix:* `ingest` requests `classification=fbs` for plays and flags every team `*_is_fbs`; `ratings.drop_non_fbs` is applied in the EPA, market and totals fits and in both `qb_epa` paths. | `test_fcs_excluded` (also checks FBS ratings are unchanged by FCS games) |
| 8 | Garbage-time filter deletes backup-QB plays | **HANDLED** | `qb.build_qb_table` pulls the backup from **unfiltered** plays (`qb.py:153-154`) and subtracts `GARBAGE_TIME_EPA_INFLATION` (`qb.py:122,125`). Caveats: the unfiltered path has no FCS filter; `qb.qb_epa` had the lookahead leak (bug A, fixed). | `test_upgrades::test_3` (tier logic only) |
| 9 | `cfbd` lib v5 auth sends no header | **HANDLED** | `ingest.cfbd_get` uses `requests` with `Authorization: Bearer` (`ingest.py:56`); the `cfbd` library is not used. Gap: a 401 surfaces as a raw `HTTPError` traceback, and `cli.cmd_pull` swallows errors (bug C). | none (Phase 1 401 test) |
| 10 | Weather: forecasts not reanalysis, shrink, domes = 0 | **PARTIAL** | Handled: dome → 0 in `weather.total_adjustment` (`weather.py:192`) and `attach_weather` (`:147`); horizon shrink in `forecast_confidence` (`:165`); `mode="forecast"` uses the historical-forecast API (`:88`). **Not handled:** coefficients are hand-set, never fitted on historical forecasts (a second, different set lives in `game_model.project_game`); for past games `hours_ahead` is negative so confidence is always 1.0, i.e. backtests ignore the horizon; ~~the dome list is a hand-typed exact-name match~~ (*fixed in Phase 2*: `attach_weather` uses CFBD's venue `dome` flag and falls back to the name list only when the flag is blank, recording `dome_source`; `project_game` takes `dome=`); ~~the weather cache never hits~~ (bug D, fixed). Still open: coefficients are not fitted (Phase 6) and the backtest horizon. | `test_upgrades::test_5` (shrink), `test_dome_zero` |
| 11 | Cache path configurable | **HANDLED** | `config.py:22` `CFBMODEL_CACHE` (and `CFBMODEL_OUTPUT`, `:23`). Caveat: bug D means the weather cache is not durable even when pointed at Drive. | none |
| 12 | `calibrate.py` claim is false | **NOT HANDLED** | `config.py:37` and `:71` still say `calibrate.py` re-fits constants. No such file; every constant is hand-set. Phase 6 replaces this and has an acceptance line for it. | none |

## Other bugs found

Ranked by how much they could hurt you. File and line are where to look.

**A. HIGH, FIXED: lookahead leak in every "as-of" rating fit.** The filter
`(season < asof_season) | (week < asof_week)` (`ratings.py:104,206,240`, `qb.py:48`, `props.py:272`,
`script.py:89`) lets in any row from a *later* season whose week number is smaller. The correct filter is
`(season < S) | ((season == S) & (week < W))`. `recency_weights` (`ratings.py`) then gives those future
rows a weight above 1 (`season_carryover ** negative`, about 1.9×). `backtest.walk_forward` concatenates
all seasons before fitting, so a 2022 week-5 rating is built partly from 2023-2025 early-season games.
Measured: ratings change by up to **3.18 points** (market) and 0.146 EPA/play (EPA) when rows that should be
invisible are removed. **Consequence: any `validate` output from the earlier build must be thrown away.**
A model that "passed" would be a mirage.

*Fix:* one shared filter, `ratings.as_of(df, season, week)`, now used at all seven sites (the six above plus
`props.build_player_priors`, whose docstring claimed "leak-safe by construction" but was not).
*Proof:* `tests/test_no_lookahead.py` (26 tests) feeds each as-of function everything vs only strictly-earlier
rows and requires identical output. Written first and run against the old code: 13 failed (every function, at
the as-of points that have a later season in the data). After the fix: all pass. The evidence script now reports
a max rating change of 0.0000. One related item is left open: `derive_depth_chart` keeps every prior-season row
(its 4-week look-back only bites within a season), so early in a season the "depth chart" is last season's.
That is a modelling choice, not a leak; revisit in Phase 3.

**B. HIGH (once bets are logged): CLV sign is inverted.** `edge.clv` (`edge.py:222`) returns the opposite
sign for spreads and totals, so "% beating the close" would be backwards. Measured on four cases, all
wrong. Fix with the sign-convention tests in Phase 8.

**C. MEDIUM, FIXED (Phases 1 and 4): errors are swallowed, which breaks rule 1 ("if an API call fails, stop and report").**
`cli.cmd_pull` (`except Exception: pass` around plays/box scores), `cli._load`, five `try/except` blocks in
`week0.build_ratings`, `weather.attach_weather` (`except Exception: wx = {}`), `ingest.season_plays`. A bad key,
a rate limit or a typo in the season looks like "no data" instead of an error. *Fix:* those sites now let errors through; only an explicit HTTP 404 on an optional dataset is reported (visibly) as "not available". The CLI turns CFBD/budget/schema errors into a plain message and exit code 2. The last one, `backtest.walk_forward`'s `except (ValueError, KeyError): continue`, was removed in Phase 4 (the function is gone; `validation.walk_forward` only skips the dedicated `NoDataBeforeAsOf` error and lists every skip).

**D. MEDIUM, FIXED in Phase 1: weather cache never hits.** `weather._cache` (`weather.py:57`) names files with Python's `hash()`,
which is randomised per process (measured: two runs, two different values). Every run re-calls Open-Meteo.
*Fix:* `weather._cache_key` uses `hashlib`; a test pins the exact file name.

**E. LOW: duplicated, hard-coded adjustments.** `game_model.project_game` carries its own wind/precip/cold/
altitude/rest/travel numbers (e.g. wind −0.42/mph over 12) that disagree with `weather.total_adjustment`
(−0.34/mph over 10). Today `project_game`'s weather arguments are never passed, so there is no double count,
but one call would create it. All of these move to the params file in Phase 6a.

**F. LOW: misleading config names.** `ridge_alpha_def` is used for the pass/rush sub-fits
(`ratings.py:168`), not for defense; `ridge_alpha_off` drives the main joint fit (`:127`).

**G. LOW: the Power-4 conference list is hard-coded in several places** (`cli.cmd_validate`, `week0.P4`,
`game_model.market_softness`). It will go stale when conferences change. Derive tiers from data in Phase 4.

**H. INFO: how good the existing tests were.** They were scripts that pytest could not collect (`SystemExit`
at import). Checks [10], [11], [13] only printed (I added minimal assertions, labelled in the file). Check
[12] passes with p = 0.0096 against a 0.01 limit: deterministic with the fixed seed, but fragile.

**I. INFO, FIXED in Phase 4: gaps in the old validation.** `backtest.market_efficiency_test` uses classical standard errors
(the plan requires heteroskedasticity-robust), tests closing lines only, and has no train/holdout split. *Fix:* `validation.py` uses HC3 standard errors, tests open and close separately, and requires the pre-registered train + holdout rule.

## Where this leaves the plan

- Phase 1 should also fix **C** and **D**.
- Phase 2's test list stands. The Phase 3 leak test already exists (`tests/test_no_lookahead.py`) because **A** was
  the most dangerous item here; Phase 3 extends it to the new rating systems and player state as they are added.
- **B** goes into Phase 8, **E/F/G** into Phase 6/4.
