# Build Plan — CFB Edge Finder

Read `CLAUDE.md` first. Do the phases in order. Each phase ends when its acceptance
criteria pass, followed by a short report to the owner: what changed, which tests prove
it, and anything uncertain.

**Season context.** It is mid-season 2026 (around week 6 in early October). Two different
rules apply to 2026 data, and they must not be confused:
- **Ratings and parameters may learn from 2026 games** — that's the point of Phases 3
  and 6 — but only walk-forward, so every score they're judged on is out-of-sample.
- **The bet-permission gate (Phase 4) stays fixed** to its pre-registered train/holdout
  seasons until the season ends. 2026 results are reported there, never used to grant
  permission mid-season.

---

## Phase 0 — Setup and audit

Goal: a clean, reproducible starting point. No new features.

Tasks
- Create a virtual environment and install `requirements.txt`.
- Confirm `.gitignore` covers `.env`, `data/cache/`, `output/`, `__pycache__`, `.venv`.
  `.env.example` contains only `CFBD_API_KEY=`.
- Key loading: a small helper that reads `.env` or the environment and fails with a clear
  message if the key is missing (ask before adding `python-dotenv`).
- Convert `tests/smoke_test.py` and `tests/test_upgrades.py` to pytest style without
  weakening any assertion.
- Audit `cfbmodel/` against every item in CLAUDE.md "Known pitfalls". Write
  `docs/AUDIT.md`: for each pitfall, the file and function where it is handled, or
  `NOT HANDLED`, plus any other bugs found.
- Add `cfbmodel/__main__.py` so `python -m cfbmodel <command>` works (wrap the commands
  currently in `run_slate.py` and `run_week0.py`).

Acceptance
- `python -m pytest -q` passes on a fresh clone with **no** API key set.
- `docs/AUDIT.md` covers all 12 pitfalls.
- `git status` shows no `.env`, cache, or output files tracked.

---

## Phase 1 — Data layer

Goal: every input the model needs — cached, budget-aware, schema-checked.

Tasks
- CFBD client: `requests` with a Bearer header against
  `https://api.collegefootballdata.com`; exponential backoff on 429; disk cache keyed by
  endpoint + params.
- Cache policy: completed seasons are cached permanently. For the current season, games
  and lines for the current and previous week are refreshable (flag or TTL). Completed
  weeks are never re-pulled unless explicitly asked.
- Budget file `data/cache/_budget.json` tracking calls per calendar month. Warn at 700;
  refuse new calls at 950 unless `--force`.
- Endpoints: `/games` (FBS), `/lines` (opening **and** closing, every provider), `/plays`
  (by week), `/games/players`, `/player/usage`, `/recruiting/teams`, `/player/returning`,
  `/player/portal`, `/venues`, `/teams/fbs`.
- Normalise each to a tidy DataFrame with documented columns. Explicit schema checks:
  required columns, dtypes, no duplicate `(gameId, provider)` rows.
- Team-name canonicalisation: one canonical name per team across every endpoint. Log
  unmatched names; never drop them silently.
- `python -m cfbmodel pull --seasons ...` prints rows per endpoint per season, calls used,
  calls remaining this month.

Acceptance
- Re-running an identical pull uses 0 calls (tested with a mocked HTTP layer).
- Schema tests pass on small, real-shaped JSON fixtures in `tests/fixtures/` (no secrets).
- A 401 produces a clear "check CFBD_API_KEY" error, not a stack trace (tested).

---

## Phase 2 — Pitfall regression tests

Goal: lock in every lesson before building more on top. Synthetic data with known truth.
Each test's docstring names the pitfall number it guards.

Minimum set
- `test_t_scale` — market-source pmf MAE within 10.5 ± 0.4, SD within 13.6 ± 0.5.
- `test_model_pmf_wider` — model-source MAE ≥ 1.25 × market-source MAE.
- `test_hfa_not_shrunk` — two-season synthetic league, alpha = 1000: fitted HFA within
  0.3 of the realised mean home margin.
- `test_rating_recovery` — EPA ridge recovers true strength with r > 0.9; market-implied
  ratings with r > 0.98.
- `test_no_positional_iterrows` — no array is indexed by `iterrows()` labels (behavioural
  test on a filtered, non-contiguous frame).
- `test_proe_time_averaged` — a 24-point favourite's lead RB mean rushing yards ≤ the
  neutral-script mean.
- `test_fcs_excluded` — rating outputs contain no FCS teams.
- `test_dome_zero` — weather adjustment is exactly 0 for dome venues.
- `test_key_numbers` — P(margin = 3) > P(margin = 2), P(7) > P(8), P(tie) = 0, pmf sums
  to 1.

Acceptance: all pass.

---

## Phase 3 — Ratings (leak-safe)

Goal: three rating systems, all as-of aware.

1. **Market-implied** ratings from closing spreads (least squares, recency-weighted).
2. **Opponent-adjusted EPA** ratings (ridge, garbage-time filtered, recency-weighted,
   pass/rush split, pace).
3. **Preseason prior**: prior-season rating, 4-year recruiting composite, returning
   production, position-weighted portal joined to prior production. Weights come from
   `fit_prior_weights` on past seasons, not hand-set values.

Blending: `rating = w_model · model + (1 − w_model) · market`, with `w_model` read from
`output/validation_status.json`. If the file is missing, `w_model = 0` — the market
until proven otherwise.

Early season: shrink model ratings toward the preseason prior with a weight that decays
over weeks 0–5. Fit the decay curve (Phase 6); don't guess it.

**Weekly updating (automatic).** After each completed week, `python -m cfbmodel update`
refits team ratings on everything before the next week and writes
`output/state/teams_<season>_w<W>.parquet`.

**Player ratings as carried-forward state.** Each player has a usage share (Beta
posterior) and per-touch efficiency (Normal posterior) that is *updated* weekly from new
box scores rather than recomputed from scratch: last week's posterior is this week's
prior. Store in `output/state/players_<season>_w<W>.parquet` with games observed and
posterior uncertainty. Transfers start from a prior built from prior-school production.
Uncertainty must shrink with games played and widen after a role change (new depth-chart
rank, injury return).

Acceptance
- **Leak test:** ratings as of (S, W) are identical whether or not rows from weeks ≥ W are
  present in the input.
- Phase 2 recovery tests still pass.
- Fitted prior weights saved to `output/prior_weights.json` with their CV score.
- A player's posterior uncertainty decreases monotonically with games observed (test).

---

## Phase 4 — Validation harness (the gate)

Goal: decide, with evidence, whether the model may influence bets at all.

Tasks
- Walk-forward: for each season and week from week 4, refit on prior data and project.
  Weeks 0–3 are evaluated separately with preseason ratings.
- Efficiency regression: `margin ~ market_margin + model_margin`, heteroskedasticity-
  robust standard errors. Run against **closing** and **opening** lines separately.
  Same structure for totals.
- Segments: tier (P4–P4, P4–G5, G5–G5), week bucket (0–3, 4–8, 9+), weekday vs Saturday,
  spread-size bucket.
- **Multiple-comparisons guard.** Testing many segments will produce false positives by
  chance. A segment passes only if (a) p < 0.05 on 2022–2024 **and** (b) the coefficient
  keeps its sign with p < 0.10 on 2025 as a holdout. 2026-to-date is reported, never
  used for decisions.
- Leak alarm: a model coefficient above 0.5 sets status `SUSPECTED_LEAK` and fails.
- Also report: MAE (model vs opener vs closer), calibration table for cover
  probabilities, ROI by edge bucket, bootstrap CI on ROI.
- Write `output/validation_status.json` and a readable `output/validation_report.md`.

`validation_status.json` schema
```json
{
  "generated": "ISO date",
  "train_seasons": [2022, 2023, 2024],
  "holdout_season": 2025,
  "markets": {
    "spread_vs_open":  {"n": 0, "coef": 0.0, "se": 0.0, "p": 1.0, "passed": false, "reason": ""},
    "spread_vs_close": {"...": "same fields"},
    "total_vs_open":   {"...": "same fields"},
    "total_vs_close":  {"...": "same fields"}
  },
  "segments": [
    {"market": "", "segment": "", "n": 0, "coef": 0.0,
     "p_train": 1.0, "p_holdout": 1.0, "passed": false, "reason": ""}
  ],
  "w_model": {"<market>|<segment>": 0.0}
}
```

Acceptance
- Running `validate` twice on the same cache produces identical JSON.
- The report says so plainly when nothing passed. That is a valid, expected outcome.

---

## Phase 5 — Pricing

- Discrete margin pmf: Student-t with calibrated scale, key-number bumps, no ties,
  `source="market" | "model"`.
- `cover_prob`, `moneyline_prob`, `total_probs`, with pushes handled explicitly.
- De-vig: multiplicative, power, Shin. Power for two-way spreads/totals; Shin for
  moneylines when the favourite is shorter than −400.
- EV, fractional Kelly (0.25, 2% cap), edge shrinkage by market softness and sample size.
- Key-number value for every line: win probability gained by moving ±0.5 and ±1.0.

Acceptance
- De-vig ordering on a lopsided market: multiplicative < Shin < power for the favourite.
- Kelly returns 0 with no edge and never exceeds the cap.
- cover + push + other side = 1 within 1e-9.

---

## Phase 6 — Learning loop (the program improves its own formula)

Goal: every tunable number in the model is learned from data, re-learned on a schedule,
and only replaced when the replacement is *proven* better out-of-sample. A learner that
promotes whatever fit last week best will chase noise and get worse; this phase exists
to make learning safe as much as to make it happen.

### 6a. Parameter registry
- Move every tunable number out of the code into `params/<version>.json`. That includes
  everything in `config.py` plus the literals currently in `script.py` (PROE slope,
  time-averaged-lead factor, play-count slopes, pull-hazard curve, benching beta),
  `weather.py` (wind, gust, precip, cold coefficients), `props.py` (shrinkage strengths,
  position priors), `game_model.py` (rest, travel, altitude, QB dropoff), and the ridge
  alphas and recency half-lives in `ratings.py`.
- Model code reads parameters from the active version only. `params/current.json` points
  to it. A test fails if a tunable literal reappears in model code (keep an allowlist for
  genuine constants such as 52.38%).

### 6b. Fitters — one per parameter group, each with a proper objective
Learn from **forecast accuracy on every FBS game** (thousands of observations), never from
the owner's bet results (a few dozen per season — far too few, and outcome-biased).

| group | objective | data |
|---|---|---|
| margin pmf: t scale, df, total slope, key-number weights | log-likelihood of actual margins given the closing spread | all FBS games 2015+ |
| HFA (league, plus shrunk per-venue), ridge alphas, recency half-lives | walk-forward log-likelihood of margins from model projections | 2015+ walk-forward |
| preseason blend weights, early-season decay curve | week-by-week walk-forward accuracy, weeks 0–5 | 2015+ |
| totals SD, pace scaling | log-likelihood of game totals | 2015+ |
| weather coefficients | totals residuals vs **historical forecasts** | 2021+ (forecast archive) |
| PROE slope, pull-hazard curve, volume-shock model, efficiency shrinkage | CRPS / PIT calibration of player stat distributions | box scores 2018+ |
| `w_model` per market/segment | Phase 4 regression coefficient | fixed train/holdout |

### 6c. Champion / challenger
`python -m cfbmodel learn` fits a **challenger** parameter set on data through the last
completed week, then scores champion and challenger on a rolling walk-forward window
neither was fit on. The challenger is promoted only if all of these hold:
- primary metric improves, with a bootstrap 95% CI that excludes zero;
- calibration does not get worse (reliability error, PIT uniformity);
- no single parameter moves more than 25% in one cycle (bounded steps prevent lurching;
  a parameter that needs to move further gets there over several cycles);
- the window has at least 300 games (or 2,000 player-games for prop parameters).

Otherwise the champion stays and the report says why.

Cadence: ratings and player state update weekly (Phase 3). `learn` runs every second week
in-season and once in full after the season. Both configurable.

### 6d. Versioning and rollback
- Each promotion writes `output/models/<version>/` with `params.json`, `metrics.json`,
  and `diff.md` — a plain-English list of what changed, by how much, and the evidence.
- `python -m cfbmodel models` lists versions and their metrics.
  `python -m cfbmodel rollback <version>` restores one.
- Every board records the params version it used.

### 6e. Drift monitoring and post-mortems
- Rolling 4-week calibration and MAE of the model vs the market. If the model degrades
  past a threshold, raise a drift alarm and force `w_model` to 0 until the next
  successful `learn`.
- Weekly post-mortem report (`output/postmortem_w<W>.md`): biggest misses and which
  component drove each (team rating, HFA, total, weather, QB). This is for the owner to
  read. It does **not** feed back into tuning automatically — that's how a model
  overreacts to one weird Saturday.

### 6f. Optional: competing model classes
Once the loop works, a different model (e.g. gradient-boosted residuals on top of the
market) can enter as a challenger under exactly the same promotion rules. No model gets
in through a side door.

Acceptance
- **Recovery test:** synthetic data generated with known parameters (e.g. HFA 3.1,
  key-number weight on 3 of 1.5); the fitters recover them within tolerance.
- **Noise test:** on data where nothing has changed, the challenger is *not* promoted
  across 20 random seeds in at least 19. This is the most important test in the phase.
- **Leak test:** `learn` as of week W produces identical parameters whether or not
  data from weeks ≥ W exists.
- **Rollback test:** rolling back reproduces the earlier version's board exactly.
- `config.py` no longer claims a `calibrate.py` that doesn't exist.

---

## Phase 7 — Edge board (the product)

Command: `python -m cfbmodel board --season 2026 --week N`

Outputs: `output/board_2026_wN.csv`, a self-contained `output/board_2026_wN.html` that
reads well on a phone, and a terminal summary.

Columns
`game, kickoff_local, tier, market, book, line, price, best_line_across_books,
no_vig_prob, model_prob, prob_edge, ev, half_point_value, key_number_note, softness,
stake_pct, status, flags, params_version`

Status values
- `BET` — edge clears thresholds **and** that market/segment passed validation.
- `INFO` — edge clears thresholds, but validation has not passed for it.
- `PASS` — no edge.
- `FCS` — FBS vs FCS; priced, never bet.
- `NO_LINE` — no market yet.

Flags (plain strings): `weather adj -4.1`, `QB uncertainty`, `large disagreement — check
ratings`, `line moved 2+ since open`, and so on.

Sanity banners at the top of every board
- Mean |model − market| spread disagreement above 6 points → "ratings likely broken".
- More than 8 `BET` rows → "too many bets — likely a data problem".
- `validation_status.json` older than 14 days → "re-run validate".
- Drift alarm active → "model weight forced to 0 — market-only pricing".

Acceptance
- With no `validation_status.json`, zero rows can be `BET` (tested).
- Golden-file test on a fixture week: board output matches the expected CSV.

---

## Phase 8 — Line logging and CLV

- Every board run snapshots the lines it used to `output/lines_log.parquet`
  (timestamp, book, line, price).
- `bets.csv` holds the owner's actual bets, added via
  `python -m cfbmodel log-bet --game ... --market ... --side ... --line ... --price ...
  --book ... --stake ...`.
- After games finish, pull CFBD closing lines and results; compute CLV in points and in
  no-vig probability, plus result and profit.
- `python -m cfbmodel clv` reports by market and segment: n, median CLV, % beating the
  close, ROI with bootstrap CI.

Acceptance
- CLV sign-convention tests for every market and side (spread home/away, total
  over/under).
- With fewer than 100 bets the report prints: "ROI is noise at this sample — read CLV."

---

## Phase 9 — Player props (optional; only after Phase 8 works)

- CFBD has no prop lines. Options: the owner enters lines in `props_lines.csv`
  (player, team, market, line, over_price, under_price, book, timestamp), or a paid odds
  API — the owner checks its NCAAF prop coverage and terms first. Never scrape books.
- Model: volume × efficiency with joint game-script simulation (`script.py`),
  role-instability volume shocks, depth chart derived from box scores, and an explicit
  `p_play` input (CFB has no injury report).
- Props become `BET`-eligible only after a backtest on logged lines shows positive CLV.

---

## Phase 10 — Dashboard

- Streamlit app (ask before adding the dependency) with pages: Edge Board, Validation,
  CLV & Bankroll, Data Health (API budget, cache freshness, unmatched team names).
- Read-only over `output/`. No betting logic lives in the UI.

Acceptance: `streamlit run app/dashboard.py` opens with every page working on fixture
data.

---

## Definition of done

- One command produces a weekly edge board with honest statuses.
- The validation gate is enforced in code and covered by tests.
- CLV is tracked for every logged bet.
- Team and player ratings update weekly; formula parameters are re-learned on a schedule
  and promoted only when proven better out-of-sample, with full version history.
- All tests pass; `docs/AUDIT.md` shows all 12 pitfalls handled.
- `README.md` explains in plain language what the board means — and what it doesn't.
