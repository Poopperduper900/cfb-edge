"""
Configuration + CFB-specific constants.

The CFBD key is read by keys.py (from .env or CFBD_API_KEY); free tier: 1000 calls/month.
Get one at https://collegefootballdata.com/key
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------- paths / api

ROOT = Path(__file__).resolve().parent.parent

# Cache and output locations are overridable so they can be pointed at durable
# storage. This matters most on Colab: the runtime wipes local disk on every
# disconnect, and re-pulling a season costs ~35 calls against a 1,000/month
# budget. Point these at Google Drive once and you never pay twice.
#     export CFBMODEL_CACHE=/content/drive/MyDrive/cfbmodel/cache
CACHE = Path(os.environ.get("CFBMODEL_CACHE", ROOT / "data" / "cache"))
OUTPUT = Path(os.environ.get("CFBMODEL_OUTPUT", ROOT / "output"))
CACHE.mkdir(parents=True, exist_ok=True)
OUTPUT.mkdir(parents=True, exist_ok=True)

CFBD_BASE = "https://api.collegefootballdata.com"

# Free tier is 1,000 calls/month. Every ingest call is cached to disk so a
# backtest re-run costs zero calls. Never loop over games — pull by season/week.
CALL_BUDGET_WARN = 25

TRAIN_SEASONS = list(range(2015, 2026))  # 2020 is handled specially (COVID)
CURRENT_SEASON = 2026

# ------------------------------------------------------------ football priors
# These are starting values. `calibrate.py` re-fits them from your own data and
# writes the fitted versions to output/calibration.json. Do not treat them as
# ground truth — they are seeds.


@dataclass
class Constants:
    # Home field. CFB HFA has compressed over the last decade (empty-ish
    # stadiums, better officiating consistency, transfer-portal parity).
    # Fit per-venue in ratings.fit_hfa(); this is the league mean.
    hfa_points: float = 2.35
    hfa_neutral: float = 0.0

    # Margin distribution.
    #
    # NOTE: this is the Student-t SCALE parameter, not a standard deviation.
    # scipy's t takes scale; the resulting sd is scale * sqrt(df/(df-2)), which
    # for df=6.5 is 1.202x. Passing an intended sd here inflates the whole
    # distribution by 20% and quietly flattens every probability toward 50%.
    #
    # Calibrated so the discretised pmf reproduces the observed error of a CFB
    # closing spread: MAE 10.5 points, implied sd 13.6.
    margin_scale_market: float = 11.22
    # A model that projects margins to 13.5 MAE needs a WIDER pmf. Pricing your
    # own projection with the market's residual is a hidden claim to market-
    # level accuracy you have not demonstrated. Use this until validate says
    # otherwise.
    margin_scale_model: float = 14.68
    margin_sd_total_coef: float = 0.040  # scale = base + coef*(total - 52)
    margin_df: float = 6.5  # Student-t dof; fatter tails than normal

    total_sd_base: float = 12.6
    # A totals projection from our own ratings is less accurate than the market's total, so its
    # distribution must be wider (same logic as pitfall 2 for margins). UNFITTED placeholder; the
    # learning loop (Phase 6) replaces it with a value fitted on past totals.
    total_sd_model_mult: float = 1.2

    # Key numbers, CFB-specific. Weight = multiplicative bump applied to the
    # discretized margin pmf. Fitted in calibrate.py from 2015-2025 margins.
    key_numbers: dict = field(
        default_factory=lambda: {
            3: 1.34,
            7: 1.28,
            10: 1.16,
            14: 1.15,
            4: 1.07,
            6: 1.06,
            17: 1.08,
            21: 1.09,
            1: 1.05,
            8: 1.04,
            24: 1.04,
            28: 1.03,
        }
    )

    # Garbage-time filter for play-level ratings. Plays outside this win-prob
    # band are dropped; CFB needs a tighter filter than the NFL.
    gt_wp_low: float = 0.05
    gt_wp_high: float = 0.95
    gt_min_quarter_4_margin: int = 22  # also drop Q4 plays past this margin

    # Recency weighting on play-by-play, in games.
    form_half_life_games: float = 7.0
    # Cross-season carryover: how much of last year's rating survives into the
    # preseason prior. Portal era -> lower than it used to be.
    season_carryover: float = 0.52

    # Ridge shrinkage for opponent adjustment (tuned by CV in ratings.py)
    ridge_alpha_off: float = 220.0
    ridge_alpha_def: float = 260.0

    # Pace: plays per team per game, league mean & spread
    pace_mean: float = 68.5

    # Betting
    min_edge_spread: float = 1.6      # model vs market, in points
    min_edge_total: float = 2.8
    min_ev_props: float = 0.045       # props carry 6-9% hold; demand more
    kelly_fraction: float = 0.25
    edge_flat_haircut: float = 0.75   # believe only 3/4 of any disagreement with the market
    sample_conf_k: float = 4.0        # weeks of data at which sample confidence is 50%
    max_bet_pct: float = 0.02


C = Constants()

# ----------------------------------------------------------- market structure
# Where CFB markets are actually soft. Used to gate bet sizing, not to
# manufacture edges.

SOFT_MARKET_TAGS = {
    "week0_week1": 1.00,       # openers: market has no current-season data either
    "g5_vs_g5": 0.85,          # low limits, less sharp money
    "weeknight_midmajor": 0.80,
    "fcs_opponent": 0.60,      # lines often untradeable / huge numbers
    "p4_marquee": 0.25,        # efficient; you are the sucker until proven otherwise
    "bowl_opt_outs": 0.30,     # roster chaos, not model territory
}

# Altitude venues (yards/points adjustment on visiting offenses late in games)
ALTITUDE_TEAMS = {
    "Air Force": 2010, "Colorado": 5360, "Colorado State": 5000,
    "Wyoming": 7220, "Utah": 4640, "BYU": 4550, "New Mexico": 5310,
    "Utah State": 4780, "Boise State": 2730, "Nevada": 4500,
}
