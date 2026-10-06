"""
Configuration + CFB-specific constants.

The CFBD key is read by keys.py (from .env or CFBD_API_KEY); free tier: 1000 calls/month.
Get one at https://collegefootballdata.com/key
"""
from __future__ import annotations

import os
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

# ------------------------------------------------------------ model parameters
# Every tunable number (home-field advantage, margin-distribution scales, key-number weights,
# ridge penalties, betting thresholds, weather and game-script coefficients, ...) now lives in the
# versioned registry params/<version>.json, read through params.py. `C` is a live view of the
# active version under the attribute names the older code used (C.hfa_points, C.margin_df, ...).
# Starting values were hand-set; `python -m cfbmodel learn` replaces them version by version, and
# only when a challenger is proven better out-of-sample (Phase 6).

from .params import Constants  # noqa: E402

C = Constants()

# ----------------------------------------------------------- market structure
# How soft each kind of market is (a multiplier on how far we trust a disagreement with the price)
# is in the registry too: P.softness.<tag>. Data that is not a tunable stays here:

# Altitude venues (yards/points adjustment on visiting offenses late in games)
ALTITUDE_TEAMS = {
    "Air Force": 2010, "Colorado": 5360, "Colorado State": 5000,
    "Wyoming": 7220, "Utah": 4640, "BYU": 4550, "New Mexico": 5310,
    "Utah State": 4780, "Boise State": 2730, "Nevada": 4500,
}
