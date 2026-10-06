"""
Turning a model distribution and a book price into an honest edge (BUILD_PLAN Phase 5).

The pieces, in order of use:

  no_vig_probs      remove the book's margin. Policy: POWER for two-way spreads/totals/props;
                    SHIN for a moneyline when the favourite is shorter than -400, where the
                    multiplicative method overstates the favourite and power over-corrects.
  shrink_prob       haircut the disagreement with the market before believing it: by how soft the
                    market is, by how much data the rating rests on, and a flat 3/4.
  size_bet          shrunk probability -> EV and fractional Kelly (0.25, capped at 2%).
  key_number_value  what half a point and a full point are worth on THIS line (3 and 7 matter),
                    in equity: a win counts 1, a push counts 1/2.

Probabilities of a bet are always (win, push, lose) with win + push + lose = 1. Edges are
compared on the "no push" scale (win / (win + lose)) because that is what a de-vigged two-way
price measures.
"""
from __future__ import annotations

import numpy as np

from . import edge, game_model
from .config import C
from .params import P

HEAVY_FAVOURITE = -400.0
KEY_NUMBERS = (3, 4, 6, 7, 10, 14, 17, 21)


# ---------------------------------------------------------------------- de-vig


def no_vig_probs(odds: list[float], market: str = "spread") -> np.ndarray:
    """Fair probabilities for a complete set of American odds.

    market: "spread" | "total" | "prop" -> power; "moneyline" -> Shin if the favourite's price is
    -400 or shorter, otherwise power.
    """
    if market in ("spread", "total", "prop"):
        method = "power"
    elif market == "moneyline":
        method = "shin" if min(odds) <= HEAVY_FAVOURITE else "power"
    else:
        raise ValueError(f"unknown market {market!r}")
    return edge.devig(list(odds), method=method)


def devig_method(odds: list[float], market: str) -> str:
    """Which method no_vig_probs will use (exposed for the board and for tests)."""
    return "shin" if market == "moneyline" and min(odds) <= HEAVY_FAVOURITE else "power"


# ----------------------------------------------------------------- shrinkage


def sample_confidence(weeks_of_data: float) -> float:
    """How much to trust a rating built from this many weeks of results: n / (n + k), never
    below betting.sample_conf_floor. Week 0 ratings are pure priors; by week 12 about 3/4 confidence."""
    n = max(float(weeks_of_data), 0.0)
    return float(max(n / (n + C.sample_conf_k), P.betting.sample_conf_floor))


def shrink_factor(softness: float, sample_conf: float) -> float:
    return float(softness * sample_conf * C.edge_flat_haircut)


def shrink_prob(model_p: float, market_p: float, softness: float = 1.0, sample_conf: float = 1.0) -> float:
    """Move the market's no-vig probability only part of the way toward the model's."""
    return float(market_p + shrink_factor(softness, sample_conf) * (model_p - market_p))


# --------------------------------------------------------------------- sizing


def size_bet(p_win: float, p_push: float, price: float, market_p: float,
             softness: float = 1.0, sample_conf: float = 1.0) -> dict:
    """Edge, EV and stake for one side of one market.

    p_win / p_push are the model's probabilities for this side (loss is the remainder);
    market_p is the book's de-vigged probability of this side on the no-push scale.
    """
    p_lose = 1.0 - p_win - p_push
    decided = p_win + p_lose
    model_cond = p_win / decided if decided > 0 else 0.5
    shrunk_cond = shrink_prob(model_cond, market_p, softness, sample_conf)
    w, l = shrunk_cond * decided, (1.0 - shrunk_cond) * decided
    ev = edge.expected_value(w, price, p_push)
    return {
        "model_prob": model_cond, "no_vig_prob": float(market_p),
        "prob_edge": model_cond - market_p, "shrunk_prob": shrunk_cond,
        "shrunk_edge": shrunk_cond - market_p, "ev": ev,
        "stake_pct": edge.kelly(w, price, p_push) if ev > 0 else 0.0,
    }


# ------------------------------------------------------------- key-number value


def side_equity(side: str, exp_margin: float, exp_total: float, spread_home: float,
                source: str = "model") -> float:
    """Win probability counting a push as half a win (a push returns the stake, which at even
    money is worth exactly half a win). This is what a line move is worth: buying half a point
    across the 3 turns the 3-point margin from a loss into a push, which a plain win probability
    would miss entirely."""
    cp = game_model.cover_prob(exp_margin, exp_total, spread_home, source=source)
    return cp[side] + 0.5 * cp["push"]


def total_side_equity(side: str, exp_total: float, exp_margin: float, line: float, source: str = "model") -> float:
    """Win probability of an over/under with a push counted as half a win."""
    tp = game_model.total_probs(exp_total, line, exp_margin, source=source)
    return tp[side] + 0.5 * tp["push"]


def key_number_value(exp_margin: float, exp_total: float, spread_home: float, side: str,
                     source: str = "model") -> dict:
    """Equity (win + half a push) gained (+) or lost (-) when the bettor's line moves 0.5 or 1.0
    points in their favour (buy) or against them (sell), for `side` ("home"/"away") at
    `spread_home` (home -7.5 -> -7.5)."""
    sign = 1.0 if side == "home" else -1.0       # buying points for home raises spread_home
    base = side_equity(side, exp_margin, exp_total, spread_home, source)
    out = {}
    for step in (0.5, 1.0):
        out[f"buy_{step}"] = side_equity(side, exp_margin, exp_total, spread_home + sign * step, source) - base
        out[f"sell_{step}"] = side_equity(side, exp_margin, exp_total, spread_home - sign * step, source) - base
    return out


def key_number_note(spread_home: float) -> str:
    """Plain-English remark when the line sits on or next to a key number, else ''."""
    line = abs(float(spread_home))
    notes = []
    for k in KEY_NUMBERS:
        d = line - k
        if d == 0:
            notes.append(f"sits on the {k}")
        elif abs(d) == 0.5:
            notes.append(f"half a point {'above' if d > 0 else 'below'} the {k}")
    return "; ".join(notes[:2])
