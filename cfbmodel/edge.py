"""
Turning a probability into a bet — and, more often, into a no-bet.

Everything here is book-facing arithmetic: strip the vig, compare, size, log.
The only opinion baked in is that CLV is the KPI. Over any sample you will
actually accumulate in one CFB season (~15 weeks), win rate tells you almost
nothing and beat-the-close tells you almost everything.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import optimize

from .config import C


# ------------------------------------------------------------------ odds math


def american_to_prob(odds: float) -> float:
    return 100.0 / (odds + 100.0) if odds > 0 else -odds / (-odds + 100.0)


def american_to_decimal(odds: float) -> float:
    return 1.0 + (odds / 100.0 if odds > 0 else 100.0 / -odds)


def prob_to_american(p: float) -> float:
    if p <= 0 or p >= 1:
        return float("nan")
    return -100 * p / (1 - p) if p >= 0.5 else 100 * (1 - p) / p


# --------------------------------------------------------------------- devig


def devig(odds: list[float], method: str = "power") -> np.ndarray:
    """
    Remove the bookmaker's margin from a set of American odds on a complete
    market.

    multiplicative — divide by the overround. Fast, but systematically
        overstates favourites.
    power — solve for k with sum(p_i^k) = 1. Best general-purpose choice and
        what you should use for two-way prop markets.
    shin — assumes the overround comes from informed bettors. Best on
        longshot-heavy markets (game moneylines with a huge favourite, which
        CFB has constantly).
    """
    raw = np.array([american_to_prob(o) for o in odds], dtype=float)

    if method == "multiplicative":
        return raw / raw.sum()

    if method == "power":
        f = lambda k: np.sum(raw ** k) - 1.0
        k = optimize.brentq(f, 0.2, 5.0)
        return raw**k

    if method == "shin":
        def obj(z):
            z = float(np.clip(z, 1e-6, 0.35))
            num = np.sqrt(z**2 + 4 * (1 - z) * raw**2 / raw.sum()) - z
            return num.sum() / (2 * (1 - z)) - 1.0

        try:
            z = optimize.brentq(obj, 1e-6, 0.35)
        except ValueError:
            z = 0.0
        num = np.sqrt(z**2 + 4 * (1 - z) * raw**2 / raw.sum()) - z
        return num / (2 * (1 - z))

    raise ValueError(method)


def no_vig_two_way(over_odds: float, under_odds: float, method: str = "power") -> tuple[float, float]:
    p = devig([over_odds, under_odds], method=method)
    return float(p[0]), float(p[1])


def hold(odds: list[float]) -> float:
    return float(sum(american_to_prob(o) for o in odds) - 1.0)


# ------------------------------------------------------------------ ev / kelly


def expected_value(p_win: float, odds: float, p_push: float = 0.0) -> float:
    """EV per unit staked."""
    d = american_to_decimal(odds)
    p_lose = 1.0 - p_win - p_push
    return p_win * (d - 1.0) - p_lose


def kelly(p_win: float, odds: float, p_push: float = 0.0, fraction: float | None = None) -> float:
    """Fractional Kelly stake as a share of bankroll, capped."""
    frac = C.kelly_fraction if fraction is None else fraction
    b = american_to_decimal(odds) - 1.0
    p_lose = 1.0 - p_win - p_push
    if b <= 0:
        return 0.0
    f = (b * p_win - p_lose) / b
    f = max(f, 0.0) * frac
    return float(min(f, C.max_bet_pct))


def edge_shrinkage(raw_edge: float, market_softness: float = 1.0, sample_conf: float = 1.0) -> float:
    """
    Haircut every edge before sizing.

    Two multiplicative discounts:
      - market_softness: a 2-point disagreement with a sharp closing number is
        mostly your error; the same disagreement with a Tuesday MAC opener is
        mostly information.
      - sample_conf: how much data the rating actually rests on (week 2 vs
        week 10).

    Then a flat 0.75 on top, because the historical failure mode of every model
    in this repo's lineage is believing its own edges at face value.
    """
    return float(raw_edge * market_softness * sample_conf * 0.75)


# ----------------------------------------------------------------- bet builder


def evaluate_spread(
    model_margin: float,
    market_spread: float,
    price: float,
    cover_p: dict,
    softness: float = 1.0,
) -> dict:
    """market_spread in home convention (home -6.5 -> -6.5)."""
    fair = -model_margin
    raw_edge_pts = fair - market_spread  # positive => market has home too cheap
    side = "home" if raw_edge_pts < 0 else "away"
    p = cover_p[side]
    ev = expected_value(p, price, cover_p.get("push", 0.0))
    edge_pts = edge_shrinkage(abs(raw_edge_pts), softness)
    bet = edge_pts >= C.min_edge_spread and ev > 0
    return {
        "market": "spread",
        "side": side,
        "line": market_spread,
        "price": price,
        "model_line": round(fair, 2),
        "edge_pts": round(edge_pts, 2),
        "p_win": round(p, 4),
        "ev": round(ev, 4),
        "stake_pct": round(kelly(p, price, cover_p.get("push", 0.0)), 4) if bet else 0.0,
        "bet": bool(bet),
    }


def evaluate_total(
    model_total: float, market_total: float, price_over: float, price_under: float, probs: dict,
    softness: float = 1.0,
) -> dict:
    raw = model_total - market_total
    side = "over" if raw > 0 else "under"
    p = probs[side]
    price = price_over if side == "over" else price_under
    ev = expected_value(p, price, probs.get("push", 0.0))
    edge_pts = edge_shrinkage(abs(raw), softness)
    bet = edge_pts >= C.min_edge_total and ev > 0
    return {
        "market": "total",
        "side": side,
        "line": market_total,
        "price": price,
        "model_line": round(model_total, 2),
        "edge_pts": round(edge_pts, 2),
        "p_win": round(p, 4),
        "ev": round(ev, 4),
        "stake_pct": round(kelly(p, price, probs.get("push", 0.0)), 4) if bet else 0.0,
        "bet": bool(bet),
    }


def evaluate_prop(
    sim_probs: dict, line: float, over_odds: float, under_odds: float, player: str, market: str
) -> dict:
    """
    Props are priced against the *devigged* book number, not against the raw
    price. CFB prop holds run 6-9% — far worse than sides — so a 3% "edge"
    against the raw price is usually a negative-EV bet against the true line.
    """
    p_over_book, p_under_book = no_vig_two_way(over_odds, under_odds, method="power")
    side = "over" if sim_probs["over"] > p_over_book else "under"
    p_model = sim_probs[side]
    odds = over_odds if side == "over" else under_odds
    ev = expected_value(p_model, odds, sim_probs.get("push", 0.0))
    book_p = p_over_book if side == "over" else p_under_book
    return {
        "market": market,
        "player": player,
        "side": side,
        "line": line,
        "price": odds,
        "model_p": round(p_model, 4),
        "book_p_novig": round(book_p, 4),
        "prob_edge": round(p_model - book_p, 4),
        "model_mean": round(sim_probs["mean"], 2),
        "ev": round(ev, 4),
        "hold": round(hold([over_odds, under_odds]), 4),
        "stake_pct": round(kelly(p_model, odds), 4) if ev >= C.min_ev_props else 0.0,
        "bet": bool(ev >= C.min_ev_props),
    }


# -------------------------------------------------------------------- logging


CLV_COLS = [
    "date", "season", "week", "game", "market", "player", "side", "line", "price",
    "model_line", "model_p", "stake_pct", "close_line", "close_price", "result",
]


def clv(bet_line: float, close_line: float, market: str, side: str) -> float:
    """
    Points of closing line value. Positive = you got a better number than close.
    """
    if market == "spread":
        return (close_line - bet_line) if side == "home" else (bet_line - close_line)
    if market == "total":
        return (close_line - bet_line) if side == "under" else (bet_line - close_line)
    return (close_line - bet_line) if side == "over" else (bet_line - close_line)


def clv_report(log: pd.DataFrame) -> pd.DataFrame:
    """
    The only report that means anything before ~500 bets.

    Read it as: if median CLV is <= 0, the model is not beating the market and
    any profit so far is variance. If median CLV is positive and stable across
    weeks, the model is beating the market and any loss so far is variance.
    """
    d = log.dropna(subset=["close_line"]).copy()
    d["clv_pts"] = [
        clv(b, c, m, s)
        for b, c, m, s in zip(d["line"], d["close_line"], d["market"], d["side"])
    ]
    d["beat_close"] = d["clv_pts"] > 0
    g = d.groupby("market").agg(
        bets=("clv_pts", "size"),
        median_clv=("clv_pts", "median"),
        mean_clv=("clv_pts", "mean"),
        pct_beat_close=("beat_close", "mean"),
    )
    return g.round(3)
