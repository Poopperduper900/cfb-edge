"""Phase 5 acceptance: pricing invariants."""
from __future__ import annotations

import itertools

import numpy as np
import pytest

from cfbmodel import edge, game_model, pricing
from cfbmodel.config import C


# ------------------------------------------------------------------- de-vig


def test_devig_ordering_on_a_lopsided_market():
    """multiplicative < Shin < power for the favourite (-2500/+1100)."""
    p = {m: edge.devig([-2500, 1100], m)[0] for m in ("multiplicative", "shin", "power")}
    assert p["multiplicative"] < p["shin"] < p["power"]


@pytest.mark.parametrize("odds", [[-110, -110], [-150, 130], [-400, 320], [-2500, 1100], [105, -125]])
@pytest.mark.parametrize("method", ["multiplicative", "power", "shin"])
def test_every_devig_method_returns_a_probability_pair_summing_to_one(odds, method):
    p = edge.devig(odds, method)
    assert p.sum() == pytest.approx(1.0, abs=1e-6) and (p > 0).all() and (p < 1).all()
    assert p[0] > p[1] if odds[0] < odds[1] else True            # favourite stays the favourite


def test_devig_policy_power_for_two_way_shin_for_heavy_moneyline_favourites():
    assert pricing.devig_method([-110, -110], "spread") == "power"
    assert pricing.devig_method([-115, -105], "total") == "power"
    assert pricing.devig_method([-300, 250], "moneyline") == "power"
    assert pricing.devig_method([-400, 320], "moneyline") == "shin"          # exactly -400 counts
    assert pricing.devig_method([-2500, 1100], "moneyline") == "shin"
    np.testing.assert_allclose(pricing.no_vig_probs([-2500, 1100], "moneyline"), edge.devig([-2500, 1100], "shin"))
    np.testing.assert_allclose(pricing.no_vig_probs([-115, -105], "spread"), edge.devig([-115, -105], "power"))
    with pytest.raises(ValueError):
        pricing.no_vig_probs([-110, -110], "parlay")


# -------------------------------------------------------------------- Kelly


def test_kelly_is_zero_with_no_edge_and_never_exceeds_the_cap():
    for odds in (-110, -150, 120, 300, -1000):
        breakeven = edge.american_to_prob(odds)
        assert edge.kelly(breakeven, odds) == 0.0
        assert edge.kelly(breakeven - 0.05, odds) == 0.0
        for p in np.linspace(0.01, 0.99, 99):
            k = edge.kelly(p, odds)
            assert 0.0 <= k <= C.max_bet_pct + 1e-12
    assert edge.kelly(0.99, 500) == C.max_bet_pct          # a huge "edge" is still capped


def test_kelly_grows_with_probability_and_uses_a_quarter_stake():
    ks = [edge.kelly(p, -110) for p in np.linspace(0.5, 0.7, 20)]
    assert all(b >= a for a, b in zip(ks, ks[1:]))
    p, odds = 0.545, -110                                    # small edge: below the cap
    b = edge.american_to_decimal(odds) - 1
    full = (b * p - (1 - p)) / b
    assert edge.kelly(p, odds) == pytest.approx(0.25 * full)


# --------------------------------------------------------- probabilities sum to 1


@pytest.mark.parametrize("source", ["market", "model"])
@pytest.mark.parametrize("exp_margin,exp_total", [(-6.5, 54.0), (0.0, 48.0), (13.0, 61.0), (-24.0, 44.0)])
def test_cover_push_and_other_side_sum_to_one(exp_margin, exp_total, source):
    for line in (-14.0, -10.5, -7.0, -6.5, -3.0, -0.5, 0.0, 2.5, 3.0, 7.0, 10.0):
        cp = game_model.cover_prob(exp_margin, exp_total, line, source=source)
        assert cp["home"] + cp["away"] + cp["push"] == pytest.approx(1.0, abs=1e-9)
        assert min(cp.values()) >= 0.0
        if float(line) != int(line) or line == 0.0:
            assert cp["push"] == 0.0     # half-point lines cannot push; neither can a pick'em (ties do not exist)
        else:
            assert cp["push"] > 0.0


@pytest.mark.parametrize("source", ["market", "model"])
def test_totals_and_moneylines_sum_to_one(source):
    for exp_total, line, exp_margin in itertools.product((40.0, 52.0, 66.0), (38.5, 47.0, 52.5, 61.0), (-20.0, 0.0, 9.0)):
        tp = game_model.total_probs(exp_total, line, exp_margin, source=source)
        assert tp["over"] + tp["under"] + tp["push"] == pytest.approx(1.0, abs=1e-9)
    ml = game_model.moneyline_prob(-6.5, 54.0, source=source)
    assert ml["home"] + ml["away"] == pytest.approx(1.0, abs=1e-9)
    assert ml["home"] < 0.5 < ml["away"]                     # home projected to lose by 6.5


def test_model_source_totals_are_wider_than_market_source():
    over_m = game_model.total_probs(52.0, 60.5, source="market")["over"]
    over_x = game_model.total_probs(52.0, 60.5, source="model")["over"]
    assert over_x > over_m                                   # fatter tail => more weight past the line
    with pytest.raises(ValueError):
        game_model.total_probs(52.0, 50.5, source="oracle")


# ------------------------------------------------------------- shrinking and sizing


def test_shrinkage_moves_the_market_probability_only_part_of_the_way():
    assert pricing.shrink_prob(0.60, 0.50, softness=1.0, sample_conf=1.0) == pytest.approx(0.50 + 0.75 * 0.10)
    assert pricing.shrink_prob(0.60, 0.50, softness=0.25, sample_conf=0.5) == pytest.approx(0.50 + 0.25 * 0.5 * 0.75 * 0.10)
    assert pricing.shrink_prob(0.50, 0.50) == 0.5
    c = [pricing.sample_confidence(w) for w in (0, 2, 4, 8, 12)]
    assert c[0] == 0.25 and all(a < b for a, b in zip(c, c[1:])) and c[-1] < 1.0


def test_size_bet_no_edge_means_no_stake_and_stake_respects_the_cap():
    flat = pricing.size_bet(0.4762, 0.0, -110, market_p=0.5, softness=1.0, sample_conf=1.0)
    assert flat["stake_pct"] == 0.0 and flat["ev"] < 0
    big = pricing.size_bet(0.80, 0.0, -110, market_p=0.5, softness=1.0, sample_conf=1.0)
    assert 0 < big["stake_pct"] <= C.max_bet_pct
    soft = pricing.size_bet(0.58, 0.0, -110, market_p=0.5, softness=1.0, sample_conf=1.0)
    hard = pricing.size_bet(0.58, 0.0, -110, market_p=0.5, softness=0.25, sample_conf=0.5)
    assert soft["ev"] > hard["ev"] and soft["stake_pct"] >= hard["stake_pct"]
    assert hard["prob_edge"] == soft["prob_edge"] and hard["shrunk_edge"] < soft["shrunk_edge"]


def test_size_bet_compares_on_the_no_push_scale():
    s = pricing.size_bet(0.45, 0.10, -110, market_p=0.5)     # 45 win, 10 push, 45 lose
    assert s["model_prob"] == pytest.approx(0.5)
    assert s["prob_edge"] == pytest.approx(0.0)


# ------------------------------------------------------------------ key numbers


def test_buying_half_a_point_onto_a_number_is_worth_half_that_numbers_probability():
    """Buying -3.5 -> -3 turns a 3-point loss into a push, worth half of P(margin = 3). So a half
    point is worth most exactly where the margin distribution piles up: 3 and 7 beat 5."""
    xs, pmf = game_model.margin_pmf(-6.0, 52.0, source="market")
    p = {m: float(pmf[xs == m][0]) for m in (3, 5, 7)}
    value = lambda line: pricing.key_number_value(-6.0, 52.0, line, "home", "market")["buy_0.5"]
    assert value(-3.5) == pytest.approx(0.5 * p[3], abs=1e-12)
    assert value(-5.5) == pytest.approx(0.5 * p[5], abs=1e-12)
    assert value(-7.5) == pytest.approx(0.5 * p[7], abs=1e-12)
    assert value(-3.5) > value(-5.5)                 # the 3 is worth more than an ordinary number


def test_buying_points_helps_and_selling_hurts_for_both_sides():
    for side in ("home", "away"):
        v = pricing.key_number_value(-4.0, 51.0, -6.5, side, "model")
        assert v["buy_0.5"] > 0 > v["sell_0.5"] and v["buy_1.0"] >= v["buy_0.5"] and v["sell_1.0"] <= v["sell_0.5"]


def test_key_number_notes_are_plain_english():
    assert pricing.key_number_note(-3.0) == "sits on the 3"
    assert "half a point below the 3" in pricing.key_number_note(-2.5)
    assert "half a point above the 7" in pricing.key_number_note(-7.5)
    assert pricing.key_number_note(-5.0) == ""
    assert pricing.key_number_note(-3.0) == pricing.key_number_note(3.0)       # either sign
