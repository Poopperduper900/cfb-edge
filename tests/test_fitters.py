"""Phase 6b: the fitters recover known parameters from synthetic data (generated with
game_model.margin_pmf, an independent implementation, not with the fitter's own likelihood)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from cfbmodel import fitters, game_model, params, script
from cfbmodel.params import P

KEYS_37 = {"margin": {"key_numbers": {"3": 1.34, "7": 1.28}}}     # fewer keys keeps the fit fast


def sample_margins(rng, exp_margin, exp_total, source="market"):
    """One margin per game drawn from margin_pmf at the ACTIVE parameters."""
    out = np.empty(len(exp_margin))
    for i, (m, t) in enumerate(zip(exp_margin, exp_total)):
        xs, pmf = game_model.margin_pmf(float(m), float(t), source=source)
        out[i] = rng.choice(xs, p=pmf)
    return out


def test_vectorised_likelihood_matches_margin_pmf():
    rng = np.random.default_rng(0)
    mu, tot = rng.normal(0, 10, 12), rng.normal(52, 8, 12)
    m = np.array([3, -7, 14, 1, -1, 10, -24, 5, 8, -3, 21, 2], dtype=float)
    lp = fitters.margin_logprob(m, mu, tot, scale=P.margin.scale_market, df=P.margin.df,
                                sd_total_coef=P.margin.sd_total_coef,
                                key_weights={int(k): v for k, v in P.margin.key_numbers.items()})
    for i in range(len(m)):
        xs, pmf = game_model.margin_pmf(mu[i], tot[i], source="market")
        assert lp[i] == pytest.approx(np.log(pmf[xs == int(m[i])][0]), abs=1e-9)


def test_recovers_margin_distribution_parameters_including_a_key_number_weight_of_1_5():
    rng = np.random.default_rng(1)
    n = 8000
    spread = np.round(rng.normal(0, 11, n) * 2) / 2
    total = rng.normal(52, 7, n)
    truth = {"margin": {"scale_market": 12.5, "df": 5.0, "sd_total_coef": 0.03,
                        "key_numbers": {"3": 1.5, "7": 1.28}}}
    with params.override(truth):
        margin = sample_margins(rng, -spread, total)
    df = pd.DataFrame({"margin": margin, "spread_close": spread, "total_close": total})
    with params.override(KEYS_37):                                   # fit starts from the defaults
        fit = fitters.fit_margin_pmf(df)
    c = fit.changes["margin"]
    assert c["key_numbers"]["3"] == pytest.approx(1.5, abs=0.15)
    assert c["key_numbers"]["7"] == pytest.approx(1.28, abs=0.2)
    assert c["scale_market"] == pytest.approx(12.5, abs=0.8)
    assert c["df"] == pytest.approx(5.0, abs=1.5)
    assert fit.info["nll_after"] < fit.info["nll_before"]


def test_recovers_home_field_advantage_of_3_1_and_the_model_error_scale():
    rng = np.random.default_rng(2)
    n = 4000
    core = rng.normal(0, 10, n)
    neutral = rng.random(n) < 0.1
    total = rng.normal(52, 7, n)
    with params.override({"margin": {"scale_model": 14.0, "key_numbers": {"3": 1.34, "7": 1.28}}}):
        margin = sample_margins(rng, core + 3.1 * (~neutral), total, source="model")
        df = pd.DataFrame({"margin": margin, "core_model": core, "neutral": neutral, "total_close": total})
        fit = fitters.fit_model_margin(df)
    assert fit.changes["margin"]["hfa_points"] == pytest.approx(3.1, abs=0.35)
    assert fit.changes["margin"]["scale_model"] == pytest.approx(14.0, abs=1.2)


def test_neutral_site_games_do_not_pull_the_home_field_estimate():
    rng = np.random.default_rng(3)
    n = 3000
    core = rng.normal(0, 10, n)
    neutral = np.zeros(n, dtype=bool); neutral[: n // 2] = True
    total = np.full(n, 52.0)
    with params.override({"margin": {"key_numbers": {"3": 1.34, "7": 1.28}}}):
        margin = sample_margins(rng, core + 3.1 * (~neutral), total, source="model")
        fit = fitters.fit_model_margin(pd.DataFrame(
            {"margin": margin, "core_model": core, "neutral": neutral, "total_close": total}))
    assert fit.changes["margin"]["hfa_points"] == pytest.approx(3.1, abs=0.5)


def test_recovers_the_early_season_decay_half_life():
    rng = np.random.default_rng(4)
    n = 2400
    week = rng.integers(0, 6, n)
    truth = rng.normal(0, 10, n)
    unshrunk = truth + rng.normal(0, 8, n)      # early in-season ratings are noisy
    prior = truth + rng.normal(0, 4, n)         # the preseason prior is better early on
    from cfbmodel import state
    pw = np.array([state.early_season_prior_weight(int(w), 3.0) for w in week])
    exp = (1 - pw) * unshrunk + pw * prior + 2.35
    total = np.full(n, 52.0)
    with params.override({"margin": {"key_numbers": {"3": 1.34, "7": 1.28}}}):
        margin = sample_margins(rng, exp, total, source="model")
    df = pd.DataFrame({"margin": margin, "week": week, "core_unshrunk": unshrunk, "core_prior": prior,
                       "neutral": False, "total_close": total})
    fit = fitters.fit_decay(df)
    assert fit.changes["state"]["decay_half_life"] == pytest.approx(3.0, abs=1.2)


def test_recovers_totals_spread_and_the_model_total_widening():
    rng = np.random.default_rng(5)
    n = 6000
    spread = rng.normal(0, 10, n)
    mu = rng.normal(52, 6, n)
    sd = 11.0 + 0.05 * np.abs(spread)
    actual = np.round(rng.normal(mu, sd))
    model_total = mu + rng.normal(0, 1, n)
    sd_model = sd * 1.3
    actual_for_model = np.round(rng.normal(model_total, sd_model))
    df = pd.DataFrame({"actual_total": actual, "total_close": mu, "spread_close": spread})
    fit = fitters.fit_totals(df)
    assert fit.changes["totals"]["sd_base"] == pytest.approx(11.0, abs=0.7)
    assert fit.changes["totals"]["sd_margin_coef"] == pytest.approx(0.05, abs=0.03)
    df2 = pd.DataFrame({"actual_total": actual_for_model, "total_close": model_total, "spread_close": spread,
                        "model_total": model_total})
    fit2 = fitters.fit_totals(df2)
    # with the widening folded into the closing-total fit the combined width is ~1.3x; the multiplier
    # fit on model totals is relative to that fitted base, so check the product
    c = fit2.changes["totals"]
    assert c["sd_base"] * c.get("sd_model_mult", 1.0) == pytest.approx(11.0 * 1.3, rel=0.15) \
        or c["sd_base"] == pytest.approx(11.0 * 1.3, rel=0.15)


def test_weather_fit_recovers_coefficients_and_ignores_dome_games():
    """Noise is 6 points here, not the real ~12: with 12 the cold term's standard error (~0.03) is as
    big as the effect, so the test would be checking luck, not the fitter."""
    rng = np.random.default_rng(6)
    n, nd = 6000, 800
    W = P.weather
    wind = np.clip(rng.gamma(3, 4, n), 0, 40)
    gust = wind + rng.gamma(2, 3, n)
    precip = np.where(rng.random(n) < 0.15, rng.gamma(2, 0.15, n), 0.0)
    temp = rng.normal(40, 20, n)         # cold enough, often enough, for the cold term to be identifiable
    conf = rng.uniform(0.4, 1.0, n)
    effect = conf * (0.5 * np.maximum(wind - W.wind_threshold_mph, 0) + 0.2 * np.maximum(gust - W.gust_threshold_mph, 0)
                     + 1.5 * np.where(precip > W.precip_threshold_in, np.minimum(precip / W.precip_ref_in, W.precip_cap_units), 0)
                     + 0.06 * np.maximum(W.cold_threshold_f - temp, 0))
    outdoor = pd.DataFrame({"actual_total": 52 - effect + rng.normal(0, 6, n), "total_close": 52.0,
                            "wind_mph": wind, "gust_mph": gust, "precip_in": precip, "temp_f": temp,
                            "wx_confidence": conf, "dome": False})
    dome = pd.DataFrame({"actual_total": 52 + rng.normal(0, 6, nd), "total_close": 52.0,
                         "wind_mph": 38.0, "gust_mph": 50.0, "precip_in": 1.0, "temp_f": 10.0,
                         "wx_confidence": 1.0, "dome": True})            # storm outside, no effect inside
    both = pd.concat([outdoor, dome], ignore_index=True)
    fit = fitters.fit_weather(both)
    c = fit.changes["weather"]
    assert c["wind_coef"] == pytest.approx(0.5, abs=0.12)
    assert c["gust_coef"] == pytest.approx(0.2, abs=0.12)
    assert c["precip_coef"] == pytest.approx(1.5, abs=0.8)
    assert c["cold_coef"] == pytest.approx(0.06, abs=0.03)
    assert fit.info["dome_rows_excluded"] == nd
    # the guard matters: counting dome games as outdoor storms dilutes the wind effect
    diluted = fitters.fit_weather(both.assign(dome=False)).changes["weather"]["wind_coef"]
    assert diluted < c["wind_coef"] * 0.8


def test_weather_coefficients_are_never_negative():
    rng = np.random.default_rng(7)
    n = 1500
    df = pd.DataFrame({"actual_total": 52 + rng.normal(0, 12, n), "total_close": 52.0,    # no effect at all
                       "wind_mph": rng.gamma(3, 4, n), "gust_mph": np.nan, "precip_in": 0.0,
                       "temp_f": rng.normal(55, 18, n), "wx_confidence": 1.0, "dome": False})
    assert all(v >= 0 for v in fitters.fit_weather(df).changes["weather"].values())


def test_pace_fit_finds_points_per_play():
    rng = np.random.default_rng(8)
    n = 3000
    pace_sum = rng.normal(137, 8, n)
    total = 52 + 0.4 * (pace_sum - 137) + rng.normal(0, 10, n)
    fit = fitters.fit_pace(pd.DataFrame({"actual_total": total, "pace_sum": pace_sum}))
    assert fit.changes["game"]["pace_total_slope"] == pytest.approx(0.4, abs=0.08)
    assert fit.changes["ratings"]["pace_mean"] == pytest.approx(68.5, abs=0.6)


# ---------------------------------------------------------------- CRPS and PIT


def test_crps_matches_the_brute_force_definition_and_prefers_the_better_forecast():
    rng = np.random.default_rng(9)
    sim = rng.normal(80, 20, 600)
    obs = 95.0
    brute = np.abs(sim - obs).mean() - 0.5 * np.abs(sim[:, None] - sim[None, :]).mean()
    assert fitters.crps_ensemble(sim, obs) == pytest.approx(brute, rel=1e-9)
    assert fitters.crps_ensemble(rng.normal(95, 20, 600), obs) < fitters.crps_ensemble(rng.normal(40, 20, 600), obs)


def test_pit_is_uniform_for_a_calibrated_forecaster_and_not_otherwise():
    rng = np.random.default_rng(10)
    good, bad = [], []
    for _ in range(400):
        mu = rng.normal(80, 25)
        obs = rng.normal(mu, 20)
        good.append(fitters.pit_value(rng.normal(mu, 20, 800), obs))
        bad.append(fitters.pit_value(rng.normal(mu, 6, 800), obs))      # overconfident
    assert fitters.pit_uniformity(np.array(good)) < 0.08
    assert fitters.pit_uniformity(np.array(bad)) > 0.2


def _rush_rows(n, seed, **over):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "exp_margin": rng.choice([-24.0, 0.0, 24.0], n), "exp_total": 55.0, "pace": 68.0,
        "carry_share": 0.55, "ypc": 5.1, "ypc_sd": 6.4, "shock_sigma": 0.18, "actual_yards": 80.0, **over})


def test_rush_prop_scoring_is_deterministic_and_fit_proe_returns_a_candidate_from_its_grid():
    rows = _rush_rows(30, 1)
    a, b = fitters.score_rush_props(rows, n_sims=300), fitters.score_rush_props(rows, n_sims=300)
    np.testing.assert_array_equal(a["crps"], b["crps"])
    now = P.script.proe_slope
    fit = fitters.fit_proe(rows, factors=(0.5, 1.0, 1.5), n_sims=300)
    assert fit.changes["script"]["proe_slope"] in (now * 0.5, now * 1.0, now * 1.5)
    assert set(fit.info["crps_by_factor"]) == {0.5, 1.0, 1.5}
    assert P.script.proe_slope == now                              # scoring did not leak parameters


def test_proe_fit_prefers_the_parameter_that_generated_the_outcomes():
    """Outcomes generated with NO pass-rate response (slope 0): the fit should not prefer 1.5x."""
    rng = np.random.default_rng(12)
    rows = _rush_rows(60, 2)
    actual = []
    with params.override({"script": {"proe_slope": 0.0}}):
        for i, r in enumerate(rows.itertuples(index=False)):
            script.RNG = np.random.default_rng(1000 + i)
            sc = script.simulate_game_script(r.exp_margin, r.exp_total, r.pace, n_sims=1)
            actual.append(float(script.simulate_rush_yards_joint(r.carry_share, sc, r.ypc, r.ypc_sd, r.shock_sigma)[0]))
    rows["actual_yards"] = actual
    fit = fitters.fit_proe(rows, factors=(0.0, 1.0, 2.0), n_sims=600)
    table = fit.info["crps_by_factor"]
    assert table[0.0] <= table[2.0]


# ------------------------------------------------------------- ratings hyper grid


def test_ratings_grid_search_returns_the_best_point_and_restores_parameters():
    seen = []

    def run():
        seen.append(P.ratings.ridge_alpha_off)
        return P.ratings.ridge_alpha_off

    res = fitters.fit_ratings_hyper(run, {("ratings", "ridge_alpha_off"): [100.0, 300.0, 500.0]},
                                    score=lambda f: -abs(f - 300.0))
    assert res.changes == {"ratings": {"ridge_alpha_off": 300.0}}
    assert seen == [100.0, 300.0, 500.0] and P.ratings.ridge_alpha_off == 220.0


def test_vectorised_home_cover_probability_matches_game_model_cover_prob():
    rng = np.random.default_rng(21)
    n = 300
    mu = rng.normal(0, 14, n)
    tot = rng.normal(52, 9, n)
    spread = np.concatenate([np.round(rng.normal(0, 12, n // 2) * 2) / 2,           # half and whole points
                             rng.choice([-90.0, -71.0, -70.5, 0.0, 70.0, 80.0], n - n // 2)])  # edge cases
    for source, scale in (("market", P.margin.scale_market), ("model", P.margin.scale_model)):
        got = fitters.home_cover_prob(mu, tot, spread, scale=scale, df=P.margin.df,
                                      sd_total_coef=P.margin.sd_total_coef,
                                      key_weights={int(k): v for k, v in P.margin.key_numbers.items()})
        want = np.array([game_model.cover_prob(float(m), float(t), float(sp), source=source)["home"]
                         for m, t, sp in zip(mu, tot, spread)])
        np.testing.assert_allclose(got, want, atol=1e-9)


def _kw():
    return dict(scale=P.margin.scale_market, df=P.margin.df, sd_total_coef=P.margin.sd_total_coef,
                key_weights={int(k): v for k, v in P.margin.key_numbers.items()})


def test_margin_pit_is_uniform_when_the_distribution_is_right_and_not_when_too_narrow_or_wide():
    rng = np.random.default_rng(31)
    n = 3000
    mu = rng.normal(0, 10, n)
    tot = rng.normal(52, 7, n)
    from synth import draw_margins
    m = draw_margins(rng, mu, tot, scale=P.margin.scale_market, df=P.margin.df, sd_total_coef=P.margin.sd_total_coef,
                     keys={int(k): v for k, v in P.margin.key_numbers.items()})
    right = fitters.pit_uniformity(fitters.margin_pit(m, mu, tot, **_kw()))
    narrow = fitters.pit_uniformity(fitters.margin_pit(m, mu, tot, **{**_kw(), "scale": 7.0}))
    wide = fitters.pit_uniformity(fitters.margin_pit(m, mu, tot, **{**_kw(), "scale": 18.0}))
    assert right < 0.03
    assert narrow > 0.05 and wide > 0.05 and min(narrow, wide) > 3 * right      # wrong widths are clearly worse


def test_total_pit_is_uniform_for_the_true_spread_and_not_for_a_wrong_one():
    rng = np.random.default_rng(32)
    n = 3000
    mu, m = rng.normal(52, 6, n), rng.normal(0, 10, n)
    actual = np.round(rng.normal(mu, 11.0 + 0.05 * np.abs(m)))
    good = fitters.pit_uniformity(fitters.total_pit(actual, mu, m, sd_base=11.0, sd_margin_coef=0.05))
    bad = fitters.pit_uniformity(fitters.total_pit(actual, mu, m, sd_base=6.0, sd_margin_coef=0.05))
    assert good < 0.03 and bad > 0.1 and bad > 4 * good


def test_the_tail_probability_is_one_at_the_far_left_and_zero_at_the_far_right_and_decreasing():
    x0 = np.array([-80.0, -20.0, -3.0, 3.0, 20.0, 80.0])
    t = fitters.margin_tail_prob(x0, np.zeros(6), np.full(6, 52.0), **_kw())
    assert t[0] == pytest.approx(1.0, abs=1e-9) and t[-1] == pytest.approx(0.0, abs=1e-9)
    assert (np.diff(t) < 0).all()
