"""
Regression tests for the known pitfalls listed in CLAUDE.md. Each docstring names the pitfall
it guards. Synthetic data with known truth only (tests/synth.py).
"""
from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from cfbmodel import game_model, props, qb, ratings, script, weather
from cfbmodel.config import C
from synth import FCS_NAMES, make_league

PKG = Path(__file__).resolve().parent.parent / "cfbmodel"


def _mae_sd(xs, pmf):
    return float((np.abs(xs) * pmf).sum()), float(np.sqrt((xs ** 2 * pmf).sum()))


@pytest.fixture(scope="module")
def league():
    return make_league(seed=11, n_teams=40, seasons=(2024, 2025), weeks=range(1, 13),
                       plays_per_team_game=68)


# --------------------------------------------------------------------- #1, #2


def test_t_scale():
    """Pitfall 1: scipy's t takes a SCALE, not an SD. The market-source pmf must reproduce a
    closing spread's error: MAE 10.5 +- 0.4, SD 13.6 +- 0.5."""
    xs, pmf = game_model.margin_pmf(0.0, 52.0, source="market")
    mae, sd = _mae_sd(xs, pmf)
    assert abs(mae - 10.5) <= 0.4, mae
    assert abs(sd - 13.6) <= 0.5, sd
    # the relation itself: sd = scale * sqrt(df / (df - 2)) = 1.202 x scale at df = 6.5
    assert C.margin_df == pytest.approx(6.5)
    assert stats.t(C.margin_df).std() == pytest.approx(1.202, abs=0.001)


def test_model_pmf_wider():
    """Pitfall 2: pricing your own projection with the market's residual overstates edge. The
    model-source pmf must be at least 1.25x the market-source MAE, and 'model' is the default."""
    xs, market = game_model.margin_pmf(0.0, 52.0, source="market")   # centred, so |x| is the error
    _, model = game_model.margin_pmf(0.0, 52.0, source="model")
    assert _mae_sd(xs, model)[0] >= 1.25 * _mae_sd(xs, market)[0]
    _, default = game_model.margin_pmf(-6.5, 54.0)
    _, explicit = game_model.margin_pmf(-6.5, 54.0, source="model")
    np.testing.assert_array_equal(default, explicit)
    assert game_model.cover_prob(-6.5, 54.0, -3.5) == game_model.cover_prob(-6.5, 54.0, -3.5, source="model")


# ------------------------------------------------------------------------- #3


def test_hfa_not_shrunk_market_fit(league):
    """Pitfall 3: ridge penalises every column including home-field advantage, so a large alpha
    collapses it (observed: 1.30 fitted vs 2.94 true). At alpha = 1000 the fitted HFA must stay
    within 0.3 of the realised home advantage."""
    ln = league["lines"]
    realised = (-ln["spread_close"]).mean() - (
        ln["home"].map(league["truth"]) - ln["away"].map(league["truth"])).mean()
    fit = ratings.fit_market_ratings(ln, 2026, 1, half_life_weeks=1e9, alpha=1000.0)
    assert abs(fit["market_hfa"].iloc[0] - realised) <= 0.3, (fit["market_hfa"].iloc[0], realised)


def test_hfa_not_shrunk_epa_fit():
    """Pitfall 3, EPA fit: a planted home EPA edge (0.5/play, exaggerated so sampling noise of
    about 0.07 is small next to it) survives a heavy ridge penalty. Without the column scaling
    the fit collapses to about 0.01, so the tolerance separates the two cases cleanly."""
    lg = make_league(seed=3, n_teams=12, seasons=(2025,), weeks=range(1, 5), plays_per_team_game=200,
                     home_epa=0.5)
    fit = ratings.fit_epa_ratings(lg["plays"], 2026, 1, split_pass_rush=False, alpha=20_000.0)
    assert abs(fit["hfa_league"].iloc[0] - 0.5) <= 0.15, fit["hfa_league"].iloc[0]


# ------------------------------------------------------------------------- #4


def test_rating_recovery(league):
    """Pitfall 4: a wrong sign in the ridge design silently gave r = 0.11 against true strength.
    EPA ridge must recover truth with r > 0.9; market-implied ratings with r > 0.98."""
    truth = league["truth"]
    epa = ratings.fit_epa_ratings(league["plays"], 2026, 1, split_pass_rush=False)
    mkt = ratings.fit_market_ratings(league["lines"], 2026, 1, half_life_weeks=1e9)
    r_epa = np.corrcoef(epa["net_epa"].reindex(truth.index), truth)[0, 1]
    r_mkt = np.corrcoef(mkt["market_rating"].reindex(truth.index), truth)[0, 1]
    assert r_epa > 0.9, r_epa
    assert r_mkt > 0.98, r_mkt


# ------------------------------------------------------------------------- #5


def _iterrows_label_names():
    """Every `for <label>, <row> in x.iterrows()` in the package: (file, line, label name)."""
    found = []
    for path in sorted(PKG.glob("*.py")):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.For) and isinstance(node.iter, ast.Call) \
                    and isinstance(node.iter.func, ast.Attribute) and node.iter.func.attr == "iterrows":
                target = node.target
                label = target.elts[0] if isinstance(target, ast.Tuple) else target
                found.append((path.name, node.lineno, getattr(label, "id", "?")))
    return found


def test_no_positional_iterrows_static():
    """Pitfall 5: iterrows() yields index LABELS, not positions. Nothing may use the label, so
    every loop must discard it (`for _, row in ...`)."""
    assert _iterrows_label_names(), "expected to find iterrows loops (is the scan broken?)"
    bad = [(f, line, name) for f, line, name in _iterrows_label_names() if name != "_"]
    assert not bad, f"iterrows label is used in: {bad}"


def test_no_positional_iterrows_behavioural(league):
    """Pitfall 5: results on a filtered, non-contiguous frame equal results on the same rows
    re-indexed 0..n-1. An array indexed by a stale label would differ."""
    plays = league["plays"]
    ln = league["lines"]
    keep = (plays["week"] % 2 == 1) & (plays.index % 3 != 0)
    sparse_plays = plays[keep]
    sparse_ln = ln[ln["week"] % 3 != 0]
    assert sparse_plays.index.to_series().diff().max() > 1       # really non-contiguous

    def same(a, b):
        pd.testing.assert_frame_equal(a, b, check_exact=False, rtol=1e-9, atol=1e-9)

    same(ratings.fit_epa_ratings(sparse_plays, 2026, 1, split_pass_rush=False),
         ratings.fit_epa_ratings(sparse_plays.reset_index(drop=True), 2026, 1, split_pass_rush=False))
    same(ratings.fit_market_ratings(sparse_ln, 2026, 1),
         ratings.fit_market_ratings(sparse_ln.reset_index(drop=True), 2026, 1))
    same(ratings.fit_total_ratings(sparse_ln, 2026, 1),
         ratings.fit_total_ratings(sparse_ln.reset_index(drop=True), 2026, 1))


# ------------------------------------------------------------------------- #6


def test_proe_time_averaged():
    """Pitfall 6: pass-rate-over-expectation must use the time-averaged lead (~0.45 x final
    margin), not the final margin, or a big favourite's RB projects ABOVE his neutral baseline."""
    neutral = script.simulate_game_script(0.0, 55.0, base_pace=68.0, n_sims=80_000)
    fav = script.simulate_game_script(24.0, 55.0, base_pace=68.0, n_sims=80_000)
    mean_rush = lambda sc: script.simulate_rush_yards_joint(0.55, sc, 5.1, 6.4, 0.18).mean()
    assert mean_rush(fav) <= mean_rush(neutral)
    # the non-simulated helper must use the same factor
    base = 0.45
    naive = base - 0.0095 * 24.0
    assert props.pass_rate_over_expectation(24.0, base) > naive + 0.05
    assert props.pass_rate_over_expectation(24.0, base) == pytest.approx(base - 0.0095 * 0.45 * 24.0)


# ------------------------------------------------------------------------- #7


def test_fcs_excluded():
    """Pitfall 7: FCS opponents distort the rating scale. No rating output may contain one, and
    adding FCS games must not change any FBS rating."""
    with_fcs = make_league(seed=5, n_teams=24, weeks=range(1, 9), fcs=True)
    without = make_league(seed=5, n_teams=24, weeks=range(1, 9), fcs=False)
    ln, pbp = with_fcs["lines"], with_fcs["plays"]
    assert set(FCS_NAMES) & (set(ln["home"]) | set(ln["away"])), "fixture must contain FCS games"

    outputs = {
        "market": ratings.fit_market_ratings(ln, 2026, 1).index,
        "total": ratings.fit_total_ratings(ln, 2026, 1).index,
        "epa": ratings.fit_epa_ratings(pbp, 2026, 1, split_pass_rush=False).index,
        "clean_plays": pd.Index(sorted(set(ratings.clean_plays(pbp)["offense"]))),
    }
    for name, idx in outputs.items():
        assert not (set(idx) & set(FCS_NAMES)), f"FCS team in {name} ratings"
        assert set(idx) == set(with_fcs["teams"]), f"FBS team missing from {name}"
    pbp_q = pbp.assign(passer=pbp["offense"] + " QB1")
    for filt in (True, False):
        q = qb.qb_epa(pbp_q, 2026, 1, filter_garbage=filt)
        assert not (set(q["team"]) & set(FCS_NAMES)), f"FCS team in qb_epa(filter_garbage={filt})"

    # same FBS-only games => same ratings, with or without the FCS games present
    clean_ln = with_fcs["lines"][with_fcs["lines"][["home_is_fbs", "away_is_fbs"]].all(axis=1)]
    pd.testing.assert_frame_equal(ratings.fit_market_ratings(ln, 2026, 1),
                                  ratings.fit_market_ratings(clean_ln, 2026, 1))


# ------------------------------------------------------------------------ #10


def test_dome_zero(monkeypatch):
    """Pitfall 10: weather adjustments must be exactly 0 for domes, whatever the outside weather."""
    stormy = {"dome": True, "wind_mph": 45.0, "gust_mph": 60.0, "precip_in": 2.0, "temp_f": -10.0,
              "wx_confidence": 1.0}
    assert weather.total_adjustment(stormy) == (0.0, 0.0)

    home, away = "Alpha State", "Beta Tech"
    rt = pd.DataFrame({"rating": [3.0, -1.0], "pace": [68.0, 68.0]}, index=[home, away])
    p = game_model.project_game(home, away, rt, wind_mph=30.0, precip=True, temp_f=5.0, dome=True)
    assert not ({"wind", "precip", "cold"} & set(p["total_adjustments"]))

    # attach_weather: the venue's dome flag (from CFBD /venues) decides, and no forecast is fetched
    def boom(*a, **k):
        raise AssertionError("weather was fetched for a dome")
    monkeypatch.setattr(weather, "game_weather", boom)
    venues = pd.DataFrame([
        {"name": "Beta Dome", "dome": True, "location.x": -90.0, "location.y": 40.0},
        {"name": "Mystery Hall", "dome": None, "location.x": -90.0, "location.y": 40.0},
    ])
    games = pd.DataFrame([
        {"id": 1, "venue": "Beta Dome", "start_date": pd.Timestamp("2025-09-06T19:00Z")},
        {"id": 2, "venue": "Kibbie Dome", "start_date": pd.Timestamp("2025-09-06T19:00Z")},  # name list
    ])
    venues = pd.concat([venues, pd.DataFrame([{"name": "Kibbie Dome", "dome": None,
                                               "location.x": -116.9, "location.y": 46.7}])])
    out = weather.attach_weather(games, venues).set_index("id")
    assert out.loc[1, "dome"] and out.loc[2, "dome"]
    assert out.loc[1, "wx_status"] == "dome" and out.loc[2, "wx_status"] == "dome"


# ----------------------------------------------------------------- key numbers


@pytest.mark.parametrize("source", ["market", "model"])
@pytest.mark.parametrize("exp_total", [40.0, 52.0, 65.0])
@pytest.mark.parametrize("exp_margin", [-24.0, -10.0, -3.0, 0.0, 3.0, 10.0, 24.0])
def test_key_numbers(exp_margin, exp_total, source):
    """Key numbers: margins pile up on 3 and 7, ties do not exist, probabilities sum to 1."""
    xs, pmf = game_model.margin_pmf(exp_margin, exp_total, source=source)
    at = lambda m: float(pmf[xs == m][0])
    assert at(3) > at(2)
    assert at(7) > at(8)
    assert at(0) == 0.0
    assert pmf.sum() == pytest.approx(1.0, abs=1e-9)
    assert (pmf >= 0).all()
