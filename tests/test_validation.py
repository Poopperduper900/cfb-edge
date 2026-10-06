"""Phase 4 acceptance: the validation gate. Synthetic data with known truth only."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from cfbmodel import config, ratings, state, status, validation
from cfbmodel.validation import PREREGISTERED as P
from synth import make_league

SEASONS = (2022, 2023, 2024, 2025, 2026)


def make_res(seed=1, n=700, model_noise=6.0, market_noise=5.0, informative=True, leak=False,
             holdout_model_noise=None, seasons=SEASONS, ytd_effect=None):
    """One row per game. truth -> noisy market (open and close) -> noisy model."""
    rng = np.random.default_rng(seed)
    frames = []
    for s in seasons:
        truth = rng.normal(0, 14, n)
        margin = truth + rng.normal(0, 13, n)
        close = truth + rng.normal(0, market_noise, n)
        mn = holdout_model_noise if (holdout_model_noise is not None and s == P["holdout_season"]) else model_noise
        model = truth + rng.normal(0, mn, n) if informative else rng.normal(0, 14, n)
        if leak:
            model = margin + rng.normal(0, 2, n)
        if ytd_effect is not None and s == P["report_season"]:
            model = margin + rng.normal(0, ytd_effect, n)
        tt = rng.normal(52, 6, n)
        total = tt + rng.normal(0, 12, n)
        tclose = tt + rng.normal(0, market_noise, n)
        tmodel = tt + rng.normal(0, mn, n) if informative else rng.normal(52, 6, n)
        kick = pd.Timestamp("2025-09-06T16:00:00Z") + pd.to_timedelta(rng.integers(0, 4, n) * 24, unit="h")
        frames.append(pd.DataFrame({
            "season": s, "week": rng.integers(0, 15, n), "gameId": np.arange(n) + s * 10_000,
            "home_conf": rng.choice(["SEC", "Big Ten", "MAC", "Sun Belt"], n),
            "away_conf": rng.choice(["SEC", "ACC", "MAC", "Sun Belt"], n),
            "start_date": kick, "neutral": False, "margin": margin, "actual_total": total,
            "model_margin": model, "model_total": tmodel,
            "spread_close": -close, "spread_open": -(close + rng.normal(0, 1.5, n)),
            "total_close": tclose, "total_open": tclose + rng.normal(0, 1.5, n)}))
    res = pd.concat(frames, ignore_index=True)
    res["market_margin_close"] = -res["spread_close"]
    res["market_margin_open"] = -res["spread_open"]
    return res


# ---------------------------------------------------------------- the regression


def test_regress_matches_least_squares_and_robust_se_beats_classical_when_variance_differs():
    rng = np.random.default_rng(0)
    n = 400
    x1, x2 = rng.normal(size=n), rng.normal(size=n)
    y = 1.0 + 2.0 * x1 + 0.3 * x2 + rng.normal(size=n) * (0.3 + np.abs(x2))   # error grows with |x2|
    r = validation.regress(y, x1, x2)
    beta = np.linalg.lstsq(np.column_stack([np.ones(n), x1, x2]), y, rcond=None)[0]
    assert r["coef"] == pytest.approx(beta[2]) and r["coef_market"] == pytest.approx(beta[1])

    robust, classical, coefs = [], [], []
    for _ in range(600):
        a, b = rng.normal(size=n), rng.normal(size=n)
        yy = 1.0 + 2.0 * a + 0.3 * b + rng.normal(size=n) * (0.3 + np.abs(b))
        rr = validation.regress(yy, a, b)
        X = np.column_stack([np.ones(n), a, b])
        res = yy - X @ np.linalg.lstsq(X, yy, rcond=None)[0]
        classical.append(np.sqrt((res @ res / (n - 3)) * np.linalg.inv(X.T @ X)[2, 2]))
        robust.append(rr["se"]); coefs.append(rr["coef"])
    truth_sd = np.std(coefs)
    assert abs(np.mean(robust) - truth_sd) / truth_sd < 0.08
    assert abs(np.mean(classical) - truth_sd) / truth_sd > 0.15          # the classical SE is visibly off


def test_regress_needs_enough_rows():
    assert validation.regress(np.arange(10.0), np.arange(10.0), np.arange(10.0) ** 2) is None


# ---------------------------------------------------------------------- outcomes


def test_an_efficient_market_means_nothing_passes_and_the_report_says_so():
    res = make_res(informative=False)
    st = validation.run_validation(res, generated="2026-10-06")
    assert not any(v["passed"] for v in st["markets"].values())
    assert not any(s["passed"] for s in st["segments"])
    assert set(st["w_model"].values()) == {0.0}
    assert st["status"] == "OK"
    report = validation.build_report(st, res)
    assert "nothing passed" in report and "valid, expected result" in report


def test_an_informative_model_passes_and_earns_a_weight_below_one_half():
    res = make_res(informative=True, model_noise=6.0, market_noise=5.0)
    st = validation.run_validation(res, generated="2026-10-06")
    m = st["markets"]["spread_vs_close"]
    assert m["passed"] and m["status"] == "PASS" and m["p"] < 0.05 and m["p_holdout"] < 0.10
    w = st["w_model"]["spread_vs_close|all"]
    assert 0.2 < w < 0.5
    assert w == pytest.approx(m["coef"] / (m["coef_market"] + m["coef"]), abs=1e-5)
    assert "some things passed" in validation.build_report(st, res)


def test_a_model_coefficient_above_half_is_flagged_as_a_leak_and_fails():
    res = make_res(leak=True)
    st = validation.run_validation(res, generated="2026-10-06")
    assert st["status"] == "SUSPECTED_LEAK"
    m = st["markets"]["spread_vs_close"]
    assert m["coef"] > 0.5 and not m["passed"] and m["status"] == "SUSPECTED_LEAK"
    assert st["w_model"]["spread_vs_close|all"] == 0.0
    assert "SUSPECTED LEAK" in validation.build_report(st, res)


def test_good_on_train_but_useless_on_the_holdout_fails():
    res = make_res(informative=True, holdout_model_noise=200.0)      # 2025 model is pure noise
    st = validation.run_validation(res, generated="2026-10-06")
    m = st["markets"]["spread_vs_close"]
    assert m["p"] < 0.05 and not m["passed"]
    assert "holdout" in m["reason"]
    assert st["w_model"]["spread_vs_close|all"] == 0.0


def test_2026_to_date_is_reported_but_never_changes_a_decision():
    base = make_res(informative=False, seasons=SEASONS)
    boosted = make_res(informative=False, seasons=SEASONS, ytd_effect=2.0)    # 2026 model is "perfect"
    a = validation.run_validation(base, generated="2026-10-06")
    b = validation.run_validation(boosted, generated="2026-10-06")
    assert a["markets"] == b["markets"] and a["w_model"] == b["w_model"] and a["segments"] == b["segments"]
    assert b["reported_not_used"]["spread_vs_close"]["coef"] > 0.5
    assert b["reported_not_used"]["spread_vs_close"]["season"] == 2026


# ------------------------------------------------- multiple-comparisons guard


def test_the_holdout_requirement_cuts_false_passes_to_a_fraction_of_the_naive_rule():
    guarded = naive = 0
    for seed in range(20):
        res = make_res(seed=100 + seed, n=400, informative=False)
        st = validation.run_validation(res, generated="2026-10-06")
        guarded += sum(s["passed"] for s in st["segments"]) + sum(v["passed"] for v in st["markets"].values())
        naive += sum(1 for s in st["segments"] if s["coef"] > 0 and s["p_train"] < 0.05)
    assert naive >= 8, f"naive rule should find plenty of chance 'edges' in pure noise (found {naive})"
    assert guarded <= naive / 4, (guarded, naive)
    assert guarded <= 5


# ------------------------------------------------------------------ determinism


def test_running_validate_twice_gives_identical_json_and_files(tmp_path):
    res = make_res(informative=True)
    s1 = validation.run_validation(res, generated="2026-10-06")
    s2 = validation.run_validation(res.copy(), generated="2026-10-06")
    assert json.dumps(s1, sort_keys=True) == json.dumps(s2, sort_keys=True)
    sp1, _ = validation.write_outputs(s1, validation.build_report(s1, res), tmp_path / "a")
    sp2, _ = validation.write_outputs(s2, validation.build_report(s2, res), tmp_path / "b")
    assert sp1.read_bytes() == sp2.read_bytes()


def test_status_file_has_the_documented_shape_and_feeds_the_blend(tmp_path, monkeypatch):
    res = make_res(informative=True)
    st = validation.run_validation(res, generated="2026-10-06")
    assert set(st) >= {"generated", "train_seasons", "holdout_season", "markets", "segments", "w_model"}
    assert set(st["markets"]) == {"spread_vs_open", "spread_vs_close", "total_vs_open", "total_vs_close"}
    assert set(st["markets"]["spread_vs_close"]) >= {"n", "coef", "se", "p", "passed", "reason"}
    assert set(st["segments"][0]) >= {"market", "segment", "n", "coef", "p_train", "p_holdout", "passed", "reason"}
    assert st["train_seasons"] == [2022, 2023, 2024] and st["holdout_season"] == 2025
    sp, _ = validation.write_outputs(st, "x", tmp_path)
    monkeypatch.setattr(config, "OUTPUT", tmp_path)
    assert status.w_model_for("spread_vs_close") == st["w_model"]["spread_vs_close|all"] > 0
    assert status.validation_age_days(now=pd.Timestamp("2026-10-10", tz="UTC").to_pydatetime()) == pytest.approx(4.0)


# ------------------------------------------------------------------------ segments


def test_segments_partition_the_games():
    res = make_res(informative=False, n=300, seasons=(2024,))
    masks = validation.segment_masks(res, "spread_vs_close")
    for group in (("tier:P4-P4", "tier:P4-G5", "tier:G5-G5"), ("weeks:0-3", "weeks:4-8", "weeks:9+"),
                  ("day:saturday", "day:other"), ("spread:<=7", "spread:7-14", "spread:>14")):
        total = sum(masks[g].astype(int) for g in group)
        assert (total == 1).all(), group


def test_tier_and_weekday_definitions():
    df = pd.DataFrame({
        "season": [2022, 2024, 2024, 2024], "week": [5, 5, 5, 5],
        "home_conf": ["Pac-12", "Pac-12", "SEC", "MAC"], "away_conf": ["SEC", "SEC", "MAC", "Sun Belt"],
        "start_date": pd.to_datetime(["2022-10-01T16:00Z",          # Saturday noon ET
                                      "2024-10-06T00:30Z",          # Sat 8:30pm ET = Sunday UTC
                                      "2024-10-04T23:00Z",          # Friday 7pm ET
                                      "2024-10-02T23:30Z"]),        # Wednesday
        "spread_close": [-3.0, -10.0, -20.0, -7.0], "spread_open": [-3.0, -10.0, -20.0, -7.0]})
    m = validation.segment_masks(df, "spread_vs_close")
    assert list(m["tier:P4-P4"]) == [True, False, False, False]          # Pac-12 counted Power in 2022 only
    assert list(m["tier:P4-G5"]) == [False, True, True, False]
    assert list(m["tier:G5-G5"]) == [False, False, False, True]
    assert list(m["day:saturday"]) == [True, True, False, False]
    assert list(m["spread:7-14"]) == [False, True, False, False]
    assert list(m["spread:<=7"]) == [True, False, False, True]


# ------------------------------------------------------------------ walk-forward


@pytest.fixture(scope="module")
def league():
    lg = make_league(seed=33, n_teams=24, seasons=(2024, 2025), weeks=range(1, 11), plays_per_team_game=30)
    g = lg["games"].copy()
    g["homePoints"] = (g["total"] + g["margin"]) / 2
    g["awayPoints"] = (g["total"] - g["margin"]) / 2
    g["start_date"] = pd.Timestamp("2025-09-06T16:00:00Z")
    g["season_type"] = "regular"
    lg["games"] = g
    return lg


def test_walk_forward_rows_do_not_depend_on_later_data(league):
    prior = {s: pd.Series(0.0, index=league["teams"]) for s in (2024, 2025)}
    kw = dict(seasons=[2025], prior_by_season=prior)
    full = validation.walk_forward(league["games"], league["lines"], league["plays"], **kw)

    def upto(df, week):
        return df[(df["season"] < 2025) | ((df["season"] == 2025) & (df["week"] < week))]
    # lines for the games being projected are still needed, so compare model output only
    # week-6 games are projected from lines/plays that stop before week 6 (plus week 6's own lines,
    # which are the market being compared against, not an input to the model)
    a = full[full["week"] == 6].set_index("gameId")["model_margin"]
    cut_lines = pd.concat([upto(league["lines"], 6), league["lines"][(league["lines"]["season"] == 2025)
                                                                      & (league["lines"]["week"] == 6)]])
    b = validation.walk_forward(league["games"][(league["games"]["week"] == 6)], cut_lines,
                                upto(league["plays"], 6), **kw)
    b = b[b["week"] == 6].set_index("gameId")["model_margin"]
    pd.testing.assert_series_equal(a.sort_index(), b.sort_index())
    assert len(a) > 5


def test_walk_forward_lists_what_it_skipped_and_never_projects_early_weeks_without_a_prior(league):
    res = validation.walk_forward(league["games"], league["lines"], league["plays"], seasons=[2024, 2025])
    assert res["week"].min() >= 4                                   # weeks 1-3 need preseason ratings
    assert any("preseason" in r for _, _, r in res.attrs["skipped"])
    prior = {2024: pd.Series(0.0, index=league["teams"]), 2025: pd.Series(0.0, index=league["teams"])}
    res2 = validation.walk_forward(league["games"], league["lines"], league["plays"], seasons=[2024, 2025],
                                   prior_by_season=prior)
    assert (2024, 1, "no plays available before the as-of point") in res2.attrs["skipped"]
    assert set(res2["week"]) >= {2, 3}                              # early weeks now evaluated


def test_the_full_report_builds_with_accuracy_calibration_and_roi_sections(league):
    prior = {s: pd.Series(0.0, index=league["teams"]) for s in (2024, 2025)}
    res = validation.walk_forward(league["games"], league["lines"], league["plays"], seasons=[2024, 2025],
                                  prior_by_season=prior)
    res = res.assign(season=res["season"].map({2024: 2024, 2025: 2025}))
    st = validation.run_validation(res, generated="2026-10-06")
    report = validation.build_report(st, res)
    for heading in ("## Markets", "## Segments", "## Accuracy", "## Calibration", "## ROI by edge size",
                    "assumed -110"):
        assert heading in report, heading
