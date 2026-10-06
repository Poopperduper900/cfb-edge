"""Phase 6c-6e: champion/challenger learning, versioning, rollback, drift, post-mortem."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cfbmodel import cli, config, learn, params, status
from cfbmodel.params import P
from synth import make_learn_data

PARAMS_SRC = Path(__file__).resolve().parent.parent / "params"
FEW_KEYS = {3: 1.34, 7: 1.28, 10: 1.16, 14: 1.15}          # four keys keep the fits fast


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    """Private copy of the parameter files and an empty output folder."""
    pdir = tmp_path / "params"
    pdir.mkdir()
    for f in PARAMS_SRC.glob("*.json"):
        shutil.copy(f, pdir / f.name)
    v1 = json.loads((pdir / "v0001.json").read_text())
    v1["margin"]["key_numbers"] = {str(k): v for k, v in FEW_KEYS.items()}       # four keys: fast fits
    (pdir / "v0001.json").write_text(json.dumps(v1, indent=2))
    monkeypatch.setenv("CFBMODEL_PARAMS", str(pdir))
    monkeypatch.setattr(config, "OUTPUT", tmp_path / "out")
    params.reload()
    yield tmp_path
    monkeypatch.delenv("CFBMODEL_PARAMS")
    params.reload()


# ----------------------------------------------------------------- the pieces


def test_every_parameter_move_is_limited_to_25_percent():
    champ = {"margin": {"scale_market": 10.0, "key_numbers": {"3": 1.4, "7": 1.2}}, "x": {"neg": -2.0, "zero": 0.0}}
    got = learn.bound_changes(champ, {"margin": {"scale_market": 20.0, "key_numbers": {"3": 1.0, "7": 1.25}},
                                      "x": {"neg": -10.0, "zero": 5.0}}, 0.25)
    assert got["margin"]["scale_market"] == pytest.approx(12.5)
    assert got["margin"]["key_numbers"]["3"] == pytest.approx(1.05)         # 1.4 * 0.75
    assert got["margin"]["key_numbers"]["7"] == pytest.approx(1.25)         # inside the band: unchanged
    assert got["x"]["neg"] == pytest.approx(-2.5)                           # negative numbers bound correctly
    assert abs(got["x"]["zero"]) <= 0.0025                                  # zero may barely move
    inside = learn.bound_changes(champ, {"margin": {"scale_market": 11.0}}, 0.25)
    assert inside["margin"]["scale_market"] == 11.0


@pytest.mark.parametrize("gain,lo,cal0,cal1,se,n,expect", [
    (0.01, 0.004, 0.02, 0.02, 0.0, 500, []),                                  # everything holds: promote
    (0.01, -0.001, 0.02, 0.02, 0.0, 500, ["includes zero"]),
    (0.0001, 0.00005, 0.02, 0.02, 0.0, 500, ["did not improve enough"]),
    (0.01, 0.004, 0.02, 0.05, 0.005, 500, ["calibration got worse"]),         # clearly worse
    (0.01, 0.004, 0.02, 0.05, 0.03, 500, []),                                 # "worse" is within sampling noise
    (0.01, 0.004, 0.02, 0.021, 0.0, 500, []),                                 # within the absolute tolerance
    (0.01, 0.004, 0.02, 0.02, 0.0, 299, ["window too small"]),
    (-0.01, -0.02, 0.02, 0.02, 0.0, 500, ["did not improve enough", "includes zero"]),
])
def test_promotion_gate(gain, lo, cal0, cal1, se, n, expect):
    ok, why = learn.decide(gain, lo, cal0, cal1, n, 300, se)
    assert ok == (not expect)
    for fragment in expect:
        assert any(fragment in w for w in why), (fragment, why)


def test_bootstrap_interval_is_deterministic_and_excludes_zero_only_for_a_real_gain():
    rng = np.random.default_rng(0)
    real, noise = rng.normal(0.02, 0.05, 600), rng.normal(0.0, 0.05, 600)
    assert learn.bootstrap_ci(real, 1000) == learn.bootstrap_ci(real, 1000)
    assert learn.bootstrap_ci(real, 1000)[1] > 0
    assert learn.bootstrap_ci(noise, 1000)[1] < 0 < learn.bootstrap_ci(noise, 1000)[2]


def test_reliability_error_is_zero_for_perfect_forecasts_and_grows_with_bias():
    rng = np.random.default_rng(1)
    p = rng.uniform(0.05, 0.95, 20000)
    y = (rng.random(20000) < p).astype(float)
    assert learn.reliability_error(p, y) < 0.02
    assert learn.reliability_error(np.clip(p + 0.15, 0, 1), y) > 0.1


def test_learning_cadence_is_every_second_week_unless_forced():
    assert learn.learn_due(2026, 6, None)[0]
    assert not learn.learn_due(2026, 7, {"season": 2026, "week": 6})[0]
    assert learn.learn_due(2026, 8, {"season": 2026, "week": 6})[0]
    assert learn.learn_due(2026, 7, {"season": 2026, "week": 6}, force=True)[0]
    assert learn.learn_due(2027, 1, {"season": 2026, "week": 14})[0]


# ------------------------------------------------------------------ promotion


def test_a_real_change_in_the_world_is_promoted_versioned_and_explained(sandbox):
    v1_before = (sandbox / "params" / "v0001.json").read_bytes()
    # the market distribution got wider than version 1 believes (scale 11.22 -> 14.0, the +25% limit)
    data = make_learn_data(1, market={"scale": 14.0}, keys=FEW_KEYS)
    rep = learn.run_learn(data, 2024, 12, force=True)
    by = {r.name: r for r in rep.results}
    assert by["margin_pmf"].status == "PROMOTED", by["margin_pmf"].reasons
    assert by["totals"].status == "KEPT" and by["model_margin"].status == "KEPT"
    assert rep.version == "v0002"
    assert params.current_version() == "v0002"
    new, old = P.margin.scale_market, params.groups_of(params.read_version("v0001"))["margin"]["scale_market"]
    assert old < new <= old * 1.25 + 1e-9
    # untouched groups really are untouched
    assert P.totals.sd_base == params.groups_of(params.read_version("v0001"))["totals"]["sd_base"]

    v = sandbox / "out" / "models" / "v0002"
    assert {f.name for f in v.iterdir()} == {"params.json", "metrics.json", "diff.md"}
    diff = (v / "diff.md").read_text()
    assert "margin.scale_market" in diff and "->" in diff and "95% interval" in diff and "not zero" in diff
    metrics = json.loads((v / "metrics.json").read_text())
    assert metrics["parent"] == "v0001" and metrics["groups"]["margin_pmf"]["status"] == "PROMOTED"
    assert metrics["groups"]["margin_pmf"]["ci_low"] > 0
    assert (sandbox / "params" / "v0001.json").read_bytes() == v1_before                    # history kept

    table = learn.models_table()
    assert list(table["version"]) == ["v0001", "v0002"] and list(table["current"]) == ["", "*"]


def test_running_again_on_the_same_data_does_not_promote_again(sandbox):
    data = make_learn_data(1, market={"scale": 14.0}, keys=FEW_KEYS)
    learn.run_learn(data, 2024, 12, force=True)
    again = learn.run_learn(data, 2024, 12, force=True)
    assert again.version is None and again.promoted == []
    assert params.current_version() == "v0002"


def test_a_window_with_too_few_games_is_skipped_not_promoted(sandbox):
    data = make_learn_data(2, market={"scale": 14.0}, keys=FEW_KEYS, games_per_week=20)   # 8 weeks x 20 = 160
    rep = learn.run_learn(data, 2024, 12, force=True)
    r = {x.name: x for x in rep.results}["margin_pmf"]
    assert r.status == "SKIPPED" and "window too small" in r.reasons[0] and rep.version is None


def test_groups_without_data_are_reported_as_skipped(sandbox):
    rep = learn.run_learn(make_learn_data(3, keys=FEW_KEYS), 2024, 12, force=True)
    by = {r.name: r for r in rep.results}
    assert by["weather"].status == "SKIPPED" and by["script_proe"].status == "SKIPPED"
    assert "early_decay" not in by                                  # season-end group, needs --full
    assert "w_model is not learned here" in (sandbox / "out" / "learn_report_2024_w12.md").read_text()


# ----------------------------------------------------------------- noise test


def test_on_data_where_nothing_changed_the_challenger_is_almost_never_promoted(sandbox):
    """The most important test in the phase: 20 random worlds identical to the champion's. A learner
    that 'improves' in more than one of them is chasing noise."""
    promoted_in = []
    for seed in range(20):
        data = make_learn_data(100 + seed, seasons=(2022, 2023, 2024), games_per_week=50, keys=FEW_KEYS)
        rep = learn.run_learn(data, 2024, 12, force=True, seed=seed)
        if rep.promoted:
            promoted_in.append((seed, rep.promoted))
            params.reload()   # a (wrong) promotion would have moved current.json; keep seeds independent
            (sandbox / "params" / "current.json").write_text('{"version": "v0001"}')
            params.reload()
    assert len(promoted_in) <= 1, f"promoted on unchanged data in {len(promoted_in)} of 20 seeds: {promoted_in}"


# ----------------------------------------------------------------- leak test


def test_learning_as_of_week_w_is_identical_with_or_without_later_data(sandbox):
    full = make_learn_data(5, seasons=(2022, 2023, 2024, 2025), market={"scale": 14.0}, keys=FEW_KEYS)
    cut = learn.LearnData(
        market=full.market[(full.market.season < 2025) | (full.market.week < 6)],
        wf=full.wf[(full.wf.season < 2025) | (full.wf.week < 6)])
    a = learn.run_learn(full, 2025, 6, force=True, output_dir=sandbox / "a")
    (sandbox / "params" / "current.json").write_text('{"version": "v0001"}')
    for f in sandbox.glob("params/v0002.json"):
        f.unlink()
    params.reload()
    b = learn.run_learn(cut, 2025, 6, force=True, output_dir=sandbox / "b")
    assert [(r.name, r.status, r.changes, r.evidence) for r in a.results] == \
           [(r.name, r.status, r.changes, r.evidence) for r in b.results]
    assert a.version == b.version == "v0002"
    assert (sandbox / "a" / "models" / "v0002" / "params.json").read_text() == \
           (sandbox / "b" / "models" / "v0002" / "params.json").read_text()


# ------------------------------------------------------------------- rollback


def test_rollback_restores_an_earlier_version_and_can_roll_forward(sandbox):
    learn.run_learn(make_learn_data(1, market={"scale": 14.0}, keys=FEW_KEYS), 2024, 12, force=True)
    assert params.current_version() == "v0002" and P.margin.scale_market > 11.22
    prev = learn.rollback("v0001")
    assert prev == "v0002" and params.current_version() == "v0001" and P.margin.scale_market == 11.22
    assert (sandbox / "params" / "v0002.json").exists()                          # nothing deleted
    assert json.loads((sandbox / "out" / "models" / "history.jsonl").read_text().splitlines()[-1])["to"] == "v0001"
    learn.rollback("v0002")
    assert P.margin.scale_market > 11.22
    with pytest.raises(learn.LearnError, match="v0099"):
        learn.rollback("v0099")


def test_the_command_line_lists_models_and_rolls_back(sandbox, capsys):
    assert cli.main(["models"]) == 0
    assert "v0001" in capsys.readouterr().out
    assert cli.main(["rollback", "v0042"]) == 2
    assert "unknown version" in capsys.readouterr().err
    assert cli.main(["rollback", "v0001"]) == 0


# ---------------------------------------------------------------------- drift


def _wf(weeks, good_weeks, n=60, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for w in weeks:
        spread = np.round(rng.normal(0, 10, n) * 2) / 2
        margin = -spread + rng.normal(0, 12, n)
        # a healthy model is wrong by about 13 points; a drifted one by far more and biased
        model = -spread + rng.normal(0, 4, n) if w in good_weeks else -spread + rng.normal(8, 14, n)
        rows.append(pd.DataFrame({"season": 2026, "week": w, "margin": margin, "model_margin": model,
                                  "spread_close": spread, "total_close": 52.0}))
    return pd.concat(rows, ignore_index=True)


def test_a_model_that_suddenly_gets_worse_raises_the_drift_alarm_and_forces_w_model_to_zero(sandbox):
    wf = _wf(range(1, 17), good_weeks=set(range(1, 12)))
    drift = learn.check_drift(wf, 2026, 17)
    assert drift["active"] and "ratio" in drift["reason"]
    (sandbox / "out").mkdir(exist_ok=True)
    (sandbox / "out" / "validation_status.json").write_text(json.dumps(
        {"generated": "2026-10-01", "w_model": {"spread_vs_close|all": 0.3}}))
    assert status.w_model_for() == 0.3
    learn.write_drift_status(drift)
    assert status.drift_active() and status.w_model_for() == 0.0


def test_a_steady_model_raises_no_alarm_and_too_little_data_cannot_judge(sandbox):
    steady = _wf(range(1, 17), good_weeks=set(range(1, 17)))
    assert not learn.check_drift(steady, 2026, 17)["active"]
    few = learn.check_drift(_wf(range(1, 3), good_weeks={1, 2}), 2026, 3)
    assert not few["active"] and "not enough games" in few["reason"]


def test_recovery_clears_the_alarm(sandbox):
    bad = learn.check_drift(_wf(range(1, 17), good_weeks=set(range(1, 12))), 2026, 17)
    learn.write_drift_status(bad)
    assert status.drift_active()
    recovered = learn.check_drift(_wf(range(1, 17), good_weeks=set(range(1, 17))), 2026, 17)
    learn.write_drift_status(recovered)
    assert not status.drift_active()


# ----------------------------------------------------------------- post-mortem


def test_postmortem_names_the_biggest_misses_and_what_drove_them():
    # game 1: the market (home -9) was right, the model said the road team wins big, and the
    # preseason prior pulled the model's rating far below its own in-season rating
    wf = pd.DataFrame({
        "season": 2026, "week": 6, "home": ["A", "C", "E"], "away": ["B", "D", "F"],
        "margin": [10.0, 10.0, 7.0], "model_margin": [-14.0, 9.0, 6.0], "spread_close": [-9.0, -9.0, -6.5],
        "core_model": [-16.35, 6.65, 3.65], "core_unshrunk": [6.0, 6.65, 3.65], "core_market": [6.7, 6.7, 4.2],
        "hfa_used": [2.35] * 3, "market_hfa": [2.3] * 3,
        "model_total": [60.0, 50.0, 55.0], "actual_total": [41.0, 51.0, 54.0], "total_close": [58.0, 50.5, 55.0]})
    text = learn.postmortem(wf, 2026, 6)
    assert text.index("B @ A") < text.index("D @ C")                             # biggest miss first
    assert "pulled toward the preseason prior (-22.4 pts)" in text
    assert "Nothing here changes the model" in text and "Biggest total misses" in text
    assert learn.postmortem(wf, 2026, 9).count("No games for this week") == 1


def test_postmortem_says_when_the_market_missed_it_too():
    wf = pd.DataFrame({"season": 2026, "week": 3, "home": ["A"], "away": ["B"], "margin": [-25.0],
                       "model_margin": [10.0], "spread_close": [-10.0], "core_model": [8.0], "core_unshrunk": [8.0],
                       "core_market": [7.5], "hfa_used": [2.0], "market_hfa": [2.0]})
    assert "the market missed it too" in learn.postmortem(wf, 2026, 3)
