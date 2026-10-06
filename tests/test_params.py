"""Phase 6a: the parameter registry. Every tunable lives in params/<version>.json, and the code
really reads it (so a new version changes the model and a rollback restores it)."""
from __future__ import annotations

import ast
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cfbmodel import config, edge, game_model, params, playerstate, pricing, props, qb, script, state, weather
from cfbmodel.config import C
from cfbmodel.params import P

PKG = Path(__file__).resolve().parent.parent / "cfbmodel"
PARAMS = Path(__file__).resolve().parent.parent / "params"


# ---------------------------------------------------------------- the files


def test_current_points_at_an_existing_version_whose_meta_matches():
    cur = json.loads((PARAMS / "current.json").read_text())["version"]
    full = json.loads((PARAMS / f"{cur}.json").read_text())
    assert full["_meta"]["version"] == cur
    assert params.current_version() == cur and params.active_version() == cur


def test_every_registry_entry_is_used_by_the_code():
    """A parameter nothing reads would be a dial wired to nothing."""
    source = "\n".join(p.read_text() for p in PKG.glob("*.py") if p.name != "params.py")
    legacy = {key for _, key in params._LEGACY.values()}
    unused = []
    for group, items in params.groups_of(json.loads((PARAMS / "v0001.json").read_text())).items():
        for key in items:
            if key in legacy and group in {g for g, _ in params._LEGACY.values()}:
                continue
            if not re.search(rf"\b{re.escape(key)}\b", source):
                unused.append(f"{group}.{key}")
    assert not unused, f"parameters that no code reads: {unused}"


def test_every_version_has_the_same_shape_as_the_first():
    ref = params.groups_of(params.read_version("v0001"))
    for f in sorted(PARAMS.glob("v*.json")):
        got = params.groups_of(json.loads(f.read_text()))
        assert set(got) == set(ref), f.name
        for g in ref:
            assert set(got[g]) == set(ref[g]), (f.name, g)


def test_config_no_longer_claims_a_calibrate_py_that_does_not_exist():
    """Pitfall 12."""
    assert "calibrate.py" not in (PKG / "config.py").read_text()
    assert not (PKG / "calibrate.py").exists()


def test_the_legacy_C_view_reads_the_active_version():
    v = params.groups_of(params.read_version(params.current_version()))
    assert C.hfa_points == v["margin"]["hfa_points"] and C.margin_df == v["margin"]["df"]
    assert C.key_numbers[3] == v["margin"]["key_numbers"]["3"]
    assert C.max_bet_pct == v["betting"]["max_bet_pct"] and C.pace_mean == v["ratings"]["pace_mean"]
    with pytest.raises(AttributeError):
        C.not_a_parameter


# ------------------------------------------------------- the code reads them


def _mae_sd(pmf, xs):
    return float((np.abs(xs) * pmf).sum()), float(np.sqrt((xs ** 2 * pmf).sum()))


def test_changing_margin_parameters_changes_the_pmf():
    xs, base = game_model.margin_pmf(0.0, 52.0, source="market")
    with params.override({"margin": {"scale_market": 15.0}}):
        _, wide = game_model.margin_pmf(0.0, 52.0, source="market")
    assert _mae_sd(wide, xs)[0] > _mae_sd(base, xs)[0] + 1.5
    with params.override({"margin": {"key_numbers": {"3": 1.0, "7": 1.0}}}):
        _, flat = game_model.margin_pmf(0.0, 52.0, source="market")
    at = lambda pmf, m: float(pmf[xs == m][0])
    assert at(flat, 3) / at(flat, 2) < at(base, 3) / at(base, 2)
    assert game_model.margin_pmf(0.0, 52.0, source="market")[1][xs == 3][0] == at(base, 3)   # restored


def test_changing_weather_and_game_parameters_changes_the_adjustments():
    wx = {"dome": False, "wind_mph": 20.0, "gust_mph": 20.0, "precip_in": 0.0, "temp_f": 60.0, "wx_confidence": 1.0}
    base, _ = weather.total_adjustment(wx)
    with params.override({"weather": {"wind_coef": 0.0}}):
        assert weather.total_adjustment(wx)[0] == 0.0
    with params.override({"weather": {"wind_coef": 0.68}}):
        assert weather.total_adjustment(wx)[0] == pytest.approx(2 * base)
    rt = pd.DataFrame({"rating": [10.0, 0.0]}, index=["A", "B"])
    d0 = game_model.qb_dropoff("A", rt)
    with params.override({"game": {"qb_dropoff_base": 4.2}}):
        assert game_model.qb_dropoff("A", rt) == pytest.approx(d0 + 1.0)


def test_changing_script_and_props_parameters_changes_the_simulation_inputs():
    assert props.pass_rate_over_expectation(24.0, 0.45) < 0.45
    with params.override({"script": {"proe_slope": 0.0}}):
        assert props.pass_rate_over_expectation(24.0, 0.45) == 0.45
    h = props.pull_hazard(30.0, 55.0)
    with params.override({"props": {"pull_blowout_weight": 0.0, "pull_near_weight": 0.0}}):
        assert props.pull_hazard(30.0, 55.0) == 0.0
    assert h > 0
    hist, mates = np.array([0.5, 0.5, 0.5]), [0.5, 0.3, 0.2]
    s0 = script.role_instability(hist, mates)
    with params.override({"script": {"role_base": 0.84}}):
        assert script.role_instability(hist, mates) > s0


def test_changing_betting_and_state_parameters_changes_sizing_and_shrinkage():
    assert edge.kelly(0.9, -110) == C.max_bet_pct
    with params.override({"betting": {"max_bet_pct": 0.01}}):
        assert edge.kelly(0.9, -110) == 0.01
    with params.override({"betting": {"edge_flat_haircut": 0.5}}):
        assert pricing.shrink_prob(0.6, 0.5) == pytest.approx(0.55)
    w = state.early_season_prior_weight(2)
    with params.override({"state": {"decay_half_life": 1.0}}):
        assert state.early_season_prior_weight(2) < w


def test_changing_qb_and_player_parameters_changes_the_state():
    d = qb.dropoff_points(0.26, 420, None, 0, 0.83, "SO")
    with params.override({"qb": {"tier3_sd": 9.0}}):
        assert qb.dropoff_points(0.26, 420, None, 0, 0.83, "SO")["sd_points"] == 9.0
    assert d["sd_points"] != 9.0
    with params.override({"player": {"new_player_share": 0.4}}):
        row = playerstate._new_row("X", "T", "X", "rush", 2025, 1, None)
        assert row["share_a"] / (row["share_a"] + row["share_b"]) == pytest.approx(0.4)
    assert playerstate.SEASON_CARRY == P.player.season_carry


# --------------------------------------------------------- registry mechanics


def test_override_restores_on_exit_even_after_an_error():
    before = P.margin.hfa_points
    with pytest.raises(RuntimeError):
        with params.override({"margin": {"hfa_points": 9.9}}):
            assert P.margin.hfa_points == 9.9
            raise RuntimeError("boom")
    assert P.margin.hfa_points == before


def test_nested_overrides_stack():
    with params.override({"margin": {"hfa_points": 1.0}}):
        with params.override({"margin": {"hfa_points": 2.0}}):
            assert P.margin.hfa_points == 2.0
        assert P.margin.hfa_points == 1.0


def test_unknown_parameters_fail_loudly_with_the_version_named():
    with pytest.raises(AttributeError, match=params.active_version()):
        P.margin.no_such_thing
    with pytest.raises(AttributeError):
        P.no_such_group.x


def test_a_different_params_directory_is_honoured(tmp_path, monkeypatch):
    full = params.read_version("v0001")
    full["margin"]["hfa_points"] = 3.1
    full["_meta"]["version"] = "v0007"
    (tmp_path / "v0007.json").write_text(json.dumps(full))
    (tmp_path / "current.json").write_text(json.dumps({"version": "v0007"}))
    monkeypatch.setenv("CFBMODEL_PARAMS", str(tmp_path))
    params.reload()
    try:
        assert P.margin.hfa_points == 3.1 and params.active_version() == "v0007" and C.hfa_points == 3.1
    finally:
        monkeypatch.delenv("CFBMODEL_PARAMS")
        params.reload()
    assert P.margin.hfa_points == 2.35


def test_a_missing_current_file_is_an_error_not_a_default(tmp_path, monkeypatch):
    monkeypatch.setenv("CFBMODEL_PARAMS", str(tmp_path))
    params.reload()
    try:
        with pytest.raises(params.ParamsError, match="current.json"):
            params.active()
    finally:
        monkeypatch.delenv("CFBMODEL_PARAMS")
        params.reload()


# ------------------------------------------------ no tunable literals in model code

# Numbers that are allowed to stay in model code because they are not tunable model parameters.
# Everything else must come from the registry. 0, 1, -1 and 2 are always allowed.
ALLOWED = {
    "game_model.py": {1000: "feet-to-thousands unit conversion", 70: "pmf support is margins -70..70",
                      0.5: "continuity correction / a push is half a win", 140: "score grid 0..139",
                      99: "week sentinel when a game has no week"},
    "weather.py": {16: "forecast days requested from Open-Meteo", 45: "HTTP timeout, seconds",
                   3: "coordinate rounding digits", 4: "hours a game lasts (weather window)",
                   3600: "seconds per hour", 70.0: "indoor placeholder temperature; has no effect (dome adjustment is 0)"},
    "ratings.py": {100.0: "HFA_SCALE: ridge column scaling (pitfall 3), not a model parameter",
                   4: "the fourth quarter", 0.5: "half-life halving", 16: "weeks per season in the recency clock"},
    "state.py": {0.5: "half-life halving"},
    "playerstate.py": {},
    "script.py": {20260829: "RNG seed", 4: "rounding digits", 40000: "default number of simulations",
                  0.05: "floor that keeps a Poisson mean above zero", 0.5: "lognormal mean correction (-sigma^2/2)",
                  0.001: "divide-by-zero guard"},
    "props.py": {20260829: "RNG seed", 1e-06: "divide-by-zero guard", 0.5: "lognormal mean correction / guard floor",
                 40000: "default number of simulations", 0.05: "floor that keeps a Poisson mean above zero",
                 0.001: "divide-by-zero guard"},
    "qb.py": {3: "evidence tier number", 4: "rounding digits"},
    "priors.py": {4: "years in the rolling recruiting window (a design choice, see BUILD_PLAN)",
                  3: "np.logspace upper exponent / rounding digits", 30: "RidgeCV grid size",
                  1000000000.0: "effectively-infinite half-life for a target fit"},
    "edge.py": {100.0: "American-odds conversion", 0.5: "half", 0.2: "power de-vig search bracket",
                5.0: "power de-vig search bracket", 1e-06: "numerical clip in the Shin solver",
                0.35: "numerical bound in the Shin solver", 1e-12: "float dust threshold in Kelly",
                4: "rounding digits", 3: "rounding digits"},
    "pricing.py": {400.0: "plan rule: Shin when the favourite is -400 or shorter",
                   3: "key number", 4: "key number", 6: "key number", 7: "key number", 10: "key number",
                   14: "key number", 17: "key number", 21: "key number", 0.5: "half (a push is half a win)"},
}


def _numeric_literals(path: Path):
    tree = ast.parse(path.read_text())
    docstrings = {id(n.body[0].value) for n in ast.walk(tree)
                  if isinstance(n, (ast.Module, ast.FunctionDef, ast.ClassDef)) and n.body
                  and isinstance(n.body[0], ast.Expr) and isinstance(n.body[0].value, ast.Constant)}
    for n in ast.walk(tree):
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)) and not isinstance(n.value, bool) \
                and id(n) not in docstrings and n.value not in (0, 1, -1, 2):
            yield n.value, n.lineno


def test_no_tunable_number_reappears_in_model_code():
    """CLAUDE.md: no tunable number lives in model code. A new numeric literal in a model file must
    either come from the registry or be added to ALLOWED above with a reason a reviewer can judge."""
    offenders = []
    for name, allowed in ALLOWED.items():
        for value, line in _numeric_literals(PKG / name):
            if value not in allowed:
                offenders.append(f"{name}:{line}  {value!r}")
    assert not offenders, "numeric literals not in the registry or the allowlist:\n  " + "\n  ".join(offenders)


def test_the_allowlist_has_no_stale_entries():
    stale = []
    for name, allowed in ALLOWED.items():
        found = {v for v, _ in _numeric_literals(PKG / name)}
        stale += [f"{name}: {v!r}" for v in allowed if v not in found]
    assert not stale, f"allowlisted numbers that no longer appear: {stale}"


def test_the_literal_scan_can_actually_catch_a_new_tunable(tmp_path):
    """Guard the guard: a synthetic file with a hidden coefficient must be flagged."""
    f = tmp_path / "bad.py"
    f.write_text('"""doc 0.123"""\nx = 3 * 0.0095 * lead\n')
    assert [v for v, _ in _numeric_literals(f)] == [3, 0.0095]
