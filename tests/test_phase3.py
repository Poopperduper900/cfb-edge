"""Phase 3 acceptance: leak-safe ratings state, blend permission, fitted priors, player posteriors."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from cfbmodel import config, playerstate, priors, ratings, state, status
from synth import box_week, make_league


@pytest.fixture
def out_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "OUTPUT", tmp_path)
    return tmp_path


@pytest.fixture(scope="module")
def league():
    return make_league(seed=21, n_teams=30, seasons=(2024, 2025), weeks=range(1, 13), plays_per_team_game=30)


def _ratings_inputs(league, season=2025, week=6):
    from cfbmodel.ratings import as_of
    epa = ratings.fit_epa_ratings(league["plays"], season, week, split_pass_rush=False)
    mkt = ratings.fit_market_ratings(league["lines"], season, week)
    return epa, mkt


# ------------------------------------------------- blend permission (w_model)


def test_blend_is_market_only_without_a_validation_file(league, out_dir):
    epa, mkt = _ratings_inputs(league)
    bl = ratings.blend(epa, mkt)
    assert (bl["w_model"] == 0.0).all()
    np.testing.assert_allclose(bl["rating"], bl["market_rating"])


def test_blend_uses_the_weight_validation_earned(league, out_dir):
    (out_dir / "validation_status.json").write_text(json.dumps(
        {"generated": "2026-10-01T00:00:00+00:00", "w_model": {"spread_vs_close|all": 0.25}}))
    epa, mkt = _ratings_inputs(league)
    bl = ratings.blend(epa, mkt)
    assert (bl["w_model"] == 0.25).all()
    np.testing.assert_allclose(bl["rating"], 0.25 * bl["model_rating_pts"] + 0.75 * bl["market_rating"])


def test_an_active_drift_alarm_forces_the_weight_to_zero(league, out_dir):
    (out_dir / "validation_status.json").write_text(json.dumps({"w_model": {"spread_vs_close|all": 0.25}}))
    (out_dir / "drift_status.json").write_text(json.dumps({"active": True}))
    epa, mkt = _ratings_inputs(league)
    assert (ratings.blend(epa, mkt)["w_model"] == 0.0).all()


def test_explicit_w_model_still_works_for_experiments(league, out_dir):
    epa, mkt = _ratings_inputs(league)
    assert (ratings.blend(epa, mkt, w_model=1.0)["w_model"] == 1.0).all()


def test_w_model_lookup_is_clipped_and_missing_keys_mean_zero(out_dir):
    (out_dir / "validation_status.json").write_text(json.dumps(
        {"generated": "2026-10-01T00:00:00+00:00", "w_model": {"spread_vs_close|all": 7.0, "total_vs_close|all": -1}}))
    assert status.w_model_for("spread_vs_close") == 1.0
    assert status.w_model_for("total_vs_close") == 0.0
    assert status.w_model_for("spread_vs_open") == 0.0


# ------------------------------------------------------ early-season shrink


def test_prior_weight_decays_over_weeks_0_to_5_then_vanishes():
    w = [state.early_season_prior_weight(k) for k in range(0, 8)]
    assert w[0] == 1.0 and w[6] == 0.0 and w[7] == 0.0
    assert all(a > b for a, b in zip(w[:5], w[1:6]))


def test_team_state_shrinks_the_model_toward_the_prior_early_and_not_late(league, out_dir):
    prior = pd.Series(np.linspace(-10, 10, len(league["teams"])), index=league["teams"])
    early = state.build_team_state(2025, 1, league["lines"], league["plays"], prior, w_model=1.0)
    pw = state.early_season_prior_weight(2)
    expect = (1 - pw) * early["model_rating_pts"] + pw * prior.reindex(early.index)
    np.testing.assert_allclose(early["model_rating_shrunk"], expect)
    np.testing.assert_allclose(early["rating"], early["model_rating_shrunk"])      # w_model = 1
    late = state.build_team_state(2025, 8, league["lines"], league["plays"], prior, w_model=1.0)
    np.testing.assert_allclose(late["model_rating_shrunk"], late["model_rating_pts"])
    assert (late["prior_weight"] == 0).all()


def test_team_state_without_a_prior_is_unshrunk_and_market_only_by_default(league, out_dir):
    t = state.build_team_state(2025, 1, league["lines"], league["plays"])
    np.testing.assert_allclose(t["model_rating_shrunk"], t["model_rating_pts"])
    np.testing.assert_allclose(t["rating"], t["market_rating"])      # nothing proven => market


# --------------------------------------------------------------- leak test


@pytest.mark.parametrize("week", [3, 6])
def test_team_state_is_identical_with_or_without_later_data(league, out_dir, week):
    """Acceptance: ratings as of (S, W) do not change if rows from later weeks (or seasons) exist."""
    prior = pd.Series(np.linspace(-8, 8, len(league["teams"])), index=league["teams"])

    def keep(df):
        return df[(df["season"] < 2025) | ((df["season"] == 2025) & (df["week"] <= week))]
    full = state.build_team_state(2025, week, league["lines"], league["plays"], prior, w_model=0.3)
    cut = state.build_team_state(2025, week, keep(league["lines"]), keep(league["plays"]), prior, w_model=0.3)
    pd.testing.assert_frame_equal(full, cut)


# ----------------------------------------------------------- preseason priors


def _multi_season_world(seed=5, n_teams=36, seasons=range(2018, 2026)):
    rng = np.random.default_rng(seed)
    teams = [f"T{i:02d}" for i in range(n_teams)]
    s = pd.Series(rng.normal(0, 10, n_teams), index=teams)
    strength, lines, rec, ret = {}, [], {}, {}
    for y in range(min(seasons) - 3, max(seasons) + 1):
        s = 0.75 * s + rng.normal(0, 6, n_teams)              # strength persists but drifts
        strength[y] = s.copy()
        rec[y] = pd.DataFrame({"team": teams, "points": 200 + 4 * s.values + rng.normal(0, 25, n_teams)})
        ret[y] = pd.DataFrame({"team": teams, "totalPPA": 0.3 * s.values + rng.normal(0, 6, n_teams),
                               "passingPPA": 0.3 * s.values + rng.normal(0, 6, n_teams)})
    gid = 0
    for y in seasons:
        for wk in range(1, 13):
            order = list(rng.permutation(teams))
            for i in range(0, n_teams, 2):
                h, a = order[i], order[i + 1]
                em = strength[y][h] - strength[y][a] + 2.4
                lines.append(dict(gameId=gid, season=y, week=wk, home=h, away=a, book="sim",
                                  spread_close=-np.round((em + rng.normal(0, 1.5)) * 2) / 2,
                                  total_close=52.0))
                gid += 1
    return pd.DataFrame(lines), rec, ret, list(seasons)


def test_prior_weights_are_fitted_cross_validated_and_saved(out_dir):
    lines, rec, ret, seasons = _multi_season_world()
    hist = priors.build_prior_history(lines, rec, ret, {}, seasons)
    assert set(hist["season"]) == set(seasons[1:])                     # first season has no prior
    weights, cv = priors.fit_prior_weights_cv(hist)
    assert sum(weights.values()) == pytest.approx(1.0, abs=0.01)
    assert weights["prior_rating"] == max(weights.values())          # persistence dominates by design
    assert cv["scheme"] == "leave-one-season-out" and cv["n"] == len(hist)
    assert cv["r_weighted_composite"] > 0.5
    # the extra inputs must be worth having: out-of-sample, not worse than last rating alone
    assert cv["r_weighted_composite"] >= cv["r_prior_rating_only"] - 0.01

    path = priors.save_prior_weights(weights, cv, cv["seasons"], generated="2026-10-06T00:00:00+00:00")
    assert path == out_dir / "prior_weights.json"
    saved = json.loads(path.read_text())
    assert saved["weights"] == weights and saved["cv"]["rmse"] == cv["rmse"]
    assert priors.load_prior_weights()[1] == "fitted"


def test_unfitted_defaults_are_labelled_as_such(out_dir):
    w, source = priors.load_prior_weights()
    assert source == "default_unfitted" and w == priors.DEFAULT_WEIGHTS
    out = priors.build_preseason_ratings(pd.Series({"A": 1.0, "B": -1.0}), pd.Series({"A": 0.5, "B": -0.5}),
                                         pd.Series({"A": 0.5, "B": -0.5}), pd.Series({"A": 0.0, "B": 0.0}))
    assert out.attrs["weights_source"] == "default_unfitted"


def test_prior_history_uses_only_earlier_seasons_for_its_inputs(out_dir):
    lines, rec, ret, seasons = _multi_season_world(seasons=range(2020, 2025))
    full = priors.build_prior_history(lines, rec, ret, {}, seasons)
    no_future = priors.build_prior_history(lines[lines["season"] <= 2022], rec, ret, {}, [2020, 2021, 2022])
    a = full[full["season"] == 2022].set_index("team")[priors.PRIOR_FEATURES]
    b = no_future[no_future["season"] == 2022].set_index("team")[priors.PRIOR_FEATURES]
    pd.testing.assert_frame_equal(a, b)


# ------------------------------------------------------------ player state


def _steady_box(weeks=range(1, 9)):
    return pd.concat([box_week(2025, w, "Alpha", rush={"RB1": (18, 90), "RB2": (12, 54)}, game_id=w)
                      for w in weeks], ignore_index=True)


def test_player_uncertainty_shrinks_with_every_game_observed():
    box = _steady_box()
    sds, effs = [], []
    for w in range(1, 9):
        st = playerstate.build_player_state(box, 2025, w)
        row = st[(st["pid"] == "RB1") & (st["group"] == "rush")].iloc[0]
        sds.append(row["share_sd"]); effs.append(row["eff_sd"])
        assert row["games"] == w
    assert all(a > b for a, b in zip(sds, sds[1:])), sds
    assert all(a > b for a, b in zip(effs, effs[1:])), effs
    st = playerstate.build_player_state(box, 2025, 8)
    rb1 = st[st["pid"] == "RB1"].iloc[0]
    assert rb1["share_mean"] == pytest.approx(0.6, abs=0.08) and rb1["eff_mean"] == pytest.approx(5.0, abs=0.35)


def test_a_role_change_widens_the_posterior():
    """RB2 takes over the job. One odd game does not move the posterior rank, so it is not a role
    change; a sustained swap eventually flips the ranks, and in exactly that week both players'
    uncertainty widens instead of shrinking."""
    st = playerstate.build_player_state(_steady_box(range(1, 6)), 2025, 5)
    assert st[st["pid"] == "RB1"].iloc[0]["rank"] == 1
    flipped_in = None
    for w in range(6, 14):
        swap = box_week(2025, w, "Alpha", rush={"RB1": (5, 20), "RB2": (25, 130)}, game_id=w)
        new = playerstate.update_player_state(st, swap, 2025, w)
        flags = {pid: bool(new[new["pid"] == pid].iloc[0]["role_change"]) for pid in ("RB1", "RB2")}
        if flipped_in is None and any(flags.values()):
            flipped_in = w
            assert all(flags.values())                       # both players' jobs changed
            for pid in ("RB1", "RB2"):
                a, b = st[st["pid"] == pid].iloc[0], new[new["pid"] == pid].iloc[0]
                assert b["share_sd"] > a["share_sd"], pid
                assert b["eff_sd"] > a["eff_sd"], pid
            assert new[new["pid"] == "RB2"].iloc[0]["rank"] == 1
        elif flipped_in is None:
            assert w < 9, "the swap should have flipped the ranks by now"
            for pid in ("RB1", "RB2"):                       # no flip yet: normal shrinkage
                assert new[new["pid"] == pid].iloc[0]["share_sd"] < st[st["pid"] == pid].iloc[0]["share_sd"]
        st = new
    assert flipped_in is not None and flipped_in > 6          # not on the very first odd game


def test_returning_after_missed_games_counts_as_a_role_change():
    box = pd.concat([
        _steady_box(range(1, 4)),
        box_week(2025, 4, "Alpha", rush={"RB2": (30, 150)}, game_id=4),     # RB1 absent
        box_week(2025, 5, "Alpha", rush={"RB2": (30, 150)}, game_id=5),     # RB1 absent
        box_week(2025, 6, "Alpha", rush={"RB1": (18, 90), "RB2": (12, 54)}, game_id=6)], ignore_index=True)
    s5 = playerstate.build_player_state(box, 2025, 5)
    assert s5[s5["pid"] == "RB1"].iloc[0]["missed"] == 2
    s6 = playerstate.build_player_state(box, 2025, 6)
    rb1 = s6[s6["pid"] == "RB1"].iloc[0]
    assert rb1["role_change"] and rb1["missed"] == 0


def test_incremental_update_equals_rebuilding_from_scratch(out_dir):
    box = _steady_box()
    inc = None
    for w in range(1, 9):
        inc = playerstate.update_player_state(inc, box[box["week"] == w], 2025, w)
    pd.testing.assert_frame_equal(inc, playerstate.build_player_state(box, 2025, 8))


def test_player_state_ignores_later_weeks():
    box = _steady_box()
    pd.testing.assert_frame_equal(playerstate.build_player_state(box, 2025, 4),
                                  playerstate.build_player_state(box[box["week"] <= 4], 2025, 4))


def test_transfers_start_from_prior_school_production_and_returners_carry_forward():
    prev_final = pd.DataFrame([
        dict(pid="STAR", team="Old U", player="STAR", group="rush", share_a=60.0, share_b=40.0,
             share_mean=0.6, eff_mean=6.6, eff_prec=8.0, eff_sd=0.35, games=11),
        dict(pid="STAY", team="Alpha", player="STAY", group="rush", share_a=60.0, share_b=40.0,
             share_mean=0.6, eff_mean=5.5, eff_prec=8.0, eff_sd=0.35, games=11)])
    ep = playerstate.EFF_PRIOR["rush"]
    xfer = playerstate._new_row("STAR", "Alpha", "STAR", "rush", 2025, 1, prev_final)
    assert xfer["prior_source"] == "prior_school"
    assert xfer["eff_mean"] == pytest.approx(0.5 * 6.6 + 0.5 * ep["mean"])      # pulled half-way to average
    assert xfer["eff_prec"] < 8.0                                              # and much less certain
    stay = playerstate._new_row("STAY", "Alpha", "STAY", "rush", 2025, 1, prev_final)
    assert stay["prior_source"] == "carried" and stay["eff_mean"] == 5.5
    assert stay["eff_prec"] == pytest.approx(8.0 * playerstate.SEASON_CARRY)
    new = playerstate._new_row("NEW", "Alpha", "NEW", "rush", 2025, 1, prev_final)
    assert new["prior_source"] == "position" and new["eff_mean"] == ep["mean"]


def test_run_update_writes_state_files_and_continues_from_last_weeks_file(league, out_dir):
    box = _steady_box(range(1, 8))
    kw = dict(lines=league["lines"], pbp=league["plays"], box=box, directory=out_dir / "state")
    r3 = state.run_update(2025, 3, **kw)
    assert r3["teams_path"].name == "teams_2025_w3.csv" and r3["players_path"].name == "players_2025_w3.csv"
    for w in (4, 5):
        res = state.run_update(2025, w, **kw)          # week 4+ continues from the saved file
    batch = playerstate.build_player_state(box, 2025, 5)
    pd.testing.assert_frame_equal(res["players"].reset_index(drop=True), batch, check_dtype=False)
    back = state.read_team_state(2025, 5, out_dir / "state")
    assert set(back.columns) >= {"rating", "market_rating", "model_rating_shrunk", "prior_weight", "w_model"}
    assert state.last_week_with_players(2025, out_dir / "state") == 5
