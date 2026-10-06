"""Phase 7 acceptance: the edge board. BET is unreachable without validation; the page is honest."""
from __future__ import annotations

import copy
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import board_fixture as F
from cfbmodel import board, cli, config, ingest, params, status
from cfbmodel.params import P
from synth import make_league

GOLDEN = Path(__file__).resolve().parent / "fixtures" / "board_golden_w6.csv"
SPEC_COLUMNS = ["game", "kickoff_local", "tier", "market", "book", "line", "price", "best_line_across_books",
                "no_vig_prob", "model_prob", "prob_edge", "ev", "half_point_value", "key_number_note",
                "softness", "stake_pct", "status", "flags", "params_version"]


def build(validation=None, drift=False, age=7.0, now=F.NOW, week=6, **kw):
    return board.build_board(2026, week, F.games(), F.lines(), F.team_state(), F.total_ratings(),
                             validation_status=validation, drift_active=drift, params_version="v0001",
                             now=now, validation_age_days=age, **kw)


# ---------------------------------------------------------- the central rule


def test_with_no_validation_file_no_row_can_be_a_bet():
    b = build(validation=None)
    assert (b.df["status"] == "BET").sum() == 0 and (b.df["stake_pct"] == 0).all()
    assert (b.df["status"] == "INFO").sum() > 0          # the model's view is still shown, labelled INFO
    assert any("No validation_status.json" in x for x in b.banners)


def test_a_failed_validation_cannot_produce_a_bet_even_with_a_weight_in_the_file():
    v = copy.deepcopy(F.PASSING_VALIDATION)
    v["markets"]["spread_vs_close"]["passed"] = False          # w_model is non-zero but it did NOT pass
    assert (build(validation=v).df["status"] == "BET").sum() == 0


def test_a_drift_alarm_forces_market_only_and_no_bets():
    b = build(validation=F.PASSING_VALIDATION, drift=True)
    assert (b.df["status"] == "BET").sum() == 0
    assert any("Drift alarm active" in x for x in b.banners)


def test_a_passed_market_produces_bets_only_in_that_market_and_never_on_fcs_or_no_line_rows():
    df = build(validation=F.PASSING_VALIDATION).df
    bets = df[df["status"] == "BET"]
    assert len(bets) >= 1 and bets["market"].str.startswith("spread").all()
    assert (bets["stake_pct"] > 0).all() and (bets["stake_pct"] <= P.betting.max_bet_pct).all()
    assert not df[df["status"].isin(["FCS", "NO_LINE"])]["stake_pct"].gt(0).any()
    assert (df[~df["market"].str.startswith("spread")]["status"] != "BET").all()    # totals/ML not validated


def test_an_unvalidated_market_stays_info_even_when_the_model_sees_a_big_edge():
    df = build(validation=F.PASSING_VALIDATION).df
    ml = df[df["market"].str.startswith("moneyline") & (df["prob_edge"] > 0.2)]
    assert len(ml) and (ml["status"] == "INFO").all()


def test_only_a_passing_segment_unlocks_bets_for_games_in_that_segment():
    """(A weight of 0.9 is far beyond anything validation allows; it is used here only so the shrunk
    edge clears the bar and the segment logic, not the threshold, is what the test is about.)"""
    v = {"generated": "2026-10-01", "markets": {"spread_vs_close": {"passed": False}},
         "segments": [{"market": "spread_vs_close", "segment": "tier:G5-G5", "passed": True}],
         "w_model": {"spread_vs_close|tier:G5-G5": 0.9, "spread_vs_close|all": 0.0}}
    df = build(validation=v).df
    bets = df[df["status"] == "BET"]
    assert set(bets["tier"]) <= {"G5-G5"} and len(bets) >= 1


def test_when_several_proofs_apply_the_most_cautious_weight_is_used():
    v = {"markets": {"spread_vs_close": {"passed": True}},
         "segments": [{"market": "spread_vs_close", "segment": "weeks:4-8", "passed": True}],
         "w_model": {"spread_vs_close|all": 0.5, "spread_vs_close|weeks:4-8": 0.2}}
    assert board.validation_weight(v, "spread_vs_close", ["weeks:4-8"]) == (0.2, True)
    assert board.validation_weight(v, "spread_vs_close", ["weeks:9+"]) == (0.5, True)
    assert board.validation_weight(v, "spread_vs_open", []) == (0.0, False)
    assert board.validation_weight(None, "spread_vs_close", []) == (0.0, False)


def test_with_zero_weight_the_actionable_view_is_the_market_so_it_cannot_bet():
    v = copy.deepcopy(F.PASSING_VALIDATION)
    v["w_model"]["spread_vs_close|all"] = 1e-9          # 'passed', but practically no earned weight
    assert (build(validation=v).df["status"] == "BET").sum() == 0


# ------------------------------------------------------------- statuses and rows


def test_every_status_appears_in_the_fixture_week():
    df = build(validation=F.PASSING_VALIDATION).df
    assert set(df["status"]) == {"BET", "INFO", "PASS", "FCS", "NO_LINE"}
    fcs = df[df["status"] == "FCS"]
    assert set(fcs["tier"]) == {"FBS-FCS"} and (fcs["stake_pct"] == 0).all()
    assert df[df["status"] == "NO_LINE"]["game"].tolist() == ["Kappa State @ Iota U"]


def test_rows_are_sorted_bet_then_info_then_pass_then_fcs_then_no_line():
    order = {"BET": 0, "INFO": 1, "PASS": 2, "FCS": 3, "NO_LINE": 4}
    ranks = build(validation=F.PASSING_VALIDATION).df["status"].map(order).tolist()
    assert ranks == sorted(ranks)


def test_columns_are_exactly_the_documented_ones_in_order():
    assert list(build().df.columns) == SPEC_COLUMNS == board.COLUMNS


def test_side_line_and_best_line_are_for_the_recommended_side():
    df = build(validation=F.PASSING_VALIDATION).df
    r = df[(df["game"] == "Gamma U @ Epsilon A&M") & (df["market"] == "spread home")].iloc[0]
    assert r["line"] == pytest.approx(-6.75)                   # median of -6.5 and -7
    assert r["best_line_across_books"] == "-6.5 (Book One)"     # the better number for a home bettor
    tot = df[(df["game"] == "Delta College @ Theta State") & (df["market"].str.startswith("total"))].iloc[0]
    assert tot["market"] == "total under" and tot["best_line_across_books"] == "56 (Book One)"
    ml = df[(df["game"] == "Beta Tech vs Eta Tech (N)") & (df["market"].str.startswith("moneyline"))].iloc[0]
    assert ml["price"] == 105                                    # the best real price, not the median


def test_flags_are_plain_strings_for_weather_disagreement_and_line_moves():
    df = build().df
    flags = " | ".join(df["flags"])
    assert "weather adj -" in flags and "large disagreement — check ratings" in flags
    assert "line moved 3+ since open" in flags
    assert df.loc[df["game"] == "Beta Tech @ Alpha State", "flags"].eq("").all()


def test_qb_uncertainty_flag_appears_when_a_starter_has_little_experience():
    qb = pd.DataFrame({"qb1_dropbacks": {"Alpha State": 500.0, "Beta Tech": 40.0}})
    df = build(qb_table=qb).df
    assert "QB uncertainty (Beta Tech)" in " ".join(df.loc[df["game"] == "Beta Tech @ Alpha State", "flags"])


def test_the_params_version_is_recorded_on_every_row():
    assert set(build().df["params_version"]) == {"v0001"}


def test_key_numbers_are_called_out_on_lines_that_sit_on_one():
    df = build().df
    r = df[(df["game"] == "Beta Tech @ Alpha State") & (df["market"] == "spread home")].iloc[0]
    assert r["key_number_note"] == "sits on the 7"
    assert r["half_point_value"] > 0


# ------------------------------------------------------------------- banners


def test_banner_when_model_and_market_disagree_by_more_than_six_points_on_average():
    assert any("ratings likely broken" in x for x in build().banners)               # the fixture is extreme
    flatter = F.team_state().assign(model_rating_shrunk=lambda d: d["model_rating_shrunk"] * 0.35)   # ratings pulled toward 0
    b = board.build_board(2026, 6, F.games(), F.lines(), flatter, F.total_ratings(), validation_status=None,
                          drift_active=False, params_version="v0001", now=F.NOW)
    assert not any("ratings likely broken" in x for x in b.banners)


def test_banner_when_there_are_too_many_bets():
    with params.override({"board": {"max_bet_rows": 0}}):
        b = build(validation=F.PASSING_VALIDATION)
    assert any("too many bets" in x for x in b.banners)


def test_banner_when_the_validation_file_is_more_than_14_days_old():
    assert any("re-run validate" in x for x in build(validation=F.PASSING_VALIDATION, age=15.0).banners)
    assert not any("re-run validate" in x for x in build(validation=F.PASSING_VALIDATION, age=13.0).banners)


def test_the_assumed_price_is_always_disclosed():
    assert any("assumed -110" in x for x in build().banners)


def test_the_board_uses_the_open_basis_for_games_far_in_the_future_and_close_when_near():
    far = datetime(2026, 10, 3, 12, 0, tzinfo=timezone.utc)        # 7 days before a Saturday kickoff
    v = copy.deepcopy(F.PASSING_VALIDATION)
    v["markets"]["spread_vs_open"] = {"passed": True}
    v["w_model"]["spread_vs_open|all"] = 0.45
    v["markets"]["spread_vs_close"]["passed"] = False
    assert (build(validation=v, now=far).df["status"] == "BET").sum() >= 1       # open market validated, used
    assert (build(validation=v, now=F.NOW).df["status"] == "BET").sum() == 0     # near kickoff: close market, not validated


# ------------------------------------------------------------------- outputs


def test_kickoff_times_follow_us_eastern_daylight_rules():
    assert board.kickoff_label(pd.Timestamp("2026-10-10T16:00:00Z")) == "Sat 12:00 PM ET"      # EDT
    assert board.kickoff_label(pd.Timestamp("2026-10-08T23:30:00Z")) == "Thu 7:30 PM ET"
    assert board.kickoff_label(pd.Timestamp("2026-11-07T17:00:00Z")) == "Sat 12:00 PM ET"      # EST after Nov 1
    assert board.kickoff_label(pd.Timestamp("2026-11-01T05:59:00Z")) == "Sun 1:59 AM ET"       # just before the change
    assert board.kickoff_label(pd.Timestamp("2026-09-05T00:00:00Z")) == "Fri 8:00 PM ET"
    assert board.kickoff_label(pd.NaT) == ""


def test_golden_file_for_the_fixture_week():
    got = board.to_csv(build(validation=F.PASSING_VALIDATION))
    assert got == GOLDEN.read_text(), "board output changed; if intended, review the diff and regenerate the golden file"


def test_html_is_one_self_contained_phone_friendly_page():
    b = build(validation=F.PASSING_VALIDATION)
    page = board.to_html(b)
    assert '<meta name="viewport"' in page and "prefers-color-scheme: dark" in page.replace("prefers-color-scheme:dark", "prefers-color-scheme: dark")
    assert "http://" not in page and "https://" not in page and "<script" not in page      # nothing external, no JS
    assert "Epsilon A&amp;M" in page and "Epsilon A&M" not in page                           # escaped
    assert page.count('class="card bet"') == int((b.df["status"] == "BET").sum())
    assert "assumed -110" in page and "an honest no-bet is a correct answer" in page


def test_terminal_summary_counts_statuses_and_prints_banners():
    text = board.summary(build(validation=F.PASSING_VALIDATION))
    assert "BET 2" in text and "NO_LINE 1" in text and "!!" in text
    empty = board.summary(board.Board(pd.DataFrame(columns=board.COLUMNS).assign(status=[]), [], {"season": 2026, "week": 1, "params_version": "v0001"}))
    assert "Nothing clears the bar" in empty


# ----------------------------------------------- the command, end to end (stubbed data)


def test_the_board_command_runs_end_to_end_on_a_synthetic_league(tmp_path, monkeypatch, capsys):
    lg = make_league(seed=3, n_teams=24, seasons=(2024, 2025, 2026), weeks=range(1, 9), plays_per_team_game=30)
    g = lg["games"].assign(homePoints=lambda d: (d["total"] + d["margin"]) / 2, awayPoints=lambda d: (d["total"] - d["margin"]) / 2,
                           start_date=pd.Timestamp("2026-10-10T16:00:00Z"), season_type="regular")
    g.loc[(g["season"] == 2026) & (g["week"] == 7), ["homePoints", "awayPoints", "margin", "total"]] = np.nan
    ln = lg["lines"].assign(ml_home=np.nan, ml_away=np.nan)
    teams = lg["teams"]
    monkeypatch.setattr(config, "OUTPUT", tmp_path)
    monkeypatch.setattr(cli, "OUTPUT", tmp_path)
    monkeypatch.setattr(ingest, "games", lambda s, **k: g[g["season"] == s].reset_index(drop=True))
    monkeypatch.setattr(ingest, "lines", lambda s, **k: ln[ln["season"] == s].reset_index(drop=True))
    monkeypatch.setattr(ingest, "season_plays", lambda s, weeks=None: lg["plays"][lg["plays"]["season"] == s].reset_index(drop=True))
    monkeypatch.setattr(ingest, "recruiting_teams", lambda y: pd.DataFrame({"team": teams, "points": np.linspace(150, 300, len(teams))}))
    monkeypatch.setattr(ingest, "returning_production", lambda s: pd.DataFrame({"team": teams, "totalPPA": np.linspace(-1, 1, len(teams)),
                                                                                "passingPPA": np.linspace(-1, 1, len(teams))}))
    monkeypatch.setattr(ingest, "portal", lambda s: pd.DataFrame())
    assert cli.main(["board", "--season", "2026", "--week", "7", "--no-weather"]) == 0
    out = capsys.readouterr().out
    csv = pd.read_csv(tmp_path / "board_2026_w7.csv")
    assert list(csv.columns) == SPEC_COLUMNS
    assert len(csv) >= 12 and "BET" not in set(csv["status"])           # no validation file => no bets
    assert (tmp_path / "board_2026_w7.html").read_text(encoding="utf-8").startswith("<!doctype html>")
    assert "No validation_status.json" in out and "wrote" in out and "BOARD season 2026 week 7" in out


# ------------------------------------------------- rollback reproduces the board exactly


def test_rolling_back_reproduces_the_earlier_versions_board_exactly(tmp_path, monkeypatch):
    """Phase 6 acceptance: promote a new parameter version, price the week (it differs), roll back,
    price again: byte-for-byte the board that version v0001 produced."""
    import shutil
    from cfbmodel import learn
    src = Path(__file__).resolve().parent.parent / "params"
    pdir = tmp_path / "params"
    pdir.mkdir()
    for f in src.glob("*.json"):
        shutil.copy(f, pdir / f.name)
    monkeypatch.setenv("CFBMODEL_PARAMS", str(pdir))
    monkeypatch.setattr(config, "OUTPUT", tmp_path / "out")
    params.reload()
    try:
        def price():
            return board.to_csv(board.build_board(2026, 6, F.games(), F.lines(), F.team_state(), F.total_ratings(),
                                                  validation_status=F.PASSING_VALIDATION, drift_active=False,
                                                  params_version=params.active_version(), now=F.NOW,
                                                  validation_age_days=7))
        before = price()
        v2 = params.read_version("v0001", pdir)
        v2["_meta"].update(version="v0002", parent="v0001")
        v2["margin"]["scale_model"] = 18.0                 # a different model-error width
        v2["margin"]["hfa_points"] = 3.1
        (pdir / "v0002.json").write_text(json.dumps(v2))
        (pdir / "current.json").write_text('{"version": "v0002"}')
        params.reload()
        during = price()
        assert during != before and ",v0002\n" in during      # a different board, stamped with its version
        learn.rollback("v0001")
        assert price() == before                              # exactly the earlier board again
    finally:
        monkeypatch.delenv("CFBMODEL_PARAMS")
        params.reload()
