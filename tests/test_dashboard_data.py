"""Phase 10: the dashboard's read-only data layer, on a folder built with the real writers."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app import data
from cfbmodel import config
from dash_fixture import make_output


@pytest.fixture
def folders(tmp_path, monkeypatch):
    out, cache = make_output(tmp_path)
    monkeypatch.setattr(config, "OUTPUT", out)
    monkeypatch.setattr(config, "CACHE", cache)
    return out, cache


def test_boards_are_listed_newest_first_and_load_with_their_banners(folders):
    assert [(s, w) for s, w, _ in data.list_boards()] == [(2026, 6), (2026, 5)]
    df, meta = data.load_board(2026, 6)
    assert len(df) == 15 and set(df["status"]) >= {"BET", "INFO", "PASS", "FCS", "NO_LINE"}
    assert any("No validation_status.json" in b for b in meta["banners"]) and meta["meta"]["params_version"] == "v0001"
    assert (df["flags"] == df["flags"].fillna("")).all()                     # blanks are empty strings, not NaN
    older, no_meta = data.load_board(2026, 5)
    assert len(older) == 15 and no_meta == {}                                # a board without a meta file still loads


def test_validation_tables(folders):
    v = data.validation()
    assert v["status"]["status"] == "OK" and v["report"].startswith("# Validation report") and v["age_days"] < 1
    m = data.markets_table(v["status"])
    assert set(m["market"]) == {"spread_vs_open", "spread_vs_close", "total_vs_open", "total_vs_close"}
    assert m.set_index("market").loc["spread_vs_close", "passed"]            # the fixture's model is informative
    passed = data.segments_table(v["status"], only_passed=True)
    assert len(passed) > 0 and passed["passed"].all()
    near = data.segments_table(v["status"], near_misses=True)
    assert not near["passed"].any()


def test_bankroll_curve_is_the_running_sum_of_settled_profit(folders):
    s = data.settled_bets()
    curve = data.bankroll_curve(s)
    assert list(curve["bet_id"]) == sorted(curve["bet_id"]) and len(curve) == 30
    assert curve["cumulative_profit"].iloc[-1] == pytest.approx(s["profit"].sum())
    assert "ROI is noise" in data.clv_report_text()
    pending = s.assign(status="pending")
    assert len(data.bankroll_curve(pending)) == 0


def test_data_health_reports_budget_cache_age_and_unmatched_names(folders):
    h = data.data_health()
    assert (h["calls_used"], h["calls_remaining"], h["limit"]) == (42, 958, 1000)
    f = h["freshness"].set_index("endpoint")
    assert f.loc["games", "files"] == 1 and 1.9 < f.loc["games", "newest_age_hours"] < 2.2
    assert 29.9 < f.loc["plays", "newest_age_hours"] < 30.2
    assert list(h["unmatched"]["name"]) == ["Tiny FCS"]
    assert h["params_version"] == "v0001" and list(h["models"]["version"]) == ["v0001"]
    assert h["last_learn"] is None and h["drift"] is None


def test_an_empty_install_returns_nothing_instead_of_crashing(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "OUTPUT", tmp_path / "empty_out")
    monkeypatch.setattr(config, "CACHE", tmp_path / "empty_cache")
    assert data.list_boards() == [] and data.load_board(2026, 1) == (None, {})
    assert data.validation() == {"status": None, "report": None, "age_days": None}
    assert data.markets_table(None).empty and data.segments_table(None).empty
    assert data.settled_bets() is None and data.clv_report_text() is None and len(data.bankroll_curve(None)) == 0
    h = data.data_health()
    assert h["calls_used"] == 0 and h["freshness"].empty and h["unmatched"].empty
