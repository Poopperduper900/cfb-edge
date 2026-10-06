"""Phase 10 acceptance: every dashboard page opens on fixture data. Needs `streamlit` (optional)."""
from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("streamlit", reason="streamlit is optional until the owner approves adding it")
from streamlit.testing.v1 import AppTest  # noqa: E402

from cfbmodel import config  # noqa: E402
from dash_fixture import make_output  # noqa: E402

APP = str(Path(__file__).resolve().parent.parent / "app" / "dashboard.py")


@pytest.fixture
def folders(tmp_path, monkeypatch):
    out, cache = make_output(tmp_path)
    monkeypatch.setattr(config, "OUTPUT", out)
    monkeypatch.setattr(config, "CACHE", cache)
    return out, cache


def open_page(name):
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception, at.exception
    if name != "Edge Board":
        at.sidebar.radio[0].set_value(name).run()
    assert not at.exception, at.exception
    return at


def test_edge_board_page(folders):
    at = open_page("Edge Board")
    assert at.header[0].value == "Edge Board"
    assert any("No validation_status.json" in w.value for w in at.warning)
    assert [m.label for m in at.metric] == ["BET", "INFO", "PASS", "FCS", "NO_LINE"]
    assert len(at.dataframe) == 1


def test_validation_page(folders):
    at = open_page("Validation")
    assert "Validation" in at.header[0].value
    assert len(at.dataframe) >= 1 and any("passed" in s.value.lower() for s in at.success)


def test_clv_and_bankroll_page(folders):
    at = open_page("CLV & Bankroll")
    assert at.header[0].value == "CLV & Bankroll" and len(at.dataframe) == 1
    assert any("ROI is noise" in m.value for m in at.markdown)


def test_data_health_page(folders):
    at = open_page("Data Health")
    assert at.header[0].value == "Data Health"
    assert len(at.dataframe) >= 2
    assert any("v0001" in m.value for m in at.markdown)


def test_every_page_says_what_to_run_when_there_is_nothing_yet(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "OUTPUT", tmp_path / "o")
    monkeypatch.setattr(config, "CACHE", tmp_path / "c")
    for page in ("Edge Board", "Validation", "CLV & Bankroll"):
        at = open_page(page)
        assert any("Nothing here yet" in i.value for i in at.info), page
    assert not open_page("Data Health").exception
