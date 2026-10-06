"""Shared test setup: a fake CFBD server, so no test ever touches the network or a real key."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from cfbmodel import ingest, keys, teams
from cfbmodel.config import CFBD_BASE

FIXTURES = Path(__file__).parent / "fixtures"
FAKE_KEY = "fake-test-key-DO-NOT-USE-0123456789"


def load_fixture(name: str):
    return json.loads((FIXTURES / name).read_text())


class _Response:
    def __init__(self, status_code: int, payload=None):
        self.status_code = status_code
        self._payload = payload
        self.text = ""

    def json(self):
        return self._payload


class FakeCfbd:
    """Stands in for ingest._http_get. Serves the tests/fixtures files and counts calls."""

    ROUTES = {
        "/teams/fbs": "teams_fbs.json", "/games": "games.json", "/lines": "lines.json",
        "/recruiting/teams": "recruiting_teams.json", "/player/returning": "returning_production.json",
        "/player/portal": "portal.json", "/player/usage": "usage.json", "/venues": "venues.json",
    }

    def __init__(self):
        self.requests: list[tuple[str, dict]] = []
        self.headers_seen: list[dict] = []
        self.status: dict[str, list[int]] = {}  # endpoint -> statuses to return first, in order

    def __call__(self, url, params, headers):
        endpoint = url[len(CFBD_BASE):]
        self.requests.append((endpoint, dict(params)))
        self.headers_seen.append(dict(headers))
        queue = self.status.get(endpoint)
        if queue:
            code = queue.pop(0)
            if code != 200:
                return _Response(code, {"error": "stub"})
        if endpoint == "/plays":
            return _Response(200, load_fixture("plays_week1.json") if params.get("week") == 1 else [])
        if endpoint == "/games/players":
            return _Response(200, load_fixture("player_box_week1.json") if params.get("week") == 1 else [])
        return _Response(200, load_fixture(self.ROUTES[endpoint]))

    def count(self, endpoint: str | None = None) -> int:
        return sum(1 for e, _ in self.requests if endpoint in (None, e))


@pytest.fixture
def cfbd(monkeypatch, tmp_path):
    """Isolated cache/output, a fake key, and a fake server. Yields the FakeCfbd."""
    fake = FakeCfbd()
    sleeps: list[float] = []
    monkeypatch.setattr(ingest, "CACHE", tmp_path / "cache")
    (tmp_path / "cache").mkdir()
    monkeypatch.setattr(teams, "OUTPUT", tmp_path / "out")
    monkeypatch.setattr(keys, "ENV_FILE", tmp_path / "no.env")   # never read a real .env in tests
    monkeypatch.setenv(keys.KEY_NAME, FAKE_KEY)
    monkeypatch.setattr(ingest, "_http_get", fake)
    monkeypatch.setattr(ingest, "_CALLS", 0)
    monkeypatch.setattr(ingest.time, "sleep", lambda s: sleeps.append(s))
    ingest.configure(force=False, refresh=False)
    ingest.set_current(season=2026, week=None)
    teams.reset_seen()
    fake.sleeps = sleeps
    fake.tmp = tmp_path
    yield fake
    ingest.configure(force=False, refresh=False)
    ingest.set_current(None, None)
    teams.reset_seen()
