"""Key loading (CLAUDE.md rule 4): readable errors, never leak the key."""
from __future__ import annotations

import pytest

from cfbmodel import cli, ingest, keys

# Obviously fake. Never put a real key in a test.
FAKE = "fake-key-for-tests-0123456789"


@pytest.fixture(autouse=True)
def _no_real_key(monkeypatch):
    monkeypatch.delenv(keys.KEY_NAME, raising=False)


def test_missing_key_raises_readable_error_without_a_value(tmp_path):
    with pytest.raises(keys.MissingKeyError) as err:
        keys.get_cfbd_key(tmp_path / "does_not_exist.env")
    msg = str(err.value)
    assert "CFBD_API_KEY" in msg and "collegefootballdata.com/key" in msg


def test_empty_value_in_env_file_counts_as_missing(tmp_path):
    f = tmp_path / ".env"
    f.write_text("CFBD_API_KEY=\n")          # exactly what .env.example contains
    with pytest.raises(keys.MissingKeyError):
        keys.get_cfbd_key(f)


def test_reads_key_from_env_file(tmp_path):
    f = tmp_path / ".env"
    f.write_text(f'CFBD_API_KEY="{FAKE}"\n')  # quotes are stripped by dotenv
    assert keys.get_cfbd_key(f) == FAKE


def test_environment_variable_beats_env_file(tmp_path, monkeypatch):
    f = tmp_path / ".env"
    f.write_text("CFBD_API_KEY=from-file\n")
    monkeypatch.setenv(keys.KEY_NAME, FAKE)
    assert keys.get_cfbd_key(f) == FAKE


def test_reading_env_file_does_not_export_the_key(tmp_path):
    import os
    f = tmp_path / ".env"
    f.write_text(f"CFBD_API_KEY={FAKE}\n")
    keys.get_cfbd_key(f)
    assert keys.KEY_NAME not in os.environ


def test_cache_hit_needs_no_key(tmp_path, monkeypatch):
    """Cached re-runs and the whole test suite must work with no key at all."""
    monkeypatch.setattr(ingest, "CACHE", tmp_path)
    path = ingest._cache_path("/games", {"year": 2099})
    path.write_text('[{"id": 1}]')
    assert ingest.cfbd_get("/games", year=2099) == [{"id": 1}]


def test_cli_missing_key_is_a_clean_exit_not_a_traceback(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(ingest, "CACHE", tmp_path)          # empty cache -> must hit network
    monkeypatch.setattr(keys, "ENV_FILE", tmp_path / "no.env")
    code = cli.main(["qb", "--season", "2099", "--week", "3"])
    err = capsys.readouterr().err
    assert code == 2
    assert "CFBD_API_KEY is not set" in err
    assert "Traceback" not in err
