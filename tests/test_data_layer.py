"""Phase 1 acceptance: cache, budget, error handling, schemas, team names. No network, no real key."""
from __future__ import annotations

import os
import time
from datetime import datetime, timezone

import pandas as pd
import pytest

from cfbmodel import budget, cli, ingest, schema, teams, weather
from conftest import FAKE_KEY, load_fixture


# ------------------------------------------------------------------ caching


def test_identical_second_pull_uses_zero_calls(cfbd, capsys):
    assert cli.main(["pull", "--seasons", "2025"]) == 0
    first = cfbd.count()
    assert first > 0
    out = capsys.readouterr().out
    assert "calls this run" in out and "remaining" in out

    ingest._CALLS = 0
    assert cli.main(["pull", "--seasons", "2025"]) == 0
    assert cfbd.count() == first, "second identical pull must not call CFBD at all"
    assert ingest.calls_used() == 0


def test_pull_prints_rows_per_endpoint_and_calls_remaining(cfbd, capsys):
    cli.main(["pull", "--seasons", "2025"])
    out = capsys.readouterr().out
    for label in ("teams/fbs", "games", "lines", "plays", "games/players", "venues"):
        assert label in out
    assert "used this month" in out and "remaining" in out


def test_completed_season_is_cached_forever_even_if_the_file_is_old(cfbd):
    ingest.games(2025)
    n = cfbd.count("/games")
    for f in ingest.CACHE.glob("*.json"):
        os.utime(f, (time.time() - 90 * 86400,) * 2)
    ingest.games(2025)
    assert cfbd.count("/games") == n


def test_current_season_refreshes_after_ttl_but_not_before(cfbd):
    ingest.games(2026)
    n = cfbd.count("/games")
    ingest.games(2026)
    assert cfbd.count("/games") == n                      # fresh -> reused
    for f in ingest.CACHE.glob("games__*.json"):
        os.utime(f, (time.time() - 13 * 3600,) * 2)       # older than the 12h TTL
    ingest.games(2026)
    assert cfbd.count("/games") == n + 1


def test_old_weeks_of_current_season_are_never_repulled(cfbd):
    ingest.set_current(season=2026, week=8)
    assert ingest.ttl_for(2026, 3) is None                # completed week -> forever
    assert ingest.ttl_for(2026, 7) == ingest.REFRESH_TTL_HOURS   # previous week -> refreshable
    assert ingest.ttl_for(2026, 8) == ingest.REFRESH_TTL_HOURS   # current week -> refreshable
    assert ingest.ttl_for(2026) == ingest.REFRESH_TTL_HOURS      # whole-season tables
    assert ingest.ttl_for(2025) is None and ingest.ttl_for(2025, 12) is None


def test_refresh_flag_forces_a_new_call(cfbd):
    ingest.games(2025)
    ingest.configure(refresh=True)
    ingest.games(2025)
    assert cfbd.count("/games") == 2


# -------------------------------------------------------------------- errors


def test_401_gives_a_clear_message_without_the_key_and_without_a_traceback(cfbd, capsys):
    cfbd.status["/teams/fbs"] = [401]
    code = cli.main(["pull", "--seasons", "2025"])
    err = capsys.readouterr().err
    assert code == 2
    assert "CFBD_API_KEY" in err and "rejected" in err
    assert "Traceback" not in err
    assert FAKE_KEY not in err


def test_401_error_object_never_contains_the_key(cfbd):
    cfbd.status["/games"] = [401]
    with pytest.raises(ingest.CfbdAuthError) as e:
        ingest.games(2025)
    assert FAKE_KEY not in str(e.value)


def test_key_is_sent_as_a_bearer_header_only(cfbd):
    ingest.venues()
    assert cfbd.headers_seen[0] == {"Authorization": f"Bearer {FAKE_KEY}"}
    assert all(FAKE_KEY not in str(p) for _, p in cfbd.requests)


def test_429_backs_off_exponentially_then_succeeds_and_counts_one_call(cfbd):
    cfbd.status["/venues"] = [429, 429, 200]
    ingest.venues()
    assert cfbd.sleeps == [1, 2]
    assert ingest.calls_used() == 1
    assert ingest.budget().used() == 1


def test_429_on_every_try_stops_the_run(cfbd):
    cfbd.status["/venues"] = [429] * 10
    with pytest.raises(ingest.CfbdError, match="rate-limited"):
        ingest.venues()


def test_a_server_error_stops_the_pull_instead_of_being_swallowed(cfbd, capsys):
    cfbd.status["/lines"] = [500]
    code = cli.main(["pull", "--seasons", "2025"])
    assert code == 2
    assert "HTTP 500" in capsys.readouterr().err


def test_a_404_on_an_optional_dataset_is_reported_not_hidden(cfbd, capsys):
    cfbd.status["/player/portal"] = [404]
    assert cli.main(["pull", "--seasons", "2025"]) == 0
    assert "not available (HTTP 404)" in capsys.readouterr().out


def test_a_404_on_games_is_an_error(cfbd):
    cfbd.status["/games"] = [404]
    with pytest.raises(ingest.CfbdNotFound):
        ingest.games(2025)


def test_missing_key_on_a_cache_miss_is_a_clean_exit(cfbd, monkeypatch, capsys):
    monkeypatch.delenv("CFBD_API_KEY")
    assert cli.main(["pull", "--seasons", "2025"]) == 2
    assert "CFBD_API_KEY is not set" in capsys.readouterr().err
    assert cfbd.count() == 0


# -------------------------------------------------------------------- budget


def _at(month: str):
    y, m = map(int, month.split("-"))
    return lambda: datetime(y, m, 15, tzinfo=timezone.utc)


def test_budget_counts_persist_between_runs(tmp_path):
    b = budget.Budget(tmp_path, now_fn=_at("2026-10"))
    b.record(3)
    assert budget.Budget(tmp_path, now_fn=_at("2026-10")).used() == 3


def test_budget_resets_in_a_new_month(tmp_path):
    budget.Budget(tmp_path, now_fn=_at("2026-10")).record(500)
    nov = budget.Budget(tmp_path, now_fn=_at("2026-11"))
    assert nov.used() == 0 and nov.remaining() == 1000


def test_budget_warns_at_700(tmp_path, capsys):
    b = budget.Budget(tmp_path, now_fn=_at("2026-10"))
    b.record(699)
    b.check()
    assert capsys.readouterr().err == ""
    b.record(1)
    b.check()
    assert "700 of 1000" in capsys.readouterr().err
    b.check()                                              # only warns once per run
    assert capsys.readouterr().err == ""


def test_budget_refuses_at_950_unless_forced(tmp_path):
    b = budget.Budget(tmp_path, now_fn=_at("2026-10"))
    b.record(949)
    b.check()                                              # 949 still allowed
    b.record(1)
    with pytest.raises(budget.BudgetExceeded, match="--force"):
        b.check()
    b.check(force=True)


def test_refused_call_never_reaches_the_server(cfbd):
    ingest.budget().record(950)
    with pytest.raises(budget.BudgetExceeded):
        ingest.venues()
    assert cfbd.count() == 0
    ingest.configure(force=True)
    ingest.venues()
    assert cfbd.count("/venues") == 1


# ------------------------------------------------------------------- schemas


def test_every_entity_loads_from_fixtures_and_passes_its_schema(cfbd):
    for name, df in {
        "teams_fbs": ingest.teams_fbs(2025), "games": ingest.games(2025), "lines": ingest.lines(2025),
        "plays": ingest.plays(2025, 1), "player_box": ingest.player_box(2025, 1),
        "recruiting_teams": ingest.recruiting_teams(2025),
        "returning_production": ingest.returning_production(2025), "portal": ingest.portal(2025),
        "usage": ingest.usage(2025), "venues": ingest.venues(),
    }.items():
        assert len(df) > 0, name
        schema.check_frame(df, name)


def test_games_columns_and_derived_values(cfbd):
    g = ingest.games(2025).set_index("id")
    assert g.loc[101, "margin"] == 10 and g.loc[101, "total"] == 50
    assert pd.isna(g.loc[104, "margin"])                    # not played yet
    assert str(g["start_date"].dt.tz) == "UTC"


def test_lines_keep_open_and_close_for_every_book(cfbd):
    ln = ingest.lines(2025)
    one = ln[(ln.gameId == 101) & (ln.book == "Book One")].iloc[0]
    assert (one.spread_close, one.spread_open, one.total_close, one.total_open) == (-6.5, -5.5, 52.5, 51.5)
    assert set(ln[ln.gameId == 101].book) == {"Book One", "Book Two"}
    assert (ln.gameId == 103).sum() == 0                    # a game with no lines has no rows


def test_lines_with_a_duplicate_game_book_pair_are_rejected(cfbd):
    ln = ingest.lines(2025)
    with pytest.raises(schema.SchemaError, match="share the same"):
        schema.check_frame(pd.concat([ln, ln.iloc[[0]]], ignore_index=True), "lines")


def test_raw_response_missing_a_required_field_names_the_field(cfbd):
    bad = load_fixture("games.json")
    del bad[1]["homeTeam"]
    with pytest.raises(schema.SchemaError, match=r"homeTeam \(missing in 1 of 5"):
        schema.check_raw(bad, "games")
    nested = load_fixture("lines.json")
    del nested[0]["lines"][0]["provider"]
    with pytest.raises(schema.SchemaError, match=r"lines\[\]\.provider"):
        schema.check_raw(nested, "lines")


def test_a_renamed_field_stops_the_pull(cfbd, capsys):
    import json
    path = ingest._cache_path("/games", {"year": 2025, "seasonType": "both"})
    data = load_fixture("games.json")
    for r in data:
        r["home_team"] = r.pop("homeTeam")
    path.write_text(json.dumps(data))
    assert cli.main(["pull", "--seasons", "2025"]) == 2
    assert "homeTeam" in capsys.readouterr().err


def test_frame_checks_catch_wrong_types_and_missing_columns(cfbd):
    g = ingest.games(2025)
    with pytest.raises(schema.SchemaError, match="missing column 'margin'"):
        schema.check_frame(g.drop(columns=["margin"]), "games")
    with pytest.raises(schema.SchemaError, match="home_is_fbs"):
        schema.check_frame(g.assign(home_is_fbs="yes"), "games")
    with pytest.raises(schema.SchemaError, match="has blanks"):
        schema.check_frame(g.assign(homeTeam=[None] * len(g)), "games")


def test_regular_weeks_skip_postseason_and_unplayed_games(cfbd):
    assert ingest.regular_weeks(2025) == [1, 2, 3]


def test_venue_dome_flag_is_kept_and_unknown_stays_unknown(cfbd):
    v = ingest.venues().set_index("name")
    assert v.loc["Beta Dome", "dome"] is True and v.loc["Alpha Field", "dome"] is False
    assert v.loc["Unknown Park", "dome"] is None


def test_data_doc_is_generated_from_the_schema():
    from pathlib import Path
    doc = Path(__file__).resolve().parent.parent / "docs" / "DATA.md"
    assert doc.read_text() == schema.render_markdown(), (
        "docs/DATA.md is out of date: run `python -m cfbmodel docs-data`")


# ---------------------------------------------------------------- team names


def test_accented_name_maps_to_the_canonical_one_and_fcs_is_kept_and_flagged(cfbd):
    g = ingest.games(2025).set_index("id")
    assert g.loc[102, "awayTeam"] == "San Jose State" and g.loc[102, "away_is_fbs"]
    assert g.loc[103, "awayTeam"] == "Tiny FCS College" and not g.loc[103, "away_is_fbs"]
    assert len(g) == 5                                      # nothing dropped


def test_unmatched_names_are_logged_with_counts(cfbd):
    ingest.games(2025)
    ingest.lines(2025)
    path = teams.write_unmatched_log()
    log = pd.read_csv(path)
    row = log[log.name == "Tiny FCS College"]
    assert len(row) >= 1 and (row.rows >= 1).all()
    assert "San José State" not in set(log.name)            # matched, so not listed


def test_the_same_name_is_used_in_every_table(cfbd):
    names = set(ingest.games(2025).awayTeam) | set(ingest.lines(2025).away)
    assert "San Jose State" in names and "San José State" not in names


def test_aliases_map_a_known_spelling_and_must_point_at_fbs_teams():
    t = teams.TeamNames(["San Jose State", "Alpha State"], aliases={"SJSU": "San Jose State"})
    assert t.canonical("SJSU") == "San Jose State"
    assert t.canonical("san jose st") is None               # no fuzzy guessing
    assert t.canonical("ALPHA  STATE") == "Alpha State"     # case/space folding only
    with pytest.raises(ValueError, match="not FBS teams"):
        teams.TeamNames(["Alpha State"], aliases={"X": "Nowhere U"})


# -------------------------------------------------------------------- weather


def test_weather_cache_key_is_stable_across_runs(tmp_path, monkeypatch):
    # Pinned value: if this changes, the cache silently stops working again.
    assert weather._cache_key("https://x.test/a", {"b": 1, "a": 2}) == "wx_96440aea4f7c68aa.json"
    monkeypatch.setattr(weather, "CACHE", tmp_path)
    calls = []

    class R:
        def raise_for_status(self): pass
        def json(self): return {"hourly": {}}

    monkeypatch.setattr(weather.requests, "get", lambda *a, **k: calls.append(1) or R())
    weather._cache("https://x.test/a", {"a": 2})
    weather._cache("https://x.test/a", {"a": 2})
    assert len(calls) == 1
