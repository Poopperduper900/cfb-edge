"""Phase 9: the player-prop board. Lines come from the owner's CSV; BET is locked behind a CLV backtest."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cfbmodel import cli, config, ingest, params, propsboard, state
from cfbmodel.params import P
from synth import box_week

T0 = pd.Timestamp("2026-10-09T15:00:00Z")


def pstate():
    def row(player, team, group, share, sd, eff, games, touches, rank, **kw):
        return dict(pid=player, team=team, player=player, group=group, share_mean=share, share_sd=sd, eff_mean=eff,
                    games=games, touches=touches, rank=rank, role_change=False, **kw)
    return pd.DataFrame([
        row("Ace Runner", "Alpha State", "rush", 0.60, 0.03, 5.2, 6, 100.0, 1),
        row("Bo Backup", "Alpha State", "rush", 0.25, 0.05, 4.8, 6, 40.0, 2),
        row("Cy Catcher", "Alpha State", "rec", 0.28, 0.04, 13.0, 6, 30.0, 1),
        row("Dan Dash", "Beta Tech", "rush", 0.55, 0.04, 4.9, 6, 90.0, 1),
        row("Eli Thin", "Beta Tech", "rush", 0.05, 0.05, 4.5, 1, 3.0, 3),
    ])


GAMES = pd.DataFrame([dict(id=1, homeTeam="Alpha State", awayTeam="Beta Tech")])


def cons(spread=-24.5, total=58.0):
    return pd.DataFrame({"spread_close": [spread], "total_close": [total]}, index=[1])


def lines(rows):
    base = dict(over_price=-115.0, under_price=-105.0, book="MyBook", p_play=1.0)
    return pd.DataFrame([{**base, **r} for r in rows])


def price(rows, spread=-24.5, eligible=False, **kw):
    return propsboard.build_props_board(lines(rows), pstate(), GAMES, cons(spread), None, week=6, eligible=eligible,
                                        params_version="v0001", **kw)


ACE = dict(player="Ace Runner", team="Alpha State", market="rush_yds", line=70.5)


# --------------------------------------------------------------------- the file


def write_csv(tmp_path, text):
    p = tmp_path / "props_lines.csv"
    p.write_text(text)
    return p


HEADER = "player,team,market,line,over_price,under_price,book,timestamp,p_play\n"


def test_a_valid_file_loads_and_blank_p_play_stays_blank(tmp_path):
    p = write_csv(tmp_path, HEADER + "Ace Runner,Alpha State,rush_yds,79.5,-115,-105,MyBook,2026-10-09T15:00:00Z,\n"
                                     "Cy Catcher,Alpha State,receptions,4.5,+105,-125,MyBook,2026-10-09T15:00:00Z,0.9\n")
    df = propsboard.load_lines(p)
    assert len(df) == 2 and pd.isna(df.loc[0, "p_play"]) and df.loc[1, "p_play"] == 0.9
    assert str(df["timestamp"].dt.tz) == "UTC"


@pytest.mark.parametrize("body,fragment", [
    ("Ace,Alpha,pass_yds,250.5,-115,-105,B,2026-10-09T15:00:00Z,1\n", "QB state"),
    ("Ace,Alpha,rush_yds,abc,-115,-105,B,2026-10-09T15:00:00Z,1\n", "non-number"),
    ("Ace,Alpha,rush_yds,70.5,-15,-105,B,2026-10-09T15:00:00Z,1\n", "American odds"),
    ("Ace,Alpha,rush_yds,70.5,-115,-105,B,not-a-time,1\n", "timestamp"),
    ("Ace,Alpha,rush_yds,70.5,-115,-105,B,2026-10-09T15:00:00Z,1.5\n", "p_play"),
])
def test_bad_rows_are_rejected_with_a_plain_message(tmp_path, body, fragment):
    with pytest.raises(propsboard.PropsError, match=fragment):
        propsboard.load_lines(write_csv(tmp_path, HEADER + body))


def test_missing_file_and_missing_columns_explain_what_to_create(tmp_path):
    with pytest.raises(propsboard.PropsError, match="Create it with the header"):
        propsboard.load_lines(tmp_path / "nope.csv")
    with pytest.raises(propsboard.PropsError, match="missing columns"):
        propsboard.load_lines(write_csv(tmp_path, "player,team,market\nA,B,rush_yds\n"))


def test_the_latest_look_per_player_market_and_book_is_what_gets_priced(tmp_path):
    p = write_csv(tmp_path, HEADER + "Ace Runner,Alpha State,rush_yds,79.5,-115,-105,B1,2026-10-08T12:00:00Z,\n"
                                     "Ace Runner,Alpha State,rush_yds,83.5,-110,-110,B1,2026-10-09T12:00:00Z,\n"
                                     "Ace Runner,Alpha State,rush_yds,80.5,-110,-110,B2,2026-10-08T12:00:00Z,\n")
    latest = propsboard.latest_lines(propsboard.load_lines(p))
    assert sorted(latest["line"]) == [80.5, 83.5]


# ------------------------------------------------------------------- pricing


def test_board_has_the_documented_columns_and_records_the_params_version():
    b = price([ACE])
    assert list(b.columns) == propsboard.BOARD_COLUMNS and set(b["params_version"]) == {"v0001"}


def test_a_blowout_lowers_the_starters_projection_and_raises_the_backups():
    rows = [ACE, dict(ACE, player="Bo Backup", line=30.5)]
    blow, even = price(rows, spread=-24.5).set_index("player"), price(rows, spread=0.0).set_index("player")
    assert blow.loc["Ace Runner", "proj_mean"] < even.loc["Ace Runner", "proj_mean"]
    assert blow.loc["Bo Backup", "proj_mean"] > even.loc["Bo Backup", "proj_mean"]


def test_p_play_scales_the_projection_and_a_blank_one_is_flagged():
    full = price([dict(ACE, p_play=1.0)]).iloc[0]
    half = price([dict(ACE, p_play=0.5)]).iloc[0]
    assert half["proj_mean"] == pytest.approx(0.5 * full["proj_mean"], rel=0.12)
    blank = price([dict(ACE, p_play=np.nan)]).iloc[0]
    assert "p_play assumed 1.0" in blank["flags"] and blank["p_play"] == 1.0
    assert "p_play assumed" not in full["flags"]


def test_receptions_and_receiving_yards_are_priced_from_the_same_simulation():
    b = price([dict(player="Cy Catcher", team="Alpha State", market="rec_yds", line=40.5),
               dict(player="Cy Catcher", team="Alpha State", market="receptions", line=3.5)]).set_index("market")
    assert 20 < b.loc["rec_yds", "proj_mean"] < 90 and 1 < b.loc["receptions", "proj_mean"] < 8


def test_unknown_players_teams_and_thin_samples_are_reported_not_priced():
    b = price([dict(ACE, player="Nobody Atall"), dict(ACE, team="Gamma U"),
               dict(player="Eli Thin", team="Beta Tech", market="rush_yds", line=10.5)]).set_index("player")
    assert b.loc["Nobody Atall", "status"] == "UNMATCHED" and "spelling" in b.loc["Nobody Atall", "flags"]
    assert b.loc["Ace Runner", "status"] == "UNMATCHED" and "no game this week" in b.loc["Ace Runner", "flags"]
    assert b.loc["Eli Thin", "status"] == "PASS" and "too little history" in b.loc["Eli Thin", "flags"]


def test_name_matching_ignores_case_accents_and_punctuation():
    b = price([dict(ACE, player="ace  RUNNER", team="alpha state")])
    assert b.iloc[0]["status"] != "UNMATCHED"


def test_a_fair_line_is_a_pass_and_a_hugely_wrong_line_is_an_edge():
    mean = price([ACE]).iloc[0]["proj_mean"]
    fair = price([dict(ACE, line=round(mean) + 0.5)]).iloc[0]
    wrong = price([dict(ACE, line=max(mean - 45, 5.5))]).iloc[0]          # book hangs a number far below the projection
    assert fair["status"] == "PASS"
    assert wrong["side"] == "over" and wrong["ev"] > P.betting.min_ev_props and wrong["status"] == "INFO"


def test_bet_is_locked_until_the_clv_backtest_unlocks_it():
    wrong = dict(ACE, line=max(price([ACE]).iloc[0]["proj_mean"] - 45, 5.5))
    assert price([wrong], eligible=False).iloc[0]["status"] == "INFO"
    unlocked = price([wrong], eligible=True).iloc[0]
    assert unlocked["status"] == "BET" and 0 < unlocked["stake_pct"] <= P.betting.max_bet_pct
    assert price([ACE | dict(line=round(price([ACE]).iloc[0]["proj_mean"]) + 0.5)], eligible=True).iloc[0]["status"] == "PASS"


def test_pricing_is_deterministic():
    pd.testing.assert_frame_equal(price([ACE, dict(ACE, player="Bo Backup")]), price([ACE, dict(ACE, player="Bo Backup")]))


def test_volume_shock_is_larger_for_shaky_roles_and_stays_within_bounds():
    s = propsboard.volume_shock_sigma
    assert s(0.60, 0.03) < s(0.20, 0.05) < s(0.08, 0.06)                   # settled starter < committee < fringe
    assert s(0.30, 0.02) < s(0.30, 0.10)                                    # more uncertainty, more swing
    for share, sd in ((0.9, 0.001), (0.01, 0.5)):
        assert P.script.role_min_sigma <= s(share, sd) <= P.script.role_max_sigma


# ---------------------------------------------------- the CLV eligibility backtest


def _log_and_history(n, move, side="over", line=70.5):
    log = pd.DataFrame({"player": [f"P{i}" for i in range(n)], "team": "T", "game": "g", "market": "rush_yds",
                        "side": side, "line": line, "book": "B", "timestamp": "2026-10-08T12:00:00+00:00"})
    hist = pd.DataFrame({"player": [f"P{i}" for i in range(n)], "team": "T", "market": "rush_yds",
                         "line": line + np.asarray(move), "timestamp": T0})
    return log, hist


def test_props_unlock_when_logged_sides_keep_beating_the_closing_number():
    rng = np.random.default_rng(0)
    log, hist = _log_and_history(150, rng.choice([2.0, 3.0, 4.0, -1.0], 150, p=[.3, .3, .2, .2]))   # over, line rose: good
    ok, why, m = propsboard.eligibility(log, hist)
    assert ok and m["n"] == 150 and m["median_clv"] > 0 and m["pct_beat_close"] > 0.5 and "BET-eligible" in why


def test_the_clv_sign_for_unders_is_the_opposite():
    log, hist = _log_and_history(150, np.full(150, 3.0), side="under")      # we said under; the line rose: worse
    ok, why, m = propsboard.eligibility(log, hist)
    assert not ok and m["median_clv"] == -3.0 and "not clearly positive" in why


def test_props_stay_locked_with_too_few_rows_even_if_every_one_was_good():
    log, hist = _log_and_history(40, np.full(40, 3.0))
    ok, why, _ = propsboard.eligibility(log, hist)
    assert not ok and "40 of 100" in why


def test_props_stay_locked_when_the_positive_clv_could_be_luck():
    rng = np.random.default_rng(1)
    log, hist = _log_and_history(120, rng.normal(0.05, 3.0, 120))           # tiny average edge, huge spread
    assert not propsboard.eligibility(log, hist)[0]


def test_a_logged_row_with_no_later_line_cannot_be_measured():
    log, hist = _log_and_history(150, np.full(150, 3.0))
    early = hist.assign(timestamp=pd.Timestamp("2026-10-01T00:00:00Z"))      # nothing newer than what the model acted on
    assert propsboard.eligibility(log, early)[2]["n"] == 0
    assert propsboard.backtest_clv(pd.DataFrame(), hist) == {"n": 0}


def test_only_model_liked_sides_are_logged_with_a_timestamp(tmp_path):
    wrong = dict(ACE, line=max(price([ACE]).iloc[0]["proj_mean"] - 45, 5.5))
    bo_mean = price([dict(ACE, player="Bo Backup", line=30.5)]).iloc[0]["proj_mean"]
    b = price([wrong, dict(ACE, player="Bo Backup", line=round(bo_mean) + 0.5)])        # one big edge, one fair line
    p = tmp_path / "props_log.csv"
    n1 = propsboard.log_recommendations(b, datetime(2026, 10, 9, 15, tzinfo=timezone.utc), p)
    n2 = propsboard.log_recommendations(b, datetime(2026, 10, 9, 16, tzinfo=timezone.utc), p)
    log = pd.read_csv(p)
    assert n1 == n2 == 1 and len(log) == 2 and list(log["player"].unique()) == ["Ace Runner"]
    assert log["timestamp"].nunique() == 2 and p.read_text().count("timestamp") == 1


# --------------------------------------------------------------- the command


def test_props_board_command_end_to_end(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(config, "OUTPUT", tmp_path)
    monkeypatch.setattr(cli, "OUTPUT", tmp_path)
    box = pd.concat([box_week(2026, w, "Alpha State", rush={"Ace Runner": (18, 95), "Bo Backup": (8, 36)},
                              rec={"Cy Catcher": (5, 66)}, game_id=w) for w in range(0, 6)], ignore_index=True)
    games = pd.DataFrame([dict(id=1, season=2026, week=6, homeTeam="Alpha State", awayTeam="Beta Tech", neutralSite=False,
                               season_type="regular")])
    lines = pd.DataFrame([dict(gameId=1, season=2026, week=6, book="B", spread_close=-14.5, spread_open=-14.5,
                               total_close=55.5, total_open=55.5, ml_home=np.nan, ml_away=np.nan)])
    monkeypatch.setattr(ingest, "games", lambda s, **k: games)
    monkeypatch.setattr(ingest, "lines", lambda s, **k: lines)
    monkeypatch.setattr(ingest, "player_box", lambda s, w, **k: box[box["week"] == w])
    (tmp_path / "props_lines.csv").write_text(
        HEADER + "Ace Runner,Alpha State,rush_yds,40.5,-115,-105,MyBook,2026-10-09T15:00:00Z,1\n"
                 "Cy Catcher,Alpha State,rec_yds,200.5,-110,-110,MyBook,2026-10-09T15:00:00Z,\n"
                 "Nobody Atall,Alpha State,rush_yds,50.5,-110,-110,MyBook,2026-10-09T15:00:00Z,1\n")
    assert cli.main(["props-board", "--season", "2026", "--week", "6"]) == 0
    out = capsys.readouterr().out
    b = pd.read_csv(tmp_path / "props_board_2026_w6.csv")
    assert list(b.columns) == propsboard.BOARD_COLUMNS and len(b) == 3
    assert "not BET-eligible yet" in out and "BET" not in set(b["status"])
    assert set(b["status"]) <= {"INFO", "PASS", "UNMATCHED"} and "UNMATCHED" in set(b["status"])
    assert (tmp_path / "props_log.csv").exists() or "INFO" not in set(b["status"])
    assert cli.main(["props-board", "--season", "2026", "--week", "6", "--lines", str(tmp_path / "nope.csv")]) == 2
