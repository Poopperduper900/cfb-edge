"""Phase 8 acceptance: bet log, line snapshots, closing-line value. The sign conventions are the point."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cfbmodel import cli, config, edge, ingest, pricing, tracking

NOW = datetime(2026, 10, 9, 15, 0, tzinfo=timezone.utc)


# ----------------------------------------------------- CLV sign, every market and side


@pytest.mark.parametrize("market,side,bet,close,expected,why", [
    # spreads are in HOME convention: home -6.5 is -6.5, and the road team at +6.5 is also -6.5
    ("spread", "home", -6.5, -7.5, +1.0, "laid 6.5 and it closed at 7.5: a point better"),
    ("spread", "home", -7.5, -6.5, -1.0, "laid 7.5 and it closed at 6.5: a point worse"),
    ("spread", "home", -3.0, -3.0, 0.0, "no move"),
    ("spread", "away", -6.5, -7.5, -1.0, "took road +6.5 and it closed +7.5: a point worse"),
    ("spread", "away", -7.5, -6.5, +1.0, "took road +7.5 and it closed +6.5: a point better"),
    ("spread", "home", -3.0, -5.0, +2.0, "the example the old code got backwards"),
    ("spread", "away", -3.0, -5.0, -2.0, "ditto"),
    ("spread", "home", +2.5, +1.5, +1.0, "underdog at home: +2.5 beats a +1.5 close"),
    # totals: an over wants a LOW number, an under wants a HIGH one
    ("total", "over", 52.0, 54.0, +2.0, "over 52 and it closed 54: better"),
    ("total", "over", 54.0, 52.0, -2.0, "over 54 and it closed 52: worse"),
    ("total", "under", 52.0, 54.0, -2.0, "under 52 and it closed 54: worse"),
    ("total", "under", 54.0, 52.0, +2.0, "under 54 and it closed 52: better"),
    ("total", "over", 50.5, 50.5, 0.0, "no move"),
    ("prop", "over", 79.5, 84.5, +5.0, "props follow the totals rule"),
    ("prop", "under", 79.5, 84.5, -5.0, "ditto"),
])
def test_clv_sign_for_every_market_and_side(market, side, bet, close, expected, why):
    assert edge.clv(bet, close, market, side) == expected, why


def test_clv_rejects_nonsense_instead_of_guessing():
    with pytest.raises(ValueError):
        edge.clv(-3, -4, "spread", "over")
    with pytest.raises(ValueError):
        edge.clv(50, 51, "total", "home")
    with pytest.raises(ValueError):
        edge.clv(-110, -120, "moneyline", "home")


def test_probability_clv_has_the_same_sign_as_points_clv():
    games = _games()
    lines = pd.DataFrame([dict(gameId=1, book="B1", spread_close=-7.5, spread_open=-7.0, total_close=49.5,
                               total_open=49.5, ml_home=-300, ml_away=250)])
    bets = pd.DataFrame([
        _bet(1, "spread", "home", -6.5), _bet(2, "spread", "away", 6.5),         # better / worse
        _bet(3, "total", "over", 48.5), _bet(4, "total", "under", 48.5)])
    s = tracking.settle(bets, games, lines).set_index("bet_id")
    assert s.loc[1, "clv_pts"] > 0 and s.loc[1, "clv_prob"] > 0
    assert s.loc[2, "clv_pts"] < 0 and s.loc[2, "clv_prob"] < 0
    assert s.loc[3, "clv_pts"] > 0 and s.loc[3, "clv_prob"] > 0
    assert s.loc[4, "clv_pts"] < 0 and s.loc[4, "clv_prob"] < 0
    # buying a point is worth a similar amount of probability on either side of the same game
    assert s.loc[1, "clv_prob"] == pytest.approx(-s.loc[2, "clv_prob"], rel=0.05)


# ---------------------------------------------------------------- settlement


def _games():
    return pd.DataFrame([
        dict(id=1, awayTeam="Road U", homeTeam="Home U", neutralSite=False, week=6, homePoints=31.0, awayPoints=20.0,
             homeConference="SEC", awayConference="MAC", start_date=pd.Timestamp("2026-10-10T16:00Z")),
        dict(id=2, awayTeam="A", homeTeam="B", neutralSite=False, week=6, homePoints=np.nan, awayPoints=np.nan,
             homeConference="MAC", awayConference="MAC", start_date=pd.Timestamp("2026-10-17T16:00Z")),
        dict(id=3, awayTeam="C", homeTeam="D", neutralSite=True, week=6, homePoints=17.0, awayPoints=24.0,
             homeConference="SEC", awayConference="SEC", start_date=pd.Timestamp("2026-10-10T23:00Z")),
    ])


def _bet(i, market, side, line, price=-110.0, stake=2.0, book="B1", game_id=1, week=6):
    return dict(bet_id=i, logged_at="2026-10-08T12:00:00+00:00", season=2026, week=week, game_id=game_id,
                game="x", market=market, side=side, line=line, price=price, book=book, stake=stake, notes="")


def _lines():
    return pd.DataFrame([
        dict(gameId=1, book="B1", spread_close=-7.5, spread_open=-7.0, total_close=49.5, total_open=49.0, ml_home=-300, ml_away=250),
        dict(gameId=1, book="B2", spread_close=-8.0, spread_open=-7.0, total_close=50.0, total_open=49.0, ml_home=-320, ml_away=260),
        dict(gameId=3, book="B1", spread_close=-1.0, spread_open=-1.0, total_close=40.0, total_open=40.0, ml_home=-115, ml_away=-105),
    ])


def test_results_and_profit_for_every_market_and_side():
    # game 1 finished home 31 - road 20: margin +11, total 51
    bets = pd.DataFrame([
        _bet(1, "spread", "home", -6.5),                       # covered by 4.5 -> win
        _bet(2, "spread", "away", 6.5),                        # road +6.5 loses by 11 -> loss
        _bet(3, "spread", "home", -11.0),                      # exactly -11 -> push
        _bet(4, "total", "over", 48.5),                        # 51 > 48.5 -> win
        _bet(5, "total", "under", 51.0),                       # exactly 51 -> push
        _bet(6, "total", "under", 52.5),                       # 51 < 52.5 -> win
        _bet(7, "moneyline", "home", -280.0, price=-280.0),    # home won -> win at -280
        _bet(8, "moneyline", "away", 230.0, price=230.0),      # road lost
        _bet(9, "spread", "away", 12.0),                       # road +12 and lost by 11 -> win
    ])
    s = tracking.settle(bets, _games(), _lines()).set_index("bet_id")
    assert list(s["result"]) == ["win", "loss", "push", "win", "push", "win", "win", "loss", "win"]
    assert s.loc[1, "profit"] == pytest.approx(2 * 100 / 110)             # +110-ish: stake 2 at -110
    assert s.loc[2, "profit"] == -2.0 and s.loc[3, "profit"] == 0.0
    assert s.loc[7, "profit"] == pytest.approx(2 * 100 / 280) and s.loc[8, "profit"] == -2.0
    assert s.loc[1, "roi"] == pytest.approx(100 / 110)


def test_a_positive_price_pays_more_than_the_stake_and_a_win_on_a_road_underdog_settles_correctly():
    g = _games()                                                          # game 3: home 17 - road 24 (neutral site)
    lines = _lines()
    s = tracking.settle(pd.DataFrame([_bet(1, "moneyline", "away", 130.0, price=130.0, stake=10, game_id=3)]), g, lines)
    assert s.loc[0, "result"] == "win" and s.loc[0, "profit"] == pytest.approx(13.0)


def test_unfinished_games_stay_pending_and_have_no_clv_or_result():
    s = tracking.settle(pd.DataFrame([_bet(1, "spread", "home", -3.0, game_id=2)]), _games(), _lines())
    assert s.loc[0, "status"] == "pending" and s.loc[0, "result"] == "" and np.isnan(s.loc[0, "clv_pts"])


def test_the_bets_own_book_is_the_close_when_available_otherwise_the_median():
    bets = pd.DataFrame([_bet(1, "spread", "home", -6.5, book="B2"), _bet(2, "spread", "home", -6.5, book="Nowhere Sportsbook")])
    s = tracking.settle(bets, _games(), _lines()).set_index("bet_id")
    assert s.loc[1, "close_line"] == -8.0 and s.loc[1, "close_basis"] == "book"
    assert s.loc[2, "close_line"] == -7.75 and s.loc[2, "close_basis"] == "consensus"     # median of -7.5 and -8
    assert s.loc[1, "clv_pts"] == pytest.approx(1.5) and s.loc[2, "clv_pts"] == pytest.approx(1.25)


def test_moneyline_clv_is_the_probability_gap_to_the_no_vig_close():
    s = tracking.settle(pd.DataFrame([_bet(1, "moneyline", "home", -250.0, price=-250.0)]), _games(), _lines())
    nv = pricing.no_vig_probs([-300.0, 250.0], "moneyline")[0]            # B1's own closing prices (the bet's book)
    assert s.loc[0, "clv_prob"] == pytest.approx(nv - edge.american_to_prob(-250.0))
    assert np.isnan(s.loc[0, "clv_pts"])                                  # no points CLV on a moneyline


# ------------------------------------------------------------------- the log


def test_add_bet_validates_and_numbers_bets_in_order(tmp_path):
    p = tmp_path / "bets.csv"
    kw = dict(season=2026, week=6, game_id=1, game="Road U @ Home U", book="B1", stake=1.0, now=NOW, path=p)
    a = tracking.add_bet(market="spread", side="home", line=-6.5, price=-110, **kw)
    b = tracking.add_bet(market="total", side="over", line=48.5, price=-105, notes="wind?", **kw)
    assert (a["bet_id"], b["bet_id"]) == (1, 2)
    back = tracking.load_bets(p)
    assert list(back["bet_id"]) == [1, 2] and list(back.columns) == tracking.BET_COLUMNS and back.loc[1, "notes"] == "wind?"
    for bad in (dict(market="spread", side="over", line=1, price=-110), dict(market="total", side="home", line=50, price=-110),
                dict(market="parlay", side="home", line=1, price=-110), dict(market="spread", side="home", line=-3, price=-50),
                dict(market="spread", side="home", line=-3, price=0)):
        with pytest.raises(ValueError):
            tracking.add_bet(**{**kw, **bad})
    with pytest.raises(ValueError, match="stake"):
        tracking.add_bet(market="spread", side="home", line=-3, price=-110, **{**kw, "stake": 0})
    assert len(tracking.load_bets(p)) == 2                                # rejected bets were not written


def test_games_can_be_found_by_board_label_or_by_id():
    g = _games()
    assert tracking.resolve_game("Road U @ Home U", g, 6) == (1, "Road U @ Home U")
    assert tracking.resolve_game("  road u   @ home u ", g, 6)[0] == 1          # case and spacing do not matter
    assert tracking.resolve_game("C vs D (N)", g, 6) == (3, "C vs D (N)")
    assert tracking.resolve_game("C vs D", g, 6)[0] == 3
    assert tracking.resolve_game("1", g, 6)[0] == 1
    for bad in ("Nobody @ Nowhere", "99"):
        with pytest.raises(ValueError):
            tracking.resolve_game(bad, g, 6)
    with pytest.raises(ValueError):
        tracking.resolve_game("Road U @ Home U", g, 7)                         # wrong week


def test_line_snapshots_append_with_a_timestamp_and_one_header(tmp_path):
    p = tmp_path / "lines_log.csv"
    assert tracking.log_lines(_lines(), 2026, 6, NOW, p) == 3
    assert tracking.log_lines(_lines().head(1), 2026, 6, datetime(2026, 10, 10, 9, 0, tzinfo=timezone.utc), p) == 1
    log = pd.read_csv(p)
    assert len(log) == 4 and list(log.columns) == tracking.LOG_COLUMNS
    assert log["logged_at"].nunique() == 2 and p.read_text().count("logged_at") == 1
    assert tracking.log_lines(pd.DataFrame(), 2026, 6, NOW, p) == 0


# ------------------------------------------------------------------- the report


def _many_settled(n):
    rng = np.random.default_rng(0)
    games, lines, bets = [], [], []
    for i in range(1, n + 1):
        close = -float(rng.integers(3, 14)) - 0.5
        games.append(dict(id=i, awayTeam=f"A{i}", homeTeam=f"H{i}", neutralSite=False, week=6,
                          homePoints=float(rng.integers(10, 45)), awayPoints=float(rng.integers(10, 45)),
                          homeConference="SEC", awayConference="MAC", start_date=pd.Timestamp("2026-10-10T16:00Z")))
        lines.append(dict(gameId=i, book="B1", spread_close=close, spread_open=close, total_close=50.5, total_open=50.5,
                          ml_home=np.nan, ml_away=np.nan))
        bets.append(_bet(i, "spread", "home", close + 1.0, game_id=i))           # always a point better than the close
    return tracking.settle(pd.DataFrame(bets), pd.DataFrame(games), pd.DataFrame(lines)), pd.DataFrame(games)


def test_report_gives_clv_by_market_and_warns_that_roi_is_noise_below_100_bets():
    s, g = _many_settled(30)
    text = tracking.report(s, g)
    assert "ROI is noise at this sample — read CLV." in text
    assert "median_clv_pts" in text and "pct_beat_close" in text and "roi_lo" in text and "By market" in text
    summary = tracking._summary(s[s["status"] == "settled"])
    assert summary["median_clv_pts"] == 1.0 and summary["pct_beat_close"] == 1.0 and summary["median_clv_prob"] > 0
    assert summary["roi_lo"] <= summary["roi"] <= summary["roi_hi"]


def test_report_has_no_noise_warning_from_100_bets_and_lists_segments():
    s, g = _many_settled(120)
    text = tracking.report(s, g)
    assert "ROI is noise" not in text and "By segment" in text and "spread / tier:P4-G5" in text


def test_report_with_nothing_settled_says_so():
    pending = tracking.settle(pd.DataFrame([_bet(1, "spread", "home", -3.0, game_id=2)]), _games(), _lines())
    assert "Nothing is settled yet" in tracking.report(pending, _games())


# ----------------------------------------------------------------- the commands


def test_log_bet_and_clv_commands_end_to_end(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(config, "OUTPUT", tmp_path)
    monkeypatch.setattr(cli, "OUTPUT", tmp_path)
    monkeypatch.setattr(ingest, "games", lambda s, **k: _games())
    monkeypatch.setattr(ingest, "lines", lambda s, **k: _lines())
    assert cli.main(["log-bet", "--season", "2026", "--week", "6", "--game", "Road U @ Home U", "--market", "spread",
                     "--side", "home", "--line", "-6.5", "--price", "-110", "--book", "B1", "--stake", "2"]) == 0
    assert "logged bet #1" in capsys.readouterr().out
    assert cli.main(["clv"]) == 0
    out = capsys.readouterr().out
    assert "median_clv_pts" in out and "ROI is noise" in out
    settled = pd.read_csv(tmp_path / "bets_settled.csv")
    assert settled.loc[0, "clv_pts"] == pytest.approx(1.0) and settled.loc[0, "result"] == "win"
    assert (tmp_path / "clv_report.md").exists()
    assert cli.main(["log-bet", "--season", "2026", "--week", "6", "--game", "Nobody @ Nowhere", "--market", "spread",
                     "--side", "home", "--line", "-6.5", "--price", "-110", "--book", "B1", "--stake", "2"]) == 2
    assert "no game matches" in capsys.readouterr().err
    assert len(tracking.load_bets(tmp_path / "bets.csv")) == 1               # the failed attempt wrote nothing


def test_clv_command_with_no_bets_explains_what_to_do(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(config, "OUTPUT", tmp_path)
    assert cli.main(["clv"]) == 0
    assert "log-bet" in capsys.readouterr().out
