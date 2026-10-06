"""One hand-built week (no randomness) that exercises every board status. Tests only."""
from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd

NOW = datetime(2026, 10, 8, 12, 0, tzinfo=timezone.utc)            # Thursday; games are Saturday: 'close' basis

RATING = {"Alpha State": 10.0, "Beta Tech": 4.0, "Gamma U": -2.0, "Delta College": -8.0, "Epsilon A&M": 12.0,
          "Zeta Poly": 1.0, "Eta Tech": -5.0, "Theta State": 6.0, "Iota U": 0.0, "Kappa State": 3.0}


def team_state() -> pd.DataFrame:
    idx = pd.Index(list(RATING), name="team")
    r = pd.Series(RATING, index=idx)
    return pd.DataFrame({"model_rating_shrunk": r, "model_rating_pts": r, "market_rating": r * 0.9,
                         "rating": r * 0.9, "w_model": 0.0}, index=idx)


def total_ratings() -> pd.DataFrame:
    return pd.DataFrame({"total_rating": {t: 26.0 + 0.2 * v for t, v in RATING.items()}}).rename_axis("team")


def _g(i, home, away, hc, ac, kick, neutral=False, home_fbs=True, away_fbs=True, **extra):
    return dict(id=i, homeTeam=home, awayTeam=away, homeConference=hc, awayConference=ac,
                start_date=pd.Timestamp(kick, tz="UTC"), neutralSite=neutral, home_is_fbs=home_fbs,
                away_is_fbs=away_fbs, **extra)


def games() -> pd.DataFrame:
    sat, thu = "2026-10-10T16:00:00", "2026-10-08T23:30:00"           # noon ET Saturday, 7:30pm ET Thursday
    rows = [
        _g(1, "Alpha State", "Beta Tech", "SEC", "SEC", sat),                          # close to market -> PASS
        _g(2, "Epsilon A&M", "Gamma U", "MAC", "MAC", sat),                            # model loves home -> big gap
        _g(3, "Theta State", "Delta College", "Sun Belt", "MAC", thu),                 # weeknight mid-major
        _g(4, "Zeta Poly", "Tiny FCS", "Big Ten", None, sat, away_fbs=False),          # FBS vs FCS
        _g(5, "Iota U", "Kappa State", "MAC", "MAC", sat),                             # no line anywhere
        _g(6, "Eta Tech", "Beta Tech", "Sun Belt", "Sun Belt", sat, neutral=True,
           dome=False, wind_mph=24.0, gust_mph=34.0, precip_in=0.0, temp_f=48.0, wx_confidence=0.9,
           wx_status="ok"),                                                            # windy, neutral site
    ]
    return pd.DataFrame(rows)


def lines() -> pd.DataFrame:
    def row(gid, book, sc, so, tc, to, mh=np.nan, ma=np.nan):
        return dict(gameId=gid, book=book, spread_close=sc, spread_open=so, total_close=tc, total_open=to,
                    ml_home=mh, ml_away=ma)
    return pd.DataFrame([
        row(1, "Book One", -7.0, -6.5, 52.5, 52.5, -280, 235), row(1, "Book Two", -7.0, -7.0, 53.0, 52.5, -290, 240),
        row(2, "Book One", -6.5, -6.5, 49.5, 49.5, -250, 205), row(2, "Book Two", -7.0, -6.5, 50.0, 49.5, -260, 215),
        row(3, "Book One", -10.5, -13.5, 56.0, 55.0, -900, 600),
        row(4, "Book One", -31.5, -31.5, 58.5, 58.5),
        row(6, "Book One", -1.5, -1.5, 46.5, 46.5, -125, 105), row(6, "Book Two", -1.0, -1.5, 47.0, 46.5, -120, 100),
    ])


PASSING_VALIDATION = {
    "generated": "2026-10-01", "status": "OK",
    "markets": {"spread_vs_close": {"passed": True}, "spread_vs_open": {"passed": False},
                "total_vs_close": {"passed": False}, "total_vs_open": {"passed": False}},
    "segments": [], "w_model": {"spread_vs_close|all": 0.45},
}
