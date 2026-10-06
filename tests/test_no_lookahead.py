"""
No-lookahead guard (CLAUDE.md rule 2; AUDIT.md bug A).

Anything computed "as of (season S, week W)" may use only games strictly before
that point: earlier seasons, or the same season with an earlier week. The old
filter `(season < S) | (week < W)` also let in LATER seasons' early weeks, so a
2024 week-5 rating was partly built from 2025 week 1-4 games.

The test: feed each as-of function (a) everything, and (b) only the rows that are
strictly before the as-of point. The two outputs must be identical. If a future
row changes the answer, it leaked.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cfbmodel import props, qb, ratings, script

SEASONS = (2023, 2024, 2025)
WEEKS = range(1, 9)
TEAMS = [f"T{i:02d}" for i in range(12)]

# Includes the as-of points that exposed the bug (a later season with a small
# week number), the first week of a season, and a week with nothing before it
# in the same season.
AS_OF = [(2024, 5), (2024, 1), (2025, 8), (2023, 4)]


def before(df: pd.DataFrame, season: int, week: int) -> pd.DataFrame:
    """Reference implementation of the correct filter, written independently."""
    keep = [(s, w) < (season, week) for s, w in zip(df["season"], df["week"])]
    return df[keep].reset_index(drop=True)


@pytest.fixture(scope="module")
def league():
    """Tiny synthetic league. Own RNG so no other test's draws are disturbed."""
    rng = np.random.default_rng(2026)
    strength = pd.Series(rng.normal(0, 8, len(TEAMS)), index=TEAMS)
    games, lines, plays, box = [], [], [], []
    gid = 0
    for s in SEASONS:
        for w in WEEKS:
            order = rng.permutation(TEAMS)
            for i in range(0, len(TEAMS), 2):
                h, a = order[i], order[i + 1]
                em = strength[h] - strength[a] + 2.4
                lines.append(dict(gameId=gid, season=s, week=w, home=h, away=a,
                                  book="sim", spread_close=-round(em * 2) / 2,
                                  total_close=float(rng.integers(44, 62))))
                for off, dfn in ((h, a), (a, h)):
                    n = 24
                    mu = (strength[off] - strength[dfn]) / 136
                    plays.append(pd.DataFrame(dict(
                        season=s, week=w, gameId=gid, offense=off, defense=dfn, home=h,
                        offenseConference="X", defenseConference="X",
                        playType=rng.choice(["Rush", "Pass Reception", "Pass Incompletion"], n),
                        passer=f"{off} QB1", epa=rng.normal(mu, 1.2, n),
                        period=rng.integers(1, 5, n), offenseScore=0, defenseScore=0,
                        homeWinProb=0.5)))
                    for p in (f"{off} RB1", f"{off} RB2"):
                        for cat, typ in (("rushing", "CAR"), ("rushing", "YDS"),
                                         ("receiving", "REC"), ("receiving", "YDS")):
                            box.append(dict(season=s, week=w, gameId=gid, team=off,
                                            player=p, athlete_id=hash(p) % 10_000,
                                            category=cat, stat_type=typ,
                                            value=float(rng.integers(1, 30))))
                gid += 1
    return dict(lines=pd.DataFrame(lines), pbp=pd.concat(plays, ignore_index=True),
                box=pd.DataFrame(box))


def _same(a: pd.DataFrame, b: pd.DataFrame):
    pd.testing.assert_frame_equal(a.sort_index(axis=1), b.sort_index(axis=1),
                                  check_exact=False, rtol=1e-9, atol=1e-9)


@pytest.mark.parametrize("season,week", AS_OF)
def test_market_ratings_ignore_future_rows(league, season, week):
    ln = league["lines"]
    _same(ratings.fit_market_ratings(ln, season, week),
          ratings.fit_market_ratings(before(ln, season, week), season, week))


@pytest.mark.parametrize("season,week", AS_OF)
def test_total_ratings_ignore_future_rows(league, season, week):
    ln = league["lines"]
    _same(ratings.fit_total_ratings(ln, season, week),
          ratings.fit_total_ratings(before(ln, season, week), season, week))


@pytest.mark.parametrize("season,week", AS_OF)
def test_epa_ratings_ignore_future_rows(league, season, week):
    pbp = league["pbp"]
    _same(ratings.fit_epa_ratings(pbp, season, week, split_pass_rush=False),
          ratings.fit_epa_ratings(before(pbp, season, week), season, week,
                                  split_pass_rush=False))


@pytest.mark.parametrize("season,week", AS_OF)
def test_qb_epa_ignores_future_rows(league, season, week):
    pbp = league["pbp"]
    for filt in (True, False):
        _same(qb.qb_epa(pbp, season, week, filter_garbage=filt),
              qb.qb_epa(before(pbp, season, week), season, week, filter_garbage=filt))


@pytest.mark.parametrize("season,week", AS_OF)
def test_player_priors_ignore_future_rows(league, season, week):
    box = league["box"]
    _same(props.build_player_priors(box, season, week),
          props.build_player_priors(before(box, season, week), season, week))


@pytest.mark.parametrize("season,week", AS_OF)
def test_depth_chart_ignores_future_rows(league, season, week):
    box = league["box"]
    team = TEAMS[0]
    _same(script.derive_depth_chart(box, team, season, week),
          script.derive_depth_chart(before(box, season, week), team, season, week))


def test_as_of_helper_matches_reference(league):
    """ratings.as_of is the one shared filter; it must equal the reference above."""
    ln = league["lines"]
    for season, week in AS_OF:
        got = ratings.as_of(ln, season, week).reset_index(drop=True)
        pd.testing.assert_frame_equal(got, before(ln, season, week))


def test_recency_weights_never_exceed_one_for_past_data(league):
    """The old bug gave later-season rows a weight above 1 (about 1.9x)."""
    pbp = league["pbp"]
    for season, week in AS_OF:
        past = before(pbp, season, week)
        if len(past):
            assert ratings.recency_weights(past, season, week).max() <= 1.0 + 1e-12
