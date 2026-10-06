"""
Team ratings.

Two independent rating systems, deliberately kept separate:

1. `fit_epa_ratings`  — opponent-adjusted EPA/play from play-by-play, via ridge
   regression with recency weights and a garbage-time filter. This is your
   *information*.

2. `fit_market_ratings` — power ratings backed out of closing spreads by least
   squares. This is the *market's* information, compressed into the same units.

The whole point is the difference between them. A model that only reproduces
market ratings has no edge; a model that ignores them has no discipline. The
backtest asks the only question that matters: does (1) explain any margin
variance that (2) has not already priced?

Note on the NFL result you already have: opponent-adjusted EPA added nothing on
top of the NFL closing spread. CFB is a different market — 130+ teams, most
games untraded until late, far more dispersion in team strength — so the same
test genuinely can come out differently here. Run it before believing it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import Ridge

from .config import C


# ------------------------------------------------------------ play filtering


def clean_plays(pbp: pd.DataFrame) -> pd.DataFrame:
    """Filter to competitive, meaningful scrimmage plays with valid EPA."""
    df = pbp.copy()

    if "ppa" in df.columns and "epa" not in df.columns:
        df = df.rename(columns={"ppa": "epa"})
    df["epa"] = pd.to_numeric(df.get("epa"), errors="coerce")
    df = df[df["epa"].notna()]

    # scrimmage plays only
    keep = {
        "Rush", "Pass Reception", "Pass Incompletion", "Pass Completion",
        "Sack", "Passing Touchdown", "Rushing Touchdown", "Interception",
        "Fumble Recovery (Own)", "Fumble Recovery (Opponent)",
        "Pass Interception Return", "Interception Return Touchdown",
        "Fumble Return Touchdown",
    }
    if "playType" in df.columns:
        df = df[df["playType"].isin(keep)]

    # garbage time: CFB needs a tighter filter than the NFL
    for col in ("homeWinProb", "home_wp"):
        if col in df.columns:
            wp = pd.to_numeric(df[col], errors="coerce")
            df = df[wp.between(C.gt_wp_low, C.gt_wp_high) | wp.isna()]
            break

    if {"period", "offenseScore", "defenseScore"} <= set(df.columns):
        gap = (df["offenseScore"] - df["defenseScore"]).abs()
        df = df[~((df["period"] >= 4) & (gap >= C.gt_min_quarter_4_margin))]

    # drop FCS opponents from the rating fit; they distort the scale badly.
    # (You still *bet* those games — see game_model.fcs_adjustment.)
    if "offenseConference" in df.columns:
        df = df[df["offenseConference"].notna() & df["defenseConference"].notna()]

    return df


def recency_weights(df: pd.DataFrame, asof_season: int, asof_week: int) -> np.ndarray:
    """Exponential decay in games-ago, plus a cross-season discount."""
    seasons_back = asof_season - df["season"].to_numpy()
    weeks_back = np.where(
        seasons_back == 0, asof_week - df["week"].to_numpy(), asof_week + 16 * seasons_back
    )
    weeks_back = np.clip(weeks_back, 0, None)
    w = 0.5 ** (weeks_back / C.form_half_life_games)
    w *= C.season_carryover ** seasons_back
    return w


# --------------------------------------------------------- EPA ratings (ridge)


def fit_epa_ratings(
    pbp: pd.DataFrame, asof_season: int, asof_week: int, split_pass_rush: bool = True
) -> pd.DataFrame:
    """
    Ridge regression:   epa_play ~ offense_team + defense_team + home + intercept

    Ridge (not OLS) because early-season sample is tiny and unbalanced — the L2
    penalty is doing empirical-Bayes shrinkage toward the league mean, which is
    exactly what you want in week 3 when a team has faced two cupcakes.

    Returns one row per team with off/def ratings in EPA-per-play units, and
    pass/rush splits if requested.
    """
    df = clean_plays(pbp)
    df = df[(df["season"] < asof_season) | (df["week"] < asof_week)]
    if df.empty:
        raise ValueError("no plays available before the as-of point")

    teams = sorted(set(df["offense"]) | set(df["defense"]))
    idx = {t: i for i, t in enumerate(teams)}
    n, p = len(df), len(teams)

    rows = np.repeat(np.arange(n), 2)
    cols = np.empty(2 * n, dtype=int)
    cols[0::2] = df["offense"].map(idx).to_numpy()
    cols[1::2] = df["defense"].map(idx).to_numpy() + p
    vals = np.empty(2 * n)
    vals[0::2] = 1.0
    vals[1::2] = 1.0   # +1: positive def coef == allows more EPA == bad defense
    X = sparse.csr_matrix((vals, (rows, cols)), shape=(n, 2 * p))

    home_col = _home_indicator(df).reshape(-1, 1)
    X = sparse.hstack([X, sparse.csr_matrix(home_col)]).tocsr()

    y = df["epa"].to_numpy()
    w = recency_weights(df, asof_season, asof_week)

    model = Ridge(alpha=C.ridge_alpha_off, fit_intercept=True, solver="sparse_cg")
    model.fit(X, y, sample_weight=w)
    coef = model.coef_

    out = pd.DataFrame(
        {
            "team": teams,
            "off_epa": coef[:p],
            "def_epa": coef[p : 2 * p],  # positive = allows more EPA = bad defense
        }
    )
    out["net_epa"] = out["off_epa"] - out["def_epa"]
    out["plays"] = df.groupby("offense").size().reindex(out["team"]).fillna(0).values

    if split_pass_rush:
        for label, mask in (
            ("pass", _is_pass(df)),
            ("rush", ~_is_pass(df)),
        ):
            sub = _sub_ratings(df[mask], teams, idx, asof_season, asof_week)
            out[f"off_epa_{label}"] = sub["off"].reindex(out["team"]).values
            out[f"def_epa_{label}"] = sub["def"].reindex(out["team"]).values

    out["pace"] = _pace(df).reindex(out["team"]).values
    out["hfa_league"] = float(coef[-1])
    return out.set_index("team")


def _sub_ratings(df, teams, idx, asof_season, asof_week) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(index=teams, columns=["off", "def"], dtype=float)
    n, p = len(df), len(teams)
    rows = np.repeat(np.arange(n), 2)
    cols = np.empty(2 * n, dtype=int)
    cols[0::2] = df["offense"].map(idx).to_numpy()
    cols[1::2] = df["defense"].map(idx).to_numpy() + p
    vals = np.empty(2 * n)
    vals[0::2] = 1.0
    vals[1::2] = 1.0
    X = sparse.csr_matrix((vals, (rows, cols)), shape=(n, 2 * p))
    w = recency_weights(df, asof_season, asof_week)
    m = Ridge(alpha=C.ridge_alpha_def, fit_intercept=True, solver="sparse_cg")
    m.fit(X, df["epa"].to_numpy(), sample_weight=w)
    return pd.DataFrame({"off": m.coef_[:p], "def": m.coef_[p:]}, index=teams)


def _is_pass(df: pd.DataFrame) -> pd.Series:
    pt = df.get("playType", pd.Series("", index=df.index)).fillna("")
    return pt.str.contains("Pass|Sack|Interception", regex=True)


def _home_indicator(df: pd.DataFrame) -> np.ndarray:
    if {"offense", "home"} <= set(df.columns):
        return (df["offense"] == df["home"]).astype(float).to_numpy()
    return np.zeros(len(df))


def _pace(df: pd.DataFrame) -> pd.Series:
    g = df.groupby(["offense", "gameId"]).size().groupby("offense").mean()
    return g.rename("pace")


# ------------------------------------------------- market-implied power ratings


def fit_market_ratings(
    lines_df: pd.DataFrame, asof_season: int, asof_week: int, half_life_weeks: float = 5.0
) -> pd.DataFrame:
    """
    Least-squares power ratings from closing spreads:
        spread_home = -(rating_home - rating_away) - hfa

    Closing spreads are the single most accurate public forecast of CFB games
    that exists. Backing ratings out of them gives you a clean, shared-scale
    benchmark and — more usefully — a way to price a *game that has no line yet*
    the way the market would. That is the actual engine of early-week edge:
    you are not beating the closer, you are beating the number before it forms.
    """
    df = lines_df.dropna(subset=["spread_close"]).copy()
    df = df[(df["season"] < asof_season) | (df["week"] < asof_week)]
    df = df.groupby(["gameId", "home", "away", "season", "week"], as_index=False)[
        "spread_close"
    ].median()
    if df.empty:
        raise ValueError("no closing lines before as-of point")

    teams = sorted(set(df["home"]) | set(df["away"]))
    idx = {t: i for i, t in enumerate(teams)}
    n, p = len(df), len(teams)

    X = np.zeros((n, p + 1))
    X[np.arange(n), df["home"].map(idx)] = 1.0
    X[np.arange(n), df["away"].map(idx)] = -1.0
    X[:, -1] = 1.0  # HFA

    y = -df["spread_close"].to_numpy()  # market's expected home margin

    back = (asof_season - df["season"]) * 16 + (asof_week - df["week"])
    w = 0.5 ** (np.clip(back, 0, None) / half_life_weeks)

    m = Ridge(alpha=1.0, fit_intercept=False)
    m.fit(X, y, sample_weight=w)

    r = pd.Series(m.coef_[:p], index=teams, name="market_rating")
    r = r - r.mean()
    out = r.to_frame()
    out["market_hfa"] = float(m.coef_[-1])
    return out


def fit_total_ratings(lines_df: pd.DataFrame, asof_season: int, asof_week: int) -> pd.DataFrame:
    """Same trick for totals: team scoring + team scoring-allowed environment."""
    df = lines_df.dropna(subset=["total_close"]).copy()
    df = df[(df["season"] < asof_season) | (df["week"] < asof_week)]
    df = df.groupby(["gameId", "home", "away", "season", "week"], as_index=False)[
        "total_close"
    ].median()
    if df.empty:
        raise ValueError("no closing totals before as-of point")

    teams = sorted(set(df["home"]) | set(df["away"]))
    idx = {t: i for i, t in enumerate(teams)}
    n, p = len(df), len(teams)
    X = np.zeros((n, p))
    X[np.arange(n), df["home"].map(idx)] = 1.0
    X[np.arange(n), df["away"].map(idx)] = 1.0
    y = df["total_close"].to_numpy()
    m = Ridge(alpha=2.0, fit_intercept=True)
    m.fit(X, y)
    return pd.DataFrame({"total_rating": m.coef_}, index=teams)


def blend(
    epa_ratings: pd.DataFrame, market_ratings: pd.DataFrame, w_model: float = 0.35
) -> pd.DataFrame:
    """
    Convert EPA ratings to points and blend with market ratings.

    w_model is the only genuinely dangerous knob in this repo. High w_model =
    you are claiming to know more than the market. Set it from the backtest's
    fitted regression coefficient, not from how confident you feel.
    """
    j = epa_ratings.join(market_ratings, how="inner")
    pts = j["net_epa"] * C.pace_mean * 2.0  # EPA/play -> points/game, both sides
    pts = pts - pts.mean()
    j["model_rating_pts"] = pts
    j["rating"] = w_model * j["model_rating_pts"] + (1 - w_model) * j["market_rating"]
    j["disagreement"] = j["model_rating_pts"] - j["market_rating"]
    return j
