"""
Preseason ratings.

Weeks 0-4 are the most exploitable stretch of the CFB calendar, for one
structural reason: the market has no current-season data either. Everyone is
working from priors. If your prior is better than the consensus prior, you have
an edge that simply does not exist in November, when 10 games of results have
been absorbed into every number.

Ingredients, in rough order of predictive weight (Connelly's SP+ work and
replications land in this neighbourhood):

    prior-season adjusted rating (multi-year, decayed)   ~ 45%
    returning production (weighted to QB + OL + secondary) ~ 25%
    recruiting composite (4-year rolling, 247 composite)  ~ 20%
    transfer portal net                                    ~ 10%

Weight them by fitting to next-season results, not by intuition. The function
below ships with fitted-ish defaults and a `fit_prior_weights` that re-derives
them from your own data.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV


DEFAULT_WEIGHTS = {
    "prior_rating": 0.45,
    "returning": 0.25,
    "recruiting": 0.20,
    "portal": 0.10,
}


def rolling_recruiting(recruit_by_year: dict[int, pd.DataFrame], season: int, n_years: int = 4) -> pd.Series:
    """
    4-year rolling 247 composite team score, weighted toward the classes most
    likely to be on the field (years 2-4 of a class matter more than the
    true freshmen who just signed).
    """
    class_weights = {0: 0.15, 1: 0.30, 2: 0.30, 3: 0.25}
    acc, wsum = {}, {}
    for back in range(n_years):
        yr = season - back
        df = recruit_by_year.get(yr)
        if df is None or df.empty:
            continue
        w = class_weights.get(back, 0.1)
        for _, r in df.iterrows():
            t = r.get("team")
            pts = float(r.get("points") or 0.0)
            acc[t] = acc.get(t, 0.0) + w * pts
            wsum[t] = wsum.get(t, 0.0) + w
    s = pd.Series({t: acc[t] / wsum[t] for t in acc if wsum[t] > 0}, name="recruit_score")
    return _z(s)


def returning_production_score(rp: pd.DataFrame) -> pd.Series:
    """
    CFBD returning production. QB continuity is worth more than the aggregate
    number suggests, so it gets an explicit extra weight.
    """
    if rp.empty:
        return pd.Series(dtype=float, name="returning")
    df = rp.copy()
    cols = {c.lower(): c for c in df.columns}
    tot = df[cols.get("totalppa", list(df.columns)[-1])].astype(float)
    pass_p = df[cols["passingppa"]].astype(float) if "passingppa" in cols else tot
    s = pd.Series((0.6 * _zv(tot) + 0.4 * _zv(pass_p)), index=df[cols.get("team", "team")])
    return s.rename("returning")


# Positional value multipliers for portal transfers. Roughly the marginal
# points-per-season a starter at each position is worth relative to a
# replacement-level starter — the same shape as NFL positional value, but with a
# steeper QB premium because CFB backup quality falls off a cliff.
POSITION_VALUE = {
    "QB": 4.6, "OT": 2.1, "EDGE": 2.0, "DE": 2.0, "CB": 1.8, "WR": 1.6,
    "DT": 1.5, "OL": 1.5, "OG": 1.3, "C": 1.3, "S": 1.2, "LB": 1.1,
    "TE": 1.0, "RB": 0.8, "K": 0.4, "P": 0.3, "LS": 0.1,
}


def portal_score(
    portal_df: pd.DataFrame,
    season: int,
    prior_production: pd.DataFrame | None = None,
) -> pd.Series:
    """
    Net portal value, position-weighted and production-joined.

    Gemini flagged that the original ignored positional value, which is correct
    and was in the docstring as a known gap. Position weights are the easy half
    of the fix and they are now in `POSITION_VALUE`.

    The half that matters more: **for portal players, recruiting rating is close
    to the wrong variable entirely.** A three-star who just threw for 3,200
    yards at a Sun Belt school is worth vastly more than his 0.84 composite, and
    a former five-star transferring as a fourth-year backup is worth far less
    than his 0.98. Recruiting rating measures what a high-school evaluator
    thought; college production measures what happened. Where a transfer has
    college snaps, production should dominate.

    `prior_production` is an optional frame with columns [player, ppa, snaps]
    from the previous season. Where a transfer is matched, production drives the
    valuation and the star rating becomes a tiebreaker. Where they aren't
    matched — true freshmen, JUCOs, walk-ons — it falls back to stars.
    """
    if portal_df.empty:
        return pd.Series(dtype=float, name="portal")

    df = portal_df.copy()
    df["rating"] = pd.to_numeric(df.get("rating"), errors="coerce").fillna(0.84)
    df["pos"] = df.get("position", pd.Series("LB", index=df.index)).fillna("LB").str.upper()
    df["pos_mult"] = df["pos"].map(POSITION_VALUE).fillna(1.0)

    # base value from stars (fallback path)
    star_val = np.where(
        df["rating"] > 0.95, 4.0,
        np.where(df["rating"] > 0.90, 2.5,
                 np.where(df["rating"] > 0.85, 1.2, 0.4)),
    )

    if prior_production is not None and not prior_production.empty:
        pp = prior_production.copy()
        pp["ppa"] = pd.to_numeric(pp["ppa"], errors="coerce")
        pp["snaps"] = pd.to_numeric(pp.get("snaps", 0), errors="coerce").fillna(0)
        pp = pp.groupby("player", as_index=False).agg({"ppa": "mean", "snaps": "sum"})
        df = df.merge(pp, left_on=df.get("firstName", "").astype(str) + " "
                      + df.get("lastName", "").astype(str),
                      right_on="player", how="left")
        # production value, z-scored within the transfer pool, credibility-
        # weighted by snaps so a 40-snap sample doesn't outrank a full season
        prod_z = _zv(df["ppa"].fillna(np.nanmean(df["ppa"])))
        cred = (df["snaps"].fillna(0) / (df["snaps"].fillna(0) + 200.0)).to_numpy()
        prod_val = 2.0 + 2.2 * np.clip(prod_z, -2.0, 2.5)
        val = cred * prod_val + (1 - cred) * star_val
    else:
        val = star_val

    df["val"] = val * df["pos_mult"]

    inc = df.groupby("destination")["val"].sum()
    out = df.groupby("origin")["val"].sum()
    net = inc.subtract(out, fill_value=0.0)
    return _z(net.rename("portal"))


def build_preseason_ratings(
    prior_season_ratings: pd.Series,
    recruiting: pd.Series,
    returning: pd.Series,
    portal: pd.Series,
    weights: dict | None = None,
    scale_points: float = 11.0,
) -> pd.DataFrame:
    """
    Combine into a points-scale preseason rating. scale_points is the SD of the
    resulting distribution in points; ~11 matches the observed spread of FBS
    team strength (best team ~ +30, worst ~ -30 against average).
    """
    w = weights or DEFAULT_WEIGHTS
    parts = {
        "prior_rating": _z(prior_season_ratings),
        "recruiting": recruiting,
        "returning": returning,
        "portal": portal,
    }
    idx = sorted(set().union(*[set(p.index) for p in parts.values() if len(p)]))
    z = pd.DataFrame({k: v.reindex(idx) for k, v in parts.items()})
    z = z.fillna(0.0)
    combined = sum(w[k] * z[k] for k in w)
    out = pd.DataFrame({"rating": _z(combined) * scale_points}, index=idx)
    out["components"] = list(z.round(3).to_dict("records"))
    return out


def fit_prior_weights(history: pd.DataFrame) -> dict:
    """
    Re-derive the blend weights from data.

    history: one row per (team, season) with columns prior_rating, recruiting,
    returning, portal, and `target` = that season's realised adjusted rating.
    """
    feats = ["prior_rating", "recruiting", "returning", "portal"]
    d = history.dropna(subset=feats + ["target"])
    m = RidgeCV(alphas=np.logspace(-2, 3, 30))
    m.fit(d[feats], d["target"])
    raw = np.maximum(m.coef_, 0)
    if raw.sum() == 0:
        return DEFAULT_WEIGHTS
    return dict(zip(feats, (raw / raw.sum()).round(3)))


def _z(s: pd.Series) -> pd.Series:
    s = s.astype(float)
    sd = s.std(ddof=0)
    return (s - s.mean()) / sd if sd > 0 else s * 0.0


def _zv(v) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    sd = np.nanstd(v)
    return (v - np.nanmean(v)) / sd if sd > 0 else v * 0.0
