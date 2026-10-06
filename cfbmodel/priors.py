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

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV

from . import config, ratings as _ratings
from .params import P


PRIOR_FEATURES = ["prior_rating", "recruiting", "returning", "portal"]

# DEFAULT_WEIGHTS (the unfitted starting blend) and POSITION_VALUE (portal position multipliers)
# live in the registry as priors.default_weights and priors.position_value; the names below are
# read through __getattr__ at the bottom of this file so older imports keep working. Real weights
# come from `python -m cfbmodel fit-priors`, which writes output/prior_weights.json;
# build_preseason_ratings uses that file when it exists and records in
# `out.attrs["weights_source"]` which of the two it used.


def rolling_recruiting(recruit_by_year: dict[int, pd.DataFrame], season: int, n_years: int = 4) -> pd.Series:
    """
    4-year rolling 247 composite team score, weighted toward the classes most
    likely to be on the field (years 2-4 of a class matter more than the
    true freshmen who just signed).
    """
    class_weights = {int(k): v for k, v in P.priors.class_weights.items()}
    acc, wsum = {}, {}
    for back in range(n_years):
        yr = season - back
        df = recruit_by_year.get(yr)
        if df is None or df.empty:
            continue
        w = class_weights.get(back, P.priors.class_weight_default)
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
    Q = P.priors
    s = pd.Series((Q.returning_total_weight * _zv(tot) + Q.returning_pass_weight * _zv(pass_p)),
                  index=df[cols.get("team", "team")])
    return s.rename("returning")


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
    Q = P.priors
    df["rating"] = pd.to_numeric(df.get("rating"), errors="coerce").fillna(Q.portal_default_rating)
    df["pos"] = df.get("position", pd.Series("LB", index=df.index)).fillna("LB").str.upper()
    df["pos_mult"] = df["pos"].map(Q.position_value).fillna(Q.position_default_mult)

    # base value from stars (fallback path)
    star_val = np.where(
        df["rating"] > Q.star_elite_rating, Q.star_elite_value,
        np.where(df["rating"] > Q.star_great_rating, Q.star_great_value,
                 np.where(df["rating"] > Q.star_good_rating, Q.star_good_value, Q.star_base_value)),
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
        cred = (df["snaps"].fillna(0) / (df["snaps"].fillna(0) + Q.production_credibility_snaps)).to_numpy()
        prod_val = Q.production_base + Q.production_slope * np.clip(prod_z, Q.production_z_lo, Q.production_z_hi)
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
    scale_points: float | None = None,
) -> pd.DataFrame:
    """
    Combine into a points-scale preseason rating. scale_points is the SD of the
    resulting distribution in points; ~11 matches the observed spread of FBS
    team strength (best team ~ +30, worst ~ -30 against average).
    """
    scale_points = P.priors.scale_points if scale_points is None else scale_points
    if weights is None:
        weights, weights_source = load_prior_weights()
    else:
        weights_source = "explicit"
    w = weights
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
    out.attrs["weights_source"] = weights_source
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
        return dict(P.priors.default_weights)
    return dict(zip(feats, (raw / raw.sum()).round(3)))


def prior_weights_path() -> Path:
    return config.OUTPUT / "prior_weights.json"


def load_prior_weights(path: Path | None = None) -> tuple[dict, str]:
    """(weights, source): the fitted weights if output/prior_weights.json exists, else the
    unfitted defaults with source "default_unfitted"."""
    p = Path(path) if path else prior_weights_path()
    if p.exists():
        return json.loads(p.read_text())["weights"], "fitted"
    return dict(P.priors.default_weights), "default_unfitted"


def save_prior_weights(weights: dict, cv: dict, seasons: list[int], path: Path | None = None,
                       generated: str | None = None) -> Path:
    p = Path(path) if path else prior_weights_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({
        "generated": generated or datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "features": PRIOR_FEATURES, "weights": weights, "cv": cv, "seasons": seasons,
    }, indent=2, sort_keys=True))
    return p


def build_prior_history(
    lines: pd.DataFrame,
    recruiting_by_year: dict[int, pd.DataFrame],
    returning_by_year: dict[int, pd.DataFrame],
    portal_by_year: dict[int, pd.DataFrame],
    seasons: list[int],
) -> pd.DataFrame:
    """
    One row per (team, season) for fitting the preseason weights: the four z-scored inputs as
    they stood BEFORE that season, and `target` = the team's market-implied rating from that
    season's own games.

    Leak safety: inputs use only data from earlier seasons (the prior rating is fit as of
    (season, week 0)); only `target` looks at the season itself, which is the point of a target.
    A season with no earlier lines has no prior rating and is skipped. A component that is not
    available for a season (e.g. no portal data) is neutral (0), as in build_preseason_ratings.
    """
    frames = []
    for s in seasons:
        if not (lines["season"] < s).any() or not (lines["season"] == s).any():
            continue
        prior = _ratings.fit_market_ratings(lines, s, 0)["market_rating"]
        target = _ratings.fit_market_ratings(
            lines[lines["season"] == s], s + 1, 0, half_life_weeks=1e9)["market_rating"]
        rp = returning_by_year.get(s)
        pt = portal_by_year.get(s)
        feats = pd.DataFrame({
            "prior_rating": _z(prior),
            "recruiting": rolling_recruiting(recruiting_by_year, s),
            "returning": returning_production_score(rp) if rp is not None and len(rp) else np.nan,
            "portal": portal_score(pt, s) if pt is not None and len(pt) else np.nan,
        }).reindex(target.index)
        feats[["recruiting", "returning", "portal"]] = feats[["recruiting", "returning", "portal"]].fillna(0.0)
        feats["target"] = target
        feats["season"] = s
        frames.append(feats.dropna(subset=["prior_rating", "target"]).rename_axis("team").reset_index())
    if not frames:
        raise ValueError("need at least two seasons of lines to build a prior history")
    return pd.concat(frames, ignore_index=True)


def fit_prior_weights_cv(history: pd.DataFrame) -> tuple[dict, dict]:
    """
    Fit the blend weights on all seasons, and score them by leave-one-season-out: for each
    season, weights are fit on the others and judged on the one left out (never on data they
    saw). Returns (weights, cv) where cv reports pooled out-of-sample numbers next to the
    "last season's rating only" baseline, so you can see whether recruiting/returning/portal
    earn their place.
    """
    d = history.dropna(subset=PRIOR_FEATURES + ["target"]).reset_index(drop=True)
    seasons = sorted(d["season"].unique())
    if len(seasons) < 2:
        raise ValueError("need at least two seasons to cross-validate the prior weights")
    weights = fit_prior_weights(d)
    ys, ridge_pred, comp_pred, base_pred = [], [], [], []
    for s in seasons:
        tr, te = d[d["season"] != s], d[d["season"] == s]
        m = RidgeCV(alphas=np.logspace(-2, 3, 30)).fit(tr[PRIOR_FEATURES], tr["target"])
        w = fit_prior_weights(tr)
        ys.append(te["target"].to_numpy())
        ridge_pred.append(m.predict(te[PRIOR_FEATURES]))
        comp_pred.append(sum(w[k] * te[k].to_numpy() for k in PRIOR_FEATURES))
        base_pred.append(te["prior_rating"].to_numpy())
    y, rp, cp, bp = map(np.concatenate, (ys, ridge_pred, comp_pred, base_pred))
    sst = float(((y - y.mean()) ** 2).sum())
    cv = {
        "scheme": "leave-one-season-out", "n": int(len(y)), "seasons": [int(x) for x in seasons],
        "rmse": float(np.sqrt(((y - rp) ** 2).mean())),
        "r2": float(1 - ((y - rp) ** 2).sum() / sst),
        "r_weighted_composite": float(np.corrcoef(y, cp)[0, 1]),
        "r_prior_rating_only": float(np.corrcoef(y, bp)[0, 1]),
    }
    return weights, cv


def _z(s: pd.Series) -> pd.Series:
    s = s.astype(float)
    sd = s.std(ddof=0)
    return (s - s.mean()) / sd if sd > 0 else s * 0.0


def _zv(v) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    sd = np.nanstd(v)
    return (v - np.nanmean(v)) / sd if sd > 0 else v * 0.0


_REGISTRY_NAMES = {"DEFAULT_WEIGHTS": "default_weights", "POSITION_VALUE": "position_value"}


def __getattr__(name):          # PEP 562: priors.DEFAULT_WEIGHTS / priors.POSITION_VALUE
    if name in _REGISTRY_NAMES:
        return getattr(P.priors, _REGISTRY_NAMES[name])
    raise AttributeError(name)
