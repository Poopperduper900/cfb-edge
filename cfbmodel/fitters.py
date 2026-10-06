"""
Parameter fitters (BUILD_PLAN Phase 6b): one per parameter group, each with a proper objective.

They learn from FORECAST ACCURACY ON EVERY GAME (thousands of observations), scored with proper
scoring rules (log-likelihood, CRPS), and never from the owner's bet results (a few dozen a
season, and biased by which bets were taken).

Every fitter takes a DataFrame and returns `FitResult(changes, info)` where `changes` is
{registry group: {name: new value}} and `info` says what was fitted and how well. A fitter only
PROPOSES numbers. Whether a proposal is adopted is decided in learn.py, by the champion/challenger
rules, on games the fitter never saw.

  group                              objective                                 data
  margin pmf (scale, df, slope, keys) log-likelihood of margins | closing line  all games
  HFA, model-margin scale             log-likelihood of margins | model margin   walk-forward
  early-season decay                  log-likelihood, weeks 0-5                  walk-forward
  totals sd, model-total widening     log-likelihood of totals | closing total   all games
  weather coefficients                non-negative least squares on total        forecast history
                                      residuals (dome games excluded)
  PROE slope                          CRPS of simulated rushing yards            player-games
  pace scaling                        least squares on totals                    games + plays
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import optimize, stats
from scipy.special import ndtr, stdtr

from . import params, script, state
from .params import P

XS = np.arange(-70, 71)          # margins the pmf covers (same grid as game_model.margin_pmf)


@dataclass
class FitResult:
    changes: dict                                  # {group: {name: value}}
    info: dict = field(default_factory=dict)


# ------------------------------------------------------------------ margin pmf


def margin_logprob(margin, exp_margin, exp_total, *, scale: float, df: float, sd_total_coef: float,
                   key_weights: dict[int, float], sd_total_ref: float | None = None,
                   sd_floor: float | None = None, sd_cap: float | None = None) -> np.ndarray:
    """log P(observed margin) under the key-number-aware Student-t pmf, for many games at once.

    Same numbers as game_model.margin_pmf (a test checks that), computed cheaply. The pmf is
    t-mass-per-integer times a weight that is 1 everywhere except on key numbers (and 0 on a tie),
    so the normalising constant is
        Z = (total mass on the grid)  -  pmf(0)  +  sum over keys k of (w_k - 1) * (pmf(k) + pmf(-k))
    and only the cells near a key number, near zero, and the observed cell are ever evaluated
    (about 15 values per game instead of 141).
    """
    M = P.margin
    ref = M.sd_total_ref if sd_total_ref is None else sd_total_ref
    lo = M.sd_floor if sd_floor is None else sd_floor
    hi = M.sd_cap if sd_cap is None else sd_cap
    margin, mu, tot = (np.asarray(a, dtype=float) for a in (margin, exp_margin, exp_total))
    n = len(margin)
    sd = np.clip(scale + sd_total_coef * (tot - ref), lo, hi)

    def cell(x):                       # pmf of the integer margin x (x may be an array of length n)
        x = np.asarray(x, dtype=float)
        return stdtr(df, (x + 0.5 - mu) / sd) - stdtr(df, (x - 0.5 - mu) / sd)

    keys = sorted(key_weights)
    mass = stdtr(df, (XS[-1] + 0.5 - mu) / sd) - stdtr(df, (XS[0] - 0.5 - mu) / sd)
    z = mass - cell(0.0)
    for k in keys:
        z = z + (key_weights[k] - 1.0) * (cell(float(k)) + cell(-float(k)))
    w_obs = np.array([key_weights.get(int(abs(m)), 1.0) for m in margin])
    p = cell(margin) * w_obs / z
    return np.log(np.clip(p, 1e-300, None))


def _cells(mu, sd, df):
    def cell(x):
        x = np.asarray(x, dtype=float)
        return stdtr(df, (x + 0.5 - mu) / sd) - stdtr(df, (x - 0.5 - mu) / sd)
    return cell


def _sd(exp_total, scale, sd_total_coef, ref, lo, hi):
    M = P.margin
    return np.clip(scale + sd_total_coef * (np.asarray(exp_total, float) - (M.sd_total_ref if ref is None else ref)),
                   M.sd_floor if lo is None else lo, M.sd_cap if hi is None else hi)


def margin_tail_prob(x0, exp_margin, exp_total, *, scale: float, df: float, sd_total_coef: float,
                     key_weights: dict[int, float], sd_total_ref: float | None = None,
                     sd_floor: float | None = None, sd_cap: float | None = None) -> np.ndarray:
    """P(margin >= x0) under the key-number pmf, for many games at once (x0 is an integer array).
    Same cell-by-cell shortcut as margin_logprob; it is the building block for cover probabilities
    and for PIT values."""
    mu, x0 = np.asarray(exp_margin, float), np.asarray(x0, float)
    sd = _sd(exp_total, scale, sd_total_coef, sd_total_ref, sd_floor, sd_cap)
    cell = _cells(mu, sd, df)
    edge_lo = np.clip(x0 - 0.5, XS[0] - 0.5, XS[-1] + 0.5)
    top = stdtr(df, (XS[-1] + 0.5 - mu) / sd)
    z = (top - stdtr(df, (XS[0] - 0.5 - mu) / sd)) - cell(0.0)
    num = np.maximum(top - stdtr(df, (edge_lo - mu) / sd), 0.0) - np.where(x0 <= 0, cell(0.0), 0.0)
    for k, w in key_weights.items():
        for m in (float(k), -float(k)):
            c = cell(m)
            z = z + (w - 1.0) * c
            num = num + (w - 1.0) * c * (m >= x0)
    return num / z


def home_cover_prob(exp_margin, exp_total, spread_home, **kw) -> np.ndarray:
    """P(home covers the spread) = P(margin > -spread), pushes excluded. Same numbers as
    game_model.cover_prob(...)["home"] (a test checks that)."""
    x0 = np.floor(-np.asarray(spread_home, float)) + 1.0     # smallest integer margin strictly above the line
    return margin_tail_prob(x0, exp_margin, exp_total, **kw)


def margin_pit(margin, exp_margin, exp_total, *, seed: int = 12345, **kw) -> np.ndarray:
    """Randomised probability integral transform of each realised margin under its predicted pmf:
    uniform on (0, 1) if the distribution is right, U-shaped if it is too narrow, hump-shaped if
    too wide. The same seed gives the same tie-breaking draws to every candidate compared."""
    m = np.asarray(margin, float)
    at = margin_tail_prob(m, exp_margin, exp_total, **kw)            # P(X >= m)
    above = margin_tail_prob(m + 1, exp_margin, exp_total, **kw)     # P(X >= m + 1)
    u = np.random.default_rng(seed).random(len(m))
    return (1.0 - at) + u * (at - above)


def total_pit(total, exp_total, exp_margin, *, sd_base: float, sd_margin_coef: float, mult: float = 1.0,
              seed: int = 12345) -> np.ndarray:
    t, mu, mm = (np.asarray(a, float) for a in (total, exp_total, exp_margin))
    sd = (sd_base + sd_margin_coef * np.abs(mm)) * mult
    lo, hi = ndtr((t - 0.5 - mu) / sd), ndtr((t + 0.5 - mu) / sd)
    return lo + np.random.default_rng(seed).random(len(t)) * (hi - lo)


def _usable(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    d = df.dropna(subset=cols)
    return d[(d["margin"].abs() <= 70) & (d["margin"] != 0)]


def _key_weights_now() -> dict[int, float]:
    return {int(k): v for k, v in P.margin.key_numbers.items()}


def fit_margin_pmf(df: pd.DataFrame, max_iter: int = 60) -> FitResult:
    """Fit scale_market, df, sd_total_coef and the key-number weights by maximum likelihood of the
    realised margins given the CLOSING line (spread_close; total_close for the width term).
    Needs columns: margin, spread_close, total_close."""
    d = _usable(df, ["margin", "spread_close"])
    d = d.assign(total_close=d["total_close"].fillna(P.margin.sd_total_ref))
    keys = sorted(_key_weights_now())
    now = _key_weights_now()
    M = P.margin
    x0 = np.array([np.log(M.scale_market), np.log(M.df), M.sd_total_coef] + [np.log(now[k]) for k in keys])
    bounds = [(np.log(6), np.log(20)), (np.log(2.5), np.log(40)), (-0.1, 0.2)] + [(np.log(0.7), np.log(2.5))] * len(keys)
    mu, tot, y = (-d["spread_close"]).to_numpy(), d["total_close"].to_numpy(), d["margin"].to_numpy()

    def nll(x):
        kw = {k: float(np.exp(v)) for k, v in zip(keys, x[3:])}
        return -margin_logprob(y, mu, tot, scale=float(np.exp(x[0])), df=float(np.exp(x[1])),
                               sd_total_coef=float(x[2]), key_weights=kw).mean()

    before = nll(x0)
    r = optimize.minimize(nll, x0, method="L-BFGS-B", bounds=bounds, options={"maxiter": max_iter})
    x = r.x
    changes = {"margin": {"scale_market": float(np.exp(x[0])), "df": float(np.exp(x[1])),
                          "sd_total_coef": float(x[2]),
                          "key_numbers": {str(k): float(np.exp(v)) for k, v in zip(keys, x[3:])}}}
    return FitResult(changes, {"n": len(d), "nll_before": float(before), "nll_after": float(r.fun)})


def fit_model_margin(df: pd.DataFrame) -> FitResult:
    """Fit home-field advantage and the width of OUR projections' error distribution (the
    model-source scale), by log-likelihood of margins given the model's projection.
    Needs columns: margin, core_model (rating difference, no HFA), neutral, total_close."""
    d = _usable(df, ["margin", "core_model"])
    tot = d["total_close"].fillna(P.margin.sd_total_ref).to_numpy()
    home = (~d["neutral"].astype(bool)).to_numpy().astype(float)
    kw, M = _key_weights_now(), P.margin
    y, core = d["margin"].to_numpy(), d["core_model"].to_numpy()

    def nll(x):
        return -margin_logprob(y, core + x[0] * home, tot, scale=float(np.exp(x[1])), df=M.df,
                               sd_total_coef=M.sd_total_coef, key_weights=kw).mean()

    x0 = np.array([M.hfa_points, np.log(M.scale_model)])
    r = optimize.minimize(nll, x0, method="L-BFGS-B", bounds=[(0.0, 6.0), (np.log(8), np.log(25))])
    return FitResult({"margin": {"hfa_points": float(r.x[0]), "scale_model": float(np.exp(r.x[1]))}},
                     {"n": len(d), "nll_before": float(nll(x0)), "nll_after": float(r.fun)})


def fit_decay(df: pd.DataFrame) -> FitResult:
    """Fit the half-life (in weeks) of the preseason prior's weight over weeks 0-5 by the
    log-likelihood of margins. Needs: margin, week, core_unshrunk, core_prior, neutral, total_close."""
    last = P.state.early_last_week
    d = _usable(df, ["margin", "core_unshrunk", "core_prior"])
    d = d[d["week"] <= last]
    tot = d["total_close"].fillna(P.margin.sd_total_ref).to_numpy()
    home = (~d["neutral"].astype(bool)).to_numpy().astype(float) * P.margin.hfa_points
    kw, M = _key_weights_now(), P.margin
    y, week = d["margin"].to_numpy(), d["week"].to_numpy()
    um, pr = d["core_unshrunk"].to_numpy(), d["core_prior"].to_numpy()

    def nll(log_h):
        h = float(np.exp(log_h))
        pw = np.array([state.early_season_prior_weight(int(w), h) for w in week])
        pred = (1 - pw) * um + pw * pr + home
        return -margin_logprob(y, pred, tot, scale=M.scale_model, df=M.df, sd_total_coef=M.sd_total_coef,
                               key_weights=kw).mean()

    r = optimize.minimize_scalar(nll, bounds=(np.log(0.3), np.log(8.0)), method="bounded")
    return FitResult({"state": {"decay_half_life": float(np.exp(r.x))}},
                     {"n": len(d), "nll_before": float(nll(np.log(P.state.decay_half_life))), "nll_after": float(r.fun)})


# --------------------------------------------------------------------- totals


def total_logprob(total, exp_total, exp_margin, *, sd_base: float, sd_margin_coef: float,
                  mult: float = 1.0) -> np.ndarray:
    """log P(observed total) under the discretised normal used by game_model.total_probs."""
    total, mu, m = (np.asarray(a, dtype=float) for a in (total, exp_total, exp_margin))
    sd = (sd_base + sd_margin_coef * np.abs(m)) * mult
    p = ndtr((total + 0.5 - mu) / sd) - ndtr((total - 0.5 - mu) / sd)
    return np.log(np.clip(p, 1e-300, None))


def fit_totals(df: pd.DataFrame) -> FitResult:
    """Fit totals sd_base and sd_margin_coef against the CLOSING total, and (if the frame has
    model totals) the extra widening sd_model_mult. Needs: actual_total, total_close, spread_close,
    optionally model_total."""
    d = df.dropna(subset=["actual_total", "total_close", "spread_close"])
    y, mu, m = d["actual_total"].to_numpy(), d["total_close"].to_numpy(), (-d["spread_close"]).to_numpy()
    T = P.totals

    def nll(x):
        return -total_logprob(y, mu, m, sd_base=x[0], sd_margin_coef=x[1]).mean()

    x0 = np.array([T.sd_base, T.sd_margin_coef])
    r = optimize.minimize(nll, x0, method="L-BFGS-B", bounds=[(6.0, 25.0), (-0.1, 0.4)])
    changes = {"totals": {"sd_base": float(r.x[0]), "sd_margin_coef": float(r.x[1])}}
    info = {"n": len(d), "nll_before": float(nll(x0)), "nll_after": float(r.fun)}
    if "model_total" in d.columns and d["model_total"].notna().sum() >= 100:
        dm = d.dropna(subset=["model_total"])
        ym, mum, mm = dm["actual_total"].to_numpy(), dm["model_total"].to_numpy(), (-dm["spread_close"]).to_numpy()
        f = lambda lm: -total_logprob(ym, mum, mm, sd_base=r.x[0], sd_margin_coef=r.x[1], mult=float(np.exp(lm))).mean()
        rm = optimize.minimize_scalar(f, bounds=(np.log(0.8), np.log(2.5)), method="bounded")
        changes["totals"]["sd_model_mult"] = float(np.exp(rm.x))
        info["n_model_totals"] = len(dm)
    return FitResult(changes, info)


def fit_pace(df: pd.DataFrame) -> FitResult:
    """Pace scaling: how many points of total each extra play (both teams combined, above the
    league norm) is worth. Needs: actual_total, pace_sum (home pace + away pace)."""
    d = df.dropna(subset=["actual_total", "pace_sum"])
    pace_mean = float(d["pace_sum"].mean() / 2.0)
    x = d["pace_sum"].to_numpy() - 2 * pace_mean
    y = d["actual_total"].to_numpy()
    slope = float(np.cov(x, y, bias=True)[0, 1] / np.var(x)) if np.var(x) > 0 else 0.0
    return FitResult({"ratings": {"pace_mean": pace_mean}, "game": {"pace_total_slope": slope}},
                     {"n": len(d)})


# -------------------------------------------------------------------- weather


def fit_weather(df: pd.DataFrame) -> FitResult:
    """Weather coefficients from total residuals against the FORECAST that was available at bet
    time (pitfall 10: never reanalysis). Needs: actual_total, total_close, wind_mph, gust_mph,
    precip_in, temp_f, dome, wx_confidence. Dome games are excluded; coefficients are constrained
    to be non-negative (weather can only lower a total)."""
    d = df[~df["dome"].astype(bool)].dropna(subset=["actual_total", "total_close", "wind_mph", "temp_f"])
    W = P.weather
    conf = d["wx_confidence"].fillna(1.0).to_numpy()
    gust = d["gust_mph"].fillna(d["wind_mph"]).to_numpy()
    precip = d["precip_in"].fillna(0.0).to_numpy()
    X = np.column_stack([
        conf * np.maximum(d["wind_mph"].to_numpy() - W.wind_threshold_mph, 0.0),
        conf * np.maximum(gust - W.gust_threshold_mph, 0.0),
        conf * np.where(precip > W.precip_threshold_in, np.minimum(precip / W.precip_ref_in, W.precip_cap_units), 0.0),
        conf * np.maximum(W.cold_threshold_f - d["temp_f"].to_numpy(), 0.0),
    ])
    y = -(d["actual_total"] - d["total_close"]).to_numpy()
    coef, _ = optimize.nnls(X, y)
    return FitResult({"weather": dict(zip(("wind_coef", "gust_coef", "precip_coef", "cold_coef"),
                                          (float(c) for c in coef)))},
                     {"n": len(d), "dome_rows_excluded": int(len(df) - len(d))})


# ---------------------------------------------------- player props: CRPS and PIT


def crps_ensemble(sim: np.ndarray, obs: float) -> float:
    """Continuous ranked probability score of an ensemble against one outcome (lower is better):
    E|X - y| - 0.5 E|X - X'|."""
    x = np.sort(np.asarray(sim, dtype=float))
    n = len(x)
    term1 = np.abs(x - obs).mean()
    term2 = (2.0 / n ** 2) * float(((2 * np.arange(1, n + 1) - n - 1) * x).sum())
    return float(term1 - 0.5 * term2)


def pit_value(sim: np.ndarray, obs: float, u: float = 0.5) -> float:
    """Probability integral transform of the outcome under the simulation (ties split by u)."""
    return float((sim < obs).mean() + u * (sim == obs).mean())


def pit_uniformity(pits: np.ndarray) -> float:
    """Kolmogorov-Smirnov distance of the PITs from Uniform(0,1). 0 = perfectly calibrated."""
    return float(stats.kstest(np.asarray(pits), "uniform").statistic)


def score_rush_props(rows: pd.DataFrame, n_sims: int = 1500, seed: int = 11) -> dict:
    """Simulate each player-game under the ACTIVE parameters and score it. Needs per row:
    exp_margin (the player's team), exp_total, pace, carry_share, ypc, ypc_sd, shock_sigma,
    actual_yards. Common random numbers (a fixed seed per row) make candidates comparable."""
    crps, pits = [], []
    old = script.RNG
    try:
        for i, r in enumerate(rows.itertuples(index=False)):
            script.RNG = np.random.default_rng(seed + i)
            sc = script.simulate_game_script(r.exp_margin, r.exp_total, r.pace, n_sims=n_sims)
            sim = script.simulate_rush_yards_joint(r.carry_share, sc, r.ypc, r.ypc_sd, r.shock_sigma)
            crps.append(crps_ensemble(sim, r.actual_yards))
            pits.append(pit_value(sim, r.actual_yards, u=0.5))
    finally:
        script.RNG = old
    return {"crps": np.asarray(crps), "pit_ks": pit_uniformity(np.asarray(pits)), "pits": np.asarray(pits)}


def fit_proe(rows: pd.DataFrame, factors=(0.5, 0.75, 1.0, 1.25, 1.5), n_sims: int = 1000) -> FitResult:
    """PROE slope by mean CRPS of simulated rushing yards (coarse grid around the current value)."""
    now = P.script.proe_slope
    best, table = None, {}
    for f in factors:
        with params.override({"script": {"proe_slope": now * f}}):
            table[f] = float(score_rush_props(rows, n_sims=n_sims)["crps"].mean())
    best = min(table, key=table.get)
    return FitResult({"script": {"proe_slope": now * best}},
                     {"n": len(rows), "crps_by_factor": table})


# --------------------------------------------- ratings hyper-parameters (costly)


def fit_ratings_hyper(run_walk_forward, grid: dict[tuple[str, str], list], score) -> FitResult:
    """Grid search over ratings parameters (ridge alphas, recency half-lives) where every candidate
    needs a full walk-forward. `run_walk_forward()` is called under params.override for each grid
    point and must return the walk-forward frame; `score(frame)` returns a number to MAXIMISE
    (mean log-likelihood of margins). Slow by nature, so `learn` runs it only with --deep."""
    names = list(grid)
    best_val, best_choice, table = -np.inf, None, []
    import itertools
    for combo in itertools.product(*(grid[n] for n in names)):
        changes: dict = {}
        for (g, k), v in zip(names, combo):
            changes.setdefault(g, {})[k] = v
        with params.override(changes):
            val = float(score(run_walk_forward()))
        table.append((dict(zip(map(".".join, names), combo)), val))
        if val > best_val:
            best_val, best_choice = val, changes
    return FitResult(best_choice, {"candidates": table, "best_score": best_val})
