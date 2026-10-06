"""
Player props.

Structure follows the decomposition that worked for you in the NFL: volume and
efficiency are different animals and must be modelled separately.

    volume      = f(usage share, team plays, game script)   -- autocorrelates
    efficiency  = g(per-touch rate, shrunk to positional mean) -- barely does

Books set CFB prop lines largely off recent *yardage* — the product of the two.
When a back gets 140 yards on 12 carries at 11.7 ypc, the line moves as if the
efficiency will persist. It will not. That single mismatch is the most reliable
prop edge in the sport.

Three CFB-specific things the NFL version does not need:

1. No snap counts, no route participation. Public CFB data has neither, so
   route participation — your strongest NFL volume predictor — is unavailable.
   Box-score usage share is a noisier substitute and the shrinkage has to be
   heavier to compensate.

2. Blowout pull risk is enormous. A 24-point favourite's starting RB may see 9
   carries. The `pull_hazard` term prices this explicitly; ignoring it is why
   naive CFB prop models lose on overs against big favourites.

3. No injury report. A "questionable" starter in the NFL is public information;
   in CFB it is a beat-writer tweet or nothing. Every projection here takes an
   explicit `p_play` you must supply from reporting.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import C
from .game_model import margin_pmf
from .ratings import as_of

RNG = np.random.default_rng(20260829)


# ------------------------------------------------------------ usage modelling


def shrink_share(
    observed_share: float, n_team_plays: int, prior_share: float, prior_strength: float = 55.0
) -> float:
    """
    Beta-binomial posterior mean for a usage share.

    prior_strength is in units of team plays. 55 is roughly one game — i.e.
    after one game you are halfway between the depth-chart prior and what you
    saw. With CFB's box-score-only usage data, being aggressive here is a
    mistake; a 3-game sample is not a role.
    """
    a = prior_share * prior_strength + observed_share * n_team_plays
    b = (1 - prior_share) * prior_strength + (1 - observed_share) * n_team_plays
    return float(a / (a + b))


def empirical_bayes_rate(
    player_total: float, player_events: float, pos_mean: float, pos_var: float, min_events: float = 12.0
) -> float:
    """
    Shrink a per-touch efficiency rate (ypc, yards/target, ypa) toward the
    positional mean. Weight = n / (n + k) with k derived from the ratio of
    within-player noise to between-player spread.
    """
    if player_events < 1:
        return pos_mean
    obs = player_total / player_events
    k = max(pos_var, 1e-6)
    k = min_events * (1.0 + 1.0 / k)
    w = player_events / (player_events + k)
    return float(w * obs + (1 - w) * pos_mean)


POS_PRIORS = {
    # (mean per-touch, between-player variance, per-touch sd) from FBS 2015-2025
    "rush_ypc": (4.72, 0.55, 6.4),
    "rec_ypt": (7.95, 1.85, 9.1),   # yards per *target*
    "pass_ypa": (7.45, 0.95, 9.6),
    "catch_rate": (0.635, 0.010, 0.0),
}


# -------------------------------------------------------------- game script


def team_play_estimate(pace_home: float, pace_away: float, exp_total: float) -> tuple[float, float]:
    """Team plays scale with both teams' pace and with scoring environment."""
    base = 0.5 * (pace_home + pace_away)
    scale = 1.0 + 0.004 * (exp_total - 52.0)
    return base * scale, base * scale


def pass_rate_over_expectation(exp_margin_for_team: float, base_pass_rate: float,
                               lead_factor: float = 0.45) -> float:
    """
    Trailing teams throw; leading teams run. Effect is steeper in CFB than the NFL because
    leads are larger and clock-killing starts earlier.

    The slope acts on the AVERAGE lead during the game, not the final margin (pitfall 6): a team
    that wins by 24 was not up 24 for four quarters. Empirically the time-averaged lead is about
    45% of the final margin (`lead_factor`; script.simulate_game_script uses the same number).
    Using the final margin makes big favourites' RBs project above their neutral baseline.
    """
    delta = -0.0095 * lead_factor * exp_margin_for_team
    return float(np.clip(base_pass_rate + delta, 0.22, 0.78))


def pull_hazard(exp_margin_for_team: float, exp_total: float, is_starter: bool = True) -> float:
    """
    Expected fraction of a starter's normal volume lost to being benched in a
    blowout — in either direction (up 35, or down 35 and the staff empties the
    bench).

    Calibrated shape: negligible below ~17 points, then rises fast. This is the
    correction that keeps you off CFB prop overs on double-digit favourites,
    which is where the public and the model both want to be.
    """
    if not is_starter:
        return 0.0
    xs, pmf = margin_pmf(exp_margin_for_team, exp_total)
    blowout = float(pmf[np.abs(xs) >= 25].sum())
    near = float(pmf[(np.abs(xs) >= 17) & (np.abs(xs) < 25)].sum())
    return float(np.clip(0.30 * blowout + 0.10 * near, 0.0, 0.35))


# ------------------------------------------------------------- prop simulation
#
# SUPERSEDED. The three simulate_* functions below apply a static mean haircut
# for blowout risk and a constant volume shock. They are kept because they are
# cheap, dependency-free, and fine for a quick sanity check on one player.
#
# For anything you would actually bet, use script.py instead:
#     script.simulate_game_script() -> simulate_rush_yards_joint / _rec_yards_joint
# It draws the game margin first and conditions pass rate, total plays and
# benching on it, so teammates correlate and blowout spots come out bimodal
# rather than merely shifted. See REVIEW_RESPONSE.md for the numbers.


def simulate_rush_yards(
    carry_share: float,
    team_plays: float,
    team_rush_rate: float,
    ypc: float,
    ypc_sd: float,
    exp_margin_for_team: float,
    exp_total: float,
    p_play: float = 1.0,
    n_sims: int = 40_000,
    volume_shock_sd: float = 0.21,
) -> np.ndarray:
    """
    Two-stage simulation with an explicit volume shock.

    The lognormal shock on expected carries is the fix that mattered most in
    your NFL build: without it the simulated distribution is far too tight,
    because it treats projected volume as known when it is the single most
    uncertain input. In CFB — no snap data, no injury report, live committee
    backfields — that uncertainty is larger, hence a wider default sd.
    """
    rush_plays = team_plays * team_rush_rate
    mu_carries = rush_plays * carry_share
    mu_carries *= 1.0 - pull_hazard(exp_margin_for_team, exp_total)

    shock = RNG.lognormal(mean=-0.5 * volume_shock_sd**2, sigma=volume_shock_sd, size=n_sims)
    lam = np.clip(mu_carries * shock, 0.05, None)
    carries = RNG.poisson(lam)

    played = RNG.random(n_sims) < p_play
    carries = np.where(played, carries, 0)

    # Per-carry yards: shifted gamma (the shift creates the TFL left tail that a
    # plain gamma cannot produce) plus an explicit breakaway component. Both are
    # mean-compensated so the realised distribution centres on the ypc you asked
    # for — otherwise every projection silently runs ~0.5 yards/carry light.
    p_break, break_shape, break_scale = 0.021, 3.0, 12.0
    shift = 1.2
    base_mean = ypc + shift - p_break * break_shape * break_scale
    base_mean = max(base_mean, 0.4)
    shape = max((base_mean / max(ypc_sd, 1e-3)) ** 2, 0.05)
    scale = max(base_mean / shape, 1e-3)

    totals = np.zeros(n_sims)
    maxc = int(carries.max()) if carries.size else 0
    if maxc == 0:
        return totals
    draws = RNG.gamma(shape, scale, size=(n_sims, maxc)) - shift
    breakaway = RNG.random((n_sims, maxc)) < p_break
    draws = np.where(
        breakaway, draws + RNG.gamma(break_shape, break_scale, size=(n_sims, maxc)), draws
    )
    mask = np.arange(maxc)[None, :] < carries[:, None]
    totals = (draws * mask).sum(axis=1)
    return totals


def simulate_rec_yards(
    target_share: float,
    team_plays: float,
    team_pass_rate: float,
    ypt: float,
    catch_rate: float,
    exp_margin_for_team: float,
    exp_total: float,
    p_play: float = 1.0,
    n_sims: int = 40_000,
    volume_shock_sd: float = 0.26,
) -> dict:
    """Returns receptions and receiving yards distributions."""
    pass_plays = team_plays * team_pass_rate
    mu_targets = pass_plays * target_share
    mu_targets *= 1.0 - 0.7 * pull_hazard(exp_margin_for_team, exp_total)

    shock = RNG.lognormal(mean=-0.5 * volume_shock_sd**2, sigma=volume_shock_sd, size=n_sims)
    tgts = RNG.poisson(np.clip(mu_targets * shock, 0.05, None))
    tgts = np.where(RNG.random(n_sims) < p_play, tgts, 0)

    recs = RNG.binomial(tgts, catch_rate)

    ypr = ypt / max(catch_rate, 1e-3)
    sd_ypr = 0.95 * ypr
    sigma = np.sqrt(np.log1p((sd_ypr / ypr) ** 2))
    mu = np.log(ypr) - 0.5 * sigma**2

    maxr = int(recs.max()) if recs.size else 0
    if maxr == 0:
        return {"receptions": recs, "yards": np.zeros(n_sims)}
    draws = RNG.lognormal(mu, sigma, size=(n_sims, maxr))
    mask = np.arange(maxr)[None, :] < recs[:, None]
    yards = (draws * mask).sum(axis=1)
    return {"receptions": recs, "yards": yards}


def simulate_pass_yards(
    attempts: float,
    ypa: float,
    exp_margin_for_team: float,
    exp_total: float,
    p_play: float = 1.0,
    n_sims: int = 40_000,
    volume_shock_sd: float = 0.16,
) -> np.ndarray:
    mu_att = attempts * (1.0 - 0.85 * pull_hazard(exp_margin_for_team, exp_total))
    shock = RNG.lognormal(mean=-0.5 * volume_shock_sd**2, sigma=volume_shock_sd, size=n_sims)
    att = RNG.poisson(np.clip(mu_att * shock, 0.5, None))
    att = np.where(RNG.random(n_sims) < p_play, att, 0)

    sd_play = POS_PRIORS["pass_ypa"][2]
    yards = RNG.normal(ypa * att, sd_play * np.sqrt(np.maximum(att, 1)))
    return np.maximum(yards, -20.0)


# ------------------------------------------------------------------ pricing


def price_prop(sim: np.ndarray, line: float) -> dict:
    """P(over)/P(under) from a simulated distribution, with a half-point push guard."""
    over = float((sim > line).mean())
    under = float((sim < line).mean())
    push = float(np.isclose(sim, line).mean())
    if push > 0:
        over += push / 2
        under += push / 2
    return {"over": over, "under": under, "push": push, "mean": float(sim.mean()),
            "p50": float(np.median(sim))}


def build_player_priors(box: pd.DataFrame, asof_season: int, asof_week: int) -> pd.DataFrame:
    """
    Turn CFBD player box scores into per-player usage shares and efficiency
    rates, shrunk appropriately, as of a point in time.

    Leak-safe by construction: only rows strictly before (season, week) are used.
    """
    df = box.copy()
    df = as_of(df, asof_season, asof_week)
    df["value"] = pd.to_numeric(df["value"], errors="coerce")

    def pick(cat, typ):
        s = df[(df["category"] == cat) & (df["stat_type"] == typ)]
        return s.groupby(["team", "player", "athlete_id"])["value"].agg(["sum", "count"])

    car = pick("rushing", "CAR").rename(columns={"sum": "carries", "count": "g_rush"})
    ryds = pick("rushing", "YDS").rename(columns={"sum": "rush_yds"})
    rec = pick("receiving", "REC").rename(columns={"sum": "rec", "count": "g_rec"})
    recy = pick("receiving", "YDS").rename(columns={"sum": "rec_yds"})

    out = car.join(ryds["rush_yds"], how="outer").join(rec[["rec", "g_rec"]], how="outer")
    out = out.join(recy["rec_yds"], how="outer").fillna(0.0)
    out = out.reset_index()

    team_car = out.groupby("team")["carries"].transform("sum")
    out["carry_share_raw"] = out["carries"] / team_car.replace(0, np.nan)
    team_rec = out.groupby("team")["rec"].transform("sum")
    out["rec_share_raw"] = out["rec"] / team_rec.replace(0, np.nan)

    m, v, _ = POS_PRIORS["rush_ypc"]
    out["ypc"] = [
        empirical_bayes_rate(y, c, m, v) for y, c in zip(out["rush_yds"], out["carries"])
    ]
    m2, v2, _ = POS_PRIORS["rec_ypt"]
    out["ypr"] = [
        empirical_bayes_rate(y, r, m2 / 0.635, v2) for y, r in zip(out["rec_yds"], out["rec"])
    ]
    return out.fillna(0.0)
