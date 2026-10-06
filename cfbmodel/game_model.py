"""
Game projection: expected margin, expected total, and — the part that actually
matters for pricing — the full *discrete* distribution over margins.

Why discrete: a spread bet at -3 is not a bet on E[margin], it is a bet on
P(margin > 3). Because CFB margins pile up on 3 and 7, a continuous normal
misprices every number near a key number by 1-3 points of implied probability.
That is bigger than most edges you will ever find, so getting the pmf right is
not a refinement — it is the bet.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from .config import ALTITUDE_TEAMS, C, SOFT_MARKET_TAGS


# --------------------------------------------------------------- point spread


def project_game(
    home: str,
    away: str,
    ratings: pd.DataFrame,
    total_ratings: pd.DataFrame | None = None,
    neutral: bool = False,
    rest_days_home: int | None = None,
    rest_days_away: int | None = None,
    travel_miles: float = 0.0,
    wind_mph: float = 0.0,
    temp_f: float | None = None,
    precip: bool = False,
    home_qb_out: bool = False,
    away_qb_out: bool = False,
    venue_hfa: float | None = None,
    dome: bool = False,
) -> dict:
    """Return projected home margin, total, and the adjustments that produced them.

    `dome=True` zeroes every weather term (wind, precipitation, cold); pitfall 10."""
    if home not in ratings.index or away not in ratings.index:
        raise KeyError(f"missing rating for {home!r} or {away!r}")

    r_home = float(ratings.loc[home, "rating"])
    r_away = float(ratings.loc[away, "rating"])
    base = r_home - r_away

    adj = {}

    # home field
    if neutral:
        adj["hfa"] = C.hfa_neutral
    else:
        adj["hfa"] = venue_hfa if venue_hfa is not None else C.hfa_points

    # rest. CFB rest edges are real but small and mostly matter at the extremes
    # (a bye vs a Thursday-to-Saturday turnaround), not 6 days vs 7.
    if rest_days_home is not None and rest_days_away is not None:
        d = np.clip(rest_days_home - rest_days_away, -9, 9)
        adj["rest"] = 0.09 * d

    # travel + altitude. Altitude is a fourth-quarter effect, so it hits the
    # *total* and the tail of the margin more than the mean.
    if travel_miles > 1200:
        adj["travel"] = -0.35 * min((travel_miles - 1200) / 1000, 2.0)
    alt = ALTITUDE_TEAMS.get(home, 0)
    if alt >= 4000 and not neutral:
        adj["altitude"] = 0.8 if alt >= 6000 else 0.45

    # QB availability. The single biggest CFB-specific line mover: no injury
    # report means this information is often *not* in the number yet.
    if home_qb_out:
        adj["qb_home"] = -qb_dropoff(home, ratings)
    if away_qb_out:
        adj["qb_away"] = qb_dropoff(away, ratings)

    margin = base + sum(adj.values())

    # ---- total
    if total_ratings is not None and home in total_ratings.index and away in total_ratings.index:
        total = float(
            total_ratings.loc[home, "total_rating"] + total_ratings.loc[away, "total_rating"]
        )
    else:
        pace = float(ratings.get("pace", pd.Series(C.pace_mean, index=ratings.index)).loc[home])
        pace2 = float(ratings.get("pace", pd.Series(C.pace_mean, index=ratings.index)).loc[away])
        total = 52.0 + 0.25 * ((pace + pace2) - 2 * C.pace_mean)

    tot_adj = {}
    if not dome:
        if wind_mph > 12:
            tot_adj["wind"] = -0.42 * (wind_mph - 12)   # steep and real
        if precip:
            tot_adj["precip"] = -1.1
        if temp_f is not None and temp_f < 32:
            tot_adj["cold"] = -0.05 * (32 - temp_f)
    if alt >= 4000:
        tot_adj["altitude"] = 1.4
    if home_qb_out or away_qb_out:
        tot_adj["qb"] = -1.5
    total = total + sum(tot_adj.values())

    return {
        "home": home,
        "away": away,
        "margin": float(margin),
        "total": float(total),
        "spread_fair": float(-margin),   # in market convention (home team)
        "margin_adjustments": adj,
        "total_adjustments": tot_adj,
    }


def qb_dropoff(team: str, ratings: pd.DataFrame) -> float:
    """
    Points lost when the starting QB is out.

    CFB dropoff is far larger and far more variable than the NFL because the
    backup is often a true freshman rather than a journeyman pro. Scaling it to
    team strength is a crude proxy for "how good was the guy who left" — replace
    it with a real QB rating the moment you have one.
    """
    r = float(ratings.loc[team, "rating"])
    return float(np.clip(3.2 + 0.18 * max(r, 0), 3.0, 11.0))


# -------------------------------------------------- discrete margin distribution


def margin_pmf(
    exp_margin: float, exp_total: float, lo: int = -70, hi: int = 70,
    source: str = "model",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Key-number-aware pmf over integer home margins.

    Student-t base (CFB margins have real tail weight — 40-point results are not
    a rounding error), scale grown with expected total, then multiplicative
    bumps on key numbers, renormalised. Margin 0 is removed: ties do not exist.

    `source` decides how wide the distribution is, and it is not cosmetic:
      "market" — exp_margin came from a betting line (MAE ~10.5)
      "model"  — exp_margin came from your own ratings (MAE ~13.5), so the
                 distribution must be ~30% wider

    Defaulting to "model" is deliberate. Using the market's residual around your
    own point estimate silently assumes your projection is as good as the
    closing line, which inflates every cover probability you compute against it.
    """
    xs = np.arange(lo, hi + 1)
    base = C.margin_scale_market if source == "market" else C.margin_scale_model
    sd = base + C.margin_sd_total_coef * (exp_total - 52.0)
    sd = float(np.clip(sd, 8.0, 20.0))

    z_hi = (xs + 0.5 - exp_margin) / sd
    z_lo = (xs - 0.5 - exp_margin) / sd
    pmf = stats.t.cdf(z_hi, C.margin_df) - stats.t.cdf(z_lo, C.margin_df)

    bump = np.ones_like(pmf)
    for k, wgt in C.key_numbers.items():
        bump[np.abs(xs) == k] *= wgt
    pmf = pmf * bump

    pmf[xs == 0] = 0.0
    pmf = pmf / pmf.sum()
    return xs, pmf


def cover_prob(exp_margin: float, exp_total: float, spread_home: float,
               source: str = "model") -> dict:
    """
    P(home covers), P(away covers), P(push) for a home spread in market
    convention (home -7.5 -> spread_home = -7.5).
    """
    xs, pmf = margin_pmf(exp_margin, exp_total, source=source)
    need = -spread_home
    push = float(pmf[np.isclose(xs, need)].sum()) if float(need).is_integer() else 0.0
    p_home = float(pmf[xs > need].sum())
    p_away = float(pmf[xs < need].sum())
    return {"home": p_home, "away": p_away, "push": push}


def moneyline_prob(exp_margin: float, exp_total: float) -> dict:
    xs, pmf = margin_pmf(exp_margin, exp_total)
    return {"home": float(pmf[xs > 0].sum()), "away": float(pmf[xs < 0].sum())}


def total_probs(exp_total: float, line: float, exp_margin: float = 0.0) -> dict:
    """
    P(over), P(under), P(push) on the game total.

    Totals sd widens in expected blowouts — a 24-point favourite means garbage
    time, and garbage time is a coin flip between kneel-downs and a backup
    throwing it 40 times.
    """
    sd = C.total_sd_base + 0.06 * abs(exp_margin)
    pts = np.arange(0, 140)
    z_hi = (pts + 0.5 - exp_total) / sd
    z_lo = (pts - 0.5 - exp_total) / sd
    pmf = stats.norm.cdf(z_hi) - stats.norm.cdf(z_lo)
    pmf /= pmf.sum()
    push = float(pmf[np.isclose(pts, line)].sum()) if float(line).is_integer() else 0.0
    return {
        "over": float(pmf[pts > line].sum()),
        "under": float(pmf[pts < line].sum()),
        "push": push,
    }


# -------------------------------------------------------------- special cases


def fcs_adjustment(fbs_rating: float) -> float:
    """
    FBS vs FCS. The market prices these off a rough talent-tier heuristic and
    they are usually untradeable (huge numbers, low limits, backups in the
    second half). Included so the pipeline does not crash, not because you
    should bet them.
    """
    return fbs_rating + 24.0


def market_softness(row: pd.Series) -> float:
    """
    Multiplier on bet size reflecting where the price actually comes from.
    A 2-point disagreement on a Tuesday MAC total is worth more than a 2-point
    disagreement on Ohio State-Michigan, because in the second case the number
    already contains everything you know and a lot you do not.
    """
    if row.get("week", 99) <= 1:
        return SOFT_MARKET_TAGS["week0_week1"]
    p4 = {"SEC", "Big Ten", "Big 12", "ACC"}
    hc, ac = row.get("home_conf"), row.get("away_conf")
    if hc in p4 and ac in p4:
        return SOFT_MARKET_TAGS["p4_marquee"]
    if hc not in p4 and ac not in p4:
        day = row.get("day_of_week", "Sat")
        if day not in ("Sat", "Saturday"):
            return SOFT_MARKET_TAGS["weeknight_midmajor"]
        return SOFT_MARKET_TAGS["g5_vs_g5"]
    return 0.5
