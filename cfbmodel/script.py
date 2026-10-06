"""
Joint game-script simulation.

This replaces the static `pull_hazard` haircut with the thing it was
approximating. Gemini's instinct here was right — the volume shock shouldn't be
a constant — but the proposed mechanism (scale the SD by opposing defense
variance and pace) is the wrong lever, for two reasons:

  1. Opposing defense variance drives *efficiency* outcomes, not *volume*
     uncertainty. Folding it into the volume shock collapses the two channels
     the model exists to keep apart.
  2. Pace is already an input to `team_play_estimate`. Adding it to the shock
     term double-counts it — and worse, it inflates variance when what pace
     actually does is shift the mean.

What genuinely drives CFB volume uncertainty is game script and role stability.
So: simulate the margin, and let volume fall out of it.

The payoff is a correctly-shaped distribution rather than a wider one. A
starting RB for a 24-point favourite has a bimodal carry distribution — ~20 in
a game that stays close, ~10 in the blowout the market expects. A static
haircut on the mean gives you 16 carries and a symmetric spread, which is a
number that never happens. It systematically misprices both tails, and the
overs it produces against big favourites are where naive CFB prop models die.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import C
from .game_model import margin_pmf

RNG = np.random.default_rng(20260829)


# --------------------------------------------------------- role stability


def role_instability(
    share_history: np.ndarray, teammate_shares: np.ndarray, own_share: float | None = None
) -> float:
    """
    Returns the volume-shock sigma to use for this player, derived from
    evidence rather than a constant.

    Three inputs, all free from box scores:
      - how many games of role we have observed
      - how volatile the share has been across those games
      - how concentrated the position group is (HHI): a bell-cow backfield is
        predictable, a three-man committee is a coin flip every Saturday

    A true freshman with two games in a committee gets sigma ~0.45. An
    established bell cow with nine games gets ~0.14. The static 0.21 the model
    shipped with was the average of two situations that should never share a
    number.
    """
    n = len(share_history)
    if n == 0:
        return 0.50

    base = 0.42 / np.sqrt(n)                      # sample-size term
    vol = float(np.std(share_history)) / max(float(np.mean(share_history)), 1e-3)
    vol_term = 0.55 * np.clip(vol, 0.0, 1.2)      # observed share volatility

    # Role security. Raw HHI is the wrong scale here: a genuine bell cow with
    # 61% of carries in a three-man room has HHI ~0.45, nowhere near 1.0, so an
    # HHI-based term never reaches zero and every player ends up noisier than
    # the constant it replaced. The player's OWN share is the direct measure of
    # whether the role is secure. 0.65 is treated as fully established.
    own = float(np.mean(share_history)) if own_share is None else float(own_share)
    committee_term = 0.28 * (1.0 - np.clip(own / 0.65, 0.0, 1.0))

    return float(np.clip(base + vol_term + committee_term, 0.10, 0.60))


def derive_depth_chart(box: pd.DataFrame, team: str, asof_season: int, asof_week: int,
                       lookback: int = 4) -> pd.DataFrame:
    """
    An empirical depth chart from who actually touched the ball, rather than a
    published one.

    This is deliberately here instead of an Ourlads scraper. Published CFB depth
    charts are famously uninformative — "OR" designations everywhere, and staffs
    that misdirect on purpose during game weeks. What a player did over the last
    four games is both more truthful and already in your pipeline.
    """
    d = box[(box["team"] == team)].copy()
    d = d[(d["season"] < asof_season) | (d["week"] < asof_week)]
    d = d[d["week"] >= asof_week - lookback]
    d["value"] = pd.to_numeric(d["value"], errors="coerce")

    out = []
    for cat, typ, label in (("rushing", "CAR", "rush"), ("receiving", "REC", "rec")):
        s = d[(d["category"] == cat) & (d["stat_type"] == typ)]
        if s.empty:
            continue
        by_game = s.pivot_table(index="player", columns="week", values="value",
                                aggfunc="sum").fillna(0.0)
        tot = by_game.sum(axis=1)
        shares = by_game.div(by_game.sum(axis=0).replace(0, np.nan), axis=1).fillna(0.0)
        group_share = (tot / tot.sum()).sort_values(ascending=False)
        for rank, (player, sh) in enumerate(group_share.items(), start=1):
            out.append({
                "team": team, "group": label, "rank": rank, "player": player,
                "share": round(float(sh), 4),
                "games": int((by_game.loc[player] > 0).sum()),
                "share_sd": round(float(shares.loc[player].std()), 4),
                "hhi": round(float((group_share ** 2).sum()), 4),
            })
    return pd.DataFrame(out)


# ------------------------------------------------------- joint simulation


def simulate_game_script(
    exp_margin_for_team: float,
    exp_total: float,
    base_pace: float,
    base_pass_rate: float = 0.45,
    n_sims: int = 40_000,
) -> dict:
    """
    Draw n_sims game realisations and return per-sim team plays, pass rate, and
    a benching multiplier for a starter.

    Everything downstream is conditioned on these draws, so a sim where the team
    wins by 40 gets fewer pass attempts AND an early benching AND a run-heavy
    script — all correlated, as they are in reality. The old code treated each
    of those as an independent average.
    """
    xs, pmf = margin_pmf(exp_margin_for_team, exp_total)
    margins = RNG.choice(xs, size=n_sims, p=pmf)

    # Total plays fall in blowouts (running clock, kneel-downs) and rise
    # modestly in shootouts. Effect is real but small; the big effect is on
    # *who* runs them.
    play_mult = 1.0 + 0.0022 * (exp_total - 52.0) - 0.0035 * np.abs(margins)
    plays = base_pace * np.clip(play_mult, 0.80, 1.15)
    plays = plays * RNG.lognormal(-0.5 * 0.07**2, 0.07, n_sims)

    # Pass rate over expectation: trailing teams throw, leading teams run.
    #
    # Critical detail: PROE must be driven by the AVERAGE in-game lead, not the
    # final margin. A team that wins by 24 was not up 24 for four quarters — the
    # lead builds. Empirically the time-averaged differential is roughly 45% of
    # the final margin. Using the final margin here inflates the run-heavy
    # script so badly that a 24-point favourite's RB projects ABOVE his neutral-
    # script baseline, which is the opposite of what happens.
    effective_lead = 0.45 * margins
    pass_rate = np.clip(base_pass_rate - 0.0095 * effective_lead, 0.20, 0.80)

    # Benching. A hazard on the realised margin, not on its expectation.
    # Below ~17 nobody sits; past ~35 the starters are in headsets.
    lead = np.abs(margins)
    p_pull = np.clip((lead - 14.0) / 22.0, 0.0, 0.95)
    pulled = RNG.random(n_sims) < p_pull
    # If pulled, you lose a random share of what remains, not a fixed one. The
    # beta is right-shifted because benchings that happen at all tend to happen
    # by early in the fourth, not with two minutes left.
    lost = np.where(pulled, RNG.beta(2.6, 1.9, n_sims) * 0.62, 0.0)
    volume_mult = 1.0 - lost

    return {
        "margin": margins,
        "plays": plays,
        "pass_rate": pass_rate,
        "volume_mult": volume_mult,
        "p_pulled": float(pulled.mean()),
    }


def simulate_rush_yards_joint(
    carry_share: float,
    script: dict,
    ypc: float,
    ypc_sd: float,
    shock_sigma: float,
    p_play: float = 1.0,
    share_pull_sensitivity: float = 1.0,
) -> np.ndarray:
    """
    Rushing yards, conditioned on the simulated script.

    share_pull_sensitivity: 1.0 for a starter who sits in blowouts, 0.0 for a
    backup whose volume *rises* in them. Set it negative for the change-of-pace
    back — that is a real and mispriced prop, because the same blowout that
    kills the starter's over makes the backup's over live.
    """
    n = len(script["margin"])
    rush_rate = 1.0 - script["pass_rate"]
    mult = 1.0 - share_pull_sensitivity * (1.0 - script["volume_mult"])

    shock = RNG.lognormal(-0.5 * shock_sigma**2, shock_sigma, n)
    lam = np.clip(script["plays"] * rush_rate * carry_share * mult * shock, 0.05, None)
    carries = RNG.poisson(lam)
    carries = np.where(RNG.random(n) < p_play, carries, 0)

    return _carry_yards(carries, ypc, ypc_sd)


def simulate_rec_yards_joint(
    target_share: float,
    script: dict,
    ypt: float,
    catch_rate: float,
    shock_sigma: float,
    p_play: float = 1.0,
    share_pull_sensitivity: float = 0.7,
) -> dict:
    n = len(script["margin"])
    mult = 1.0 - share_pull_sensitivity * (1.0 - script["volume_mult"])
    shock = RNG.lognormal(-0.5 * shock_sigma**2, shock_sigma, n)
    lam = np.clip(script["plays"] * script["pass_rate"] * target_share * mult * shock, 0.05, None)
    tgts = RNG.poisson(lam)
    tgts = np.where(RNG.random(n) < p_play, tgts, 0)
    recs = RNG.binomial(tgts, catch_rate)

    ypr = ypt / max(catch_rate, 1e-3)
    sigma = np.sqrt(np.log1p(0.95**2))
    mu = np.log(ypr) - 0.5 * sigma**2
    maxr = int(recs.max()) if recs.size else 0
    if maxr == 0:
        return {"targets": tgts, "receptions": recs, "yards": np.zeros(n)}
    draws = RNG.lognormal(mu, sigma, size=(n, maxr))
    mask = np.arange(maxr)[None, :] < recs[:, None]
    return {"targets": tgts, "receptions": recs, "yards": (draws * mask).sum(axis=1)}


def _carry_yards(carries: np.ndarray, ypc: float, ypc_sd: float) -> np.ndarray:
    n = len(carries)
    p_break, bshape, bscale, shift = 0.021, 3.0, 12.0, 1.2
    base_mean = max(ypc + shift - p_break * bshape * bscale, 0.4)
    shape = max((base_mean / max(ypc_sd, 1e-3)) ** 2, 0.05)
    scale = max(base_mean / shape, 1e-3)
    maxc = int(carries.max()) if carries.size else 0
    if maxc == 0:
        return np.zeros(n)
    draws = RNG.gamma(shape, scale, size=(n, maxc)) - shift
    brk = RNG.random((n, maxc)) < p_break
    draws = np.where(brk, draws + RNG.gamma(bshape, bscale, size=(n, maxc)), draws)
    mask = np.arange(maxc)[None, :] < carries[:, None]
    return (draws * mask).sum(axis=1)
