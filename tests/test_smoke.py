"""
Synthetic end-to-end smoke test. No API key required.

Generates a fake league with known true team strengths, simulates play-by-play
and closing lines, then checks that the pipeline recovers the truth. If this
fails, the plumbing is broken and nothing downstream is trustworthy.

Each numbered check from the original script is its own test. The random draws
are made in the same order as before, so the printed numbers are unchanged.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cfbmodel import backtest, edge, game_model, props, ratings
from cfbmodel.config import C

rng = np.random.default_rng(11)
N_TEAMS, N_SEASONS, N_WEEKS = 60, 2, 13
TEAMS = [f"Team{i:02d}" for i in range(N_TEAMS)]
TRUE = pd.Series(rng.normal(0, 10, N_TEAMS), index=TEAMS)  # points vs average
HFA = 2.4


def make_schedule():
    rows, gid = [], 0
    for s in range(2024, 2024 + N_SEASONS):
        for w in range(1, N_WEEKS + 1):
            order = rng.permutation(TEAMS)
            for i in range(0, N_TEAMS, 2):
                h, a = order[i], order[i + 1]
                em = TRUE[h] - TRUE[a] + HFA
                margin = int(np.round(rng.standard_t(7) * 15.5 + em))
                margin = margin + 1 if margin == 0 else margin
                total = int(np.round(rng.normal(52, 12)))
                rows.append(dict(id=gid, season=s, week=w, homeTeam=h, awayTeam=a,
                                 margin=margin, total=total, neutralSite=False,
                                 homeConference="X", awayConference="X"))
                gid += 1
    return pd.DataFrame(rows)


def make_lines(g):
    """Market knows the truth plus noise — i.e. a near-efficient market."""
    em = g["homeTeam"].map(TRUE) - g["awayTeam"].map(TRUE) + HFA
    noise = rng.normal(0, 1.6, len(g))
    close = -np.round((em + noise) * 2) / 2
    open_ = close + np.round(rng.normal(0, 1.2, len(g)) * 2) / 2
    return pd.DataFrame(dict(gameId=g["id"], season=g["season"], week=g["week"],
                             home=g["homeTeam"], away=g["awayTeam"], book="sim",
                             spread_close=close, spread_open=open_,
                             total_close=52.0, total_open=52.0))


def make_pbp(g, plays_per_game=68):
    """Play-level EPA driven by true offense/defense strength."""
    off = TRUE / 2 / 68
    frames = []
    for _, row in g.iterrows():
        for o, d, is_home in ((row.homeTeam, row.awayTeam, 1), (row.awayTeam, row.homeTeam, 0)):
            n = plays_per_game
            mu = off[o] - (-off[d]) * 0 + off[o] * 0  # offense-driven
            mu = (TRUE[o] - TRUE[d]) / (2 * 68)
            frames.append(pd.DataFrame(dict(
                season=row.season, week=row.week, gameId=row.id,
                offense=o, defense=d, home=row.homeTeam,
                offenseConference="X", defenseConference="X",
                playType=rng.choice(["Rush", "Pass Reception", "Pass Incompletion"], n),
                epa=rng.normal(mu, 1.25, n), period=rng.integers(1, 5, n),
                offenseScore=0, defenseScore=0, homeWinProb=0.5)))
    return pd.concat(frames, ignore_index=True)


@pytest.fixture(scope="module")
def world():
    """Synthetic league, fitted ratings, and the noise used by checks 12-13."""
    g = make_schedule()
    ln = make_lines(g)
    pbp = make_pbp(g)
    # drawn here, in the original order, so checks 12 and 13 see the same numbers
    sloppy_noise = rng.normal(0, 5.0, len(g))
    coinflips = rng.choice([0.909, -1.0], 60, p=[0.53, 0.47])
    print(f"synthetic league: {len(g)} games, {len(pbp):,} plays")

    mkt = ratings.fit_market_ratings(ln, 2025, N_WEEKS, half_life_weeks=40)
    epa = ratings.fit_epa_ratings(pbp, 2025, N_WEEKS, split_pass_rush=False)
    bl = ratings.blend(epa, mkt, w_model=0.35)
    return dict(g=g, ln=ln, pbp=pbp, mkt=mkt, epa=epa, bl=bl,
                sloppy_noise=sloppy_noise, coinflips=coinflips)


def _model_vs_market_frame(w):
    g, ln, bl = w["g"], w["ln"], w["bl"]
    df = g.copy()
    df["market_margin"] = -ln.set_index("gameId").loc[df["id"], "spread_close"].values
    rmap = bl["model_rating_pts"]
    df["model_margin"] = df["homeTeam"].map(rmap) - df["awayTeam"].map(rmap) + HFA
    return df


def test_01_market_ratings_recover_truth(world):
    mkt = world["mkt"]
    corr = np.corrcoef(mkt["market_rating"].reindex(TEAMS), TRUE)[0, 1]
    print(f"[1] market ratings vs truth      r={corr:.3f}  hfa={mkt['market_hfa'].iloc[0]:.2f}")
    assert corr > 0.95


def test_02_epa_ratings_recover_truth(world):
    corr2 = np.corrcoef(world["epa"]["net_epa"].reindex(TEAMS), TRUE)[0, 1]
    print(f"[2] EPA ratings vs truth         r={corr2:.3f}")
    assert corr2 > 0.80


def test_03_blend_covers_every_team(world):
    # The original script only printed this check; the assertion is new.
    bl = world["bl"]
    print(f"[3] blended rating sd            {bl['rating'].std():.2f} pts "
          f"(truth sd {TRUE.std():.2f})")
    assert set(bl.index) == set(TEAMS) and bl["rating"].notna().all()


def test_04_margin_pmf_key_numbers_and_no_ties():
    xs, pmf = game_model.margin_pmf(-6.5, 54.0)
    print(f"[4] pmf sum={pmf.sum():.6f}  P(tie)={pmf[xs==0].sum():.6f}  "
          f"P(m=3)={pmf[xs==3][0]:.4f} vs P(m=2)={pmf[xs==2][0]:.4f}")
    assert abs(pmf.sum() - 1) < 1e-9 and pmf[xs == 0].sum() == 0
    assert pmf[xs == 3][0] > pmf[xs == 2][0]


def test_05_cover_probabilities_sum_to_one():
    cp = game_model.cover_prob(-6.5, 54.0, -6.5)
    print(f"[5] cover prob home={cp['home']:.4f} away={cp['away']:.4f} "
          f"sum={cp['home']+cp['away']+cp['push']:.6f}")
    assert abs(cp["home"] + cp["away"] + cp["push"] - 1) < 1e-9


def test_06_devig_methods():
    for m in ("multiplicative", "power", "shin"):
        p = edge.devig([-110, -110], m)
        assert abs(p.sum() - 1) < 1e-6, m
    lop = {m: edge.devig([-2500, 1100], m)[0] for m in ("multiplicative", "power", "shin")}
    print(f"[6] devig heavy fav: " + "  ".join(f"{k}={v:.4f}" for k, v in lop.items()))
    # known ordering on longshot-heavy markets: multiplicative < shin < power
    assert lop["multiplicative"] < lop["shin"] < lop["power"]


def test_07_kelly_sanity():
    k_edge = edge.kelly(0.56, -110)
    k_none = edge.kelly(0.50, -110)
    print(f"[7] kelly p=.56 -> {k_edge:.4f} of bankroll;  p=.50 -> {k_none:.4f}")
    assert k_none == 0.0 and 0 < k_edge <= 0.02


def test_08_blowout_pull_hazard():
    h_close = props.pull_hazard(0.0, 52.0)
    h_blow = props.pull_hazard(28.0, 62.0)
    print(f"[8] pull hazard  even game={h_close:.3f}   -28 favourite={h_blow:.3f}")
    assert h_blow > h_close * 1.5


def test_08b_rush_sim_returns_input_ypc():
    # mean-compensation check: with no pull risk and known volume, realised
    # yards/carry must come back to the input ypc
    chk = props.simulate_rush_yards(0.55, 68, 0.55, 5.1, 6.4, 0.0, 52.0,
                                    n_sims=40_000, volume_shock_sd=1e-6)
    exp_car = 68 * 0.55 * 0.55 * (1 - props.pull_hazard(0.0, 52.0))
    print(f"[8b] realised ypc={chk.mean()/exp_car:.2f} (input 5.10)")
    assert abs(chk.mean() / exp_car - 5.10) < 0.35


def test_09_rb_sim_is_plausible():
    sim = props.simulate_rush_yards(0.55, 68, 0.55, 5.1, 6.4, 24.0, 58.0, n_sims=20_000)
    pr = props.price_prop(sim, 79.5)
    print(f"[9] RB sim mean={pr['mean']:.1f} median={pr['p50']:.1f} "
          f"P(over 79.5)={pr['over']:.3f}")
    assert 30 < pr["mean"] < 160


def test_10_prop_evaluation_runs():
    # The original script only printed this check; the assertions are new.
    sim = props.simulate_rush_yards(0.55, 68, 0.55, 5.1, 6.4, 24.0, 58.0, n_sims=20_000)
    pr = props.price_prop(sim, 79.5)
    ev = edge.evaluate_prop(pr, 79.5, -115, -115, "Back", "rush_yds")
    print(f"[10] prop eval  side={ev['side']} ev={ev['ev']:+.4f} "
          f"hold={ev['hold']:.3f} stake={ev['stake_pct']:.4f}")
    assert ev["side"] in ("over", "under")
    assert 0.0 <= ev["stake_pct"] <= C.max_bet_pct


def test_11_market_efficiency_test_runs_on_efficient_market(world):
    # The original script only printed this check; the assertion is new.
    res = backtest.market_efficiency_test(_model_vs_market_frame(world))
    print("[11] market efficiency test (in-sample, near-efficient sim market):")
    print(res.to_string())
    assert list(res.index) == ["intercept", "market_margin", "model_margin"]


def test_12_efficiency_test_detects_signal_vs_sloppy_opener(world):
    """The test must be ABLE to find signal, or [11] means nothing. Re-run against
    a deliberately sloppy "opening" line (noise sd 5 instead of 1.6): the model
    coefficient should now be strongly significant."""
    df = _model_vs_market_frame(world)
    sloppy = -np.round((df["homeTeam"].map(TRUE) - df["awayTeam"].map(TRUE) + HFA
                        + world["sloppy_noise"]) * 2) / 2
    df2 = df.copy()
    df2["market_margin"] = -sloppy.values
    res2 = backtest.market_efficiency_test(df2)
    print(f"[12] power check vs sloppy opener: model coef="
          f"{res2.loc['model_margin','coef']:.3f} t={res2.loc['model_margin','t']:.2f} "
          f"p={res2.loc['model_margin','p']:.4f}")
    assert res2.loc["model_margin", "p"] < 0.01


def test_13_bootstrap_roi_interval_contains_estimate(world):
    # The original script only printed this check; the assertion is new.
    boot = backtest.bootstrap_roi(world["coinflips"])
    print(f"[13] bootstrap ROI on 60 coinflip-ish bets: "
          f"roi={boot['roi']:+.3f} ci=[{boot['ci_low']:+.3f},{boot['ci_high']:+.3f}] "
          f"P(profitable)={boot['p_profitable']:.2f}")
    assert boot["ci_low"] <= boot["roi"] <= boot["ci_high"]
