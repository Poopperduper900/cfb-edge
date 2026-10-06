"""
Synthetic leagues with KNOWN truth, for tests only (CLAUDE.md rule 1: synthetic data lives
under tests/). Every call builds its own random generator from `seed`, so no test can
disturb another by consuming shared random numbers.

The league: `n_teams` FBS teams with true strengths (points vs average), a double
round-robin-ish schedule, closing lines = truth + noise (an efficient market), final margins
drawn around the truth with heavy tails, and play-level EPA driven by the same strengths.
Optionally some FCS opponents, flagged exactly the way ingest.py flags them.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

FCS_NAMES = ("Tiny FCS A", "Tiny FCS B")


def make_league(seed: int = 7, n_teams: int = 40, seasons=(2024, 2025), weeks=range(1, 13),
                hfa: float = 2.4, plays_per_team_game: int = 40, market_noise: float = 1.6,
                strength_sd: float = 10.0, home_epa: float = 0.0, fcs: bool = False,
                with_plays: bool = True) -> dict:
    rng = np.random.default_rng(seed)
    names = [f"Team{i:02d}" for i in range(n_teams)]
    truth = pd.Series(rng.normal(0, strength_sd, n_teams), index=names)
    if fcs:
        truth = pd.concat([truth, pd.Series(-35.0, index=list(FCS_NAMES))])
    fbs = set(names)

    games, lines, plays = [], [], []
    gid = 0
    for season in seasons:
        for week in weeks:
            order = list(rng.permutation(names))
            pairs = [(order[i], order[i + 1]) for i in range(0, len(order) - 1, 2)]
            if fcs and week in (1, 5):
                pairs = pairs[2:] + [(order[0], FCS_NAMES[0]), (order[1], FCS_NAMES[1])]
            for home, away in pairs:
                em = truth[home] - truth[away] + hfa
                margin = int(np.round(rng.standard_t(7) * 15.5 + em)) or 1
                total = int(np.round(rng.normal(52, 12)))
                is_fbs_h, is_fbs_a = home in fbs, away in fbs
                games.append(dict(id=gid, season=season, week=week, homeTeam=home, awayTeam=away,
                                  margin=margin, total=total, neutralSite=False,
                                  homeConference="X" if is_fbs_h else None,
                                  awayConference="X" if is_fbs_a else None,
                                  home_is_fbs=is_fbs_h, away_is_fbs=is_fbs_a))
                close = -np.round((em + rng.normal(0, market_noise)) * 2) / 2
                lines.append(dict(gameId=gid, season=season, week=week, home=home, away=away,
                                  book="sim", spread_close=close,
                                  spread_open=close + np.round(rng.normal(0, 1.2) * 2) / 2,
                                  total_close=float(total + rng.integers(-3, 4)),
                                  total_open=float(total + rng.integers(-4, 5)),
                                  home_is_fbs=is_fbs_h, away_is_fbs=is_fbs_a))
                if with_plays:
                    for off, dfn, is_home in ((home, away, 1), (away, home, 0)):
                        n = plays_per_team_game
                        mu = (truth[off] - truth[dfn]) / (2 * 68) + home_epa * is_home
                        ok = off in fbs and dfn in fbs
                        plays.append(pd.DataFrame(dict(
                            season=season, week=week, gameId=gid, offense=off, defense=dfn,
                            home=home, offense_is_fbs=off in fbs, defense_is_fbs=dfn in fbs,
                            offenseConference="X" if off in fbs else None,
                            defenseConference="X" if dfn in fbs else None,
                            playType=rng.choice(["Rush", "Pass Reception", "Pass Incompletion"], n),
                            epa=rng.normal(mu, 1.25, n), period=rng.integers(1, 5, n),
                            offenseScore=0, defenseScore=0, homeWinProb=0.5)))
                gid += 1
    out = dict(truth=truth, teams=names, hfa=hfa, games=pd.DataFrame(games),
               lines=pd.DataFrame(lines))
    out["plays"] = pd.concat(plays, ignore_index=True) if plays else pd.DataFrame()
    return out


def box_week(season: int, week: int, team: str, rush: dict | None = None, rec: dict | None = None,
             game_id: int = 1) -> pd.DataFrame:
    """Box-score rows (the shape ingest.player_box returns) for one team-game.
    rush: {player: (carries, yards)}   rec: {player: (receptions, yards)}.
    The athlete id is the player's name, so the same name means the same person."""
    rows = []
    for cat, spec, (touch, yds) in (("rushing", rush or {}, ("CAR", "YDS")),
                                    ("receiving", rec or {}, ("REC", "YDS"))):
        for player, (n, y) in spec.items():
            for stat, v in ((touch, n), (yds, y)):
                rows.append(dict(gameId=game_id, season=season, week=week, team=team, conference="X",
                                 category=cat, stat_type=stat, athlete_id=player, player=player,
                                 value=str(v), team_is_fbs=True))
    return pd.DataFrame(rows)


# ------------------------------------------------------------- learning-loop data
from scipy import stats as _st

_XS = np.arange(-70, 71)


def draw_margins(rng, exp_margin, exp_total, *, scale, df, sd_total_coef, keys, sd_ref=52.0, floor=8.0, cap=20.0):
    """Margins from the key-number Student-t pmf, written independently of cfbmodel's own likelihood
    code (scipy.stats.t, a plain grid, a plain normalisation)."""
    exp_margin, exp_total = np.asarray(exp_margin, float), np.asarray(exp_total, float)
    sd = np.clip(scale + sd_total_coef * (exp_total - sd_ref), floor, cap)[:, None]
    mu = exp_margin[:, None]
    pmf = _st.t.cdf((_XS[None, :] + 0.5 - mu) / sd, df) - _st.t.cdf((_XS[None, :] - 0.5 - mu) / sd, df)
    bump = np.ones(len(_XS))
    for k, w in keys.items():
        bump[np.abs(_XS) == int(k)] *= w
    bump[_XS == 0] = 0.0
    pmf = pmf * bump[None, :]
    cum = np.cumsum(pmf / pmf.sum(axis=1, keepdims=True), axis=1)
    return _XS[(cum < rng.random(len(exp_margin))[:, None]).sum(axis=1).clip(0, len(_XS) - 1)].astype(float)


def make_learn_data(seed: int, *, seasons=(2022, 2023, 2024), weeks=range(1, 13), games_per_week=60,
                    market=None, model=None, totals=None, keys=None, with_early=False):
    """LearnData-shaped frames whose true parameters are given. `market`/`model`/`totals` are dicts
    of the truth; anything omitted uses the version-1 values (so 'nothing has changed')."""
    from cfbmodel import learn
    from cfbmodel.params import P
    rng = np.random.default_rng(seed)
    keys = keys or {int(k): v for k, v in P.margin.key_numbers.items()}
    m = {"scale": P.margin.scale_market, "df": P.margin.df, "coef": P.margin.sd_total_coef, **(market or {})}
    md = {"scale": P.margin.scale_model, "hfa": P.margin.hfa_points, **(model or {})}
    t = {"sd_base": P.totals.sd_base, "coef": P.totals.sd_margin_coef, **(totals or {})}
    rows_m, rows_w = [], []
    for s in seasons:
        for w in weeks:
            n = games_per_week
            spread = np.round(rng.normal(0, 11, n) * 2) / 2
            tot = np.round(rng.normal(52, 7, n) * 2) / 2
            margin = draw_margins(rng, -spread, tot, scale=m["scale"], df=m["df"], sd_total_coef=m["coef"], keys=keys)
            sd_t = t["sd_base"] + t["coef"] * np.abs(spread)
            actual = np.round(rng.normal(tot, sd_t))
            rows_m.append(pd.DataFrame({"season": s, "week": w, "margin": margin, "actual_total": actual,
                                        "spread_close": spread, "total_close": tot}))
            core = rng.normal(0, 10, n)
            neutral = rng.random(n) < 0.08
            exp = core + md["hfa"] * (~neutral)
            mm = draw_margins(rng, exp, tot, scale=md["scale"], df=m["df"], sd_total_coef=m["coef"], keys=keys)
            line = -(np.round((exp + rng.normal(0, 5, n)) * 2) / 2)
            wf = pd.DataFrame({"season": s, "week": w, "margin": mm, "actual_total": actual, "spread_close": line,
                               "total_close": tot, "core_model": core, "neutral": neutral,
                               "model_margin": exp, "core_unshrunk": core, "core_prior": np.nan})
            rows_w.append(wf)
    return learn.LearnData(market=pd.concat(rows_m, ignore_index=True), wf=pd.concat(rows_w, ignore_index=True))
