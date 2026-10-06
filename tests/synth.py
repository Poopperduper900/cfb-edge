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
