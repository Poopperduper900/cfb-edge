"""
Weekly team and player state (BUILD_PLAN Phase 3).

`python -m cfbmodel update --season S --week W` runs after week W has finished. It refits team
ratings on everything strictly before week W+1 and writes

    output/state/teams_<S>_w<W>.csv      one row per team (ratings and their ingredients)
    output/state/players_<S>_w<W>.csv    one row per player-group (usage + efficiency posteriors)

The BUILD_PLAN names .parquet files; CSV is used instead because parquet needs the pyarrow
package, which this project has not been given permission to add (CLAUDE.md rule 7). Switching
is a one-line change in write_/read_ below if you decide you want it.

The blended rating a game is priced with:

    model_shrunk = (1 - pw) * model + pw * preseason_prior     pw decays over weeks 0-5
    rating       = w_model * model_shrunk + (1 - w_model) * market

w_model comes from output/validation_status.json (0 if absent). `half_life` of the decay is an
a registry parameter (state.decay_half_life) that `learn` re-fits.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import config, playerstate, priors, ratings, status
from .config import C
from .params import P


def state_dir() -> Path:
    return config.OUTPUT / "state"


def early_season_prior_weight(week: int, half_life: float | None = None,
                              last_week: int | None = None) -> float:
    """Weight on the preseason prior when week `week` of games is about to be played (so `week`
    weeks of results are in). 1.0 before any games, halving every `half_life` weeks, 0 after
    week `last_week`."""
    half_life = P.state.decay_half_life if half_life is None else half_life
    last_week = P.state.early_last_week if last_week is None else last_week
    if week > last_week:
        return 0.0
    return float(0.5 ** (week / half_life))


# ------------------------------------------------------------------ team state


def build_team_state(season: int, week_completed: int, lines: pd.DataFrame, pbp: pd.DataFrame,
                     prior_pts: pd.Series | None = None, w_model: float | None = None,
                     half_life: float | None = None) -> pd.DataFrame:
    """Team ratings after `week_completed`, using only games strictly before the next week."""
    asof_week = week_completed + 1
    epa = ratings.fit_epa_ratings(pbp, season, asof_week, split_pass_rush=False)
    mkt = ratings.fit_market_ratings(lines, season, asof_week)
    j = ratings.blend(epa, mkt, w_model=0.0)          # model points + market, no weighting yet

    pw = early_season_prior_weight(asof_week, half_life)
    prior = (prior_pts.reindex(j.index) if prior_pts is not None
             else pd.Series(np.nan, index=j.index))
    weight = pd.Series(np.where(prior.notna(), pw, 0.0), index=j.index)   # no prior => no shrink
    shrunk = (1 - weight) * j["model_rating_pts"] + weight * prior.fillna(0.0)

    w = status.w_model_for("spread_vs_close") if w_model is None else float(w_model)
    out = pd.DataFrame({
        "season": season, "week_completed": week_completed,
        "model_rating_pts": j["model_rating_pts"], "prior_rating": prior, "prior_weight": weight,
        "model_rating_shrunk": shrunk, "market_rating": j["market_rating"], "w_model": w,
        "rating": w * shrunk + (1 - w) * j["market_rating"],
        "disagreement": shrunk - j["market_rating"],
        "off_epa": j["off_epa"], "def_epa": j["def_epa"], "net_epa": j["net_epa"],
        "pace": j["pace"], "plays": j["plays"], "market_hfa": j["market_hfa"],
    })
    return out.rename_axis("team")


def preseason_prior_points(season: int, lines: pd.DataFrame, recruiting: dict, returning: pd.DataFrame,
                           portal: pd.DataFrame, weights: dict | None = None) -> pd.Series:
    """Preseason rating in points for every team, from last season's market rating, 4-year
    recruiting, returning production and portal, weighted by output/prior_weights.json
    (unfitted defaults, and a note in the result's attrs, if that file is missing)."""
    prior_rating = ratings.fit_market_ratings(lines, season, 0)["market_rating"]
    out = priors.build_preseason_ratings(
        prior_season_ratings=prior_rating,
        recruiting=priors.rolling_recruiting(recruiting, season),
        returning=priors.returning_production_score(returning),
        portal=priors.portal_score(portal, season),
        weights=weights,
    )
    s = out["rating"].rename("prior_rating")
    s.attrs["weights_source"] = out.attrs.get("weights_source")
    return s


# ---------------------------------------------------------------------- file io


def _path(kind: str, season: int, week: int, directory: Path | None) -> Path:
    return Path(directory or state_dir()) / f"{kind}_{season}_w{week}.csv"


def write_team_state(df: pd.DataFrame, season: int, week: int, directory: Path | None = None) -> Path:
    p = _path("teams", season, week, directory)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p)
    return p


def read_team_state(season: int, week: int, directory: Path | None = None) -> pd.DataFrame:
    return pd.read_csv(_path("teams", season, week, directory), index_col="team")


def write_player_state(df: pd.DataFrame, season: int, week: int, directory: Path | None = None) -> Path:
    p = _path("players", season, week, directory)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, index=False)
    return p


def read_player_state(season: int, week: int, directory: Path | None = None) -> pd.DataFrame | None:
    p = _path("players", season, week, directory)
    return pd.read_csv(p, dtype={"pid": str}) if p.exists() else None


def last_week_with_players(season: int, directory: Path | None = None) -> int | None:
    d = Path(directory or state_dir())
    weeks = [int(f.stem.rsplit("_w", 1)[1]) for f in d.glob(f"players_{season}_w*.csv")] if d.exists() else []
    return max(weeks) if weeks else None


# ------------------------------------------------------------------- the update


def run_update(season: int, week: int, lines: pd.DataFrame, pbp: pd.DataFrame, box: pd.DataFrame,
               prior_pts: pd.Series | None = None, w_model: float | None = None,
               directory: Path | None = None) -> dict:
    """Team + player state after completed week `week`. The player state continues from last
    week's file when there is one (last week's posterior is this week's prior), and is rebuilt
    from the season's earlier weeks otherwise; both give the same answer (tested)."""
    teams_df = build_team_state(season, week, lines, pbp, prior_pts, w_model)

    prev_final = None
    last_prev = last_week_with_players(season - 1, directory)
    if last_prev is not None:
        prev_final = read_player_state(season - 1, last_prev, directory)

    prev = read_player_state(season, week - 1, directory) if week > 0 else None
    this_week = box[(box["season"] == season) & (box["week"] == week)]
    if prev is not None:
        players_df = playerstate.update_player_state(prev, this_week, season, week, prev_final)
    else:
        players_df = playerstate.build_player_state(box, season, week, prev_final)

    return {
        "teams": teams_df, "players": players_df,
        "teams_path": write_team_state(teams_df, season, week, directory),
        "players_path": write_player_state(players_df, season, week, directory),
    }
