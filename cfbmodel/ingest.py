"""
Data ingest. Everything is cached to disk on first pull so a full backtest
re-run costs zero API calls (free tier = 1,000 calls/month).

Sources
-------
CFBD v2 (api.collegefootballdata.com) — plays, games, lines, player box scores,
    usage, recruiting, returning production, portal, venues, advanced stats.
    This is the only source you actually need; the sites you linked
    (Sports-Reference, TeamRankings, Ourlads, ESPN FPI) are either derived from
    the same underlying data, are scrape-hostile, or are competitor *outputs*
    rather than inputs. Two exceptions worth pulling manually:
      - 247 composite team rankings -> already in CFBD /recruiting/teams
      - Ourlads depth charts -> the one thing CFBD lacks. See depth_charts().
"""
from __future__ import annotations

import hashlib
import json
import time
from typing import Any

import pandas as pd
import requests

from .config import CACHE, CFBD_BASE
from .keys import get_cfbd_key

_SESSION = requests.Session()
_CALLS = 0


def _cache_path(endpoint: str, params: dict) -> "Any":
    key = endpoint + json.dumps(params, sort_keys=True)
    h = hashlib.md5(key.encode()).hexdigest()[:16]
    safe = endpoint.strip("/").replace("/", "_")
    return CACHE / f"{safe}__{h}.json"


def cfbd_get(endpoint: str, refresh: bool = False, **params) -> list[dict]:
    """GET a CFBD endpoint with disk caching. Returns list of records."""
    global _CALLS
    params = {k: v for k, v in params.items() if v is not None}
    path = _cache_path(endpoint, params)

    if path.exists() and not refresh:
        return json.loads(path.read_text())

    # Only reached on a cache miss, so cached re-runs and tests need no key.
    key = get_cfbd_key()

    for attempt in range(4):
        r = _SESSION.get(
            f"{CFBD_BASE}{endpoint}",
            params=params,
            headers={"Authorization": f"Bearer {key}"},
            timeout=60,
        )
        if r.status_code == 429:
            time.sleep(2 ** attempt)
            continue
        r.raise_for_status()
        data = r.json()
        path.write_text(json.dumps(data))
        _CALLS += 1
        return data
    raise RuntimeError(f"rate limited on {endpoint}")


def calls_used() -> int:
    return _CALLS


# ------------------------------------------------------------------- entities


def games(season: int, season_type: str = "both") -> pd.DataFrame:
    df = pd.DataFrame(cfbd_get("/games", year=season, seasonType=season_type))
    if df.empty:
        return df
    df = df.rename(columns={"startDate": "start_date"})
    df["start_date"] = pd.to_datetime(df["start_date"], utc=True, errors="coerce")
    for c in ("homePoints", "awayPoints", "week"):
        if c in df:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df["margin"] = df["homePoints"] - df["awayPoints"]
    df["total"] = df["homePoints"] + df["awayPoints"]
    df["season"] = season
    return df


def plays(season: int, week: int, season_type: str = "regular") -> pd.DataFrame:
    """Play-by-play. One call per week — 16ish calls per season."""
    df = pd.DataFrame(
        cfbd_get("/plays", year=season, week=week, seasonType=season_type)
    )
    if not df.empty:
        df["season"] = season
        df["week"] = week
    return df


def season_plays(season: int, weeks: range | None = None) -> pd.DataFrame:
    weeks = weeks or range(0, 16)
    frames = []
    for w in weeks:
        try:
            d = plays(season, w)
        except requests.HTTPError:
            continue
        if not d.empty:
            frames.append(d)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def lines(season: int, season_type: str = "regular") -> pd.DataFrame:
    """
    Betting lines, one row per game per book. CFBD carries opening AND current
    /closing numbers for several books — the opener is the more useful column
    for edge-hunting and the closer is your CLV benchmark.
    """
    raw = cfbd_get("/lines", year=season, seasonType=season_type)
    rows = []
    for g in raw:
        for ln in g.get("lines", []) or []:
            rows.append(
                {
                    "gameId": g.get("id"),
                    "season": g.get("season"),
                    "week": g.get("week"),
                    "home": g.get("homeTeam"),
                    "away": g.get("awayTeam"),
                    "homePoints": g.get("homeScore"),
                    "awayPoints": g.get("awayScore"),
                    "book": ln.get("provider"),
                    "spread_close": _num(ln.get("spread")),
                    "spread_open": _num(ln.get("spreadOpen")),
                    "total_close": _num(ln.get("overUnder")),
                    "total_open": _num(ln.get("overUnderOpen")),
                    "ml_home": _num(ln.get("homeMoneyline")),
                    "ml_away": _num(ln.get("awayMoneyline")),
                }
            )
    return pd.DataFrame(rows)


def player_box(season: int, week: int, season_type: str = "regular") -> pd.DataFrame:
    """
    Per-game player stats. This is the backbone of the props model — CFB has no
    public snap counts or route participation, so box-score usage share is the
    best volume signal available.
    """
    raw = cfbd_get("/games/players", year=season, week=week, seasonType=season_type)
    rows = []
    for g in raw:
        for team in g.get("teams", []):
            for cat in team.get("categories", []):
                for typ in cat.get("types", []):
                    for ath in typ.get("athletes", []):
                        rows.append(
                            {
                                "gameId": g.get("id"),
                                "season": season,
                                "week": week,
                                "team": team.get("team"),
                                "conference": team.get("conference"),
                                "category": cat.get("name"),
                                "stat_type": typ.get("name"),
                                "athlete_id": ath.get("id"),
                                "player": ath.get("name"),
                                "value": ath.get("stat"),
                            }
                        )
    return pd.DataFrame(rows)


def usage(season: int) -> pd.DataFrame:
    """CFBD's own usage rates (overall / pass / rush, plus situational)."""
    return pd.json_normalize(cfbd_get("/player/usage", year=season))


def recruiting_teams(season: int) -> pd.DataFrame:
    """247 composite team rankings — same data as the link you sent."""
    return pd.DataFrame(cfbd_get("/recruiting/teams", year=season))


def returning_production(season: int) -> pd.DataFrame:
    """Bill Connelly's returning production. Strongest single preseason input."""
    return pd.DataFrame(cfbd_get("/player/returning", year=season))


def portal(season: int) -> pd.DataFrame:
    return pd.DataFrame(cfbd_get("/player/portal", year=season))


def sp_ratings(season: int) -> pd.DataFrame:
    """SP+ — use as a sanity check / benchmark, never as an input feature."""
    return pd.json_normalize(cfbd_get("/ratings/sp", year=season))


def venues() -> pd.DataFrame:
    return pd.json_normalize(cfbd_get("/venues"))


def talent(season: int) -> pd.DataFrame:
    return pd.DataFrame(cfbd_get("/talent", year=season))


def advanced_team_stats(season: int) -> pd.DataFrame:
    return pd.json_normalize(cfbd_get("/stats/season/advanced", year=season))


def depth_charts(path: str) -> pd.DataFrame:
    """
    CFBD has no depth charts. Ourlads has them but blocks automated access, so
    this reads a CSV you export by hand (team,pos,rank,player) once a week.

    In practice CFB has *no mandated injury report* — this is the single
    biggest information asymmetry in the sport and it is a reporting problem,
    not a modelling one. Beat writers and team Twitter are your feed; the model
    just needs to know who is expected to play.
    """
    df = pd.read_csv(path)
    need = {"team", "pos", "rank", "player"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"depth chart csv missing columns: {missing}")
    return df


def _num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None
