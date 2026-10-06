"""
Data ingest. CFBD is the only source for games, lines, plays and rosters; weather
comes from Open-Meteo (weather.py). Everything is cached to disk so a re-run costs
zero API calls (free tier = 1,000 calls/month, tracked in budget.py).

How a call works
----------------
cfbd_get(endpoint, **params)
  1. cache hit (and not stale)      -> return the saved JSON, no call, no key needed
  2. budget check                   -> warn at 700, refuse at 950 unless --force
  3. key check                      -> clear error if CFBD_API_KEY is missing
  4. HTTP GET with Bearer header    -> 429: wait 1,2,4,8s and retry; 401: "check your key";
                                       404: CfbdNotFound; anything else: CfbdError
  5. save to cache, count the call

Cache policy (ttl_for): finished seasons are cached forever. In the current season
the whole-season tables (games, lines) can be refreshed after REFRESH_TTL_HOURS, and
weekly tables (plays, box scores) for the current and previous week likewise; older
weeks are never re-pulled unless you pass --refresh.

Nothing here swallows an error. If CFBD fails, the run stops and says why.

Every entity returns the documented columns (see schema.py / docs/DATA.md) with team
names mapped to the canonical FBS name (teams.py). Non-FBS names are kept and flagged,
never dropped.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from typing import Any

import pandas as pd
import requests

from . import budget as _budget
from . import schema, teams
from .config import CACHE, CFBD_BASE, CURRENT_SEASON
from .keys import get_cfbd_key

REFRESH_TTL_HOURS = 12.0
MAX_RETRIES = 5

_SESSION = requests.Session()
_CALLS = 0
_STATE: dict[str, Any] = {"force": False, "refresh": False,
                          "current_season": None, "current_week": None}


class CfbdError(RuntimeError):
    """CFBD answered with something other than data."""


class CfbdAuthError(CfbdError):
    """CFBD rejected the key (HTTP 401/403)."""


class CfbdNotFound(CfbdError):
    """CFBD has no such resource (HTTP 404), e.g. a dataset that does not exist for an old year."""


# --------------------------------------------------------------- run settings


def configure(force: bool = False, refresh: bool = False) -> None:
    """Set once per run by the command line: --force (go past 950 calls), --refresh."""
    _STATE["force"] = bool(force)
    _STATE["refresh"] = bool(refresh)


def set_current(season: int | None = None, week: int | None = None) -> None:
    """Tell the cache policy which season/week is 'now' (default: config.CURRENT_SEASON)."""
    _STATE["current_season"] = season
    _STATE["current_week"] = week


def calls_used() -> int:
    """API calls made by this process (cache hits are free and not counted)."""
    return _CALLS


def budget() -> _budget.Budget:
    return _budget.Budget(CACHE)


def ttl_for(season: int, week: int | None = None) -> float | None:
    """Hours a cached answer stays fresh; None means 'cache forever'."""
    cur = _STATE["current_season"] or CURRENT_SEASON
    if season < cur:
        return None
    if season > cur:
        return REFRESH_TTL_HOURS
    if week is None:
        return REFRESH_TTL_HOURS
    cw = _STATE["current_week"]
    if cw is None or week >= cw - 1:
        return REFRESH_TTL_HOURS
    return None


# ------------------------------------------------------------------- the call


def _cache_path(endpoint: str, params: dict) -> "Any":
    key = endpoint + json.dumps(params, sort_keys=True)
    h = hashlib.md5(key.encode()).hexdigest()[:16]
    safe = endpoint.strip("/").replace("/", "_")
    return CACHE / f"{safe}__{h}.json"


def _is_stale(path, ttl_hours: float | None) -> bool:
    if ttl_hours is None:
        return False
    return (time.time() - path.stat().st_mtime) / 3600.0 >= ttl_hours


def _http_get(url: str, params: dict, headers: dict):
    """The one place that touches the network (tests replace this)."""
    return _SESSION.get(url, params=params, headers=headers, timeout=60)


def cfbd_get(endpoint: str, refresh: bool = False, ttl_hours: float | None = None, **params) -> list[dict]:
    """GET a CFBD endpoint through the cache. Returns the parsed JSON (a list of records)."""
    global _CALLS
    params = {k: v for k, v in params.items() if v is not None}
    path = _cache_path(endpoint, params)

    if path.exists() and not (refresh or _STATE["refresh"]) and not _is_stale(path, ttl_hours):
        return json.loads(path.read_text())

    bud = budget()
    bud.check(force=_STATE["force"])
    key = get_cfbd_key()  # only reached on a cache miss, so tests/cached runs need no key

    for attempt in range(MAX_RETRIES):
        r = _http_get(f"{CFBD_BASE}{endpoint}", params, {"Authorization": f"Bearer {key}"})
        if r.status_code == 429:
            time.sleep(2 ** attempt)
            continue
        bud.record()
        _CALLS += 1
        if r.status_code in (401, 403):
            raise CfbdAuthError(
                f"CFBD rejected the API key (HTTP {r.status_code}). Check CFBD_API_KEY in your "
                ".env file: no spaces, no quotes, and a key that is still active "
                "(a fresh one is free at https://collegefootballdata.com/key)."
            )
        if r.status_code == 404:
            raise CfbdNotFound(f"CFBD has nothing at {endpoint} {params} (HTTP 404).")
        if r.status_code != 200:
            raise CfbdError(f"CFBD {endpoint} {params} returned HTTP {r.status_code}.")
        data = r.json()
        tmp = path.with_suffix(".tmp")
        CACHE.mkdir(parents=True, exist_ok=True)
        tmp.write_text(json.dumps(data))
        os.replace(tmp, path)
        return data
    raise CfbdError(f"CFBD rate-limited {endpoint} {params} on all {MAX_RETRIES} tries.")


# ------------------------------------------------------------------- helpers


def _raw(name: str, endpoint: str, ttl: float | None, **params) -> list[dict]:
    data = cfbd_get(endpoint, ttl_hours=ttl, **params)
    schema.check_raw(data, name)
    return data


def _num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _num_col(df: pd.DataFrame, col: str) -> pd.DataFrame:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def fbs_names(season: int) -> teams.TeamNames:
    return teams.TeamNames(teams_fbs(season)["school"])


# ------------------------------------------------------------------- entities


def teams_fbs(season: int) -> pd.DataFrame:
    raw = _raw("teams_fbs", "/teams/fbs", ttl_for(season), year=season)
    df = pd.DataFrame(raw)
    return schema.check_frame(df, "teams_fbs")


def games(season: int, season_type: str = "both") -> pd.DataFrame:
    raw = _raw("games", "/games", ttl_for(season), year=season, seasonType=season_type)
    df = pd.DataFrame(raw)
    if df.empty:
        return schema.empty_frame("games")
    df = df.rename(columns={"startDate": "start_date", "seasonType": "season_type"})
    df["start_date"] = pd.to_datetime(df["start_date"], utc=True, errors="coerce")
    for c in ("homePoints", "awayPoints", "week"):
        _num_col(df, c)
    df["season"] = season
    df["neutralSite"] = (df["neutralSite"] if "neutralSite" in df else False)
    df["neutralSite"] = df["neutralSite"].fillna(False).astype(bool)
    for c in ("homePoints", "awayPoints"):
        if c not in df.columns:
            df[c] = float("nan")
    df["margin"] = df["homePoints"] - df["awayPoints"]
    df["total"] = df["homePoints"] + df["awayPoints"]
    df = fbs_names(season).canonicalize_frame(
        df, ["homeTeam", "awayTeam"], ["home_is_fbs", "away_is_fbs"], "games")
    return schema.check_frame(df, "games")


def regular_weeks(season: int) -> list[int]:
    """Weeks of the regular season that have at least one finished game."""
    g = games(season)
    if g.empty:
        return []
    if "season_type" in g.columns:
        g = g[g["season_type"] == "regular"]
    g = g[g["homePoints"].notna() & g["week"].notna()]
    return sorted(int(w) for w in g["week"].unique())


def plays(season: int, week: int, season_type: str = "regular") -> pd.DataFrame:
    """Play-by-play. One call per week. FBS classification requested (pitfall 7); non-FBS
    names that still appear are flagged, and ratings.clean_plays excludes them."""
    raw = _raw("plays", "/plays", ttl_for(season, week), year=season, week=week,
               seasonType=season_type, classification="fbs")
    df = pd.DataFrame(raw)
    if df.empty:
        return schema.empty_frame("plays")
    df["season"] = season
    df["week"] = week
    for c in ("ppa", "period", "offenseScore", "defenseScore"):
        _num_col(df, c)
    df = fbs_names(season).canonicalize_frame(
        df, ["offense", "defense"], ["offense_is_fbs", "defense_is_fbs"], "plays")
    df = fbs_names(season).canonicalize_frame(df, ["home", "away"], None, "plays")
    return schema.check_frame(df, "plays")


def season_plays(season: int, weeks=None) -> pd.DataFrame:
    weeks = list(weeks) if weeks is not None else regular_weeks(season)
    frames = [d for d in (plays(season, w) for w in weeks) if not d.empty]
    return pd.concat(frames, ignore_index=True) if frames else schema.empty_frame("plays")


def lines(season: int, season_type: str = "regular") -> pd.DataFrame:
    """
    Betting lines, one row per game per book. CFBD carries opening AND current/closing
    numbers for several books. The opener is the more useful column for edge-hunting and
    the closer is the CLV benchmark. CFBD's lines carry no prices (juice); callers that
    need one must say where it came from.
    """
    raw = _raw("lines", "/lines", ttl_for(season), year=season, seasonType=season_type)
    rows = []
    for g in raw:
        for ln in g.get("lines", []) or []:
            rows.append({
                "gameId": g.get("id"), "season": g.get("season"), "week": g.get("week"),
                "home": g.get("homeTeam"), "away": g.get("awayTeam"),
                "homePoints": g.get("homeScore"), "awayPoints": g.get("awayScore"),
                "book": ln.get("provider"),
                "spread_close": _num(ln.get("spread")), "spread_open": _num(ln.get("spreadOpen")),
                "total_close": _num(ln.get("overUnder")), "total_open": _num(ln.get("overUnderOpen")),
                "ml_home": _num(ln.get("homeMoneyline")), "ml_away": _num(ln.get("awayMoneyline")),
                "formatted_spread": ln.get("formattedSpread"),
            })
    df = pd.DataFrame(rows)
    if df.empty:
        return schema.empty_frame("lines")
    for c in ("gameId", "season", "week", "homePoints", "awayPoints"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = fbs_names(season).canonicalize_frame(
        df, ["home", "away"], ["home_is_fbs", "away_is_fbs"], "lines")
    return schema.check_frame(df, "lines")


def player_box(season: int, week: int, season_type: str = "regular") -> pd.DataFrame:
    """
    Per-game player stats. This is the backbone of the props model: CFB has no public
    snap counts or route participation, so box-score usage share is the best volume
    signal available.
    """
    raw = _raw("player_box", "/games/players", ttl_for(season, week), year=season, week=week,
               seasonType=season_type)
    rows = []
    for g in raw:
        for team in g.get("teams", []):
            for cat in team.get("categories", []):
                for typ in cat.get("types", []):
                    for ath in typ.get("athletes", []):
                        rows.append({
                            "gameId": g.get("id"), "season": season, "week": week,
                            "team": team.get("team"), "conference": team.get("conference"),
                            "category": cat.get("name"), "stat_type": typ.get("name"),
                            "athlete_id": None if ath.get("id") is None else str(ath.get("id")),
                            "player": ath.get("name"), "value": ath.get("stat"),
                        })
    df = pd.DataFrame(rows)
    if df.empty:
        return schema.empty_frame("player_box")
    df["gameId"] = pd.to_numeric(df["gameId"], errors="coerce")
    df = fbs_names(season).canonicalize_frame(df, ["team"], ["team_is_fbs"], "player_box")
    return schema.check_frame(df, "player_box")


def usage(season: int) -> pd.DataFrame:
    """CFBD's own usage rates (overall / pass / rush, plus situational)."""
    raw = cfbd_get("/player/usage", ttl_hours=ttl_for(season), year=season)
    df = pd.json_normalize(raw)
    if "team" in df.columns:
        df = fbs_names(season).canonicalize_frame(df, ["team"], None, "usage")
    return schema.check_frame(df, "usage")


def recruiting_teams(season: int) -> pd.DataFrame:
    """247 composite team rankings."""
    raw = _raw("recruiting_teams", "/recruiting/teams", ttl_for(season), year=season)
    df = pd.DataFrame(raw)
    if df.empty:
        return schema.empty_frame("recruiting_teams")
    df = _num_col(df, "points")
    df = fbs_names(season).canonicalize_frame(df, ["team"], None, "recruiting_teams")
    return schema.check_frame(df, "recruiting_teams")


def returning_production(season: int) -> pd.DataFrame:
    """Bill Connelly's returning production. Strongest single preseason input."""
    raw = _raw("returning_production", "/player/returning", ttl_for(season), year=season)
    df = pd.DataFrame(raw)
    if df.empty:
        return schema.empty_frame("returning_production")
    df = fbs_names(season).canonicalize_frame(df, ["team"], None, "returning_production")
    return schema.check_frame(df, "returning_production")


def portal(season: int) -> pd.DataFrame:
    raw = _raw("portal", "/player/portal", ttl_for(season), year=season)
    df = pd.DataFrame(raw)
    if df.empty:
        return schema.empty_frame("portal")
    df = _num_col(df, "rating")
    df = fbs_names(season).canonicalize_frame(df, ["origin", "destination"], None, "portal")
    return schema.check_frame(df, "portal")


def sp_ratings(season: int) -> pd.DataFrame:
    """SP+: a sanity check / benchmark, never an input feature."""
    return pd.json_normalize(cfbd_get("/ratings/sp", ttl_hours=ttl_for(season), year=season))


def venues() -> pd.DataFrame:
    """Stadiums. Venues rarely change, so this is cached until you pass --refresh."""
    raw = _raw("venues", "/venues", None)
    df = pd.json_normalize(raw)
    if df.empty:
        return schema.empty_frame("venues")
    if "dome" in df.columns:
        df["dome"] = df["dome"].map(lambda v: v if isinstance(v, bool) else None).astype(object)
    return schema.check_frame(df, "venues")


def talent(season: int) -> pd.DataFrame:
    return pd.DataFrame(cfbd_get("/talent", ttl_hours=ttl_for(season), year=season))


def advanced_team_stats(season: int) -> pd.DataFrame:
    return pd.json_normalize(cfbd_get("/stats/season/advanced", ttl_hours=ttl_for(season), year=season))


def depth_charts(path: str) -> pd.DataFrame:
    """
    CFBD has no depth charts, and sites that do block automated access, so this reads a CSV
    you export by hand (team,pos,rank,player). CFB also has no mandated injury report: who is
    expected to play is a reporting problem, not a modelling one.
    """
    df = pd.read_csv(path)
    need = {"team", "pos", "rank", "player"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"depth chart csv missing columns: {missing}")
    return df
