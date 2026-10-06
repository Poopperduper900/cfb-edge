"""
Schema checks for every table the data layer produces.

Two layers of checking, both loud:

1. `check_raw(records, name)` looks at the JSON exactly as CFBD sent it and fails if
   a field the code depends on is missing. This is the early warning for "CFBD
   renamed a field": the error names the field and the endpoint instead of letting
   a column quietly fill with blanks.
2. `check_frame(df, name)` looks at the cleaned DataFrame: required columns exist,
   have the documented type, and (for lines) have no duplicate (gameId, book) rows.

Columns not listed here pass through unchanged; the listed ones are the contract the
rest of the code relies on. `render_markdown()` produces docs/DATA.md from this file,
and a test keeps the two in step.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


class SchemaError(ValueError):
    """The data does not match its documented shape."""


# kind -> human description (used in docs/DATA.md)
KINDS = {
    "id": "whole number, never blank",
    "int_null": "whole number, blank allowed",
    "float": "number, blank allowed",
    "str": "text, never blank",
    "str_null": "text, blank allowed",
    "bool": "True/False",
    "bool_null": "True/False/blank (unknown)",
    "datetime": "UTC timestamp",
}


@dataclass(frozen=True)
class Schema:
    name: str
    doc: str
    required: dict = field(default_factory=dict)      # col -> kind
    optional: dict = field(default_factory=dict)      # col -> kind (checked if present)
    unique: tuple = ()
    raw_required: tuple = ()                          # keys that must be in every raw record


SCHEMAS: dict[str, Schema] = {s.name: s for s in [
    Schema(
        "teams_fbs", "One row per FBS team for a season (/teams/fbs). The source of canonical names.",
        required={"school": "str"}, optional={"conference": "str_null", "classification": "str_null"},
        unique=("school",), raw_required=("school",)),
    Schema(
        "games", "One row per game (/games), regular season and postseason.",
        required={"id": "id", "season": "id", "week": "int_null", "start_date": "datetime",
                  "homeTeam": "str", "awayTeam": "str", "neutralSite": "bool",
                  "homePoints": "float", "awayPoints": "float", "margin": "float", "total": "float",
                  "home_is_fbs": "bool", "away_is_fbs": "bool"},
        optional={"season_type": "str_null", "homeConference": "str_null",
                  "awayConference": "str_null", "venue": "str_null"},
        unique=("id",), raw_required=("id", "season", "week", "homeTeam", "awayTeam", "startDate")),
    Schema(
        "lines", "One row per game per sportsbook (/lines): opening and closing numbers.",
        required={"gameId": "id", "season": "int_null", "week": "int_null", "home": "str",
                  "away": "str", "book": "str", "spread_close": "float", "spread_open": "float",
                  "total_close": "float", "total_open": "float", "ml_home": "float",
                  "ml_away": "float", "home_is_fbs": "bool", "away_is_fbs": "bool"},
        optional={"formatted_spread": "str_null", "homePoints": "float", "awayPoints": "float"},
        unique=("gameId", "book"),
        raw_required=("id", "season", "week", "homeTeam", "awayTeam", "lines", "lines[].provider")),
    Schema(
        "plays", "One row per play (/plays), fetched one week at a time, FBS classification.",
        required={"season": "id", "week": "id", "offense": "str", "defense": "str",
                  "offense_is_fbs": "bool", "defense_is_fbs": "bool"},
        optional={"gameId": "id", "playType": "str_null", "ppa": "float", "period": "float",
                  "offenseScore": "float", "defenseScore": "float", "home": "str_null",
                  "away": "str_null", "offenseConference": "str_null",
                  "defenseConference": "str_null"},
        raw_required=("gameId", "offense", "defense", "playType", "ppa", "period")),
    Schema(
        "player_box", "One row per player per stat per game (/games/players), one week at a time.",
        required={"gameId": "id", "season": "id", "week": "id", "team": "str", "category": "str",
                  "stat_type": "str", "player": "str", "team_is_fbs": "bool"},
        optional={"athlete_id": "str_null", "value": "str_null", "conference": "str_null"},
        raw_required=("id", "teams", "teams[].team", "teams[].categories")),
    Schema(
        "usage", "Player usage rates (/player/usage). Passed through as CFBD sends it.",
        optional={"team": "str_null"}),
    Schema(
        "recruiting_teams", "247 composite team recruiting class (/recruiting/teams).",
        required={"team": "str", "points": "float"}, optional={"rank": "float", "year": "float"},
        raw_required=("team", "points")),
    Schema(
        "returning_production", "Returning production by team (/player/returning).",
        required={"team": "str"}, raw_required=("team",)),
    Schema(
        "portal", "Transfer portal entries (/player/portal).",
        required={"origin": "str"},
        optional={"destination": "str_null", "position": "str_null", "rating": "float",
                  "firstName": "str_null", "lastName": "str_null"},
        raw_required=("origin",)),
    Schema(
        "venues", "Stadiums (/venues), with the dome flag used to zero weather effects.",
        required={"name": "str"}, optional={"dome": "bool_null", "latitude": "float",
                                            "longitude": "float", "timezone": "str_null"},
        raw_required=("name",)),
]}


# ----------------------------------------------------------------- raw checks


def _get_all(rec, path: str):
    """Yield every value at `path` ('a', or 'a[].b' for a list of dicts)."""
    if "[]." in path:
        head, tail = path.split("[].", 1)
        for item in rec.get(head) or []:
            if not isinstance(item, dict):
                yield None
            else:
                yield from _get_all(item, tail)
    else:
        yield rec[path] if path in rec else _MISSING


_MISSING = object()


def check_raw(records, name: str) -> None:
    """Every raw record must carry the keys the code relies on (null values are fine)."""
    sch = SCHEMAS[name]
    if not isinstance(records, list):
        raise SchemaError(f"{name}: CFBD returned {type(records).__name__}, expected a list "
                          "of records. The API may have changed.")
    missing: dict[str, int] = {}
    for rec in records:
        if not isinstance(rec, dict):
            raise SchemaError(f"{name}: a record is {type(rec).__name__}, expected an object.")
        for key in sch.raw_required:
            if any(v is _MISSING for v in _get_all(rec, key)):
                missing[key] = missing.get(key, 0) + 1
    if missing:
        detail = ", ".join(f"{k} (missing in {n} of {len(records)} records)" for k, n in missing.items())
        raise SchemaError(f"{name}: CFBD response is missing field(s): {detail}. "
                          "The API may have changed; do not guess, tell Claude.")


# ---------------------------------------------------------------- frame checks


def _kind_problem(s: pd.Series, kind: str) -> str | None:
    nn = s.dropna()
    if kind == "id":
        if len(nn) != len(s):
            return "has blanks"
        if not pd.api.types.is_numeric_dtype(s) or (len(nn) and not (nn == nn.round()).all()):
            return "is not whole numbers"
    elif kind == "int_null":
        if not pd.api.types.is_numeric_dtype(s) or (len(nn) and not (nn == nn.round()).all()):
            return "is not whole numbers"
    elif kind == "float":
        if not pd.api.types.is_numeric_dtype(s):
            return "is not numeric"
    elif kind in ("str", "str_null"):
        if kind == "str" and len(nn) != len(s):
            return "has blanks"
        if len(nn) and not nn.map(lambda v: isinstance(v, str)).all():
            return "is not all text"
    elif kind == "bool":
        if s.dtype != bool:
            return f"has dtype {s.dtype}, expected bool"
    elif kind == "bool_null":
        if len(nn) and not nn.map(lambda v: isinstance(v, (bool, np.bool_))).all():
            return "is not True/False/blank"
    elif kind == "datetime":
        if not pd.api.types.is_datetime64_any_dtype(s):
            return f"has dtype {s.dtype}, expected datetime"
    else:  # pragma: no cover
        raise ValueError(kind)
    return None


def check_frame(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """Raise SchemaError listing every problem; return df unchanged if all is well."""
    sch = SCHEMAS[name]
    problems = []
    for col in sch.required:
        if col not in df.columns:
            problems.append(f"missing column '{col}'")
    if not problems:
        for col, kind in {**sch.required, **{c: k for c, k in sch.optional.items() if c in df.columns}}.items():
            bad = _kind_problem(df[col], kind)
            if bad:
                problems.append(f"column '{col}' {bad} (expected {KINDS[kind]})")
        if sch.unique and len(df):
            dup = df.duplicated(list(sch.unique), keep=False)
            if dup.any():
                ex = df.loc[dup, list(sch.unique)].head(3).to_dict("records")
                problems.append(f"{int(dup.sum())} rows share the same {sch.unique}, e.g. {ex}")
    if problems:
        raise SchemaError(f"{name}: " + "; ".join(problems))
    return df


def empty_frame(name: str) -> pd.DataFrame:
    """A zero-row frame with the documented columns (for 'CFBD returned nothing')."""
    sch = SCHEMAS[name]
    cols = {}
    for col, kind in sch.required.items():
        if kind == "bool":
            cols[col] = pd.Series(dtype=bool)
        elif kind == "datetime":
            cols[col] = pd.Series(dtype="datetime64[ns, UTC]")
        elif kind in ("id", "int_null", "float"):
            cols[col] = pd.Series(dtype=float)
        else:
            cols[col] = pd.Series(dtype=object)
    return pd.DataFrame(cols)


# --------------------------------------------------------------------- docs


def render_markdown() -> str:
    lines = ["# Data tables", "",
             "Generated from `cfbmodel/schema.py` (a test keeps this file in step). "
             "Columns CFBD sends that are not listed here are passed through unchanged.", ""]
    for sch in SCHEMAS.values():
        lines += [f"## `{sch.name}`", "", sch.doc, ""]
        if sch.required or sch.optional:
            lines += ["| column | meaning of the type | required |", "|---|---|---|"]
            for col, kind in sch.required.items():
                lines.append(f"| `{col}` | {KINDS[kind]} | yes |")
            for col, kind in sch.optional.items():
                lines.append(f"| `{col}` | {KINDS[kind]} | no |")
            lines.append("")
        if sch.unique:
            lines += [f"No two rows may share the same ({', '.join(sch.unique)}).", ""]
        if sch.raw_required:
            lines += ["Fields CFBD must send: " + ", ".join(f"`{k}`" for k in sch.raw_required) + ".", ""]
    return "\n".join(lines).rstrip() + "\n"
