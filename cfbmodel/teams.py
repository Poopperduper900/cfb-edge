"""
Team-name canonicalisation.

One canonical name per team across every endpoint: the `school` name CFBD uses in
`/teams/fbs`. A name that does not match is NEVER dropped or guessed: it is kept
as it came, flagged `*_is_fbs = False`, and written to `output/unmatched_names.csv`
so you can see it. Most of those are FCS opponents (expected); a few may be real
spelling differences, which are fixed by adding a line to `data/team_aliases.csv`.

Matching is deterministic and conservative: exact name, then the alias file, then
the same name with accents, case, punctuation and extra spaces folded away. There
is no fuzzy matching, because a wrong guess silently mixes two teams' ratings.
"""
from __future__ import annotations

import re
import unicodedata
from collections import Counter
from pathlib import Path

import pandas as pd

from .config import OUTPUT, ROOT

ALIASES_FILE = ROOT / "data" / "team_aliases.csv"

# Names seen by `canonicalize_frame` in this process: {(source, column, name): rows}
_SEEN: Counter = Counter()


def fold(name: str) -> str:
    """Accent/case/punctuation-insensitive key: 'San José St.' -> 'sanjosest'."""
    s = unicodedata.normalize("NFKD", str(name))
    s = "".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9]", "", s.casefold())


def load_aliases(path: Path | None = None) -> dict[str, str]:
    p = Path(path) if path else ALIASES_FILE
    if not p.exists():
        return {}
    df = pd.read_csv(p, comment="#", skip_blank_lines=True)
    if not {"alias", "canonical"} <= set(df.columns):
        raise ValueError(f"{p} must have columns: alias,canonical")
    return dict(zip(df["alias"].astype(str), df["canonical"].astype(str)))


class TeamNames:
    def __init__(self, fbs_names, aliases: dict[str, str] | None = None):
        self.fbs = sorted({str(n) for n in fbs_names if pd.notna(n)})
        if not self.fbs:
            raise ValueError("TeamNames needs the FBS team list; /teams/fbs returned nothing")
        self._exact = set(self.fbs)
        self._folded = {}
        for n in self.fbs:
            self._folded.setdefault(fold(n), n)
        self.aliases = dict(load_aliases() if aliases is None else aliases)
        bad = {a: c for a, c in self.aliases.items() if c not in self._exact}
        if bad:
            raise ValueError(f"alias file points at names that are not FBS teams: {bad}")

    def canonical(self, name) -> str | None:
        """Canonical FBS name, or None if this is not (recognisably) an FBS team."""
        if name is None or (isinstance(name, float) and pd.isna(name)):
            return None
        name = str(name)
        if name in self._exact:
            return name
        if name in self.aliases:
            return self.aliases[name]
        return self._folded.get(fold(name))

    def canonicalize_frame(self, df: pd.DataFrame, cols: list[str], flag_cols: list[str] | None,
                           source: str) -> pd.DataFrame:
        """Replace names in `cols` by canonical ones where known and add a bool flag
        column for each name in `flag_cols` (same order as `cols`). Unknown names are
        kept unchanged, flagged False, and recorded."""
        df = df.copy()
        for i, col in enumerate(cols):
            if col not in df.columns:
                continue
            cache: dict = {}
            for nm in df[col].dropna().unique():
                cache[nm] = self.canonical(nm)
            mapped = df[col].map(cache)
            known = mapped.notna()
            for nm, n in df.loc[~known & df[col].notna(), col].value_counts().items():
                _SEEN[(source, col, str(nm))] += int(n)
            if flag_cols:
                df[flag_cols[i]] = known.astype(bool)
            df[col] = mapped.where(known, df[col])
        return df


# ------------------------------------------------------------- unmatched log


def seen_unmatched() -> dict[str, int]:
    """Unmatched names this process, summed over sources: {name: rows}."""
    out: Counter = Counter()
    for (_, _, name), n in _SEEN.items():
        out[name] += n
    return dict(out)


def write_unmatched_log(path: Path | None = None) -> Path:
    """Merge this process's unmatched names into output/unmatched_names.csv."""
    p = Path(path) if path else OUTPUT / "unmatched_names.csv"
    new = pd.DataFrame(
        [(s, c, n, rows) for (s, c, n), rows in sorted(_SEEN.items())],
        columns=["source", "column", "name", "rows"],
    )
    if p.exists():
        old = pd.read_csv(p)
        new = pd.concat([old, new], ignore_index=True)
        new = new.groupby(["source", "column", "name"], as_index=False)["rows"].max()
    p.parent.mkdir(parents=True, exist_ok=True)
    new.to_csv(p, index=False)
    return p


def reset_seen() -> None:
    _SEEN.clear()
