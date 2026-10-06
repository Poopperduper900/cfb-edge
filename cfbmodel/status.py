"""
Reading the two "permission" files that decide how much the model may matter.

output/validation_status.json  written by `validate` (Phase 4). Says, per market and
                               segment, whether the model has PROVEN it adds information
                               over the betting line, and how much weight it earned.
output/drift_status.json       written by `learn` / the drift monitor (Phase 6). When
                               `"active": true`, model weight is forced to 0 until the next
                               successful `learn`.

Rule of this project: the market until proven otherwise. A missing file, a missing key, or an
active drift alarm all mean w_model = 0.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from . import config


def validation_path() -> Path:
    return config.OUTPUT / "validation_status.json"


def drift_path() -> Path:
    return config.OUTPUT / "drift_status.json"


def read_json(path: Path | None) -> dict | None:
    if path is None or not Path(path).exists():
        return None
    return json.loads(Path(path).read_text())


def read_validation(path: Path | None = None) -> dict | None:
    return read_json(path or validation_path())


def drift_active(path: Path | None = None) -> bool:
    d = read_json(path or drift_path())
    return bool(d and d.get("active"))


def w_model_for(market: str = "spread_vs_close", segment: str = "all",
                path: Path | None = None, drift_file: Path | None = None) -> float:
    """Weight (0..1) the model has earned for this market/segment. 0 unless proven."""
    if drift_active(drift_file):
        return 0.0
    v = read_validation(path)
    if not v:
        return 0.0
    w = (v.get("w_model") or {}).get(f"{market}|{segment}", 0.0)
    return float(min(max(float(w), 0.0), 1.0))


def validation_age_days(path: Path | None = None, now: datetime | None = None) -> float | None:
    """Days since validation_status.json was generated, or None if there is no file."""
    v = read_validation(path)
    if not v or not v.get("generated"):
        return None
    gen = datetime.fromisoformat(v["generated"])
    if gen.tzinfo is None:
        gen = gen.replace(tzinfo=timezone.utc)
    return ((now or datetime.now(timezone.utc)) - gen).total_seconds() / 86400.0
