"""
Read-only data access for the dashboard. Plain pandas, no Streamlit, no betting logic: it only reads
what the commands already wrote to output/ (and the cache folder) and shapes it for display. Every
function returns None or an empty frame when a file is missing, so a fresh install shows
"nothing here yet" pages instead of crashing.
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd

from cfbmodel import budget as budget_mod
from cfbmodel import config, learn, params, status


def _out(out) -> Path:
    return Path(out or config.OUTPUT)


def _read_json(p: Path):
    return json.loads(p.read_text()) if p.exists() else None


# ------------------------------------------------------------------ edge board


def list_boards(out=None) -> list[tuple[int, int, Path]]:
    """(season, week, csv path) for every saved board, newest first."""
    found = []
    for p in _out(out).glob("board_*_w*.csv"):
        m = re.fullmatch(r"board_(\d{4})_w(\d+)\.csv", p.name)
        if m:
            found.append((int(m.group(1)), int(m.group(2)), p))
    return sorted(found, reverse=True)


def load_board(season: int, week: int, out=None) -> tuple[pd.DataFrame | None, dict]:
    p = _out(out) / f"board_{season}_w{week}.csv"
    if not p.exists():
        return None, {}
    df = pd.read_csv(p)
    for col in ("book", "flags", "key_number_note", "best_line_across_books"):
        if col in df:
            df[col] = df[col].fillna("")
    return df, (_read_json(p.with_suffix(".json")) or {})


# ------------------------------------------------------------------ validation


def validation(out=None) -> dict:
    o = _out(out)
    v = _read_json(o / "validation_status.json")
    rp = o / "validation_report.md"
    return {"status": v, "report": rp.read_text() if rp.exists() else None,
            "age_days": status.validation_age_days(o / "validation_status.json") if v else None}


def markets_table(v: dict | None) -> pd.DataFrame:
    if not v:
        return pd.DataFrame()
    rows = [{"market": k, **{c: m.get(c) for c in ("n", "coef", "se", "p", "p_holdout", "passed", "status", "reason")}}
            for k, m in v["markets"].items()]
    return pd.DataFrame(rows)


def segments_table(v: dict | None, only_passed: bool = False, near_misses: bool = False) -> pd.DataFrame:
    if not v or not v.get("segments"):
        return pd.DataFrame()
    d = pd.DataFrame(v["segments"])
    if only_passed:
        return d[d["passed"]].reset_index(drop=True)
    if near_misses:        # significant on the train seasons but did not survive the holdout
        return d[(~d["passed"]) & (d["coef"] > 0) & (d["p_train"] < v.get("thresholds", {}).get("p_train", 0.05))].reset_index(drop=True)
    return d


# ------------------------------------------------------------------ CLV & bankroll


def settled_bets(out=None) -> pd.DataFrame | None:
    p = _out(out) / "bets_settled.csv"
    return pd.read_csv(p) if p.exists() else None


def clv_report_text(out=None) -> str | None:
    p = _out(out) / "clv_report.md"
    return p.read_text() if p.exists() else None


def bankroll_curve(settled: pd.DataFrame | None) -> pd.DataFrame:
    """Cumulative profit in the order bets were logged (settled bets only)."""
    if settled is None or settled.empty:
        return pd.DataFrame(columns=["bet_id", "cumulative_profit"])
    d = settled[settled["status"] == "settled"].sort_values("bet_id")
    return pd.DataFrame({"bet_id": d["bet_id"].to_numpy(), "cumulative_profit": d["profit"].cumsum().to_numpy()})


# ------------------------------------------------------------------ data health


def cache_freshness(cache=None, now: float | None = None) -> pd.DataFrame:
    """Per CFBD endpoint: how many cached answers and how old the newest one is."""
    c = Path(cache or config.CACHE)
    now = now or time.time()
    rows: dict[str, list[float]] = {}
    for f in c.glob("*__*.json"):
        rows.setdefault(f.name.split("__")[0], []).append(f.stat().st_mtime)
    return pd.DataFrame(sorted(
        ((k, len(v), round((now - max(v)) / 3600.0, 1)) for k, v in rows.items()),
        key=lambda r: r[0]), columns=["endpoint", "files", "newest_age_hours"])


def data_health(out=None, cache=None) -> dict:
    o = _out(out)
    c = Path(cache or config.CACHE)
    bud = budget_mod.Budget(c)
    unmatched = pd.read_csv(o / "unmatched_names.csv") if (o / "unmatched_names.csv").exists() else pd.DataFrame()
    try:
        version, models = params.active_version(), learn.models_table(output_dir=o)
    except params.ParamsError:
        version, models = None, pd.DataFrame()
    return {
        "calls_used": bud.used(), "calls_remaining": bud.remaining(), "limit": budget_mod.MONTHLY_LIMIT,
        "warn_at": budget_mod.WARN_AT, "refuse_at": budget_mod.REFUSE_AT,
        "freshness": cache_freshness(c), "unmatched": unmatched,
        "params_version": version, "models": models,
        "last_learn": _read_json(learn.models_dir(o) / "last_learn.json"),
        "drift": _read_json(o / "drift_status.json"),
    }
