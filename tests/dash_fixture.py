"""A realistic output/ folder for dashboard tests (built with the real writers, not hand-typed)."""
from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path

import pandas as pd

from cfbmodel import budget, tracking, validation
from test_tracking import _many_settled
from test_validation import make_res
import board_fixture as F
from cfbmodel import board

FIXTURES = Path(__file__).resolve().parent / "fixtures"


def make_output(tmp: Path) -> tuple[Path, Path]:
    out, cache = tmp / "output", tmp / "cache"
    out.mkdir()
    cache.mkdir()
    # boards (two weeks) with banners next to the CSV
    for week in (5, 6):
        shutil.copy(FIXTURES / "board_golden_w6.csv", out / f"board_2026_w{week}.csv")
    b = board.build_board(2026, 6, F.games(), F.lines(), F.team_state(), F.total_ratings(), validation_status=None,
                          drift_active=False, params_version="v0001", now=F.NOW)
    (out / "board_2026_w6.json").write_text(json.dumps(board.to_meta(b)))
    # validation
    res = make_res(informative=True, n=700)
    st = validation.run_validation(res, generated=pd.Timestamp.now(tz="UTC").date().isoformat())
    validation.write_outputs(st, validation.build_report(st, res), out)
    # settled bets and the CLV report
    settled, games = _many_settled(30)
    settled.to_csv(out / "bets_settled.csv", index=False)
    (out / "clv_report.md").write_text(tracking.report(settled, games))
    # data health inputs
    pd.DataFrame([{"source": "games", "column": "awayTeam", "name": "Tiny FCS", "rows": 12}]).to_csv(out / "unmatched_names.csv", index=False)
    budget.Budget(cache).record(42)
    for name, age_h in (("games__aaaa.json", 2), ("plays__bbbb.json", 30)):
        f = cache / name
        f.write_text("[]")
        os.utime(f, (time.time() - age_h * 3600,) * 2)
    return out, cache
