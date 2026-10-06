"""
CFBD call budget (CLAUDE.md rule 5).

The free tier allows 1,000 calls a month. Every call that actually reaches CFBD
is counted in `<cache>/_budget.json`, keyed by calendar month (UTC), so the count
survives restarts and a new month starts from zero by itself.

  * at 700 calls a warning is printed (once per run);
  * at 950 calls new calls are refused unless the run was started with --force.

The 50-call gap below 1,000 is deliberate headroom: a refused call is annoying,
a blocked key for the rest of the month is worse.
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

MONTHLY_LIMIT = 1000
WARN_AT = 700
REFUSE_AT = 950


class BudgetExceeded(RuntimeError):
    """Raised when a call would go past the refuse threshold and --force is off."""


def current_month(now: datetime | None = None) -> str:
    return (now or datetime.now(timezone.utc)).strftime("%Y-%m")


class Budget:
    def __init__(self, cache_dir: Path, now_fn=None):
        self.path = Path(cache_dir) / "_budget.json"
        self._now = now_fn or (lambda: datetime.now(timezone.utc))
        self._warned = False

    # ------------------------------------------------------------- file io
    def _read(self) -> dict:
        if not self.path.exists():
            return {}
        return json.loads(self.path.read_text())

    def _write(self, data: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2, sort_keys=True))
        os.replace(tmp, self.path)

    # ------------------------------------------------------------- queries
    def used(self) -> int:
        return int(self._read().get(current_month(self._now()), 0))

    def remaining(self) -> int:
        return max(MONTHLY_LIMIT - self.used(), 0)

    # ------------------------------------------------------------- actions
    def check(self, force: bool = False) -> None:
        """Call before each network request. Warns at 700, refuses at 950."""
        used = self.used()
        if used >= REFUSE_AT and not force:
            raise BudgetExceeded(
                f"{used} CFBD calls used this month; the limit is {MONTHLY_LIMIT} and "
                f"new calls stop at {REFUSE_AT}. Wait for next month, or re-run with "
                "--force if you really mean it."
            )
        if used >= WARN_AT and not self._warned:
            self._warned = True
            print(f"WARNING: {used} of {MONTHLY_LIMIT} CFBD calls used this month "
                  f"({self.remaining()} left).", file=sys.stderr)

    def record(self, n: int = 1) -> int:
        """Count n calls that reached CFBD. Returns the new monthly total."""
        data = self._read()
        month = current_month(self._now())
        data[month] = int(data.get(month, 0)) + n
        self._write(data)
        return data[month]
