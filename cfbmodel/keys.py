"""
API key loading.

The CFBD key lives in a local `.env` file (git-ignored) or in the CFBD_API_KEY
environment variable. The environment variable wins, so a key set in the shell
or in Colab secrets always overrides the file.

Rules this module exists to enforce (see CLAUDE.md rule 4):
  * The key is never printed, logged, or put in an error message.
  * A missing key is a loud, readable error, not a stack trace and not a
    silent fallback to "no data".
  * The key is looked up only when a network call is actually needed, so
    tests and cached re-runs work with no key at all.
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = ROOT / ".env"
KEY_NAME = "CFBD_API_KEY"


class MissingKeyError(RuntimeError):
    """Raised when a CFBD call needs a key and none is configured."""


def get_cfbd_key(env_file: Path | None = None) -> str:
    """Return the CFBD key from the environment, else from `.env`.

    `env_file` exists so tests can point at a temporary file; normal code
    leaves it unset.
    """
    path = ENV_FILE if env_file is None else Path(env_file)

    key = os.environ.get(KEY_NAME, "").strip()
    if not key and path.is_file():
        # dotenv_values reads the file without touching os.environ, so loading
        # the key here cannot leak it into child processes by accident.
        key = (dotenv_values(path).get(KEY_NAME) or "").strip()

    if not key:
        raise MissingKeyError(
            f"{KEY_NAME} is not set.\n"
            "  1. Get a free key at https://collegefootballdata.com/key\n"
            f"  2. Copy .env.example to .env and put the key after {KEY_NAME}=\n"
            f"     (expected file: {path})\n"
            "  The key never goes in chat, in code, or in git."
        )
    return key
