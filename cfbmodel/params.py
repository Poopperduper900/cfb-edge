"""
The parameter registry (BUILD_PLAN Phase 6a).

Every tunable number in the model lives in a versioned file, params/<version>.json, and
params/current.json says which version is active. Model code reads numbers only through this
module (`P.margin.scale_market`, or the legacy `config.C.hfa_points`), never from literals, so a
version change really changes the model and a rollback really restores it.

  params/v0001.json   the starting values (most were hand-set, not fitted)
  params/v0002.json   written by `python -m cfbmodel learn` only when a challenger is proven better
  params/current.json {"version": "v0002"}

`override()` swaps parameters for the duration of a `with` block without touching any file; the
learning loop uses it to score candidate parameters, and tests use it to prove the code reads them.
"""
from __future__ import annotations

import copy
import json
import os
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

_LOADED: dict = {}          # {(dir, version): groups}; cleared by reload()
_STACK: list[dict] = []     # override() stack of full parameter dicts


class ParamsError(RuntimeError):
    """The parameter files are missing or malformed."""


def params_dir() -> Path:
    return Path(os.environ.get("CFBMODEL_PARAMS") or ROOT / "params")


def version_path(version: str, directory: Path | None = None) -> Path:
    return Path(directory or params_dir()) / f"{version}.json"


def current_version(directory: Path | None = None) -> str:
    p = Path(directory or params_dir()) / "current.json"
    if not p.exists():
        raise ParamsError(f"{p} is missing; it must name the active parameter version.")
    return json.loads(p.read_text())["version"]


def read_version(version: str, directory: Path | None = None) -> dict:
    """The full file for a version, including `_meta`."""
    p = version_path(version, directory)
    if not p.exists():
        raise ParamsError(f"parameter version {version!r} not found at {p}")
    return json.loads(p.read_text())


def groups_of(full: dict) -> dict:
    return {k: v for k, v in full.items() if not k.startswith("_")}


def reload() -> None:
    """Forget cached files (after current.json or a version file changed in this process)."""
    _LOADED.clear()


def active() -> dict:
    """The active parameter groups: the innermost override(), else the version in current.json."""
    if _STACK:
        return _STACK[-1]
    d = params_dir()
    key = (str(d), current_version(d))
    if key not in _LOADED:
        _LOADED[key] = groups_of(read_version(key[1], d))
    return _LOADED[key]


def active_version() -> str:
    return current_version()


@contextmanager
def override(changes: dict | None = None, base: dict | None = None):
    """Use different parameters inside the block. `changes` is {group: {name: value}} merged over
    `base` (default: the active parameters). Nothing is written to disk."""
    new = copy.deepcopy(base if base is not None else active())
    for group, items in (changes or {}).items():
        new.setdefault(group, {}).update(items)
    _STACK.append(new)
    try:
        yield new
    finally:
        _STACK.pop()


class _Group:
    def __init__(self, name: str):
        self._name = name

    def __getattr__(self, key: str):
        group = active().get(self._name)
        if group is None or key not in group:
            raise AttributeError(f"parameter {self._name}.{key} is not in version {active_version()}")
        return group[key]


class _Params:
    """P.margin.scale_market  ->  the active value."""

    def __getattr__(self, group: str) -> _Group:
        if group.startswith("_"):
            raise AttributeError(group)
        return _Group(group)


P = _Params()


# Names the older code used on config.C, mapped to (group, name) in the registry.
_LEGACY = {
    "hfa_points": ("margin", "hfa_points"), "hfa_neutral": ("margin", "hfa_neutral"),
    "margin_scale_market": ("margin", "scale_market"), "margin_scale_model": ("margin", "scale_model"),
    "margin_sd_total_coef": ("margin", "sd_total_coef"), "margin_df": ("margin", "df"),
    "total_sd_base": ("totals", "sd_base"), "total_sd_model_mult": ("totals", "sd_model_mult"),
    "gt_wp_low": ("ratings", "gt_wp_low"), "gt_wp_high": ("ratings", "gt_wp_high"),
    "gt_min_quarter_4_margin": ("ratings", "gt_min_q4_margin"),
    "form_half_life_games": ("ratings", "form_half_life_games"),
    "season_carryover": ("ratings", "season_carryover"),
    "ridge_alpha_off": ("ratings", "ridge_alpha_off"), "ridge_alpha_def": ("ratings", "ridge_alpha_def"),
    "pace_mean": ("ratings", "pace_mean"),
    "min_edge_spread": ("betting", "min_edge_spread"), "min_edge_total": ("betting", "min_edge_total"),
    "min_ev_props": ("betting", "min_ev_props"), "kelly_fraction": ("betting", "kelly_fraction"),
    "max_bet_pct": ("betting", "max_bet_pct"), "edge_flat_haircut": ("betting", "edge_flat_haircut"),
    "sample_conf_k": ("betting", "sample_conf_k"),
}


class Constants:
    """config.C: a live view of the active parameter version under the older attribute names."""

    def __getattr__(self, name: str):
        if name == "key_numbers":
            return {int(k): v for k, v in active()["margin"]["key_numbers"].items()}
        if name in _LEGACY:
            group, key = _LEGACY[name]
            return active()[group][key]
        raise AttributeError(name)
