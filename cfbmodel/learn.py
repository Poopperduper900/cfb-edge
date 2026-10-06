"""
The learning loop (BUILD_PLAN Phase 6c-6e): the program improves its own formula, safely.

A learner that adopts whatever fit last week best chases noise and gets worse. So every parameter
group goes through the same gate:

  1. Fit a CHALLENGER on data before a held-out window of recent weeks.
  2. Limit it: no single parameter may move more than 25% in one cycle (a parameter that needs to
     go further gets there over several cycles).
  3. Score champion (the active version) and challenger on the window, which neither was fit on.
  4. Promote only if ALL hold:
       - the primary score improves, and the 95% bootstrap CI of the improvement excludes zero;
       - calibration does not get worse;
       - the window is big enough (300 games, or 2,000 player-games for prop parameters).
     Otherwise the champion stays and the report says exactly why.

Groups are judged independently; the new version contains only the groups that passed. A promotion
writes params/<version>.json, moves params/current.json, and writes output/models/<version>/ with
params.json, metrics.json and diff.md (a plain-English list of what changed, by how much, and the
evidence). `rollback` points current.json back at any earlier version.

Also here: the drift monitor (a model that suddenly gets worse than it was is switched off, w_model
= 0, until a `learn` run sees it recovered) and the weekly post-mortem (what drove the biggest
misses; read by the owner, never fed back into tuning, so one weird Saturday cannot move the model).
"""
from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from . import config, fitters, game_model, params, ratings, state, status, weather
from .params import P


class LearnError(RuntimeError):
    """The learning loop was asked to do something it cannot do."""


# ----------------------------------------------------------------------- data


@dataclass
class LearnData:
    """The frames the fitters need. Only `market` and `wf` are required.

    market   one row per game with a closing line: season, week, margin, actual_total,
             spread_close, total_close
    wf       the walk-forward frame from validation.walk_forward (model projections + components)
    weather  optional: forecast history joined to totals (see fitters.fit_weather)
    props    optional: player-games for rushing yards (see fitters.score_rush_props)
    """
    market: pd.DataFrame
    wf: pd.DataFrame
    weather: pd.DataFrame | None = None
    props: pd.DataFrame | None = None


# -------------------------------------------------------------- scoring helpers


def reliability_error(p: np.ndarray, y: np.ndarray, bins: int = 10) -> float:
    """Calibration error: the weighted average gap between predicted and realised frequency across
    probability bins (0 = perfectly calibrated). Lower is better."""
    p, y = np.asarray(p, float), np.asarray(y, float)
    if len(p) < bins * 5:
        bins = max(2, len(p) // 5)
    order = np.argsort(p, kind="stable")
    err = 0.0
    for chunk in np.array_split(order, bins):
        if len(chunk):
            err += len(chunk) * abs(p[chunk].mean() - y[chunk].mean())
    return float(err / len(p))


def bootstrap_ci(diff: np.ndarray, n_boot: int, seed: int = 7) -> tuple[float, float, float]:
    """(mean, lo, hi): mean of per-row differences and its 95% bootstrap interval."""
    diff = np.asarray(diff, float)
    rng = np.random.default_rng(seed)
    means = diff[rng.integers(0, len(diff), size=(n_boot, len(diff)))].mean(axis=1)
    return float(diff.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def bound_changes(champion: dict, changes: dict, max_step: float) -> dict:
    """Clip every proposed value to within +-max_step (relative) of the champion's value.
    A champion value of exactly 0 may only move by max_step * 0.01 (an absolute floor)."""
    def clip(old, new):
        lo, hi = sorted((old * (1 - max_step), old * (1 + max_step)))
        if old == 0:
            lo, hi = -max_step * 0.01, max_step * 0.01
        return float(min(max(new, lo), hi))

    out: dict = {}
    for group, items in changes.items():
        out[group] = {}
        for name, new in items.items():
            old = champion[group][name]
            if isinstance(new, dict):
                out[group][name] = {k: clip(old[k], v) for k, v in new.items()}
            else:
                out[group][name] = clip(old, new)
    return out


def diff_lines(champion: dict, changes: dict) -> list[tuple[str, float, float]]:
    rows = []
    for group, items in changes.items():
        for name, new in items.items():
            old = champion[group][name]
            if isinstance(new, dict):
                rows += [(f"{group}.{name}[{k}]", float(old[k]), float(v)) for k, v in new.items()]
            else:
                rows.append((f"{group}.{name}", float(old), float(new)))
    return rows


# ------------------------------------------------------------------ group specs


def _cover_y(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """(home covered the closing spread?, row is not a push)."""
    c = (df["margin"] + df["spread_close"]).to_numpy()
    return (c > 0).astype(float), c != 0


def _pmf_kwargs(source: str) -> dict:
    M = P.margin
    return dict(scale=M.scale_market if source == "market" else M.scale_model, df=M.df,
                sd_total_coef=M.sd_total_coef, key_weights={int(k): v for k, v in M.key_numbers.items()})


def _margin_pmf_rows(df: pd.DataFrame, exp_margin: np.ndarray, source: str) -> tuple[np.ndarray, np.ndarray]:
    """(per-row log score of the realised margin, home-cover probability) under ACTIVE parameters."""
    tot = df["total_close"].fillna(P.margin.sd_total_ref).to_numpy()
    kw = _pmf_kwargs(source)
    lp = fitters.margin_logprob(df["margin"].to_numpy(), exp_margin, tot, **kw)
    return lp, fitters.home_cover_prob(exp_margin, tot, df["spread_close"].to_numpy(), **kw)


def _pit(df, exp_margin, source) -> np.ndarray:
    tot = df["total_close"].fillna(P.margin.sd_total_ref).to_numpy()
    return fitters.margin_pit(df["margin"].to_numpy(), exp_margin, tot, **_pmf_kwargs(source))


def _score_market_margin(df):
    return _margin_pmf_rows(df, (-df["spread_close"]).to_numpy(), "market")[0]


def _model_exp_margin(df):
    home = (~df["neutral"].astype(bool)).to_numpy().astype(float)
    return df["core_model"].to_numpy() + P.margin.hfa_points * home


def _score_total(df, mult: float = 1.0):
    T = P.totals
    return fitters.total_logprob(df["actual_total"], df["total_close"], -df["spread_close"],
                                 sd_base=T.sd_base, sd_margin_coef=T.sd_margin_coef, mult=mult)


def _pit_total(df) -> np.ndarray:
    T = P.totals
    return fitters.total_pit(df["actual_total"], df["total_close"], -df["spread_close"],
                             sd_base=T.sd_base, sd_margin_coef=T.sd_margin_coef)


def _decay_prediction(df) -> np.ndarray:
    pw = np.array([state.early_season_prior_weight(int(w)) for w in df["week"]])
    home = (~df["neutral"].astype(bool)).to_numpy().astype(float) * P.margin.hfa_points
    return (1 - pw) * df["core_unshrunk"].to_numpy() + pw * df["core_prior"].to_numpy() + home


def _score_decay(df):
    return _margin_pmf_rows(df, _decay_prediction(df), "model")[0]


def _pit_decay(df) -> np.ndarray:
    return _pit(df, _decay_prediction(df), "model")


def _early_weeks(d: "LearnData"):
    """Weeks 0..early_last_week only: later weeks give the prior zero weight, so champion and
    challenger predict identically there and the rows would only dilute the comparison."""
    return d.wf[d.wf["week"] <= P.state.early_last_week]


def _score_weather(df):
    rows = []
    for r in df.itertuples(index=False):
        adj, _ = weather.total_adjustment({"dome": bool(r.dome), "wind_mph": r.wind_mph, "gust_mph": r.gust_mph,
                                           "precip_in": r.precip_in, "temp_f": r.temp_f,
                                           "wx_confidence": r.wx_confidence})
        rows.append(r.total_close + adj)
    T = P.totals
    return fitters.total_logprob(df["actual_total"], np.array(rows), np.zeros(len(df)),
                                 sd_base=T.sd_base, sd_margin_coef=0.0)


def _score_props(df):
    return -fitters.score_rush_props(df)["crps"]


def _pit_props(df) -> np.ndarray:
    return fitters.score_rush_props(df)["pits"]


@dataclass
class GroupSpec:
    name: str
    frame: Callable[[LearnData], pd.DataFrame | None]
    fit: Callable[[pd.DataFrame], fitters.FitResult]
    score: Callable[[pd.DataFrame], np.ndarray]          # per row, higher is better
    calibration: Callable[[pd.DataFrame], np.ndarray | None]   # PIT values (uniform = calibrated)
    min_n_key: str = "min_window_games"
    season_end: bool = False                             # only judged by a --full (season-end) run
    required: tuple = ()


SPECS = [
    GroupSpec("margin_pmf", lambda d: d.market, fitters.fit_margin_pmf, _score_market_margin,
              lambda df: _pit(df, (-df["spread_close"]).to_numpy(), "market"),
              required=("margin", "spread_close", "total_close")),
    GroupSpec("model_margin", lambda d: d.wf, fitters.fit_model_margin,
              lambda df: _margin_pmf_rows(df, _model_exp_margin(df), "model")[0],
              lambda df: _pit(df, _model_exp_margin(df), "model"),
              required=("margin", "core_model", "neutral", "spread_close", "total_close")),
    GroupSpec("totals", lambda d: d.market, fitters.fit_totals, _score_total, _pit_total,
              required=("actual_total", "total_close", "spread_close")),
    GroupSpec("early_decay", _early_weeks, fitters.fit_decay, _score_decay, _pit_decay, season_end=True,
              required=("margin", "week", "core_unshrunk", "core_prior", "neutral", "spread_close", "total_close")),
    GroupSpec("weather", lambda d: d.weather, fitters.fit_weather, _score_weather,
              lambda df: None, required=("actual_total", "total_close", "wind_mph", "dome")),
    GroupSpec("script_proe", lambda d: d.props, fitters.fit_proe, _score_props, _pit_props,
              min_n_key="min_window_player_games", required=("exp_margin", "actual_yards")),
]


# ------------------------------------------------------------------- decisions


@dataclass
class GroupResult:
    name: str
    status: str                       # PROMOTED | KEPT | SKIPPED
    reasons: list[str]
    changes: dict = field(default_factory=dict)       # bounded challenger changes (what would change)
    evidence: dict = field(default_factory=dict)


def decide(mean_gain: float, ci_lo: float, cal_champ: float, cal_chal: float, n_window: int,
           min_n: int, cal_se: float = 0.0) -> tuple[bool, list[str]]:
    """The promotion gate. Returns (promote, reasons the champion stays; empty if promoted).

    Calibration is the PIT-uniformity distance (0 = perfectly calibrated). It "gets worse" when the
    challenger's distance exceeds the champion's by more than `cal_tolerance` AND by more than
    1.645 standard errors of that paired difference, so ordinary sampling noise in a 480-game
    window cannot block a genuine improvement, while a real deterioration still does."""
    L = P.learn
    why = []
    if n_window < min_n:
        why.append(f"window too small ({n_window} < {min_n})")
    if not (mean_gain > L.min_improvement):
        why.append(f"primary score did not improve enough (gain {mean_gain:+.5f}, need > {L.min_improvement})")
    if not (ci_lo > 0):
        why.append(f"95% CI of the improvement includes zero (lower bound {ci_lo:+.5f})")
    if cal_chal - cal_champ > max(L.cal_tolerance, 1.645 * cal_se):
        why.append(f"calibration got worse ({cal_champ:.4f} -> {cal_chal:.4f})")
    return (not why), why


def split_window(frame: pd.DataFrame, calendar: list[tuple[int, int]], season_end: bool):
    """(train, window): the window is the last `window_weeks` calendar weeks (or, for season-end
    groups, the latest season); everything earlier is training."""
    key = list(zip(frame["season"].astype(int), frame["week"].astype(int)))
    if season_end:
        latest = max(s for s, _ in key)
        in_win = np.array([s == latest for s, _ in key])
    else:
        win = set(calendar[-int(P.learn.window_weeks):])
        in_win = np.array([k in win for k in key])
    return frame[~in_win], frame[in_win]


def evaluate_group(spec: GroupSpec, data: LearnData, calendar, seed: int) -> GroupResult:
    frame = spec.frame(data)
    if frame is None or len(frame) == 0:
        return GroupResult(spec.name, "SKIPPED", ["no data supplied for this group"])
    missing = [c for c in spec.required if c not in frame.columns]
    if missing:
        return GroupResult(spec.name, "SKIPPED", [f"missing columns {missing}"])
    frame = frame.dropna(subset=list(spec.required))
    train, window = split_window(frame, calendar, spec.season_end)
    L = P.learn
    min_n = int(getattr(L, spec.min_n_key))
    if len(window) < min_n:
        return GroupResult(spec.name, "SKIPPED", [f"window too small ({len(window)} < {min_n})"],
                           evidence={"n_window": len(window)})
    if len(train) < min_n:
        return GroupResult(spec.name, "SKIPPED", [f"not enough training data ({len(train)} < {min_n})"],
                           evidence={"n_train": len(train)})

    champion = params.active()
    fit = spec.fit(train)
    changes = bound_changes(champion, fit.changes, L.max_step)

    s_champ, pit_champ = spec.score(window), spec.calibration(window)
    with params.override(changes):
        s_chal, pit_chal = spec.score(window), spec.calibration(window)
    mean_gain, lo, hi = bootstrap_ci(s_chal - s_champ, int(L.bootstrap_n), seed)
    if pit_champ is None:
        c_champ = c_chal = cal_se = 0.0
    else:
        c_champ, c_chal = fitters.pit_uniformity(pit_champ), fitters.pit_uniformity(pit_chal)
        rng = np.random.default_rng(seed + 1)
        diffs = []
        for _ in range(300):                                       # paired bootstrap of the calibration gap
            idx = rng.integers(0, len(pit_champ), len(pit_champ))
            diffs.append(fitters.pit_uniformity(pit_chal[idx]) - fitters.pit_uniformity(pit_champ[idx]))
        cal_se = float(np.std(diffs))
    promote, why = decide(mean_gain, lo, c_champ, c_chal, len(window), min_n, cal_se)
    evidence = {"n_train": len(train), "n_window": len(window), "mean_gain": mean_gain, "ci_low": lo,
                "ci_high": hi, "calibration_champion": c_champ, "calibration_challenger": c_chal,
                "calibration_se": cal_se,
                "fit_info": {k: v for k, v in fit.info.items() if isinstance(v, (int, float, str))}}
    return GroupResult(spec.name, "PROMOTED" if promote else "KEPT", why, changes, evidence)


# --------------------------------------------------------------------- versions


def models_dir(output_dir: Path | None = None) -> Path:
    return Path(output_dir or config.OUTPUT) / "models"


def list_versions(directory: Path | None = None) -> list[str]:
    d = Path(directory or params.params_dir())
    return sorted(p.stem for p in d.glob("v[0-9][0-9][0-9][0-9].json"))


def next_version(directory: Path | None = None) -> str:
    vs = list_versions(directory)
    return f"v{int(vs[-1][1:]) + 1:04d}" if vs else "v0001"


def _diff_markdown(version: str, parent: str, asof: tuple[int, int], results: list[GroupResult],
                   champion: dict) -> str:
    L = [f"# {version}: what changed and why", "",
         f"Parent: {parent}. Learned with data through season {asof[0]}, week {asof[1] - 1}. "
         "Only groups that passed every test are listed as changed; the rest stayed as they were.", ""]
    for r in results:
        if r.status != "PROMOTED":
            continue
        e = r.evidence
        L += [f"## {r.name}", ""]
        for name, old, new in diff_lines(champion, r.changes):
            pct = (new - old) / old * 100 if old else float("nan")
            L.append(f"- `{name}`: {old:.4g} -> {new:.4g} ({pct:+.1f}%)")
        L += ["", f"Evidence: on {e['n_window']} games the model has not seen, average score improved by "
                  f"{e['mean_gain']:+.5f} per game (95% interval {e['ci_low']:+.5f} to {e['ci_high']:+.5f}, "
                  f"so it is not zero). Calibration error {e['calibration_champion']:.4f} -> "
                  f"{e['calibration_challenger']:.4f} (not worse). No parameter moved more than "
                  f"{P.learn.max_step:.0%} in this cycle.", ""]
    skipped = [r for r in results if r.status != "PROMOTED"]
    if skipped:
        L += ["## Not changed", ""] + [f"- {r.name}: {r.status.lower()}; {'; '.join(r.reasons) or 'n/a'}" for r in skipped]
    return "\n".join(L) + "\n"


def write_version(champion: dict, results: list[GroupResult], asof: tuple[int, int], directory: Path | None = None,
                  output_dir: Path | None = None, today: str | None = None) -> str:
    """Create the next version from the champion plus the promoted changes and make it current."""
    d = Path(directory or params.params_dir())
    parent = params.current_version(d)
    new = copy.deepcopy(champion)
    for r in results:
        if r.status == "PROMOTED":
            for g, items in r.changes.items():
                new[g].update(items)
    version = next_version(d)
    created = today or date.today().isoformat()
    promoted = [r.name for r in results if r.status == "PROMOTED"]
    full = {"_meta": {"version": version, "parent": parent, "created": created,
                      "note": f"learned: {', '.join(promoted)} (data through season {asof[0]} week {asof[1] - 1})"},
            **new}
    (d / f"{version}.json").write_text(json.dumps(full, indent=2, sort_keys=False) + "\n")
    (d / "current.json").write_text(json.dumps({"version": version}) + "\n")
    params.reload()

    out = models_dir(output_dir) / version
    out.mkdir(parents=True, exist_ok=True)
    (out / "params.json").write_text(json.dumps(full, indent=2) + "\n")
    (out / "metrics.json").write_text(json.dumps({
        "version": version, "parent": parent, "created": created, "asof_season": asof[0],
        "asof_week": asof[1], "groups": {r.name: {"status": r.status, "reasons": r.reasons, **r.evidence}
                                         for r in results}}, indent=2, sort_keys=True, default=float) + "\n")
    (out / "diff.md").write_text(_diff_markdown(version, parent, asof, results, champion))
    return version


def models_table(directory: Path | None = None, output_dir: Path | None = None) -> pd.DataFrame:
    d = Path(directory or params.params_dir())
    cur = params.current_version(d)
    rows = []
    for v in list_versions(d):
        meta = params.read_version(v, d)["_meta"]
        m_path = models_dir(output_dir) / v / "metrics.json"
        gains = {}
        if m_path.exists():
            gains = {k: g.get("mean_gain") for k, g in json.loads(m_path.read_text())["groups"].items()
                     if g.get("status") == "PROMOTED"}
        rows.append({"version": v, "current": "*" if v == cur else "", "created": meta.get("created"),
                     "parent": meta.get("parent"), "what": meta.get("note", ""),
                     "gains": ", ".join(f"{k} {x:+.4f}" for k, x in gains.items())})
    return pd.DataFrame(rows)


def rollback(version: str, directory: Path | None = None, output_dir: Path | None = None) -> str:
    """Make `version` current again. Version files are never deleted, so you can roll forward too."""
    d = Path(directory or params.params_dir())
    if version not in list_versions(d):
        raise LearnError(f"unknown version {version!r}; known versions: {', '.join(list_versions(d))}")
    previous = params.current_version(d)
    (d / "current.json").write_text(json.dumps({"version": version}) + "\n")
    params.reload()
    hist = models_dir(output_dir) / "history.jsonl"
    hist.parent.mkdir(parents=True, exist_ok=True)
    with hist.open("a") as f:
        f.write(json.dumps({"event": "rollback", "from": previous, "to": version, "date": date.today().isoformat()}) + "\n")
    return previous


# ------------------------------------------------------------------ the main run


@dataclass
class LearnReport:
    asof: tuple[int, int]
    due: bool
    results: list[GroupResult]
    version: str | None
    notes: list[str] = field(default_factory=list)

    @property
    def promoted(self) -> list[str]:
        return [r.name for r in self.results if r.status == "PROMOTED"]


def _calendar(market: pd.DataFrame) -> list[tuple[int, int]]:
    return sorted(set(zip(market["season"].astype(int), market["week"].astype(int))))


def _as_of(df: pd.DataFrame | None, season: int, week: int):
    return None if df is None else ratings.as_of(df, season, week)


def learn_due(season: int, week: int, last: dict | None, force: bool = False) -> tuple[bool, str]:
    """Learning runs every `every_n_weeks` in-season (and once in full after the season)."""
    if force or not last:
        return True, "forced" if force else "never run"
    gap = (season - last["season"]) * 52 + (week - last["week"])
    if gap >= P.learn.every_n_weeks:
        return True, f"{gap} weeks since the last run"
    return False, f"only {gap} week(s) since the last run; next due at week {last['week'] + int(P.learn.every_n_weeks)}"


def run_learn(data: LearnData, season: int, week: int, *, full: bool = False, force: bool = False,
              directory: Path | None = None, output_dir: Path | None = None, seed: int = 7,
              today: str | None = None) -> LearnReport:
    """Learn using ONLY data strictly before (season, week). Writes a new version if (and only if)
    some group passes the promotion gate; always writes output/learn_report_<season>_w<week>.md."""
    out = Path(output_dir or config.OUTPUT)
    last_path = models_dir(out) / "last_learn.json"
    last = json.loads(last_path.read_text()) if last_path.exists() else None
    due, why_due = learn_due(season, week, last, force)
    if not due:
        return LearnReport((season, week), False, [], None, [f"not due: {why_due}"])

    d = LearnData(_as_of(data.market, season, week), _as_of(data.wf, season, week),
                  _as_of(data.weather, season, week), _as_of(data.props, season, week))
    if d.market is None or len(d.market) == 0:
        raise LearnError("no games before the as-of point; nothing to learn from")
    calendar = _calendar(d.market)
    champion = copy.deepcopy(params.active())

    results = [evaluate_group(spec, d, calendar, seed) for spec in SPECS if full or not spec.season_end]
    version = None
    if any(r.status == "PROMOTED" for r in results):
        version = write_version(champion, results, (season, week), directory, out, today)
    notes = [f"learning run: {why_due}"]

    out.mkdir(parents=True, exist_ok=True)
    (out / f"learn_report_{season}_w{week}.md").write_text(report_markdown(season, week, results, version, champion, full))
    last_path.parent.mkdir(parents=True, exist_ok=True)
    last_path.write_text(json.dumps({"season": season, "week": week}))
    return LearnReport((season, week), True, results, version, notes)


def report_markdown(season, week, results, version, champion, full) -> str:
    L = [f"# Learning run, season {season} week {week}", "",
         ("**A new parameter version was written: " + version + "**" if version else
          "**No change: the current parameters stay.** That is the normal outcome. A learner that "
          "changes the model every cycle is chasing noise."), ""]
    for r in results:
        L.append(f"## {r.name}: {r.status}")
        e = r.evidence
        if r.status != "SKIPPED" and e:
            L.append(f"- window {e['n_window']} games, trained on {e['n_train']}; gain {e['mean_gain']:+.5f}/game, "
                     f"95% CI [{e['ci_low']:+.5f}, {e['ci_high']:+.5f}]; calibration {e['calibration_champion']:.4f} -> "
                     f"{e['calibration_challenger']:.4f}")
        for name, old, new in (diff_lines(champion, r.changes) if r.status != "SKIPPED" else []):
            L.append(f"  - would change `{name}`: {old:.4g} -> {new:.4g}")
        L += [f"- why not: {w}" for w in r.reasons]
        L.append("")
    if not full:
        L.append("(Season-end groups, such as the early-season decay, are judged only by `learn --full`.)")
    L.append("w_model is not learned here: it comes from `validate` on its pre-registered seasons.")
    return "\n".join(L) + "\n"


# ------------------------------------------------------------------ drift monitor


def _recent_and_baseline(wf: pd.DataFrame, season: int, week: int):
    d = ratings.as_of(wf, season, week)
    cal = sorted(set(zip(d["season"].astype(int), d["week"].astype(int))))
    D = P.drift
    recent_w, base_w = cal[-int(D.recent_weeks):], cal[-int(D.recent_weeks) - int(D.baseline_weeks):-int(D.recent_weeks)]
    key = list(zip(d["season"].astype(int), d["week"].astype(int)))
    pick = lambda ws: d[[k in set(ws) for k in key]]
    return pick(recent_w), pick(base_w)


def _drift_metrics(df: pd.DataFrame) -> dict:
    model_mae = float((df["margin"] - df["model_margin"]).abs().mean())
    market_mae = float((df["margin"] + df["spread_close"]).abs().mean())
    tot = df["total_close"].fillna(P.margin.sd_total_ref) if "total_close" in df else pd.Series(P.margin.sd_total_ref, index=df.index)
    p = np.array([game_model.cover_prob(float(m), float(t), float(s), source="model")["home"]
                  for m, t, s in zip(df["model_margin"], tot, df["spread_close"])])
    y, ok = _cover_y(df)
    return {"n": int(len(df)), "mae_ratio": model_mae / market_mae, "model_mae": model_mae, "market_mae": market_mae,
            "cal_error": reliability_error(p[ok], y[ok])}


def check_drift(wf: pd.DataFrame, season: int, week: int) -> dict:
    """Compare the last `recent_weeks` of model-vs-market accuracy and calibration with the weeks
    before them. The model is normally WORSE than the closing line (MAE ~13 vs ~10.5), so the test
    is whether it got worse relative to ITS OWN recent past, not whether it beats the market."""
    D = P.drift
    recent, base = _recent_and_baseline(wf, season, week)
    if len(recent) < D.min_games or len(base) < D.min_games:
        return {"active": False, "reason": "not enough games to judge drift", "n_recent": len(recent), "n_baseline": len(base)}
    r, b = _drift_metrics(recent), _drift_metrics(base)
    reasons = []
    if (r["mae_ratio"] - b["mae_ratio"]) / b["mae_ratio"] > D.mae_ratio_increase:
        reasons.append(f"model/market MAE ratio rose from {b['mae_ratio']:.3f} to {r['mae_ratio']:.3f}")
    if r["cal_error"] - b["cal_error"] > D.cal_error_increase:
        reasons.append(f"calibration error rose from {b['cal_error']:.3f} to {r['cal_error']:.3f}")
    return {"active": bool(reasons), "reason": "; ".join(reasons) or "no drift", "recent": r, "baseline": b,
            "asof_season": season, "asof_week": week}


def write_drift_status(result: dict, path: Path | None = None) -> Path:
    p = Path(path or status.drift_path())
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(result, indent=2, sort_keys=True, default=float) + "\n")
    return p


# ------------------------------------------------------------------ post-mortem


def postmortem(wf: pd.DataFrame, season: int, week: int, top: int = 8) -> str:
    """What drove the biggest spread misses of one week. For the owner to read; it never feeds back
    into tuning, because reacting to one Saturday is how a model overreacts."""
    d = wf[(wf["season"] == season) & (wf["week"] == week)].copy()
    L = [f"# Post-mortem: {season} week {week}", "",
         "For reading only. Nothing here changes the model: one weird Saturday is not evidence.", ""]
    if d.empty:
        return "\n".join(L + ["No games for this week."]) + "\n"
    d["model_miss"] = d["margin"] - d["model_margin"]
    d["market_miss"] = d["margin"] + d["spread_close"]
    comp = pd.DataFrame({
        "pulled toward the preseason prior": d["core_model"] - d["core_unshrunk"],
        "team ratings (model vs market)": d["core_unshrunk"] - d["core_market"],
        "home-field number": d["hfa_used"] - d["market_hfa"],
    })
    d["driver"] = [
        ("the market missed it too (variance, not model error)" if abs(mk) >= 0.8 * abs(mm) else
         f"{comp.loc[i].abs().idxmax()} ({comp.loc[i, comp.loc[i].abs().idxmax()]:+.1f} pts)")
        for i, mm, mk in zip(d.index, d["model_miss"], d["market_miss"])]
    worst = d.reindex(d["model_miss"].abs().sort_values(ascending=False).index).head(top)
    L += ["## Biggest spread misses", "",
          "| game | model | market | actual | model miss | market miss | biggest driver |", "|---|---|---|---|---|---|---|"]
    for r in worst.itertuples():
        L.append(f"| {r.away} @ {r.home} | {r.model_margin:+.1f} | {-r.spread_close:+.1f} | {r.margin:+.0f} | "
                 f"{r.model_miss:+.1f} | {r.market_miss:+.1f} | {r.driver} |")
    L += ["", f"Week MAE: model {d['model_miss'].abs().mean():.2f}, closing line {d['market_miss'].abs().mean():.2f} "
              f"over {len(d)} games.", ""]
    if "model_total" in d.columns and d["model_total"].notna().any():
        t = d.dropna(subset=["model_total"]).assign(miss=lambda x: x["actual_total"] - x["model_total"])
        t = t.reindex(t["miss"].abs().sort_values(ascending=False).index).head(top // 2)
        L += ["## Biggest total misses", "", "| game | model total | closing total | actual | miss |", "|---|---|---|---|---|"]
        L += [f"| {r.away} @ {r.home} | {r.model_total:.1f} | {r.total_close:.1f} | {r.actual_total:.0f} | {r.miss:+.1f} |"
              for r in t.itertuples()]
    return "\n".join(L) + "\n"


def write_postmortem(text: str, season: int, week: int, output_dir: Path | None = None) -> Path:
    p = Path(output_dir or config.OUTPUT) / f"postmortem_w{week}.md"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)
    return p
