"""
The validation gate (BUILD_PLAN Phase 4): may the model influence a bet at all?

Question asked, for every market:    margin = b0 + b1 * market + b2 * model + error
If b2 is not clearly above zero, the model contains nothing the betting line has not already
priced, and the right answer is "the market until proven otherwise". That is a valid and
expected outcome (the owner's NFL model got exactly that).

Rules fixed IN ADVANCE (PREREGISTERED below; changing them is a code change, on purpose):

  * train seasons 2022-2024, holdout 2025. 2026-to-date is REPORTED and never used to decide.
  * Heteroskedasticity-robust standard errors (HC3).
  * A market or segment passes only if the model coefficient is positive and significant on the
    train seasons (p < 0.05) AND keeps its sign with p < 0.10 on the 2025 holdout. Testing dozens
    of segments produces false positives by chance; the holdout requirement is the guard.
  * A model coefficient above 0.5 is not plausible for a real edge. It sets status
    SUSPECTED_LEAK (look for lookahead) and fails.

Outputs: output/validation_status.json (read by the blend and the board) and
output/validation_report.md. Both are deterministic given the same cached data.
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from . import backtest, config, game_model, ratings, state
from .config import C

PREREGISTERED = {
    "train_seasons": [2022, 2023, 2024],
    "holdout_season": 2025,
    "report_season": 2026,          # reported, never used for decisions
    "p_train": 0.05,
    "p_holdout": 0.10,
    "leak_coef": 0.5,
    "min_n_train": 100,             # fewer games than this cannot be judged
    "min_n_holdout": 30,
    "early_weeks": [0, 1, 2, 3],    # judged with preseason ratings; week 4+ with in-season ratings
}

MARKETS = {
    "spread_vs_open": dict(y="margin", market="market_margin_open", model="model_margin", line="spread_open"),
    "spread_vs_close": dict(y="margin", market="market_margin_close", model="model_margin", line="spread_close"),
    "total_vs_open": dict(y="actual_total", market="total_open", model="model_total", line="spread_open"),
    "total_vs_close": dict(y="actual_total", market="total_close", model="model_total", line="spread_close"),
}

POWER_CONFS = {"SEC", "Big Ten", "Big 12", "ACC"}
POWER_CONFS_BY_SEASON = {y: POWER_CONFS | {"Pac-12"} for y in range(2000, 2024)}   # Pac-12 was Power through 2023
EDGE_BUCKETS = [0, 1, 2, 3, 5, 8, 100]
ASSUMED_PRICE = -110   # CFBD carries no prices, so ROI below assumes -110 and says so


# -------------------------------------------------------------------- walk-forward


def _market_numbers(lines: pd.DataFrame) -> pd.DataFrame:
    cols = ["spread_close", "spread_open", "total_close", "total_open"]
    return lines.groupby("gameId")[cols].median()


def walk_forward(games: pd.DataFrame, lines: pd.DataFrame, pbp: pd.DataFrame, seasons: list[int],
                 prior_by_season: dict[int, pd.Series] | None = None, hfa: float | None = None) -> pd.DataFrame:
    """Project every played regular-season game using ONLY data before its week.

    For (season, week) the team ratings are `state.build_team_state(season, week - 1, ...)` with
    w_model = 1 (the model's own opinion, not the blend). Weeks 0-3 use the preseason prior
    (supplied in `prior_by_season`); with none supplied those weeks are skipped and listed in
    `result.attrs["skipped"]`, never silently evaluated without it.
    """
    hfa = C.hfa_points if hfa is None else hfa
    prior_by_season = prior_by_season or {}
    mk = _market_numbers(lines)
    rows, skipped, unrated = [], [], 0
    g = games.copy()
    if "season_type" in g.columns:
        g = g[g["season_type"] == "regular"]
    g = g[g["homePoints"].notna() & g["week"].notna()]

    for season in seasons:
        sg = g[g["season"] == season]
        for week in sorted(sg["week"].unique()):
            week = int(week)
            prior = prior_by_season.get(season)
            if week in PREREGISTERED["early_weeks"] and prior is None:
                skipped.append((season, week, "no preseason ratings supplied for the early weeks"))
                continue
            try:
                ts = state.build_team_state(season, week - 1, lines, pbp, prior_pts=prior, w_model=1.0)
                tot = ratings.fit_total_ratings(lines, season, week)
            except ratings.NoDataBeforeAsOf as e:
                skipped.append((season, week, str(e)))
                continue
            for r in sg[sg["week"] == week].itertuples(index=False):
                h, a = r.homeTeam, r.awayTeam
                if h not in ts.index or a not in ts.index:
                    unrated += 1
                    continue
                neutral = bool(getattr(r, "neutralSite", False))
                m = mk.loc[r.id] if r.id in mk.index else pd.Series(np.nan, index=mk.columns)
                have_tot = h in tot.index and a in tot.index
                rows.append({
                    "season": season, "week": week, "gameId": r.id, "home": h, "away": a,
                    "home_conf": getattr(r, "homeConference", None),
                    "away_conf": getattr(r, "awayConference", None),
                    "start_date": getattr(r, "start_date", pd.NaT), "neutral": neutral,
                    "margin": float(r.margin), "actual_total": float(r.total),
                    "model_margin": float(ts.loc[h, "model_rating_shrunk"] - ts.loc[a, "model_rating_shrunk"]
                                          + (0.0 if neutral else hfa)),
                    "model_total": float(tot.loc[h, "total_rating"] + tot.loc[a, "total_rating"]) if have_tot else np.nan,
                    "spread_close": m["spread_close"], "spread_open": m["spread_open"],
                    "total_close": m["total_close"], "total_open": m["total_open"],
                })
    res = pd.DataFrame(rows)
    if not res.empty:
        res["market_margin_close"] = -res["spread_close"]
        res["market_margin_open"] = -res["spread_open"]
    res.attrs["skipped"] = skipped
    res.attrs["unrated_games"] = unrated
    return res


# -------------------------------------------------------------------- the regression


def regress(y: np.ndarray, market: np.ndarray, model: np.ndarray) -> dict | None:
    """OLS of y on [1, market, model] with HC3 robust standard errors. None if not estimable."""
    n = len(y)
    if n < 30:
        return None
    X = np.column_stack([np.ones(n), market, model])
    try:
        xtx_inv = np.linalg.inv(X.T @ X)
    except np.linalg.LinAlgError:
        return None
    beta = xtx_inv @ X.T @ y
    e = y - X @ beta
    h = np.einsum("ij,jk,ik->i", X, xtx_inv, X)
    w = (e / (1.0 - np.clip(h, 0.0, 1.0 - 1e-8))) ** 2
    cov = xtx_inv @ ((X * w[:, None]).T @ X) @ xtx_inv
    se = np.sqrt(np.diag(cov))
    t = beta / se
    p = 2 * (1 - stats.t.cdf(np.abs(t), n - 3))
    return {"n": int(n), "coef": float(beta[2]), "se": float(se[2]), "p": float(p[2]),
            "coef_market": float(beta[1]), "se_market": float(se[1])}


def judge(train: dict | None, holdout: dict | None) -> tuple[bool, str, str]:
    """(passed, status, reason) under the pre-registered rules."""
    P = PREREGISTERED
    if train is None or train["n"] < P["min_n_train"]:
        n = 0 if train is None else train["n"]
        return False, "FAIL", f"too few games in the train seasons to judge (n={n})"
    if train["coef"] > P["leak_coef"]:
        return False, "SUSPECTED_LEAK", (f"model coefficient {train['coef']:.2f} is above {P['leak_coef']}; "
                                         "a real edge is never this big, so look for lookahead")
    if train["coef"] <= 0:
        return False, "FAIL", f"model coefficient is not positive ({train['coef']:+.3f})"
    if train["p"] >= P["p_train"]:
        return False, "FAIL", f"not significant on the train seasons (p={train['p']:.3f}, need < {P['p_train']})"
    if holdout is None or holdout["n"] < P["min_n_holdout"]:
        n = 0 if holdout is None else holdout["n"]
        return False, "FAIL", f"too few holdout games to confirm (n={n})"
    if holdout["coef"] <= 0:
        return False, "FAIL", f"sign flipped on the {P['holdout_season']} holdout ({holdout['coef']:+.3f})"
    if holdout["p"] >= P["p_holdout"]:
        return False, "FAIL", (f"did not hold up on the {P['holdout_season']} holdout "
                               f"(p={holdout['p']:.3f}, need < {P['p_holdout']})")
    return True, "PASS", "significant on train seasons and confirmed on the holdout"


def _evaluate(df: pd.DataFrame, market_name: str) -> dict:
    spec, P = MARKETS[market_name], PREREGISTERED
    d = df.dropna(subset=[spec["y"], spec["market"], spec["model"]])
    parts = {}
    for key, sub in (("train", d[d["season"].isin(P["train_seasons"])]),
                     ("holdout", d[d["season"] == P["holdout_season"]])):
        parts[key] = regress(sub[spec["y"]].to_numpy(), sub[spec["market"]].to_numpy(), sub[spec["model"]].to_numpy())
    passed, status, reason = judge(parts["train"], parts["holdout"])
    t, h = parts["train"], parts["holdout"]
    return {
        "n": t["n"] if t else int(d["season"].isin(P["train_seasons"]).sum()),
        "coef": t["coef"] if t else 0.0, "se": t["se"] if t else 0.0, "p": t["p"] if t else 1.0,
        "coef_market": t["coef_market"] if t else 0.0,
        "n_holdout": h["n"] if h else int((d["season"] == P["holdout_season"]).sum()),
        "coef_holdout": h["coef"] if h else 0.0, "p_holdout": h["p"] if h else 1.0,
        "passed": bool(passed), "status": status, "reason": reason,
    }


# ------------------------------------------------------------------------- segments


def _tier(row_home, row_away, season) -> str:
    power = POWER_CONFS_BY_SEASON.get(season, POWER_CONFS)
    hp, ap = row_home in power, row_away in power
    return "P4-P4" if hp and ap else ("G5-G5" if not hp and not ap else "P4-G5")


def segment_masks(df: pd.DataFrame, market_name: str) -> dict[str, pd.Series]:
    """Boolean masks for every pre-registered segment, in a fixed order."""
    out: dict[str, pd.Series] = {}
    tiers = pd.Series([_tier(h, a, s) for h, a, s in zip(df["home_conf"], df["away_conf"], df["season"])],
                      index=df.index)
    for t in ("P4-P4", "P4-G5", "G5-G5"):
        out[f"tier:{t}"] = tiers == t
    wk = df["week"]
    out["weeks:0-3"] = wk <= 3
    out["weeks:4-8"] = (wk >= 4) & (wk <= 8)
    out["weeks:9+"] = wk >= 9
    # kickoff is UTC; shifting by 5h puts every US kickoff (incl. late Hawaii games) on the right day
    local = pd.to_datetime(df["start_date"], utc=True, errors="coerce") - pd.Timedelta(hours=5)
    sat = local.dt.dayofweek == 5
    out["day:saturday"] = sat
    out["day:other"] = ~sat & local.notna()
    size = df[MARKETS[market_name]["line"]].abs()
    out["spread:<=7"] = size <= 7
    out["spread:7-14"] = (size > 7) & (size <= 14)
    out["spread:>14"] = size > 14
    return out


# --------------------------------------------------------------------- the whole run


def run_validation(res: pd.DataFrame, generated: str | None = None) -> dict:
    """Evaluate every market and segment. Pure function of `res` (and `generated`)."""
    P = PREREGISTERED
    status = {
        "generated": generated or date.today().isoformat(),
        "status": "OK",
        "train_seasons": list(P["train_seasons"]), "holdout_season": P["holdout_season"],
        "thresholds": {k: P[k] for k in ("p_train", "p_holdout", "leak_coef", "min_n_train", "min_n_holdout")},
        "markets": {}, "segments": [], "w_model": {}, "reported_not_used": {},
    }
    for name in MARKETS:
        ev = _evaluate(res, name) if len(res) else _evaluate(pd.DataFrame(
            columns=["season", "week", "margin", "actual_total", "model_margin", "model_total",
                     "market_margin_open", "market_margin_close", "total_open", "total_close"]), name)
        status["markets"][name] = ev
        status["w_model"][f"{name}|all"] = _weight(ev)
        if ev["status"] == "SUSPECTED_LEAK":
            status["status"] = "SUSPECTED_LEAK"
        if not len(res):
            continue
        for seg, mask in segment_masks(res, name).items():
            sev = _evaluate(res[mask.fillna(False)], name)
            status["segments"].append({"market": name, "segment": seg, **{
                k: sev[k] for k in ("n", "coef", "p", "n_holdout", "coef_holdout", "p_holdout",
                                    "passed", "status", "reason")}})
            status["segments"][-1]["p_train"] = status["segments"][-1].pop("p")
            status["w_model"][f"{name}|{seg}"] = _weight(sev)
            if sev["status"] == "SUSPECTED_LEAK":
                status["status"] = "SUSPECTED_LEAK"
        ytd = res[res["season"] == P["report_season"]]
        spec = MARKETS[name]
        r = regress(*(ytd.dropna(subset=[spec["y"], spec["market"], spec["model"]])[c].to_numpy()
                      for c in (spec["y"], spec["market"], spec["model"])))
        status["reported_not_used"][name] = ({"season": P["report_season"], "n": r["n"], "coef": r["coef"], "p": r["p"]}
                                             if r else {"season": P["report_season"], "n": int(len(ytd)), "coef": None, "p": None})
    return _rounded(status)


def _weight(ev: dict) -> float:
    """The blend weight a passed market/segment has earned: b2 / (b1 + b2), clipped to [0, 1]."""
    if not ev["passed"]:
        return 0.0
    return float(min(max(ev["coef"] / (ev["coef_market"] + ev["coef"]), 0.0), 1.0))


def _rounded(obj, nd: int = 6):
    if isinstance(obj, float):
        return round(obj, nd)
    if isinstance(obj, dict):
        return {k: _rounded(v, nd) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_rounded(v, nd) for v in obj]
    return obj


# ------------------------------------------------------------------- extra reporting


def _eval_rows(res: pd.DataFrame) -> pd.DataFrame:
    P = PREREGISTERED
    return res[res["season"].isin(list(P["train_seasons"]) + [P["holdout_season"]])]


def mae_table(res: pd.DataFrame) -> pd.DataFrame:
    d = _eval_rows(res).dropna(subset=["margin", "model_margin", "spread_close", "spread_open"])
    t = _eval_rows(res).dropna(subset=["actual_total", "model_total", "total_close", "total_open"])
    return pd.DataFrame({
        "market": ["spread", "total"], "n": [len(d), len(t)],
        "model": [(d["margin"] - d["model_margin"]).abs().mean(), (t["actual_total"] - t["model_total"]).abs().mean()],
        "opener": [(d["margin"] + d["spread_open"]).abs().mean(), (t["actual_total"] - t["total_open"]).abs().mean()],
        "closer": [(d["margin"] + d["spread_close"]).abs().mean(), (t["actual_total"] - t["total_close"]).abs().mean()],
    }).round(2)


def calibration(res: pd.DataFrame) -> pd.DataFrame:
    """Predicted vs realised home-cover rate against the closing spread (pushes dropped)."""
    d = _eval_rows(res).dropna(subset=["margin", "model_margin", "spread_close"])
    d = d[(d["margin"] + d["spread_close"]) != 0]
    if len(d) < 50:
        return pd.DataFrame()
    tot = d["model_total"].fillna(d["total_close"]).fillna(52.0)
    p = np.array([game_model.cover_prob(m, t, s, source="model")["home"]
                  for m, t, s in zip(d["model_margin"], tot, d["spread_close"])])
    return backtest.calibration_table(p, ((d["margin"] + d["spread_close"]) > 0).astype(float).to_numpy())


def roi_table(res: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """If every game with model-vs-market disagreement were bet on the model's side of the
    closing spread at an ASSUMED -110, how would each edge bucket have done?"""
    d = _eval_rows(res).dropna(subset=["margin", "model_margin", "spread_close"]).copy()
    diff = d["model_margin"] + d["spread_close"]            # >0: model likes the home side
    cover = d["margin"] + d["spread_close"]
    win = np.where(diff > 0, cover > 0, cover < 0)
    d["profit"] = np.where(cover == 0, 0.0, np.where(win, 100.0 / abs(ASSUMED_PRICE), -1.0))
    d["edge"] = diff.abs()
    d["bucket"] = pd.cut(d["edge"], EDGE_BUCKETS, right=False)
    rows = []
    for b, g in d.groupby("bucket", observed=True):
        boot = backtest.bootstrap_roi(g["profit"].to_numpy()) if len(g) >= 30 else {}
        rows.append({"edge (pts)": str(b), "n": len(g), "roi": round(g["profit"].mean(), 4),
                     "ci_low": round(boot.get("ci_low", np.nan), 3), "ci_high": round(boot.get("ci_high", np.nan), 3)})
    overall = backtest.bootstrap_roi(d["profit"].to_numpy()) if len(d) else {}
    return pd.DataFrame(rows), overall


# --------------------------------------------------------------------------- report


def build_report(status: dict, res: pd.DataFrame | None = None) -> str:
    P = PREREGISTERED
    markets, segs = status["markets"], status["segments"]
    passed_m = [k for k, v in markets.items() if v["passed"]]
    passed_s = [s for s in segs if s["passed"]]
    L = [f"# Validation report ({status['generated']})", ""]
    if status["status"] == "SUSPECTED_LEAK":
        L += ["## STATUS: SUSPECTED LEAK", "",
              "A model coefficient came out above 0.5. Real edges are never that large; this almost always "
              "means the model saw information from the future. Nothing below can be trusted until that is "
              "found. No market or segment has been given any weight.", ""]
    if not passed_m and not passed_s:
        L += ["## Result: nothing passed", "",
              "The model has **not** been shown to add information beyond the betting line, in any market or "
              "segment. That is a valid, expected result. It means the app prices games off the market "
              f"(model weight 0) and every row on the board is `INFO` or `PASS`, never `BET`. "
              "Key-number values and line tracking still work.", ""]
    else:
        L += ["## Result: some things passed", "",
              f"Markets: {', '.join(passed_m) or 'none'}. Segments: {len(passed_s)}. "
              "Only these may produce `BET` rows, at the weights listed in `validation_status.json`. "
              "Passing means a statistical test was cleared in past seasons, not that bets will win.", ""]
    L += [f"Rules, fixed in advance: model coefficient must be positive with p < {P['p_train']} on seasons "
          f"{P['train_seasons'][0]}-{P['train_seasons'][-1]} **and** keep its sign with p < {P['p_holdout']} on "
          f"{P['holdout_season']}. {P['report_season']}-to-date is shown for information only.", "",
          "## Markets", "", "| market | n (train) | model coef | se | p | holdout coef | holdout p | result |",
          "|---|---|---|---|---|---|---|---|"]
    for k, v in markets.items():
        L.append(f"| {k} | {v['n']} | {v['coef']:+.3f} | {v['se']:.3f} | {v['p']:.3f} | {v['coef_holdout']:+.3f} | "
                 f"{v['p_holdout']:.3f} | {v['status']}: {v['reason']} |")
    n_tests = len(segs)
    L += ["", f"## Segments ({n_tests} tests)", "",
          f"With {n_tests} segment tests, about {0.05 * n_tests:.0f} would clear p < 0.05 on the train seasons by "
          "pure luck. That is why the holdout is required.", ""]
    near = [s for s in segs if s["status"] == "FAIL" and s["coef"] > 0 and s["p_train"] < P["p_train"]]
    if passed_s:
        L += ["Passed:", ""] + [f"- {s['market']} / {s['segment']} (n={s['n']}, coef {s['coef']:+.3f}, "
                                f"p_train {s['p_train']:.3f}, p_holdout {s['p_holdout']:.3f})" for s in passed_s] + [""]
    if near:
        L += ["Looked good on the train seasons but did not survive the holdout (this is the guard working):", ""]
        L += [f"- {s['market']} / {s['segment']}: p_train {s['p_train']:.3f}, {s['reason']}" for s in near] + [""]
    if not passed_s and not near:
        L += ["No segment was even significant on the train seasons.", ""]
    L += ["## Not used for decisions", ""]
    for k, v in status["reported_not_used"].items():
        L.append(f"- {k}, {v['season']}-to-date: " + (f"n={v['n']}, coef {v['coef']:+.3f}, p {v['p']:.3f}"
                                                      if v.get("coef") is not None else f"n={v['n']} (too few to estimate)"))
    if res is not None and len(res):
        L += ["", "## Accuracy (evaluation seasons, MAE in points, lower is better)", "",
              mae_table(res).to_markdown(index=False) if _has_tabulate() else mae_table(res).to_string(index=False)]
        cal = calibration(res)
        if len(cal):
            L += ["", "## Calibration of the model's home-cover probability vs the closing spread", "",
                  "pred = what the model said, actual = how often it happened. Look for the two columns moving together.", "",
                  cal.reset_index(drop=True).to_string(index=False)]
        roi, overall = roi_table(res)
        if len(roi):
            L += ["", f"## ROI by edge size (betting the model's side of the close at an assumed {ASSUMED_PRICE}, "
                      "evaluation seasons; CFBD has no real prices)", "", roi.to_string(index=False)]
            if overall:
                L += ["", f"All games: ROI {overall['roi']:+.3f}, bootstrap 95% CI [{overall['ci_low']:+.3f}, {overall['ci_high']:+.3f}]. "
                          "A CI that includes 0 means no profit has been demonstrated."]
        skipped = res.attrs.get("skipped") or []
        if skipped:
            L += ["", f"Skipped (season, week): {len(skipped)}. First reasons: " + "; ".join(
                f"{s}-wk{w}: {r}" for s, w, r in skipped[:5])]
        L += ["", f"Games without a rating for one side (FCS or new teams): {res.attrs.get('unrated_games', 0)}."]
    return "\n".join(L).rstrip() + "\n"


def _has_tabulate() -> bool:
    try:
        import tabulate  # noqa: F401
        return True
    except ImportError:
        return False


def write_outputs(status: dict, report: str, out_dir: Path | None = None) -> tuple[Path, Path]:
    d = Path(out_dir or config.OUTPUT)
    d.mkdir(parents=True, exist_ok=True)
    sp, rp = d / "validation_status.json", d / "validation_report.md"
    sp.write_text(json.dumps(status, indent=2, sort_keys=True) + "\n")
    rp.write_text(report)
    return sp, rp
