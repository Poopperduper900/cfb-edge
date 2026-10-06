"""
Walk-forward backtest.

The headline function is `market_efficiency_test`. Run it before anything else.
It asks one question:

    margin ~ b0 + b1 * (market implied margin) + b2 * (model implied margin)

If b2 is not significantly different from zero, your model contains no
information the closing line has not already priced, and every "edge" the rest
of this repo reports on sides is noise wearing a number. That is exactly the
result you got in the NFL. It is a real possible outcome here too.

The CFB-specific reason to run it anyway rather than assume: 136 FBS teams
means the market cannot price every game with equal care, and the tests should
be run *segmented* — by week, by conference tier, by day of week, and openers
vs closers. A model can be worthless against the Saturday P4 closer and
genuinely profitable against a Tuesday MAC opener. Aggregate results hide that.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from .config import C
from .edge import american_to_decimal, expected_value


# ----------------------------------------------------- the test that matters


def market_efficiency_test(df: pd.DataFrame, model_col: str = "model_margin") -> pd.DataFrame:
    """
    df needs: margin (actual home margin), market_margin (= -closing spread),
    and a model column. Returns coefficient table with t-stats.
    """
    d = df.dropna(subset=["margin", "market_margin", model_col]).copy()
    if len(d) < 50:
        raise ValueError(f"only {len(d)} rows; need a real sample")

    X = np.column_stack(
        [np.ones(len(d)), d["market_margin"].to_numpy(), d[model_col].to_numpy()]
    )
    y = d["margin"].to_numpy()

    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    dof = len(d) - X.shape[1]
    s2 = resid @ resid / dof
    cov = s2 * np.linalg.inv(X.T @ X)
    se = np.sqrt(np.diag(cov))
    t = beta / se
    p = 2 * (1 - stats.t.cdf(np.abs(t), dof))

    return pd.DataFrame(
        {
            "term": ["intercept", "market_margin", model_col],
            "coef": beta.round(4),
            "se": se.round(4),
            "t": t.round(3),
            "p": p.round(4),
        }
    ).set_index("term")


def segmented_efficiency(df: pd.DataFrame, by: str, model_col: str = "model_margin") -> pd.DataFrame:
    """Same test, split by a column. This is where CFB edge hides, if it exists."""
    rows = []
    for key, sub in df.groupby(by):
        if len(sub) < 80:
            continue
        try:
            res = market_efficiency_test(sub, model_col)
        except (ValueError, np.linalg.LinAlgError):
            continue
        rows.append(
            {
                by: key,
                "n": len(sub),
                "model_coef": res.loc[model_col, "coef"],
                "model_t": res.loc[model_col, "t"],
                "model_p": res.loc[model_col, "p"],
                "market_coef": res.loc["market_margin", "coef"],
            }
        )
    return pd.DataFrame(rows).sort_values("model_t", ascending=False)


def opener_vs_closer(df: pd.DataFrame) -> pd.DataFrame:
    """
    Does the model beat the *opening* number? This is the realistic target.
    Beating the close means beating the aggregate of everyone who bet the game;
    beating the open means being early. In CFB, early is where the money is.
    """
    d = df.dropna(subset=["spread_open", "spread_close", "margin", "model_margin"]).copy()
    d["open_margin"] = -d["spread_open"]
    d["close_margin"] = -d["spread_close"]
    d["market_moved_toward_model"] = np.sign(d["close_margin"] - d["open_margin"]) == np.sign(
        d["model_margin"] - d["open_margin"]
    )
    d["model_vs_open"] = (d["model_margin"] - d["open_margin"]).abs()
    return pd.DataFrame(
        {
            "n": [len(d)],
            "pct_line_moved_toward_model": [d["market_moved_toward_model"].mean().round(4)],
            "mean_abs_model_vs_open": [d["model_vs_open"].mean().round(3)],
            "mae_model": [(d["margin"] - d["model_margin"]).abs().mean().round(3)],
            "mae_open": [(d["margin"] - d["open_margin"]).abs().mean().round(3)],
            "mae_close": [(d["margin"] - d["close_margin"]).abs().mean().round(3)],
        }
    )


# ---------------------------------------------------------------- calibration


def calibration_table(p: np.ndarray, y: np.ndarray, bins: int = 10) -> pd.DataFrame:
    """Predicted vs realised, in probability buckets. Look for monotonicity."""
    d = pd.DataFrame({"p": p, "y": y}).dropna()
    d["bucket"] = pd.qcut(d["p"], bins, duplicates="drop")
    g = d.groupby("bucket", observed=True).agg(
        n=("y", "size"), pred=("p", "mean"), actual=("y", "mean")
    )
    g["gap"] = (g["actual"] - g["pred"]).round(4)
    return g.round(4)


def roi_by_edge(bets: pd.DataFrame, edge_col: str = "edge_pts") -> pd.DataFrame:
    """
    ROI bucketed by claimed edge. A working model shows ROI rising with edge.
    A broken one shows the opposite — big claimed edges are usually big data
    errors, stale lines, or a team you have mis-rated.
    """
    d = bets.dropna(subset=[edge_col, "result"]).copy()
    d["bucket"] = pd.cut(d[edge_col], [0, 1, 2, 3, 5, 8, 100])
    d["profit"] = np.where(
        d["result"] == 1,
        d["price"].map(american_to_decimal) - 1,
        np.where(d["result"] == 0, 0.0, -1.0),
    )
    g = d.groupby("bucket", observed=True).agg(
        n=("profit", "size"), win_rate=("result", "mean"), roi=("profit", "mean")
    )
    return g.round(4)


def bootstrap_roi(profits: np.ndarray, n_boot: int = 10_000, seed: int = 7) -> dict:
    """
    CI on ROI. Run this before believing any backtest number. A 15-week CFB
    season at 4 bets/week is 60 bets; the 95% CI on 60 bets at -110 spans
    roughly +/-13% ROI, which is wider than any real edge.
    """
    rng = np.random.default_rng(seed)
    n = len(profits)
    if n == 0:
        return {}
    means = rng.choice(profits, size=(n_boot, n), replace=True).mean(axis=1)
    return {
        "n": n,
        "roi": float(profits.mean()),
        "ci_low": float(np.percentile(means, 2.5)),
        "ci_high": float(np.percentile(means, 97.5)),
        "p_profitable": float((means > 0).mean()),
    }
