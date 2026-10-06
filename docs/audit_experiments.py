"""
Evidence behind docs/AUDIT.md. Synthetic data only; needs no API key.

    python docs/audit_experiments.py

Not part of the test suite (it demonstrates bugs that the suite does not yet
guard against). Phase 2/3 turn the useful ones into real regression tests.
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

import numpy as np                                   # noqa: E402
from sklearn.linear_model import Ridge               # noqa: E402

import test_smoke as T                               # noqa: E402
from cfbmodel import edge, game_model, ratings       # noqa: E402

g = T.make_schedule()
ln = T.make_lines(g)
pbp = T.make_pbp(g)
S, W = 2024, 5


def before(d, s=S, w=W):
    """The CORRECT as-of filter: strictly earlier season, or same season and earlier week."""
    return d[(d.season < s) | ((d.season == s) & (d.week < w))]


print("A. LOOKAHEAD: ratings as of (2024, week 5), with vs without later rows")
for name, full, fn in (
    ("market ratings", ln, lambda d: ratings.fit_market_ratings(d, S, W)["market_rating"]),
    ("EPA ratings", pbp, lambda d: ratings.fit_epa_ratings(d, S, W, split_pass_rush=False)["net_epa"]),
):
    a, b = fn(full), fn(before(full))
    diff = (a - b.reindex(a.index)).abs().max()
    print(f"   {name}: max rating change from rows that should be invisible = "
          f"{diff:.4f} -> {'LEAK' if diff > 1e-9 else 'ok'}")
print(f"   (the OLD filter would have let in {int(((ln.season > S) & (ln.week < W)).sum())} future-season line rows;"
      f" ratings.as_of lets in 0)")

print("\nB. RIDGE vs HOME FIELD (market fit, all 2024 + 2025 wk 1-13; true HFA 2.4)")
S2, W2 = 2025, 14
d = before(ln, S2, W2).groupby(["gameId", "home", "away", "season", "week"],
                               as_index=False)["spread_close"].median()
teams = sorted(set(d.home) | set(d.away))
idx = {t: i for i, t in enumerate(teams)}
n, p = len(d), len(teams)
X = np.zeros((n, p + 1))
X[np.arange(n), d.home.map(idx)] = 1
X[np.arange(n), d.away.map(idx)] = -1
X[:, -1] = 1
y = -d.spread_close.to_numpy()
for a in (1.0, 100.0, 1000.0):
    plain = Ridge(alpha=a, fit_intercept=False).fit(X, y).coef_[-1]
    Xs = X.copy()
    Xs[:, -1] *= 100
    scaled = Ridge(alpha=a, fit_intercept=False).fit(Xs, y).coef_[-1] * 100
    print(f"   alpha={a:>6}: HFA as shipped (unscaled)={plain:.2f}   with x100 scaling={scaled:.2f}")

print("\nC. CLV SIGN (edge.clv). Positive must mean we beat the close.")
cases = (
    ("spread, home bet -3, closes -5 (better number)", edge.clv(-3, -5, "spread", "home"), +2),
    ("spread, away bet home-3, closes -5 (worse)", edge.clv(-3, -5, "spread", "away"), -2),
    ("total, over 52, closes 54 (better)", edge.clv(52, 54, "total", "over"), +2),
    ("total, under 52, closes 54 (worse)", edge.clv(52, 54, "total", "under"), -2),
)
for label, got, want in cases:
    print(f"   {label:<50} got {got:+} want {want:+}  {'ok' if got == want else 'WRONG SIGN'}")

print("\nD. MARGIN PMF vs pitfall 1/2 targets (market MAE 10.5+-0.4, SD 13.6+-0.5)")
for src in ("market", "model"):
    xs, pm = game_model.margin_pmf(0.0, 52.0, source=src)
    print(f"   source={src:<6} MAE={(np.abs(xs) * pm).sum():.2f}  SD={np.sqrt((xs ** 2 * pm).sum()):.2f}")

print("\nE. WEATHER CACHE KEY: python's hash() differs on every run")
for _ in range(2):
    out = subprocess.run([sys.executable, "-c", "print(hash('same text'))"],
                         capture_output=True, text=True).stdout.strip()
    print("   ", out)
