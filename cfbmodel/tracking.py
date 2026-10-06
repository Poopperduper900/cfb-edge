"""
Bet log, line snapshots and closing-line value (BUILD_PLAN Phase 8).

CLV (closing line value) is the primary KPI. Over the few dozen bets a season produces, win rate
and ROI are mostly luck; whether you keep getting better numbers than the market ends up at is the
signal. So this module records every bet you actually place and, once games are over, answers:

  * did you beat the closing line, and by how many points?  (median CLV, % beating the close)
  * what is that worth in probability?                       (CLV in win probability)
  * what happened, and what was the profit?                  (result, ROI with a bootstrap interval)

Files (CSV, not parquet: parquet needs the pyarrow package, which has not been approved):
  output/lines_log.csv   every board run appends the lines it used, with a timestamp
  output/bets.csv        the owner's real bets, added with `python -m cfbmodel log-bet`

Conventions
  spread lines are stored in HOME convention (home -6.5 = -6.5; the road team at +6.5 is also -6.5)
  and `line` in bets.csv is the number you SAW for your side (Auburn +6.5 is entered as 6.5, side away).
  CFBD publishes no prices for spreads/totals, so probability-CLV for those markets measures the value of
  the LINE only (price is assumed equal at bet time and close). For moneylines the price is part of it.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from . import backtest, config, edge, pricing, validation

BET_COLUMNS = ["bet_id", "logged_at", "season", "week", "game_id", "game", "market", "side", "line", "price",
               "book", "stake", "notes"]
LOG_COLUMNS = ["logged_at", "season", "week", "gameId", "home", "away", "book", "spread_close", "spread_open",
               "total_close", "total_open", "ml_home", "ml_away"]
SIDES = {"spread": ("home", "away"), "total": ("over", "under"), "moneyline": ("home", "away")}
MIN_BETS_FOR_ROI = 100
NOISE_MESSAGE = "ROI is noise at this sample — read CLV."


class TrackingError(ValueError):
    """A bet could not be logged or matched to a game."""


def bets_path() -> Path:
    return config.OUTPUT / "bets.csv"


def lines_log_path() -> Path:
    return config.OUTPUT / "lines_log.csv"


# ------------------------------------------------------------------ line snapshots


def log_lines(lines: pd.DataFrame, season: int, week: int, now: datetime | None = None,
              path: Path | None = None) -> int:
    """Append the lines a board run used. Returns the number of rows written."""
    p = Path(path or lines_log_path())
    if lines.empty:
        return 0
    stamp = (now or datetime.now(timezone.utc)).isoformat(timespec="seconds")
    d = lines.assign(logged_at=stamp, season=season, week=week)
    for c in LOG_COLUMNS:
        if c not in d.columns:
            d[c] = np.nan
    p.parent.mkdir(parents=True, exist_ok=True)
    d[LOG_COLUMNS].to_csv(p, mode="a", header=not p.exists(), index=False)
    return len(d)


# ------------------------------------------------------------------- the bet log


def resolve_game(game: str, games: pd.DataFrame, week: int) -> tuple[int, str]:
    """A CFBD game id (digits) or the label shown on the board ('Away @ Home', 'Away vs Home (N)')."""
    wk = games[games["week"] == week]
    if str(game).strip().isdigit():
        hit = wk[wk["id"] == int(game)]
        if hit.empty:
            raise TrackingError(f"no game with id {game} in week {week}")
    else:
        want = " ".join(str(game).split()).casefold()
        labels = [(" ".join(f"{r.awayTeam} {'vs' if r.neutralSite else '@'} {r.homeTeam}"
                            f"{' (N)' if r.neutralSite else ''}".split()).casefold()) for r in wk.itertuples()]
        idx = [i for i, lab in enumerate(labels) if lab == want or lab.replace(" (n)", "") == want]
        if len(idx) != 1:
            raise TrackingError(f"{'no' if not idx else 'more than one'} game matches {game!r} in week {week}; "
                             "use the label from the board or the numeric game id")
        hit = wk.iloc[idx]
    r = hit.iloc[0]
    return int(r["id"]), f"{r['awayTeam']} {'vs' if r['neutralSite'] else '@'} {r['homeTeam']}" + \
        (" (N)" if r["neutralSite"] else "")


def add_bet(*, season: int, week: int, game_id: int, game: str, market: str, side: str, line: float,
            price: float, book: str, stake: float, notes: str = "", now: datetime | None = None,
            path: Path | None = None) -> dict:
    """Validate and append one bet; returns the stored record."""
    if market not in SIDES:
        raise TrackingError(f"market must be one of {sorted(SIDES)}, not {market!r}")
    if side not in SIDES[market]:
        raise TrackingError(f"for {market} the side must be one of {SIDES[market]}, not {side!r}")
    if stake <= 0:
        raise TrackingError("stake must be positive")
    if price == 0 or -100 < price < 100:
        raise TrackingError(f"price must be American odds like -110 or +150, not {price}")
    p = Path(path or bets_path())
    existing = load_bets(p)
    rec = {"bet_id": int(existing["bet_id"].max() + 1) if len(existing) else 1,
           "logged_at": (now or datetime.now(timezone.utc)).isoformat(timespec="seconds"),
           "season": season, "week": week, "game_id": game_id, "game": game, "market": market, "side": side,
           "line": float(line), "price": float(price), "book": book, "stake": float(stake), "notes": notes}
    p.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([rec])[BET_COLUMNS].to_csv(p, mode="a", header=not p.exists(), index=False)
    return rec


def load_bets(path: Path | None = None) -> pd.DataFrame:
    p = Path(path or bets_path())
    if not p.exists():
        return pd.DataFrame(columns=BET_COLUMNS)
    return pd.read_csv(p)


# --------------------------------------------------------------------- settlement


def home_line(market: str, side: str, line: float) -> float:
    """The number in HOME convention for a spread bet entered as seen for its own side."""
    if market == "spread":
        return float(line) if side == "home" else -float(line)
    return float(line)


def _result(market: str, side: str, hl: float, margin: float, total: float) -> str:
    """win / loss / push for a settled game. hl is the home-convention spread, or the total."""
    if market == "spread":
        d = margin + hl                                  # >0: home covered
        return "push" if d == 0 else ("win" if (d > 0) == (side == "home") else "loss")
    if market == "total":
        d = total - hl
        return "push" if d == 0 else ("win" if (d > 0) == (side == "over") else "loss")
    return "push" if margin == 0 else ("win" if (margin > 0) == (side == "home") else "loss")


def profit_units(result: str, price: float, stake: float) -> float:
    if result == "push":
        return 0.0
    return stake * (edge.american_to_decimal(price) - 1.0) if result == "win" else -stake


def settle(bets: pd.DataFrame, games: pd.DataFrame, lines: pd.DataFrame) -> pd.DataFrame:
    """Join each bet to its game's result and closing line. Games not final yet stay 'pending'.

    The closing line is the bet's own book at the close when the lines table has it, otherwise the
    median across books (close_basis says which)."""
    rows = []
    g = games.set_index("id")
    for b in bets.itertuples(index=False):
        r = b._asdict()
        r.update(status="pending", close_line=np.nan, close_basis="", clv_pts=np.nan, clv_prob=np.nan,
                 result="", profit=np.nan, roi=np.nan, home_line=np.nan)
        if b.game_id not in g.index or pd.isna(g.loc[b.game_id, "homePoints"]):
            rows.append(r)
            continue
        gm = g.loc[b.game_id]
        margin, total = float(gm["homePoints"] - gm["awayPoints"]), float(gm["homePoints"] + gm["awayPoints"])
        ln = lines[lines["gameId"] == b.game_id]
        mine = ln[ln["book"].astype(str).str.casefold() == str(b.book).casefold()]
        use, basis = (mine, "book") if len(mine) else (ln, "consensus")
        r.update(status="settled", close_basis=basis if len(ln) else "none", home_line=home_line(b.market, b.side, b.line))
        if b.market in ("spread", "total") and len(ln):
            col = "spread_close" if b.market == "spread" else "total_close"
            close = float(use[col].median()) if use[col].notna().any() else np.nan
            if not np.isnan(close):
                r["close_line"] = close
                r["clv_pts"] = edge.clv(r["home_line"], close, b.market, b.side)
                close_total = float(use["total_close"].median()) if use["total_close"].notna().any() else 52.0
                close_margin = -close if b.market == "spread" else -float(use["spread_close"].median()) \
                    if use["spread_close"].notna().any() else 0.0
                if b.market == "spread":
                    eq = lambda L: pricing.side_equity(b.side, -close, close_total, L, "market")
                else:
                    eq = lambda L: pricing.total_side_equity(b.side, close, close_margin, L, "market")
                r["clv_prob"] = eq(r["home_line"]) - eq(close)
        elif b.market == "moneyline" and len(ln) and ln["ml_home"].notna().any() and ln["ml_away"].notna().any():
            mh, ma = float(use["ml_home"].median()), float(use["ml_away"].median())
            nv = pricing.no_vig_probs([mh, ma], "moneyline")
            r["clv_prob"] = float(nv[0] if b.side == "home" else nv[1]) - edge.american_to_prob(b.price)
        r["result"] = _result(b.market, b.side, r["home_line"], margin, total)
        r["profit"] = profit_units(r["result"], b.price, b.stake)
        r["roi"] = r["profit"] / b.stake
        rows.append(r)
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------ report


def _summary(d: pd.DataFrame) -> dict:
    clv = d["clv_pts"].dropna()
    prob = d["clv_prob"].dropna()
    roi = d["roi"].dropna().to_numpy()
    boot = backtest.bootstrap_roi(roi) if len(roi) >= 10 else {}
    return {"n": len(d), "median_clv_pts": float(clv.median()) if len(clv) else np.nan,
            "pct_beat_close": float((clv > 0).mean()) if len(clv) else np.nan,
            "median_clv_prob": float(prob.median()) if len(prob) else np.nan,
            "roi": float(d["profit"].sum() / d["stake"].sum()) if d["stake"].sum() else np.nan,
            "roi_lo": boot.get("ci_low", np.nan), "roi_hi": boot.get("ci_high", np.nan)}


def segment_labels(settled: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    """Tier / week bucket / weekday / spread-size labels for each settled bet."""
    g = games.set_index("id")
    d = settled.copy()
    d["home_conf"] = d["game_id"].map(g["homeConference"])
    d["away_conf"] = d["game_id"].map(g["awayConference"])
    d["start_date"] = d["game_id"].map(g["start_date"])
    d["spread_close"] = np.where(d["market"] == "spread", d["close_line"], np.nan)
    d["spread_open"] = d["spread_close"]
    return d


def report(settled: pd.DataFrame, games: pd.DataFrame | None = None) -> str:
    done = settled[settled["status"] == "settled"]
    pending = len(settled) - len(done)
    L = [f"# CLV report ({len(done)} settled bets, {pending} pending)", ""]
    if done.empty:
        return "\n".join(L + ["Nothing is settled yet. Log bets with `python -m cfbmodel log-bet`, then run this again "
                               "after the games."]) + "\n"
    tab = pd.DataFrame({m: _summary(g) for m, g in done.groupby("market")}).T
    L += ["Positive CLV means you got a better number than the market ended at. It is the number to watch.", "",
          "## By market", "", tab.round(4).to_string(), ""]
    if games is not None and len(done) >= 10:
        seg = segment_labels(done, games)
        rows = {}
        for market, g in seg.groupby("market"):
            if market == "moneyline":
                continue
            for name, mask in validation.segment_masks(g.assign(season=g["season"], week=g["week"]),
                                                       "spread_vs_close").items():
                sub = g[mask.fillna(False)]
                if len(sub) >= 10:
                    rows[f"{market} / {name}"] = _summary(sub)
        if rows:
            L += ["## By segment (at least 10 bets)", "", pd.DataFrame(rows).T.round(4).to_string(), ""]
    L.append(f"Closing line basis: the bet's own book when available, otherwise the median of all books "
             f"(see close_basis in bets_settled.csv). {sum(done['close_basis'].eq('consensus'))} bets used the consensus.")
    if len(done) < MIN_BETS_FOR_ROI:
        L += ["", f"{NOISE_MESSAGE} ({len(done)} settled bets; ROI needs several hundred before it means anything.)"]
    return "\n".join(L) + "\n"
