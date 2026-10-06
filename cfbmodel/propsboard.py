"""
Player-prop board (BUILD_PLAN Phase 9; optional, and only after the main board works).

CFBD has no prop lines, and scraping sportsbooks is off the table (CLAUDE.md rule 8). So the lines
come from YOU: type them into output/props_lines.csv, one row per player per market per look, e.g.

    player,team,market,line,over_price,under_price,book,timestamp,p_play
    Ace Runner,Alpha State,rush_yds,79.5,-115,-105,MyBook,2026-10-09T15:00:00Z,1.0

  market    rush_yds | rec_yds | receptions   (passing yards need QB state, which is not built)
  p_play    your own probability the player plays. CFB has no injury report, so this is an input,
            never a guess: leave it blank and the row is flagged "p_play assumed 1.0".
  timestamp when you looked. Add a NEW row (same player/market, later time) as the line moves; the
            last row before kickoff is the closing line the CLV backtest uses.

Model: volume x efficiency under a JOINT game-script simulation (script.py): one simulated game per
team, shared by all of its players, so a blowout lowers a starter's volume and raises the backup's.
Volume comes from the player's usage-share posterior, yards per touch from the efficiency posterior,
and the volume shock from how settled the role is (playerstate.py).

BET is locked until a backtest on YOUR logged lines shows positive CLV. Every run logs the sides the
model likes (output/props_log.csv); once you have added later lines for those players, `eligibility`
measures whether the model's side kept beating the closing number. Until it has done so on at least
100 rows, with a median CLV above zero, more than half beating the close and a bootstrap interval
above zero, every prop is INFO at best.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from . import backtest, config, edge, playerstate, pricing, script, teams
from .params import P

MARKETS = {"rush_yds": "rush", "rec_yds": "rec", "receptions": "rec"}
LINE_COLUMNS = ["player", "team", "market", "line", "over_price", "under_price", "book", "timestamp"]
BOARD_COLUMNS = ["player", "team", "game", "market", "side", "line", "over_price", "under_price", "book", "p_play",
                 "rank", "share", "games", "proj_mean", "p_over", "no_vig_over", "prob_edge", "ev", "stake_pct",
                 "status", "flags", "params_version"]


class PropsError(ValueError):
    """props_lines.csv is malformed."""


def lines_path() -> Path:
    return config.OUTPUT / "props_lines.csv"


def log_path() -> Path:
    return config.OUTPUT / "props_log.csv"


# ------------------------------------------------------------------- the input file


def load_lines(path: Path | None = None) -> pd.DataFrame:
    p = Path(path or lines_path())
    if not p.exists():
        raise PropsError(f"{p} does not exist. Create it with the header: "
                         + ",".join(LINE_COLUMNS) + ",p_play")
    df = pd.read_csv(p)
    missing = [c for c in LINE_COLUMNS if c not in df.columns]
    if missing:
        raise PropsError(f"{p.name} is missing columns: {missing}")
    bad = df[~df["market"].isin(MARKETS)]
    if len(bad):
        raise PropsError(f"unsupported market(s) {sorted(bad['market'].unique())} (row {bad.index[0] + 2}); "
                         f"supported: {sorted(MARKETS)}. Passing yards need QB state, which is not built.")
    for col in ("line", "over_price", "under_price"):
        if pd.to_numeric(df[col], errors="coerce").isna().any():
            raise PropsError(f"column {col} has a non-number (row {df[pd.to_numeric(df[col], errors='coerce').isna()].index[0] + 2})")
        df[col] = pd.to_numeric(df[col])
    for col in ("over_price", "under_price"):
        if ((df[col].abs() < 100)).any():
            raise PropsError(f"{col} must be American odds like -115 or +105")
    ts = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    if ts.isna().any():
        raise PropsError(f"timestamp not understood on row {df[ts.isna()].index[0] + 2}; use e.g. 2026-10-09T15:00:00Z")
    df["timestamp"] = ts
    if "p_play" not in df.columns:
        df["p_play"] = np.nan
    pp = pd.to_numeric(df["p_play"], errors="coerce")
    if ((pp < 0) | (pp > 1)).any():
        raise PropsError("p_play must be between 0 and 1")
    df["p_play"] = pp
    return df


def latest_lines(history: pd.DataFrame) -> pd.DataFrame:
    """The most recent look per player/market/book: what to price now."""
    key = ["player", "team", "market", "book"]
    return history.sort_values("timestamp").groupby(key, as_index=False).tail(1).reset_index(drop=True)


# ---------------------------------------------------------------- the eligibility gate


def backtest_clv(log: pd.DataFrame, history: pd.DataFrame) -> dict:
    """CLV of the model's logged sides: first line it liked vs the last line you recorded for that
    player and market (the close). Points, with positive meaning a better number than the close."""
    if log.empty:
        return {"n": 0}
    log = log.assign(timestamp=pd.to_datetime(log["timestamp"], utc=True))
    first = log.sort_values("timestamp").groupby(["player", "team", "market", "game"], as_index=False).head(1)
    last = history.sort_values("timestamp").groupby(["player", "team", "market"], as_index=False).tail(1)
    last = last.rename(columns={"line": "close_line", "timestamp": "close_time"})[["player", "team", "market", "close_line", "close_time"]]
    d = first.merge(last, on=["player", "team", "market"], how="inner")
    d = d[d["close_time"] > d["timestamp"]]              # a later look than the one the model acted on
    if d.empty:
        return {"n": 0}
    clv = np.array([edge.clv(b, c, "prop", s) for b, c, s in zip(d["line"], d["close_line"], d["side"])])
    boot = backtest.bootstrap_roi(clv, n_boot=2000)
    return {"n": int(len(clv)), "median_clv": float(np.median(clv)), "mean_clv": float(clv.mean()),
            "pct_beat_close": float((clv > 0).mean()), "ci_low": boot["ci_low"], "ci_high": boot["ci_high"]}


def eligibility(log: pd.DataFrame, history: pd.DataFrame) -> tuple[bool, str, dict]:
    """May props produce BET rows? Only if a backtest on logged lines shows positive CLV."""
    m = backtest_clv(log, history)
    need = P.props.backtest_min_n
    if m["n"] < need:
        return False, f"props are not BET-eligible yet: {m['n']} of {need} logged rows have a later line to measure CLV against", m
    ok = m["median_clv"] > 0 and m["pct_beat_close"] > 0.5 and m["ci_low"] > 0
    why = (f"props are BET-eligible: {m['n']} logged rows, median CLV {m['median_clv']:+.2f}, "
           f"{m['pct_beat_close']:.0%} beat the close" if ok else
           f"props are not BET-eligible: CLV on {m['n']} logged rows is not clearly positive "
           f"(median {m['median_clv']:+.2f}, {m['pct_beat_close']:.0%} beat the close, interval low {m['ci_low']:+.2f})")
    return ok, why, m


# --------------------------------------------------------------------- pricing


def volume_shock_sigma(share_mean: float, share_sd: float) -> float:
    """How much a player's volume can swing, from how uncertain and how settled the role is: the
    relative uncertainty of the usage-share posterior, plus the committee term (a player with a small
    share is a coin flip each Saturday). Bounded by the same limits as script.role_instability."""
    S = P.script
    rel = share_sd / max(share_mean, 1e-3)
    committee = S.role_committee * (1.0 - np.clip(share_mean / S.role_secure_share, 0.0, 1.0))
    return float(np.clip(np.hypot(rel, committee), S.role_min_sigma, S.role_max_sigma))


def _prop_sim(row: pd.Series, sc: dict, p_play: float) -> tuple[np.ndarray, float]:
    """Simulated outcome for one player-market under the team's shared script."""
    sigma = volume_shock_sigma(row["share_mean"], row["share_sd"])
    sens = P.script.starter_pull_sensitivity if row["rank"] == 1 else P.script.backup_pull_sensitivity
    if row["group"] == "rush":
        return script.simulate_rush_yards_joint(float(row["share_mean"]), sc, float(row["eff_mean"]),
                                                P.props.rush_ypc_sd, sigma, p_play=p_play,
                                                share_pull_sensitivity=sens), sigma
    c = P.props.catch_rate_mean
    out = script.simulate_rec_yards_joint(float(row["share_mean"]), sc, float(row["eff_mean"]) * c, c, sigma,
                                          p_play=p_play, share_pull_sensitivity=P.script.rec_pull_sensitivity * sens)
    return out, sigma


def build_props_board(lines: pd.DataFrame, pstate: pd.DataFrame, games: pd.DataFrame, cons: pd.DataFrame,
                      team_pace: pd.Series | None, *, week: int, eligible: bool, params_version: str,
                      seed: int = 2026) -> pd.DataFrame:
    """Price each recorded prop line.

    lines   latest props_lines rows (see load_lines)
    pstate  playerstate rows (pid, team, player, group, share_mean, share_sd, eff_mean, games, touches, rank)
    games   this week's games (id, homeTeam, awayTeam)
    cons    consensus market numbers by game id (spread_close, total_close)
    """
    sample_k = P.betting.sample_conf_k
    names = pstate.assign(_t=pstate["team"].map(teams.fold), _p=pstate["player"].map(teams.fold))
    gm = games.reset_index(drop=True)
    rows, sims = [], {}
    for n, r in enumerate(lines.reset_index(drop=True).to_dict("records")):
        base = {"player": r["player"], "team": r["team"], "market": r["market"], "line": r["line"],
                "over_price": r["over_price"], "under_price": r["under_price"], "book": r["book"],
                "params_version": params_version, "stake_pct": 0.0}
        flags = []
        p_play = r["p_play"]
        if pd.isna(p_play):
            p_play = P.props.default_p_play
            flags.append("p_play assumed 1.0")
        base["p_play"] = p_play

        tf = teams.fold(r["team"])
        hit = gm[(gm["homeTeam"].map(teams.fold) == tf) | (gm["awayTeam"].map(teams.fold) == tf)]
        if hit.empty:
            rows.append({**base, "status": "UNMATCHED", "flags": "; ".join(flags + ["team has no game this week"])})
            continue
        g = hit.iloc[0]
        is_home = teams.fold(g["homeTeam"]) == tf
        base["game"] = f"{g['awayTeam']} @ {g['homeTeam']}"
        grp = MARKETS[r["market"]]
        cand = names[(names["_t"] == tf) & (names["_p"] == teams.fold(r["player"])) & (names["group"] == grp)]
        if cand.empty:
            rows.append({**base, "status": "UNMATCHED", "flags": "; ".join(flags + ["no player state: check the spelling, or run update"])})
            continue
        ps = cand.iloc[0]
        need = P.props.min_rush_touches if grp == "rush" else P.props.min_rec_touches
        base.update(rank=int(ps["rank"]), share=round(float(ps["share_mean"]), 3), games=int(ps["games"]))
        if ps["touches"] < need:
            rows.append({**base, "status": "PASS", "flags": "; ".join(flags + [f"too little history ({ps['touches']:.0f} touches)"])})
            continue
        if bool(ps.get("role_change", False)):
            flags.append("role changed recently")

        # the team's shared script, from the market's own numbers when we have them
        c = cons.loc[g["id"]] if g["id"] in cons.index else None
        if c is not None and pd.notna(c["spread_close"]) and pd.notna(c["total_close"]):
            margin_home, total = -float(c["spread_close"]), float(c["total_close"])
        else:
            margin_home, total = 0.0, P.game.total_base
            flags.append("no game line: script assumed even")
        margin_for = margin_home if is_home else -margin_home
        pace = float(team_pace.get(r["team"], P.ratings.pace_mean)) if team_pace is not None else P.ratings.pace_mean
        if r["team"] not in sims:
            script.RNG = np.random.default_rng(seed + len(sims))
            sims[r["team"]] = script.simulate_game_script(margin_for, total, pace, n_sims=int(P.props.board_sims))
        script.RNG = np.random.default_rng(seed + 1000 + n)
        out, sigma = _prop_sim(ps, sims[r["team"]], p_play)
        if r["market"] == "rush_yds":
            sim = out
        elif r["market"] == "rec_yds":
            sim = out["yards"]
        else:
            sim = out["receptions"].astype(float)

        pr = _price(sim, r["line"])
        nv_over, nv_under = pricing.no_vig_probs([r["over_price"], r["under_price"]], "prop")
        conf = max(float(ps["games"]) / (float(ps["games"]) + sample_k), P.betting.sample_conf_floor)
        sizes = {"over": pricing.size_bet(pr["over"], pr["push"], r["over_price"], nv_over, P.props.softness, conf),
                 "under": pricing.size_bet(pr["under"], pr["push"], r["under_price"], nv_under, P.props.softness, conf)}
        side = max(sizes, key=lambda k: sizes[k]["shrunk_edge"])
        z = sizes[side]
        clears = z["ev"] >= P.betting.min_ev_props
        status = "BET" if (clears and eligible) else ("INFO" if clears else "PASS")
        rows.append({**base, "side": side, "proj_mean": round(float(sim.mean()), 2), "p_over": round(z["model_prob"] if side == "over" else 1 - z["model_prob"], 4),
                     "no_vig_over": round(float(nv_over), 4), "prob_edge": round(z["prob_edge"], 4), "ev": round(z["ev"], 4),
                     "stake_pct": round(z["stake_pct"], 4) if status == "BET" else 0.0, "status": status,
                     "flags": "; ".join(flags + ([f"hold {edge.hold([r['over_price'], r['under_price']]):.1%}"]))})
    df = pd.DataFrame(rows)
    for col in BOARD_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan if col not in ("flags", "game", "side") else ""
    order = {"BET": 0, "INFO": 1, "PASS": 2, "UNMATCHED": 3}
    df["_o"] = df["status"].map(order)
    df["_e"] = -df["ev"].fillna(-9)
    return df.sort_values(["_o", "_e", "player"], kind="stable").drop(columns=["_o", "_e"])[BOARD_COLUMNS].reset_index(drop=True)


def _price(sim: np.ndarray, line: float) -> dict:
    over, under = float((sim > line).mean()), float((sim < line).mean())
    return {"over": over, "under": under, "push": max(0.0, 1.0 - over - under)}


def log_recommendations(board: pd.DataFrame, now: datetime | None = None, path: Path | None = None) -> int:
    """Append the sides the model liked (INFO or BET) so a later backtest can measure their CLV."""
    liked = board[board["status"].isin(["INFO", "BET"])]
    if liked.empty:
        return 0
    p = Path(path or log_path())
    stamp = (now or datetime.now(timezone.utc)).isoformat(timespec="seconds")
    out = liked[["player", "team", "game", "market", "side", "line", "book"]].assign(timestamp=stamp)
    p.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(p, mode="a", header=not p.exists(), index=False)
    return len(out)
