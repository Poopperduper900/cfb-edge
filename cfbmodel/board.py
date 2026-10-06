"""
The edge board (BUILD_PLAN Phase 7): the weekly product.

One command, `python -m cfbmodel board --season 2026 --week N`, writes
    output/board_2026_wN.csv    every priced market, one row each
    output/board_2026_wN.html   the same, readable on a phone (nothing to install, no internet needed)
and prints a short summary. Most rows should be PASS. An honest "no bet" is a correct output.

Two views of every game are priced:

  model view       what the model says on its own (its projected margin / total). This is what makes
                   an INFO row: "the model sees something, but nothing has proven it may be trusted".
  actionable view  (1 - w) * market + w * model, where w is the weight the VALIDATION gate earned for
                   this market and segment (0 unless it passed). This is the only view allowed to
                   produce a BET, and with w = 0 it IS the market, so it can never disagree with the
                   price. No validation file, a failed validation, or an active drift alarm all mean
                   w = 0, so zero BET rows. That rule is enforced here, in code, and in tests.

Statuses
  BET      actionable view clears the thresholds AND this market/segment passed validation
  INFO     model view clears the thresholds, but validation has not passed for it (cannot be bet)
  PASS     no edge
  FCS      FBS vs FCS: priced, never bet
  NO_LINE  no price for this market yet

CFBD publishes no prices (juice) for spreads and totals, so those are priced at an ASSUMED -110 and
the board says so. Moneylines use the real prices CFBD carries.
"""
from __future__ import annotations

import html
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from . import game_model, pricing, status as status_mod, validation, weather
from .config import C
from .params import P

COLUMNS = ["game", "kickoff_local", "tier", "market", "book", "line", "price", "best_line_across_books",
           "no_vig_prob", "model_prob", "prob_edge", "ev", "half_point_value", "key_number_note",
           "softness", "stake_pct", "status", "flags", "params_version"]
STATUS_ORDER = {"BET": 0, "INFO": 1, "PASS": 2, "FCS": 3, "NO_LINE": 4}


@dataclass
class Board:
    df: pd.DataFrame
    banners: list[str] = field(default_factory=list)
    meta: dict = field(default_factory=dict)


# ------------------------------------------------------------------- small helpers


def _nth_sunday(year: int, month: int, n: int) -> datetime:
    d = datetime(year, month, 1)
    first = d + timedelta(days=(6 - d.weekday()) % 7)
    return first + timedelta(weeks=n - 1)


def eastern(ts) -> pd.Timestamp:
    """UTC timestamp -> US Eastern wall-clock time (daylight rule: 2nd Sunday of March to 1st Sunday
    of November, 2:00 local). Written by hand so no time-zone database has to be installed."""
    ts = pd.Timestamp(ts)
    ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
    start = _nth_sunday(ts.year, 3, 2) + timedelta(hours=7)        # 2:00 EST = 07:00 UTC
    end = _nth_sunday(ts.year, 11, 1) + timedelta(hours=6)         # 2:00 EDT = 06:00 UTC
    naive = ts.tz_localize(None)
    dst = start <= naive < end
    return (naive - timedelta(hours=4 if dst else 5))


def kickoff_label(ts) -> str:
    if pd.isna(ts):
        return ""
    e = eastern(ts)
    return f"{e.strftime('%a')} {e.strftime('%I:%M').lstrip('0')} {e.strftime('%p')} ET"


def validation_weight(v: dict | None, market: str | None, segments: list[str]) -> tuple[float, bool]:
    """(weight, validated) for one market and the segments a game falls in. The weight is the
    SMALLEST weight among the passing entries that apply (the market as a whole, and each passing
    segment the game belongs to): when several proofs apply, trust the most cautious one."""
    if not v or not market:
        return 0.0, False
    weights = []
    m = (v.get("markets") or {}).get(market)
    if m and m.get("passed"):
        weights.append(float(v["w_model"].get(f"{market}|all", 0.0)))
    for s in v.get("segments") or []:
        if s["market"] == market and s["passed"] and s["segment"] in segments:
            weights.append(float(v["w_model"].get(f"{market}|{s['segment']}", 0.0)))
    weights = [w for w in weights if w > 0]
    return (min(weights), True) if weights else (0.0, False)


def _consensus(lines: pd.DataFrame) -> pd.DataFrame:
    cols = ["spread_close", "spread_open", "total_close", "total_open", "ml_home", "ml_away"]
    return lines.groupby("gameId")[cols].median()


def _best_spread(book_lines: pd.DataFrame, side: str):
    d = book_lines.dropna(subset=["spread_close"])
    if d.empty:
        return None
    r = d.loc[d["spread_close"].idxmax()] if side == "home" else d.loc[d["spread_close"].idxmin()]
    num = r["spread_close"] if side == "home" else -r["spread_close"]
    return f"{num:+g} ({r['book']})"


def _best_total(book_lines: pd.DataFrame, side: str):
    d = book_lines.dropna(subset=["total_close"])
    if d.empty:
        return None
    r = d.loc[d["total_close"].idxmin()] if side == "over" else d.loc[d["total_close"].idxmax()]
    return f"{r['total_close']:g} ({r['book']})"


def _best_ml(book_lines: pd.DataFrame, side: str):
    col = "ml_home" if side == "home" else "ml_away"
    d = book_lines.dropna(subset=[col])
    if d.empty:
        return None
    r = d.loc[d[col].idxmax()]
    return f"{r[col]:+g} ({r['book']})"


def _total_equity(model_total, margin, line, side, source="model") -> float:
    tp = game_model.total_probs(model_total, line, margin, source=source)
    return tp[side] + 0.5 * tp["push"]


# ---------------------------------------------------------------------- the builder


def build_board(season: int, week: int, games: pd.DataFrame, lines: pd.DataFrame, ts: pd.DataFrame,
                total_ratings: pd.DataFrame | None, *, validation_status: dict | None, drift_active: bool,
                params_version: str, now: datetime | None = None, qb_table: pd.DataFrame | None = None,
                validation_age_days: float | None = None, prior_source: str | None = None) -> Board:
    """Price every game of one week.

    games   one row per game: id, homeTeam, awayTeam, start_date, neutralSite, homeConference,
            awayConference, home_is_fbs, away_is_fbs (+ weather columns from weather.attach_weather)
    lines   one row per game per book (ingest.lines): spread_close/open, total_close/open, ml_*, book
    ts      team state (state.build_team_state with w_model=1): model_rating_shrunk per team
    """
    now = now or datetime.now(timezone.utc)
    games = games.reset_index(drop=True)      # row number == label, and we iterate over plain dicts (pitfall 5)
    B, G = P.board, P.game
    cons = _consensus(lines) if len(lines) else pd.DataFrame(columns=["spread_close", "spread_open", "total_close",
                                                                      "total_open", "ml_home", "ml_away"])
    sample_conf = pricing.sample_confidence(week)
    drift_off = bool(drift_active)
    rows, disagreement = [], []

    # segments per game, per validation market (spread bucket depends on the market's own line)
    seg_df = games.assign(home_conf=games["homeConference"], away_conf=games["awayConference"], season=season,
                          week=week)
    seg_df["spread_close"] = seg_df["id"].map(cons["spread_close"]) if len(cons) else np.nan
    seg_df["spread_open"] = seg_df["id"].map(cons["spread_open"]) if len(cons) else np.nan
    masks = {m: validation.segment_masks(seg_df, m) for m in ("spread_vs_open", "spread_vs_close",
                                                              "total_vs_open", "total_vs_close")}

    for gi, g in enumerate(games.to_dict("records")):
        gid, h, a = g["id"], g["homeTeam"], g["awayTeam"]
        neutral = bool(g.get("neutralSite", False))
        fbs_both = bool(g.get("home_is_fbs", True)) and bool(g.get("away_is_fbs", True))
        label = f"{a} {'vs' if neutral else '@'} {h}" + (" (N)" if neutral else "")
        kick = kickoff_label(g.get("start_date"))
        hours_out = ((pd.Timestamp(g["start_date"]) - pd.Timestamp(now)).total_seconds() / 3600
                     if pd.notna(g.get("start_date")) else 0.0)
        basis = "open" if hours_out > B.open_basis_hours else "close"
        tier = validation._tier(g.get("homeConference"), g.get("awayConference"), season) if fbs_both else "FBS-FCS"
        soft = game_model.market_softness(pd.Series({"week": week, "home_conf": g.get("homeConference"),
                                                     "away_conf": g.get("awayConference")}))
        book_lines = lines[lines["gameId"] == gid]
        c = cons.loc[gid] if gid in cons.index else None
        base = dict(game=label, kickoff_local=kick, tier=tier, softness=round(float(soft), 2),
                    params_version=params_version)
        flags: list[str] = []

        # ---- model view of margin and total
        rate = lambda t: float(ts.loc[t, "model_rating_shrunk"]) if t in ts.index else np.nan
        hfa = 0.0 if neutral else C.hfa_points
        if fbs_both:
            model_margin = rate(h) - rate(a) + hfa
        elif bool(g.get("home_is_fbs", True)):
            model_margin = rate(h) + G.fcs_rating_adjust + hfa
        else:
            model_margin = -(rate(a) + G.fcs_rating_adjust) + hfa
        wx_adj = 0.0
        if "wx_status" in g:
            wx_adj, _ = weather.total_adjustment({k: g.get(k) for k in ("dome", "wind_mph", "gust_mph", "precip_in",
                                                                         "temp_f", "wx_confidence") if k in g})
        if total_ratings is not None and h in total_ratings.index and a in total_ratings.index:
            model_total = float(total_ratings.loc[h, "total_rating"] + total_ratings.loc[a, "total_rating"]) + wx_adj
        else:
            model_total = np.nan
        if abs(wx_adj) >= B.weather_flag_pts:
            flags.append(f"weather adj {wx_adj:+.1f}")
        if qb_table is not None:
            for t in (h, a):
                if t in qb_table.index and float(qb_table.loc[t, "qb1_dropbacks"]) < B.qb_min_dropbacks:
                    flags.append(f"QB uncertainty ({t})")
        if np.isnan(model_margin):
            flags.append("no rating for a team")

        has_spread = c is not None and pd.notna(c["spread_close"])
        has_total = c is not None and pd.notna(c["total_close"])
        has_ml = c is not None and pd.notna(c["ml_home"]) and pd.notna(c["ml_away"])
        if has_spread and not np.isnan(model_margin):
            disagreement.append(abs(model_margin + c["spread_close"]))
            if abs(model_margin + c["spread_close"]) >= B.large_disagreement_pts:
                flags.append("large disagreement — check ratings")
        if has_spread and pd.notna(c["spread_open"]) and abs(c["spread_close"] - c["spread_open"]) >= B.line_move_pts:
            flags.append(f"line moved {abs(c['spread_close'] - c['spread_open']):.0f}+ since open")

        if not (has_spread or has_total or has_ml):
            rows.append({**base, "market": "all", "status": "NO_LINE", "stake_pct": 0.0, "flags": "; ".join(flags)})
            continue

        def finish(row, market_key, clears_model, clears_act, w, validated):
            if not fbs_both:
                st = "FCS"
            elif clears_act and validated and not drift_off and w > 0:
                st = "BET"
            elif clears_model:
                st = "INFO"
            else:
                st = "PASS"
            row["status"] = st
            if st != "BET":
                row["stake_pct"] = 0.0
            row["flags"] = "; ".join(flags + ([f"validated w={w:.2f}"] if validated and st == "BET" else []))
            rows.append(row)

        # ================================================================== spread
        if has_spread and not np.isnan(model_margin):
            s_home = float(c["spread_close"])
            mk = f"spread_vs_{basis}"
            segs = [n for n, m in masks[mk].items() if bool(m.loc[gi])]
            w, validated = validation_weight(validation_status, mk, segs)
            mkt_margin = -s_home
            tot_ref = c["total_close"] if has_total else (model_total if not np.isnan(model_total) else 52.0)
            tot_m = model_total if not np.isnan(model_total) else tot_ref
            side = "home" if model_margin > mkt_margin else "away"
            line = s_home if side == "home" else -s_home
            price = float(B.assumed_price)
            act_margin = (1 - w) * mkt_margin + w * model_margin
            src_act = "market" if w == 0 else "model"
            cp_m = game_model.cover_prob(model_margin, tot_m, s_home, source="model")
            cp_a = game_model.cover_prob(act_margin, tot_ref, s_home, source=src_act)
            nv = float(pricing.no_vig_probs([price, price], "spread")[0])
            sz_m = pricing.size_bet(cp_m[side], cp_m["push"], price, nv, soft, sample_conf)
            sz_a = pricing.size_bet(cp_a[side], cp_a["push"], price, nv, soft, sample_conf)
            raw_pts = abs(model_margin - mkt_margin)
            sh = pricing.shrink_factor(soft, sample_conf)
            clears_m = (raw_pts * sh >= P.betting.min_edge_spread) and sz_m["ev"] > 0
            clears_a = (abs(act_margin - mkt_margin) * sh >= P.betting.min_edge_spread) and sz_a["ev"] > 0
            kv = pricing.key_number_value(model_margin, tot_m, s_home, side, "model")
            use = sz_a if (clears_a and validated and w > 0 and not drift_off and fbs_both) else sz_m
            finish({**base, "market": f"spread {side}", "book": ", ".join(sorted(book_lines["book"].dropna().unique()[:3])),
                    "line": line, "price": int(price),
                    "best_line_across_books": _best_spread(book_lines, side),
                    "no_vig_prob": round(nv, 4), "model_prob": round(sz_m["model_prob"], 4),
                    "prob_edge": round(sz_m["prob_edge"], 4), "ev": round(use["ev"], 4),
                    "half_point_value": round(kv["buy_0.5"], 4),
                    "key_number_note": pricing.key_number_note(s_home),
                    "stake_pct": round(sz_a["stake_pct"], 4)}, mk, clears_m, clears_a, w, validated)

        # ================================================================== total
        if has_total and not np.isnan(model_total):
            t_line = float(c["total_close"])
            mk = f"total_vs_{basis}"
            segs = [n for n, m in masks[mk].items() if bool(m.loc[gi])]
            w, validated = validation_weight(validation_status, mk, segs)
            side = "over" if model_total > t_line else "under"
            price = float(B.assumed_price)
            act_total = (1 - w) * t_line + w * model_total
            tp_m = game_model.total_probs(model_total, t_line, model_margin if not np.isnan(model_margin) else 0.0, "model")
            tp_a = game_model.total_probs(act_total, t_line, model_margin if not np.isnan(model_margin) else 0.0,
                                          "market" if w == 0 else "model")
            nv = float(pricing.no_vig_probs([price, price], "total")[0])
            sz_m = pricing.size_bet(tp_m[side], tp_m["push"], price, nv, soft, sample_conf)
            sz_a = pricing.size_bet(tp_a[side], tp_a["push"], price, nv, soft, sample_conf)
            sh = pricing.shrink_factor(soft, sample_conf)
            clears_m = (abs(model_total - t_line) * sh >= P.betting.min_edge_total) and sz_m["ev"] > 0
            clears_a = (abs(act_total - t_line) * sh >= P.betting.min_edge_total) and sz_a["ev"] > 0
            m_for_equity = model_margin if not np.isnan(model_margin) else 0.0
            buy = _total_equity(model_total, m_for_equity, t_line + (-0.5 if side == "over" else 0.5), side) \
                - _total_equity(model_total, m_for_equity, t_line, side)
            use = sz_a if (clears_a and validated and w > 0 and not drift_off and fbs_both) else sz_m
            finish({**base, "market": f"total {side}", "book": ", ".join(sorted(book_lines["book"].dropna().unique()[:3])),
                    "line": t_line, "price": int(price), "best_line_across_books": _best_total(book_lines, side),
                    "no_vig_prob": round(nv, 4), "model_prob": round(sz_m["model_prob"], 4),
                    "prob_edge": round(sz_m["prob_edge"], 4), "ev": round(use["ev"], 4),
                    "half_point_value": round(buy, 4), "key_number_note": "",
                    "stake_pct": round(sz_a["stake_pct"], 4)}, mk, clears_m, clears_a, w, validated)
        elif has_total:
            rows.append({**base, "market": "total", "book": "", "line": float(c["total_close"]),
                         "status": "PASS" if fbs_both else "FCS", "stake_pct": 0.0,
                         "flags": "; ".join(flags + ["no model total for a team"])})

        # ================================================================ moneyline
        if has_ml and not np.isnan(model_margin):
            ml_h, ml_a = float(c["ml_home"]), float(c["ml_away"])
            nvp = pricing.no_vig_probs([ml_h, ml_a], "moneyline")
            tot_m = model_total if not np.isnan(model_total) else 52.0
            mp_m = game_model.moneyline_prob(model_margin, tot_m, source="model")
            side = "home" if mp_m["home"] - nvp[0] > mp_m["away"] - nvp[1] else "away"
            col = "ml_home" if side == "home" else "ml_away"
            price = float(book_lines[col].max())          # the price you could actually get
            nv = float(nvp[0] if side == "home" else nvp[1])
            sz_m = pricing.size_bet(mp_m[side], 0.0, price, nv, soft, sample_conf)
            clears_m = sz_m["shrunk_edge"] >= B.min_ml_prob_edge and sz_m["ev"] > 0
            finish({**base, "market": f"moneyline {side}", "book": ", ".join(sorted(book_lines["book"].dropna().unique()[:3])),
                    "line": price, "price": int(price), "best_line_across_books": _best_ml(book_lines, side),
                    "no_vig_prob": round(nv, 4), "model_prob": round(sz_m["model_prob"], 4),
                    "prob_edge": round(sz_m["prob_edge"], 4), "ev": round(sz_m["ev"], 4),
                    "half_point_value": np.nan, "key_number_note": "", "stake_pct": 0.0},
                   None, clears_m, False, 0.0, False)

    df = pd.DataFrame(rows)
    for col in COLUMNS:
        if col not in df.columns:
            df[col] = np.nan if col not in ("book", "key_number_note", "flags", "best_line_across_books") else ""
    df["_o"] = df["status"].map(STATUS_ORDER)
    df["_e"] = -df["ev"].fillna(-9)
    df = df.sort_values(["_o", "_e", "game"], kind="stable").drop(columns=["_o", "_e"])[COLUMNS].reset_index(drop=True)

    # ---- sanity banners
    banners = []
    n_bet = int((df["status"] == "BET").sum())
    if disagreement and float(np.mean(disagreement)) > B.disagreement_banner_pts:
        banners.append(f"Mean |model - market| spread disagreement is {np.mean(disagreement):.1f} points "
                       f"(limit {B.disagreement_banner_pts:g}): ratings likely broken. Do not act on this board.")
    if n_bet > B.max_bet_rows:
        banners.append(f"{n_bet} BET rows (limit {B.max_bet_rows}): too many bets, likely a data problem.")
    if validation_status is None:
        banners.append("No validation_status.json: the model has not been validated, so the board is market-only "
                       "and no row can be a BET. Run `python -m cfbmodel validate`.")
    elif validation_age_days is not None and validation_age_days > B.validation_max_age_days:
        banners.append(f"validation_status.json is {validation_age_days:.0f} days old (limit "
                       f"{B.validation_max_age_days}): re-run validate.")
    if drift_off:
        banners.append("Drift alarm active: model weight forced to 0, market-only pricing. No BET rows.")
    banners.append(f"Prices: CFBD publishes no prices for spreads and totals, so they are priced at an assumed "
                   f"{int(B.assumed_price)}. Check your book's real price before betting. Moneylines use real prices.")
    if prior_source == "default_unfitted" and week <= P.state.early_last_week:
        banners.append("Preseason weights are the unfitted defaults (run `python -m cfbmodel fit-priors`).")
    return Board(df, banners, {"season": season, "week": week, "n_bet": n_bet, "params_version": params_version,
                               "mean_disagreement": float(np.mean(disagreement)) if disagreement else float("nan")})


# ------------------------------------------------------------------------- outputs


def to_csv(board: Board) -> str:
    return board.df.to_csv(index=False, float_format="%.4f")


def summary(board: Board, top: int = 12) -> str:
    df = board.df
    counts = df["status"].value_counts().reindex(list(STATUS_ORDER), fill_value=0)
    out = [f"BOARD season {board.meta['season']} week {board.meta['week']}   params {board.meta['params_version']}",
           "  " + "   ".join(f"{k} {v}" for k, v in counts.items()), ""]
    for b in board.banners:
        out.append(f"!! {b}")
    show = df[df["status"].isin(["BET", "INFO"])].head(top)
    if len(show):
        out += ["", show[["game", "market", "line", "model_prob", "no_vig_prob", "ev", "status", "flags"]].to_string(index=False)]
    else:
        out += ["", "No BET or INFO rows. Nothing clears the bar this week, which is a normal result."]
    return "\n".join(out)


def to_html(board: Board) -> str:
    df, e = board.df, html.escape
    counts = df["status"].value_counts().reindex(list(STATUS_ORDER), fill_value=0)
    cards = []
    for r in df.itertuples(index=False):
        def f(v, spec="{:.3f}"):
            return "" if pd.isna(v) or v == "" else (spec.format(v) if not isinstance(v, str) else e(v))
        cards.append(
            f'<article class="card {r.status.lower()}"><header><span class="st">{r.status}</span> '
            f'<strong>{e(str(r.game))}</strong><span class="ko">{e(str(r.kickoff_local))}</span></header>'
            f'<div class="mk">{e(str(r.market))} <b>{f(r.line, "{:+g}") if str(r.market).startswith(("spread", "moneyline")) else f(r.line, "{:g}")}</b>'
            f' <span class="muted">{f(r.price, "{:+.0f}")} · best {e(str(r.best_line_across_books or ""))}</span></div>'
            f'<dl><dt>model</dt><dd>{f(r.model_prob)}</dd><dt>market</dt><dd>{f(r.no_vig_prob)}</dd>'
            f'<dt>edge</dt><dd>{f(r.prob_edge, "{:+.3f}")}</dd><dt>EV</dt><dd>{f(r.ev, "{:+.3f}")}</dd>'
            f'<dt>½-pt</dt><dd>{f(r.half_point_value, "{:+.3f}")}</dd><dt>stake</dt><dd>{f(r.stake_pct, "{:.2%}")}</dd></dl>'
            + (f'<p class="note">{e(str(r.key_number_note))}</p>' if r.key_number_note else "")
            + (f'<p class="flags">{e(str(r.flags))}</p>' if r.flags else "") + "</article>")
    banners = "".join(f'<p class="banner">{e(b)}</p>' for b in board.banners)
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>CFB Edge Board {board.meta['season']} week {board.meta['week']}</title>
<style>
:root{{--bg:#fafafa;--fg:#1c1c1e;--card:#fff;--line:#e2e2e6;--muted:#6b6b72;--bet:#1a7f37;--info:#9a6700;--pass:#6b6b72;--warn:#b42318}}
@media(prefers-color-scheme:dark){{:root{{--bg:#111113;--fg:#ececf0;--card:#1b1b1f;--line:#2e2e35;--muted:#9a9aa3;--bet:#3fb950;--info:#d29922;--pass:#9a9aa3;--warn:#f85149}}}}
body{{margin:0;background:var(--bg);color:var(--fg);font:16px/1.4 system-ui,sans-serif;padding:0 16px 40px;max-width:760px;margin-inline:auto}}
h1{{font-size:1.25rem;margin:18px 0 4px}} .sub{{color:var(--muted);margin:0 0 12px}}
.banner{{border:1px solid var(--warn);border-left-width:5px;border-radius:6px;padding:8px 10px;margin:8px 0;background:var(--card)}}
.counts{{display:flex;gap:8px;flex-wrap:wrap;margin:10px 0}} .counts span{{border:1px solid var(--line);border-radius:12px;padding:2px 10px;background:var(--card)}}
.card{{background:var(--card);border:1px solid var(--line);border-left-width:5px;border-radius:8px;padding:10px 12px;margin:10px 0}}
.card.bet{{border-left-color:var(--bet)}} .card.info{{border-left-color:var(--info)}} .card.pass,.card.fcs,.card.no_line{{border-left-color:var(--pass);opacity:.85}}
header{{display:flex;gap:8px;align-items:baseline;flex-wrap:wrap}} .st{{font-size:.75rem;font-weight:700;letter-spacing:.04em}}
.card.bet .st{{color:var(--bet)}} .card.info .st{{color:var(--info)}} .ko{{margin-left:auto;color:var(--muted);font-size:.85rem}}
.mk{{margin:4px 0}} .muted,.note{{color:var(--muted);font-size:.85rem}} .flags{{color:var(--warn);font-size:.85rem;margin:4px 0 0}}
dl{{display:grid;grid-template-columns:repeat(6,auto);gap:2px 10px;margin:6px 0;font-size:.85rem}} dt{{color:var(--muted)}} dd{{margin:0;font-variant-numeric:tabular-nums}}
@media(max-width:480px){{dl{{grid-template-columns:repeat(4,auto)}}}}
footer{{color:var(--muted);font-size:.8rem;margin-top:24px}}
</style></head><body>
<h1>CFB Edge Board: {board.meta['season']} week {board.meta['week']}</h1>
<p class="sub">Parameters {e(str(board.meta['params_version']))}. Most rows should be PASS: an honest no-bet is a correct answer.</p>
{banners}
<div class="counts">{"".join(f"<span>{k} {v}</span>" for k, v in counts.items())}</div>
{"".join(cards)}
<footer>BET = clears the bar AND the model has been proven out-of-sample for this market. INFO = the model sees
something nobody has proven it may be trusted on: not a bet. This page is a research tool, not betting advice.</footer>
</body></html>
"""
