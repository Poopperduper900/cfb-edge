#!/usr/bin/env python3
"""
Week 0 end-to-end, one command.

    export CFBD_API_KEY=...
    python run_week0.py --season 2026

Does the whole chain: pull -> preseason ratings -> project -> compare to market
-> bet card. Roughly 40 API calls the first time, zero on re-runs.

WHAT THIS CAN AND CANNOT KNOW
-----------------------------
Week 0 is the one week where the model has no current-season data — and neither
does the market. That symmetry is why weeks 0-3 are the most exploitable stretch
of the calendar, and it is also why the ratings here rest entirely on priors:

    2025 closing lines   -> market-implied power ratings (the strongest anchor)
    2025 play-by-play    -> opponent-adjusted EPA
    recruiting/returning -> preseason prior for roster turnover
    transfer portal      -> position-weighted net movement

None of that has seen a 2026 snap. Treat every number below as a prior with a
wide distribution around it, which is exactly how the pmf prices it.
"""
from __future__ import annotations

import argparse
import numpy as np
import pandas as pd

from cfbmodel import edge, game_model, ingest, priors, ratings, weather
from cfbmodel.config import C, OUTPUT

P4 = {"SEC", "Big Ten", "Big 12", "ACC"}


def build_ratings(season: int, w_model: float):
    prev = season - 1
    print(f"pulling {prev-1}-{prev} lines and plays...")
    lines = pd.concat([ingest.lines(y) for y in (prev - 1, prev)], ignore_index=True)

    frames = []
    for y in (prev - 1, prev):
        for wk in range(0, 16):
            try:
                d = ingest.plays(y, wk)
            except Exception:
                continue
            if not d.empty:
                frames.append(d)
    pbp = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    # as-of week 99 of the previous season = "everything that ever happened"
    mkt = ratings.fit_market_ratings(lines, prev, 99)
    print(f"  market ratings: {len(mkt)} teams, HFA {mkt['market_hfa'].iloc[0]:+.2f}")

    epa = ratings.fit_epa_ratings(pbp, prev, 99, split_pass_rush=False)
    rt = ratings.blend(epa, mkt, w_model=w_model)

    # roster turnover: last season's ratings do not carry over intact
    print("pulling recruiting / returning production / portal...")
    rec = {}
    for y in range(season - 4, season + 1):
        try:
            rec[y] = ingest.recruiting_teams(y)
        except Exception:
            pass
    try:
        rp = ingest.returning_production(season)
    except Exception:
        rp = pd.DataFrame()
    try:
        pt = ingest.portal(season)
    except Exception:
        pt = pd.DataFrame()

    pre = priors.build_preseason_ratings(
        prior_season_ratings=rt["rating"],
        recruiting=priors.rolling_recruiting(rec, season),
        returning=priors.returning_production_score(rp),
        portal=priors.portal_score(pt, season),
    )

    out = rt.join(pre[["rating"]].rename(columns={"rating": "preseason"}), how="left")
    out["preseason"] = out["preseason"].fillna(out["rating"])
    # blend last year's measured strength with the turnover-aware prior
    out["rating"] = 0.55 * out["rating"] + 0.45 * out["preseason"]
    out["rating"] -= out["rating"].mean()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, default=2026)
    ap.add_argument("--week", type=int, default=0)
    ap.add_argument("--w-model", type=float, default=0.30)
    ap.add_argument("--no-weather", action="store_true")
    args = ap.parse_args()
    s, w = args.season, args.week

    rt = build_ratings(s, args.w_model)
    print(f"\ntop 15 preseason:")
    print(rt[["rating"]].sort_values("rating", ascending=False).head(15).round(2).to_string())

    games = ingest.games(s)
    slate = games[games["week"] == w]
    if slate.empty:
        print(f"\nno games found for {s} week {w}")
        return
    cur = ingest.lines(s)
    cur = cur[cur["week"] == w]

    if not args.no_weather:
        try:
            slate = weather.attach_weather(slate, ingest.venues(), mode="forecast")
        except Exception as e:
            print(f"weather unavailable ({e})")

    rows = []
    for _, g in slate.iterrows():
        h, a = g["homeTeam"], g["awayTeam"]
        if h not in rt.index or a not in rt.index:
            print(f"  skipping {a} @ {h} — no rating (FCS or name mismatch)")
            continue
        neutral = bool(g.get("neutralSite"))
        proj = game_model.project_game(h, a, rt, neutral=neutral)

        wx_pts, _ = weather.total_adjustment(g.to_dict()) if not args.no_weather else (0.0, 0.0)
        proj["total"] += wx_pts

        book = cur[cur["gameId"] == g["id"]]
        if book.empty:
            print(f"  no line yet: {a} @ {h}  model {proj['margin']:+.1f}")
            continue
        spread = float(book["spread_close"].median())
        total_line = float(book["total_close"].median())

        soft = game_model.market_softness(pd.Series({
            "week": w, "home_conf": g.get("homeConference"),
            "away_conf": g.get("awayConference")}))

        # source="model": our projection is a prior, not a closing line, so the
        # distribution must be wider than the market's residual
        cp = game_model.cover_prob(proj["margin"], proj["total"], spread, source="model")
        tp = game_model.total_probs(proj["total"], total_line, proj["margin"])

        sp_card = edge.evaluate_spread(proj["margin"], spread, -110, cp, soft)
        tt_card = edge.evaluate_total(proj["total"], total_line, -110, -110, tp, soft)

        rows.append({
            "game": f"{a} @ {h}" + (" (N)" if neutral else ""),
            "mkt_spread": spread, "model_spread": sp_card["model_line"],
            "spread_diff": round(sp_card["model_line"] - spread, 1),
            "spread_side": sp_card["side"], "spread_edge": sp_card["edge_pts"],
            "spread_bet": sp_card["bet"],
            "mkt_total": total_line, "model_total": tt_card["model_line"],
            "total_diff": round(tt_card["model_line"] - total_line, 1),
            "total_side": tt_card["side"], "total_edge": tt_card["edge_pts"],
            "total_bet": tt_card["bet"],
            "wx": round(wx_pts, 1), "soft": soft,
        })

    df = pd.DataFrame(rows)
    if df.empty:
        print("\nnothing priced")
        return

    pd.set_option("display.width", 200)
    print(f"\n{'='*100}\nWEEK {w} — MODEL vs MARKET\n{'='*100}")
    print(df.sort_values("spread_edge", ascending=False).to_string(index=False))
    df.to_csv(OUTPUT / f"week{w}_{s}.csv", index=False)

    bets = df[df["spread_bet"] | df["total_bet"]]
    print(f"\n{len(bets)} of {len(df)*2} markets clear the threshold.")
    print(f"mean |spread disagreement| = {df['spread_diff'].abs().mean():.2f} pts")

    if df["spread_diff"].abs().mean() > 6:
        print("\n!! Mean disagreement above 6 points means the ratings are probably")
        print("   broken, not that the market is. Check the top-15 list above against")
        print("   your own eyes before acting on any of this.")

    print("""
BEFORE YOU ACT ON ANY OF THIS
  Run:  python run_slate.py validate --seasons 2022 2023 2024 2025
  If the model_margin coefficient is not significant, these disagreements are
  noise with a decimal point. Week 0 is the softest market of the year, but soft
  is not the same as wrong — the market's Week 0 number is also built from
  priors, and it has more of them than you do.
""")


if __name__ == "__main__":
    main()
