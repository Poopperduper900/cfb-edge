"""
Command line for the whole project:  python -m cfbmodel <command>

    # the key is read from .env (or the CFBD_API_KEY environment variable)

    # one-time / weekly data pull (cached; costs ~20 API calls per season)
    python -m cfbmodel pull --seasons 2019 2020 2021 2022 2023 2024 2025

    # the test you run BEFORE betting anything
    python -m cfbmodel validate --seasons 2022 2023 2024 2025

    # preseason ratings for weeks 0-2 (no current-season games yet)
    python -m cfbmodel preseason --season 2026

    # in-season slate
    python -m cfbmodel slate --season 2026 --week 5

    # props for one game
    python -m cfbmodel props --season 2026 --week 5 --home Auburn --away Georgia
"""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import pandas as pd

from . import (backtest, edge, game_model, ingest, priors, props, qb,
               ratings, script, week0, weather)
from .config import C, OUTPUT
from .keys import MissingKeyError


# ------------------------------------------------------------------- commands


def cmd_pull(args):
    total = 0
    for s in args.seasons:
        ingest.games(s)
        ingest.lines(s)
        ingest.recruiting_teams(s)
        try:
            ingest.returning_production(s)
        except Exception as e:  # endpoint occasionally 404s for old seasons
            print(f"  returning production {s}: {e}")
        for w in range(0, 16):
            try:
                ingest.plays(s, w)
                ingest.player_box(s, w)
            except Exception:
                pass
        print(f"pulled {s}  (cumulative API calls this run: {ingest.calls_used()})")
        total = ingest.calls_used()
    print(f"\ndone. {total} calls used. Free tier is 1,000/month; everything is "
          f"cached, so re-runs cost zero.")


def _load(seasons):
    g = pd.concat([ingest.games(s) for s in seasons], ignore_index=True)
    ln = pd.concat([ingest.lines(s) for s in seasons], ignore_index=True)
    pbp = {}
    for s in seasons:
        frames = []
        for w in range(0, 16):
            try:
                d = ingest.plays(s, w)
            except Exception:
                continue
            if not d.empty:
                frames.append(d)
        if frames:
            pbp[s] = pd.concat(frames, ignore_index=True)
    return g, ln, pbp


def cmd_validate(args):
    g, ln, pbp = _load(args.seasons)
    print(f"loaded {len(g)} games, {sum(len(v) for v in pbp.values()):,} plays")

    res = backtest.walk_forward(pbp, ln, g, args.seasons, start_week=args.start_week,
                                w_model=1.0)
    res = res.dropna(subset=["margin", "market_margin", "model_margin"])
    res.to_csv(OUTPUT / "walkforward.csv", index=False)
    print(f"\nwalk-forward rows: {len(res)}")

    print("\n=== MARKET EFFICIENCY TEST (the one that decides everything) ===")
    print(backtest.market_efficiency_test(res).to_string())
    print("""
Read it like this:
  model_margin p > 0.10  -> your ratings add nothing over the closing line.
                            Do not bet sides or totals. Go to props.
  model_margin p < 0.05  -> there is signal. The coef is your w_model:
                            a coef of 0.15 means blend 15% model / 85% market.
                            It is NOT permission to bet your raw number.
""")

    print("=== SEGMENTED (where CFB edge actually lives, if anywhere) ===")
    res["day_of_week"] = pd.to_datetime(res.get("start_date"), errors="coerce").dt.day_name() \
        if "start_date" in res else "Saturday"
    res["tier"] = np.where(
        res["home_conf"].isin(["SEC", "Big Ten", "Big 12", "ACC"])
        & res["away_conf"].isin(["SEC", "Big Ten", "Big 12", "ACC"]),
        "P4 vs P4",
        np.where(
            ~res["home_conf"].isin(["SEC", "Big Ten", "Big 12", "ACC"])
            & ~res["away_conf"].isin(["SEC", "Big Ten", "Big 12", "ACC"]),
            "G5 vs G5", "mixed"),
    )
    for by in ("tier", "week"):
        print(f"\n-- by {by} --")
        print(backtest.segmented_efficiency(res, by).to_string(index=False))

    print("\n=== OPENER vs CLOSER ===")
    print(backtest.opener_vs_closer(res).to_string(index=False))
    print("\nwrote output/walkforward.csv")


def cmd_preseason(args):
    s = args.season
    rec = {y: ingest.recruiting_teams(y) for y in range(s - 4, s + 1)}
    rp = ingest.returning_production(s)
    try:
        pt = ingest.portal(s)
    except Exception:
        pt = pd.DataFrame()

    prev_pbp = pd.concat(
        [d for y in (s - 1, s - 2) for d in [_season_pbp(y)] if d is not None and not d.empty],
        ignore_index=True,
    )
    prior = ratings.fit_epa_ratings(prev_pbp, s, 0, split_pass_rush=False)["net_epa"]

    out = priors.build_preseason_ratings(
        prior_season_ratings=prior,
        recruiting=priors.rolling_recruiting(rec, s),
        returning=priors.returning_production_score(rp),
        portal=priors.portal_score(pt, s),
    )
    out = out[["rating"]].sort_values("rating", ascending=False)
    out.to_csv(OUTPUT / f"preseason_{s}.csv")
    print(out.head(30).round(2).to_string())
    print(f"\nwrote output/preseason_{s}.csv")
    print("""
Sanity check this list against your own eyes before it prices a single game.
If a team you know is bad shows up top-15, the input is wrong, not the sport.
""")


def _season_pbp(year):
    frames = []
    for w in range(0, 16):
        try:
            d = ingest.plays(year, w)
        except Exception:
            continue
        if not d.empty:
            frames.append(d)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def cmd_slate(args):
    s, w = args.season, args.week
    seasons = [s - 2, s - 1, s]
    g, ln, pbp = _load([x for x in seasons if x >= 2015])
    allp = pd.concat(pbp.values(), ignore_index=True)

    epa = ratings.fit_epa_ratings(allp, s, w)
    mkt = ratings.fit_market_ratings(ln, s, w)
    rt = ratings.blend(epa, mkt, w_model=args.w_model)
    tot = ratings.fit_total_ratings(ln, s, w)

    slate = g[(g["season"] == s) & (g["week"] == w)]
    cur = ln[(ln["season"] == s) & (ln["week"] == w)]

    # weather: forecast-based, horizon-shrunk, domes zeroed
    if not args.no_weather:
        try:
            slate = weather.attach_weather(slate, ingest.venues(), mode="forecast")
        except Exception as e:
            print(f"weather unavailable ({e}); continuing without it")

    cards = []
    for _, row in slate.iterrows():
        h, a = row["homeTeam"], row["awayTeam"]
        if h not in rt.index or a not in rt.index:
            continue
        proj = game_model.project_game(
            h, a, rt, tot, neutral=bool(row.get("neutralSite")),
        )
        book = cur[cur["gameId"] == row["id"]]
        if book.empty:
            continue
        spread = float(book["spread_close"].median())
        total_line = float(book["total_close"].median())

        wx_pts, wx_sd = weather.total_adjustment(row.to_dict())
        proj["total"] += wx_pts
        proj["total_adjustments"]["weather"] = round(wx_pts, 2)
        soft = game_model.market_softness(
            pd.Series({"week": w, "home_conf": row.get("homeConference"),
                       "away_conf": row.get("awayConference")})
        )
        cp = game_model.cover_prob(proj["margin"], proj["total"], spread)
        tp = game_model.total_probs(proj["total"], total_line, proj["margin"])
        for card in (
            edge.evaluate_spread(proj["margin"], spread, -110, cp, soft),
            edge.evaluate_total(proj["total"], total_line, -110, -110, tp, soft),
        ):
            card.update(game=f"{a} @ {h}", season=s, week=w, softness=soft,
                        wx_adj=round(wx_pts, 2))
            cards.append(card)

    df = pd.DataFrame(cards)
    if df.empty:
        print("no games priced — check that lines exist for this week")
        return
    df = df.sort_values("edge_pts", ascending=False)
    df.to_csv(OUTPUT / f"slate_{s}_w{w}.csv", index=False)
    bets = df[df["bet"]]
    print(df.head(25).to_string(index=False))
    print(f"\n{len(bets)} plays clear the threshold out of {len(df)} priced markets.")
    if len(bets) > 8:
        print("More than 8 plays on one slate almost always means a data problem "
              "or a mis-rated team, not a good week. Check the top edges by hand.")
    print(f"wrote output/slate_{s}_w{w}.csv")


def cmd_props(args):
    s, w = args.season, args.week
    box = pd.concat([ingest.player_box(s, x) for x in range(0, w)], ignore_index=True)
    pri = props.build_player_priors(box, s, w)

    g, ln, pbp = _load([s - 1, s])
    allp = pd.concat(pbp.values(), ignore_index=True)
    rt = ratings.blend(
        ratings.fit_epa_ratings(allp, s, w),
        ratings.fit_market_ratings(ln, s, w),
        w_model=args.w_model,
    )

    # QB availability feeds the margin, which feeds every player's game script.
    qbt = qb.build_qb_table(allp, s, w)
    proj = game_model.project_game(
        args.home, args.away, rt,
        home_qb_out=args.home_qb_out, away_qb_out=args.away_qb_out,
    )
    if args.home_qb_out and args.home in qbt.index:
        proj["margin"] -= qbt.loc[args.home, "dropoff_points"] - game_model.qb_dropoff(args.home, rt)
    if args.away_qb_out and args.away in qbt.index:
        proj["margin"] += qbt.loc[args.away, "dropoff_points"] - game_model.qb_dropoff(args.away, rt)

    rows = []
    for team in (args.home, args.away):
        is_home = team == args.home
        margin_for = proj["margin"] if is_home else -proj["margin"]
        pace = float(rt.loc[team, "pace"]) if "pace" in rt and not pd.isna(
            rt.loc[team, "pace"]) else C.pace_mean

        # one script draw per team, shared by every player on it, so teammates'
        # projections stay correlated the way they are on Saturday
        sc = script.simulate_game_script(margin_for, proj["total"], pace, n_sims=40_000)
        depth = script.derive_depth_chart(box, team, s, w)

        for _, d in depth.iterrows():
            row = pri[(pri["team"] == team) & (pri["player"] == d["player"])]
            if row.empty:
                continue
            r = row.iloc[0]
            hist = np.array([d["share"]] * max(int(d["games"]), 1))
            mates = depth[depth["group"] == d["group"]]["share"].to_numpy()
            sigma = script.role_instability(hist, mates, own_share=float(d["share"]))
            # starters lose volume in blowouts; backups gain it
            sens = 1.0 if d["rank"] == 1 else (-1.2 if d["rank"] >= 2 else 0.5)

            if d["group"] == "rush" and float(r["carries"]) >= 10:
                sim = script.simulate_rush_yards_joint(
                    float(d["share"]), sc, float(r["ypc"]), 6.4, sigma,
                    p_play=args.p_play, share_pull_sensitivity=sens)
                mk = "rush_yds"
            elif d["group"] == "rec" and float(r["rec"]) >= 6:
                out = script.simulate_rec_yards_joint(
                    float(d["share"]), sc, float(r["ypr"]) * 0.635, 0.635, sigma,
                    p_play=args.p_play, share_pull_sensitivity=0.7 * sens)
                sim, mk = out["yards"], "rec_yds"
            else:
                continue

            rows.append({
                "player": d["player"], "team": team, "market": mk,
                "rank": int(d["rank"]), "share": round(float(d["share"]), 3),
                "sigma": round(sigma, 3),
                "proj": round(float(sim.mean()), 1),
                "p25": round(float(np.percentile(sim, 25)), 1),
                "p75": round(float(np.percentile(sim, 75)), 1),
                "close_game": round(float(sim[np.abs(sc["margin"]) < 14].mean()), 1),
                "blowout": round(float(sim[np.abs(sc["margin"]) >= 28].mean()), 1),
            })

    df = pd.DataFrame(rows).sort_values(["team", "market", "rank"])
    print(f"\n{args.away} @ {args.home}   proj margin {proj['margin']:+.1f}  "
          f"total {proj['total']:.1f}")
    if not df.empty:
        print(df.to_string(index=False))
        df.to_csv(OUTPUT / f"props_{s}_w{w}_{args.home}.csv", index=False)

    print(f"""
close_game / blowout show the same player under two scripts. When those two
numbers straddle the book's line, the prop is a bet on the GAME, not the player,
and it should be sized accordingly.

These are PROJECTIONS, not bets. Paste the book's line and both prices into
edge.evaluate_prop() for a bet card — props run 6-9% hold and a projection that
beats the line by 3 yards is usually still negative EV.

Confirm the player is playing. CFB has no injury report. Pass --p-play 0.5 for a
genuine game-time decision rather than betting it at face value.
""")


def cmd_weather(args):
    g = ingest.games(args.season)
    g = g[g["week"] == args.week]
    v = ingest.venues()
    wx = weather.attach_weather(g, v, mode="forecast")
    rows = []
    for _, r in wx.iterrows():
        pts, sd = weather.total_adjustment(r.to_dict())
        if abs(pts) < 0.5:
            continue
        rows.append({"game": f"{r['awayTeam']} @ {r['homeTeam']}",
                     "wind": round(r.get("wind_mph") or 0, 1),
                     "gust": round(r.get("gust_mph") or 0, 1),
                     "temp": round(r.get("temp_f") or 0, 0),
                     "precip": round(r.get("precip_in") or 0, 2),
                     "conf": round(r.get("wx_confidence") or 0, 2),
                     "total_adj": round(pts, 2)})
    df = pd.DataFrame(rows).sort_values("total_adj")
    print(df.to_string(index=False) if not df.empty else "no weather-relevant games")
    print("\nRe-run Friday. Forecast confidence below ~0.7 means the adjustment "
          "is mostly noise and the market has not priced it yet either.")


def cmd_qb(args):
    g, ln, pbp = _load([args.season - 1, args.season])
    allp = pd.concat(pbp.values(), ignore_index=True)
    t = qb.build_qb_table(allp, args.season, args.week)
    cols = ["qb1", "qb1_dropbacks", "qb1_epa", "qb2", "qb2_dropbacks",
            "dropoff_points", "dropoff_sd_points", "dropoff_tier"]
    print(t[cols].sort_values("dropoff_points", ascending=False).head(30).to_string())
    t.to_csv(OUTPUT / f"qb_{args.season}_w{args.week}.csv")
    print("\nTier 3 rows are priors, not measurements. The sd column is the one "
          "that should decide whether you act.")


# ---------------------------------------------------------------------- main


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m cfbmodel",
                                 description="CFB game + prop model")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("pull"); p.add_argument("--seasons", type=int, nargs="+", required=True)
    p.set_defaults(func=cmd_pull)

    p = sub.add_parser("validate")
    p.add_argument("--seasons", type=int, nargs="+", required=True)
    p.add_argument("--start-week", type=int, default=4)
    p.set_defaults(func=cmd_validate)

    p = sub.add_parser("preseason"); p.add_argument("--season", type=int, required=True)
    p.set_defaults(func=cmd_preseason)

    p = sub.add_parser("slate")
    p.add_argument("--season", type=int, required=True)
    p.add_argument("--week", type=int, required=True)
    p.add_argument("--w-model", type=float, default=0.35)
    p.add_argument("--no-weather", action="store_true")
    p.set_defaults(func=cmd_slate)

    p = sub.add_parser("props")
    p.add_argument("--season", type=int, required=True)
    p.add_argument("--week", type=int, required=True)
    p.add_argument("--home", required=True)
    p.add_argument("--away", required=True)
    p.add_argument("--w-model", type=float, default=0.35)
    p.add_argument("--p-play", type=float, default=1.0,
                   help="probability the player suits up (CFB has no injury report)")
    p.add_argument("--home-qb-out", action="store_true")
    p.add_argument("--away-qb-out", action="store_true")
    p.set_defaults(func=cmd_props)

    p = sub.add_parser("weather")
    p.add_argument("--season", type=int, required=True)
    p.add_argument("--week", type=int, required=True)
    p.set_defaults(func=cmd_weather)

    p = sub.add_parser("qb")
    p.add_argument("--season", type=int, required=True)
    p.add_argument("--week", type=int, required=True)
    p.set_defaults(func=cmd_qb)

    p = sub.add_parser("week0", help="pull -> preseason ratings -> bet card, one command")
    p.add_argument("--season", type=int, default=2026)
    p.add_argument("--week", type=int, default=0)
    p.add_argument("--w-model", type=float, default=0.30)
    p.add_argument("--no-weather", action="store_true")
    p.set_defaults(func=week0.run)

    args = ap.parse_args(argv)
    try:
        return args.func(args) or 0
    except MissingKeyError as e:
        # A missing key is a setup problem, not a bug: say what to do, exit 2.
        print(f"\n{e}", file=sys.stderr)
        return 2
