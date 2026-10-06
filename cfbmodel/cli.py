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
from pathlib import Path

import numpy as np
import pandas as pd

from . import board as board_mod
from . import (backtest, budget, edge, game_model, ingest, learn, params, priors, props, qb,
               playerstate, propsboard, ratings, schema, script, state, status, teams, tracking, validation, week0, weather)
from .config import C, OUTPUT
from .params import P
from .keys import MissingKeyError


# ------------------------------------------------------------------- commands


def cmd_pull(args):
    """Fetch (or reuse from cache) everything the model needs, and say what it cost.

    Any failure stops the run with its reason. The only thing treated as "not there" is
    an explicit HTTP 404 on a dataset that may not exist for a given year.
    """
    ingest.set_current(week=args.week)
    table = []  # (season, endpoint, rows, calls, note)

    def step(season, label, fn, optional=False):
        before = ingest.calls_used()
        try:
            n, note = len(fn()), ""
        except ingest.CfbdNotFound:
            if not optional:
                raise
            n, note = 0, "not available (HTTP 404)"
        table.append((season, label, n, ingest.calls_used() - before, note))

    def weekly(season, label, fn):
        before, rows, weeks = ingest.calls_used(), 0, ingest.regular_weeks(season)
        for w in weeks:
            rows += len(fn(season, w))
        table.append((season, f"{label} ({len(weeks)} weeks)", rows, ingest.calls_used() - before, ""))

    for s in args.seasons:
        step(s, "teams/fbs", lambda s=s: ingest.teams_fbs(s))
        step(s, "games", lambda s=s: ingest.games(s))
        step(s, "lines", lambda s=s: ingest.lines(s))
        step(s, "recruiting/teams", lambda s=s: ingest.recruiting_teams(s), optional=True)
        step(s, "player/returning", lambda s=s: ingest.returning_production(s), optional=True)
        step(s, "player/portal", lambda s=s: ingest.portal(s), optional=True)
        step(s, "player/usage", lambda s=s: ingest.usage(s), optional=True)
        weekly(s, "plays", ingest.plays)
        weekly(s, "games/players", ingest.player_box)
    step("-", "venues", ingest.venues)

    print(f"\n{'season':<7}{'endpoint':<26}{'rows':>10}{'calls':>7}  note")
    for season, label, n, calls, note in table:
        print(f"{season!s:<7}{label:<26}{n:>10,}{calls:>7}  {note}")
        if n == 0 and not note:
            print(f"{'':<7}  ^ zero rows: expected for a season that has not started, otherwise check")
    bud = ingest.budget()
    print(f"\ncalls this run: {ingest.calls_used()}   "
          f"used this month: {bud.used()} of {budget.MONTHLY_LIMIT}   remaining: {bud.remaining()}")
    unmatched = teams.seen_unmatched()
    if unmatched:
        path = teams.write_unmatched_log()
        top = ", ".join(f"{n} ({r})" for n, r in sorted(unmatched.items(), key=lambda kv: -kv[1])[:8])
        print(f"\nnames that are not FBS teams: {len(unmatched)} (mostly FCS opponents). "
              f"Biggest: {top}\nfull list: {path}\n"
              "If an FBS team is on that list under another spelling, add it to data/team_aliases.csv.")


def cmd_docs_data(args):
    """Rewrite docs/DATA.md from schema.py (a test fails if the two drift apart)."""
    path = Path(__file__).resolve().parent.parent / "docs" / "DATA.md"
    path.write_text(schema.render_markdown())
    print(f"wrote {path}")


def _load(seasons):
    g = pd.concat([ingest.games(s) for s in seasons], ignore_index=True)
    ln = pd.concat([ingest.lines(s) for s in seasons], ignore_index=True)
    pbp = {}
    for s in seasons:
        d = ingest.season_plays(s)
        if not d.empty:
            pbp[s] = d
    return g, ln, pbp


def _prior_weights_by_season(seasons, ln, rec, rp, pt):
    """Preseason blend weights for each season, fit ONLY on earlier seasons (so judging the
    early weeks of season S never uses S's own results). Falls back to the labelled unfitted
    defaults when fewer than two earlier seasons exist."""
    out = {}
    loaded = sorted(ln["season"].unique())
    for s in seasons:
        earlier = [y for y in loaded if y < s]
        weights, source = dict(priors.DEFAULT_WEIGHTS), "default_unfitted"
        if len(earlier) >= 3:
            hist = priors.build_prior_history(ln[ln["season"] < s], rec, rp, pt, earlier)
            if hist["season"].nunique() >= 2:
                weights, source = priors.fit_prior_weights(hist), f"fitted on seasons < {s}"
        out[s] = (weights, source)
    return out


def _walk_forward_data(seasons):
    """Load everything and run the walk-forward for `seasons`. Returns (games, lines, res)."""
    load = [seasons[0] - 1] + seasons
    ingest.set_current(season=max(seasons))
    g, ln, pbp = _load(load)
    allp = pd.concat(pbp.values(), ignore_index=True)
    print(f"loaded {len(g)} games, {len(allp):,} plays for {load[0]}-{load[-1]}")
    rec = {y: ingest.recruiting_teams(y) for y in range(load[0] - 3, load[-1] + 1)}
    rp = {y: ingest.returning_production(y) for y in load}
    pt = {y: _portal_or_empty(y) for y in load}
    weights = _prior_weights_by_season(seasons, ln, rec, rp, pt)
    prior = {s: state.preseason_prior_points(s, ln, rec, rp[s], pt[s], weights=weights[s][0]) for s in seasons}
    res = validation.walk_forward(g, ln, allp, seasons, prior_by_season=prior)
    res.attrs["prior_weight_sources"] = {s: w[1] for s, w in weights.items()}
    return g, ln, res


def cmd_validate(args):
    """The gate: does the model add information beyond the betting line? See validation.py."""
    seasons = sorted(args.seasons)
    g, ln, res = _walk_forward_data(seasons)
    res.to_csv(OUTPUT / "walkforward.csv", index=False)
    status_ = validation.run_validation(res)
    report = validation.build_report(status_, res)
    report += "\nPreseason weights used per season: " + "; ".join(
        f"{s}: {src}" for s, src in res.attrs["prior_weight_sources"].items()) + "\n"
    sp, rp_path = validation.write_outputs(status_, report)

    passed = [k for k, v in status_["markets"].items() if v["passed"]]
    passed_segs = [x for x in status_["segments"] if x["passed"]]
    print(f"\nwalk-forward rows: {len(res)}   skipped weeks: {len(res.attrs.get('skipped', []))}")
    if status_["status"] == "SUSPECTED_LEAK":
        print("!! SUSPECTED_LEAK: a model coefficient is above 0.5. Do not trust anything until this is explained.")
    if not passed and not passed_segs:
        print("RESULT: nothing passed. The market is not beaten (yet). The board will show no BET rows.")
    else:
        print(f"RESULT: markets passed: {passed or 'none'}; segments passed: {len(passed_segs)}")
    print(f"wrote {sp}\nwrote {rp_path}")


def _market_frame(g: pd.DataFrame, ln: pd.DataFrame) -> pd.DataFrame:
    """Every FBS-vs-FBS regular-season game that has a closing line and a result."""
    mk = validation._market_numbers(ln)
    d = g[g["homePoints"].notna() & g["week"].notna()]
    if "season_type" in d.columns:
        d = d[d["season_type"] == "regular"]
    d = d[d["home_is_fbs"] & d["away_is_fbs"]]
    d = d.merge(mk, left_on="id", right_index=True)
    d = d.rename(columns={"total": "actual_total"})
    return d.dropna(subset=["spread_close"])[["id", "season", "week", "margin", "actual_total", "spread_close",
                                              "total_close", "spread_open", "total_open"]]


def _optional_csv(name):
    p = OUTPUT / name
    return pd.read_csv(p) if p.exists() else None


def cmd_learn(args):
    """After week W has finished: re-learn formula parameters, but adopt a change only if it is
    proven better out-of-sample (see learn.py). Usually the answer is 'no change'."""
    s, w = args.season, args.week
    seasons = sorted(args.seasons)
    g, ln, res = _walk_forward_data(seasons)
    data = learn.LearnData(market=_market_frame(g, ln), wf=res,
                           weather=_optional_csv("weather_history.csv"), props=_optional_csv("prop_games.csv"))
    rep = learn.run_learn(data, s, w + 1, full=args.full, force=args.even_if_not_due)
    if not rep.due:
        print("; ".join(rep.notes))
        return
    print(f"learning run for season {s}, after week {w}  (current version: {params.active_version()})\n")
    for r in rep.results:
        line = f"  {r.name:<14}{r.status:<9}"
        if r.evidence.get("n_window"):
            line += (f"gain {r.evidence['mean_gain']:+.5f}/game  CI [{r.evidence['ci_low']:+.5f}, "
                     f"{r.evidence['ci_high']:+.5f}]  window {r.evidence['n_window']}")
        print(line)
        for why in r.reasons:
            print(f"      - {why}")
    print(f"\nNEW VERSION {rep.version}: see output/models/{rep.version}/diff.md" if rep.version
          else "\nNo change: the current parameters stay.")

    drift = learn.check_drift(res, s, w + 1)
    learn.write_drift_status(drift)
    print(f"drift check: {'ALARM, model weight forced to 0: ' if drift['active'] else 'ok. '}{drift['reason']}")
    path = learn.write_postmortem(learn.postmortem(res, s, w), s, w)
    print(f"wrote {path}")


def cmd_board(args):
    """The weekly product: every game priced, honest statuses, one CSV and one phone-friendly page."""
    s, w = args.season, args.week
    ingest.set_current(week=w)
    games_all = ingest.games(s)
    wk = games_all[games_all["week"] == w]
    if "season_type" in wk.columns:
        wk = wk[wk["season_type"] == "regular"]
    if wk.empty:
        print(f"no games found for season {s} week {w}")
        return 1
    g, ln, pbp = _load([x for x in (s - 2, s - 1, s) if x >= 2015])
    allp = pd.concat(pbp.values(), ignore_index=True)
    rec = {y: ingest.recruiting_teams(y) for y in range(s - 3, s + 1)}
    prior = state.preseason_prior_points(s, ln, rec, ingest.returning_production(s), _portal_or_empty(s))
    ts = state.build_team_state(s, w - 1, ln, allp, prior_pts=prior, w_model=1.0)      # the model's own view
    tot = ratings.fit_total_ratings(ln, s, w)

    wk_games = wk.reset_index(drop=True)
    if not args.no_weather:
        wk_games = weather.attach_weather(wk_games, ingest.venues(), mode="forecast")
    qb_table = None
    if "passer" in allp.columns:
        qb_table = qb.build_qb_table(allp, s, w)
    else:
        print("note: plays have no 'passer' column, so QB-uncertainty flags are unavailable.")

    wk_lines = ln[(ln["season"] == s) & (ln["week"] == w)]
    b = board_mod.build_board(
        s, w, wk_games, wk_lines, ts, tot, validation_status=status.read_validation(),
        drift_active=status.drift_active(), params_version=params.active_version(), qb_table=qb_table,
        validation_age_days=status.validation_age_days(), prior_source=prior.attrs.get("weights_source"))
    csv_path, html_path = OUTPUT / f"board_{s}_w{w}.csv", OUTPUT / f"board_{s}_w{w}.html"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.write_text(board_mod.to_csv(b))
    html_path.write_text(board_mod.to_html(b), encoding="utf-8")
    (OUTPUT / f"board_{s}_w{w}.json").write_text(json.dumps(board_mod.to_meta(b), indent=2))
    print(board_mod.summary(b))
    n_logged = tracking.log_lines(wk_lines, s, w)
    print(f"\nwrote {csv_path}\nwrote {html_path}   (open it in a browser, or send it to your phone)"
          f"\nlogged {n_logged} book lines to {tracking.lines_log_path()}")


def cmd_log_bet(args):
    """Record a bet you actually placed, so CLV and ROI can be tracked after the game."""
    games = ingest.games(args.season)
    game_id, label = tracking.resolve_game(args.game, games, args.week)
    rec = tracking.add_bet(season=args.season, week=args.week, game_id=game_id, game=label, market=args.market,
                           side=args.side, line=args.line, price=args.price, book=args.book, stake=args.stake,
                           notes=args.notes or "")
    print(f"logged bet #{rec['bet_id']}: {label}  {args.market} {args.side} {args.line:+g} at {args.price:+g} "
          f"({args.book}), stake {args.stake:g}\nsaved to {tracking.bets_path()}")


def cmd_clv(args):
    """After the games: closing line value, results and ROI for every logged bet."""
    bets = tracking.load_bets()
    if bets.empty:
        print(f"no bets logged yet ({tracking.bets_path()}). Add one with `python -m cfbmodel log-bet`.")
        return
    games = pd.concat([ingest.games(s) for s in sorted(bets["season"].unique())], ignore_index=True)
    lines = pd.concat([ingest.lines(s) for s in sorted(bets["season"].unique())], ignore_index=True)
    settled = tracking.settle(bets, games, lines)
    text = tracking.report(settled, games)
    settled.to_csv(OUTPUT / "bets_settled.csv", index=False)
    (OUTPUT / "clv_report.md").write_text(text)
    print(text)
    print(f"wrote {OUTPUT / 'bets_settled.csv'}\nwrote {OUTPUT / 'clv_report.md'}")


def cmd_props_board(args):
    """Price the prop lines YOU typed into props_lines.csv. BET stays locked until a backtest on your
    logged lines shows positive CLV (see propsboard.py)."""
    s, w = args.season, args.week
    ingest.set_current(week=w)
    history = propsboard.load_lines(Path(args.lines) if args.lines else None)
    latest = propsboard.latest_lines(history)
    games_all = ingest.games(s)
    wk = games_all[games_all["week"] == w]
    if "season_type" in wk.columns:
        wk = wk[wk["season_type"] == "regular"]
    ps = state.read_player_state(s, w - 1)
    if ps is None:
        print(f"no saved player state for week {w - 1}; building it from box scores (run `update` to save it).")
        box = pd.concat([ingest.player_box(s, x) for x in range(0, w)], ignore_index=True)
        prev_last = state.last_week_with_players(s - 1)
        ps = playerstate.build_player_state(box, s, w - 1, state.read_player_state(s - 1, prev_last) if prev_last is not None else None)
    cons = board_mod._consensus(ingest.lines(s).query("week == @w")) if len(wk) else pd.DataFrame()
    pace = None
    try:
        pace = state.read_team_state(s, w - 1)["pace"]
    except FileNotFoundError:
        print("note: no saved team state, so every team gets the league-average pace.")
    log_p = propsboard.log_path()
    log = pd.read_csv(log_p) if log_p.exists() else pd.DataFrame()
    ok, why, _ = propsboard.eligibility(log, history)
    b = propsboard.build_props_board(latest, ps, wk, cons, pace, week=w, eligible=ok, params_version=params.active_version())
    out = OUTPUT / f"props_board_{s}_w{w}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    b.to_csv(out, index=False, float_format="%.4f")
    n = propsboard.log_recommendations(b)
    print(b[["player", "market", "side", "line", "p_play", "proj_mean", "p_over", "ev", "status", "flags"]].head(25).to_string(index=False))
    print(f"\n{why}\n{b['status'].value_counts().to_dict()}   wrote {out}   logged {n} model-liked sides to {log_p}")
    print("Remember: CFB has no injury report. p_play is YOUR number; blank means 1.0 and is flagged.")


def cmd_models(args):
    t = learn.models_table()
    print(t.to_string(index=False))
    print("\n* = current. Details of each learned version: output/models/<version>/diff.md")


def cmd_rollback(args):
    prev = learn.rollback(args.version)
    print(f"current parameter version: {prev} -> {args.version}  (nothing was deleted; you can roll forward again)")


def cmd_postmortem(args):
    res = pd.read_csv(OUTPUT / "walkforward.csv")
    text = learn.postmortem(res, args.season, args.week)
    path = learn.write_postmortem(text, args.season, args.week)
    print(text)
    print(f"wrote {path}")


def _portal_or_empty(season):
    try:
        return ingest.portal(season)
    except ingest.CfbdNotFound:
        print(f"note: CFBD has no transfer-portal data for {season}; portal term is zero.")
        return pd.DataFrame()


def cmd_preseason(args):
    s = args.season
    ln = pd.concat([ingest.lines(y) for y in (s - 2, s - 1)], ignore_index=True)
    rec = {y: ingest.recruiting_teams(y) for y in range(s - 3, s + 1)}
    prior = state.preseason_prior_points(s, ln, rec, ingest.returning_production(s), _portal_or_empty(s))
    out = prior.to_frame("rating").sort_values("rating", ascending=False)
    out.to_csv(OUTPUT / f"preseason_{s}.csv")
    print(out.head(30).round(2).to_string())
    print(f"\nwrote output/preseason_{s}.csv   (blend weights: {prior.attrs.get('weights_source')})")
    if prior.attrs.get("weights_source") == "default_unfitted":
        print("NOTE: using unfitted default weights. Run `python -m cfbmodel fit-priors` once "
              "several seasons are pulled.")
    print("""
Sanity check this list against your own eyes before it prices a single game.
If a team you know is bad shows up top-15, the input is wrong, not the sport.
""")


def cmd_fit_priors(args):
    """Fit the preseason blend weights on past seasons and save them with their CV score."""
    seasons = sorted(args.seasons)
    ln = pd.concat([ingest.lines(y) for y in range(min(seasons) - 1, max(seasons) + 1)], ignore_index=True)
    rec = {y: ingest.recruiting_teams(y) for y in range(min(seasons) - 3, max(seasons) + 1)}
    rp = {y: ingest.returning_production(y) for y in seasons}
    pt = {y: _portal_or_empty(y) for y in seasons}
    hist = priors.build_prior_history(ln, rec, rp, pt, seasons)
    weights, cv = priors.fit_prior_weights_cv(hist)
    path = priors.save_prior_weights(weights, cv, cv["seasons"])
    print("fitted preseason weights:", json.dumps(weights))
    print(f"out-of-sample (leave-one-season-out, n={cv['n']}):  r={cv['r_weighted_composite']:.3f}  "
          f"vs last-season-rating-only r={cv['r_prior_rating_only']:.3f}   rmse={cv['rmse']:.2f}  r2={cv['r2']:.3f}")
    print(f"wrote {path}")


def cmd_update(args):
    """After week W has finished: refit team ratings and update player posteriors."""
    s, w = args.season, args.week
    ingest.set_current(week=w + 1)
    g, ln, pbp = _load([x for x in (s - 2, s - 1, s) if x >= 2015])
    allp = pd.concat(pbp.values(), ignore_index=True)
    box = pd.concat([ingest.player_box(s, x) for x in range(0, w + 1)], ignore_index=True)
    rec = {y: ingest.recruiting_teams(y) for y in range(s - 3, s + 1)}
    prior = state.preseason_prior_points(s, ln, rec, ingest.returning_production(s), _portal_or_empty(s))
    res = state.run_update(s, w, ln, allp, box, prior_pts=prior)
    t = res["teams"].sort_values("rating", ascending=False)
    print(t[["rating", "market_rating", "model_rating_shrunk", "prior_weight"]].head(15).round(2).to_string())
    print(f"\nw_model = {t['w_model'].iloc[0]:.2f}  (0 means: market only, nothing proven yet)")
    print(f"teams:   {res['teams_path']}  ({len(res['teams'])} teams)")
    print(f"players: {res['players_path']}  ({len(res['players'])} player-groups)")


def _season_pbp(year):
    return ingest.season_plays(year)


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
        slate = weather.attach_weather(slate, ingest.venues(), mode="forecast")

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
            sens = P.script.starter_pull_sensitivity if d["rank"] == 1 else P.script.backup_pull_sensitivity

            if d["group"] == "rush" and float(r["carries"]) >= 10:
                sim = script.simulate_rush_yards_joint(
                    float(d["share"]), sc, float(r["ypc"]), P.props.rush_ypc_sd, sigma,
                    p_play=args.p_play, share_pull_sensitivity=sens)
                mk = "rush_yds"
            elif d["group"] == "rec" and float(r["rec"]) >= 6:
                out = script.simulate_rec_yards_joint(
                    float(d["share"]), sc, float(r["ypr"]) * P.props.catch_rate_mean, P.props.catch_rate_mean, sigma,
                    p_play=args.p_play, share_pull_sensitivity=P.script.rec_pull_sensitivity * sens)
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
    net = argparse.ArgumentParser(add_help=False)
    net.add_argument("--force", action="store_true",
                     help="allow CFBD calls past the 950-per-month safety stop")
    net.add_argument("--refresh", action="store_true",
                     help="ignore the cache and re-fetch everything this run touches")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("pull", parents=[net]); p.add_argument("--seasons", type=int, nargs="+", required=True)
    p.add_argument("--week", type=int, default=None,
                   help="current week of the season in progress (weeks before it are never re-pulled)")
    p.set_defaults(func=cmd_pull)

    p = sub.add_parser("validate", parents=[net])
    p.add_argument("--seasons", type=int, nargs="+", required=True)
    p.set_defaults(func=cmd_validate)

    p = sub.add_parser("preseason", parents=[net]); p.add_argument("--season", type=int, required=True)
    p.set_defaults(func=cmd_preseason)

    p = sub.add_parser("fit-priors", parents=[net], help="fit and save the preseason blend weights")
    p.add_argument("--seasons", type=int, nargs="+", required=True)
    p.set_defaults(func=cmd_fit_priors)

    p = sub.add_parser("learn", parents=[net], help="re-learn formula parameters; adopt only if proven better")
    p.add_argument("--season", type=int, required=True)
    p.add_argument("--week", type=int, required=True, help="the week that just finished")
    p.add_argument("--seasons", type=int, nargs="+", required=True, help="seasons for the walk-forward, e.g. 2021 2022 ... 2026")
    p.add_argument("--full", action="store_true", help="season-end run: also judge the early-season decay")
    p.add_argument("--even-if-not-due", action="store_true", help="run even if it is not due yet")
    p.set_defaults(func=cmd_learn)

    p = sub.add_parser("board", parents=[net], help="price a week: the edge board (CSV + HTML)")
    p.add_argument("--season", type=int, required=True)
    p.add_argument("--week", type=int, required=True, help="the week about to be played")
    p.add_argument("--no-weather", action="store_true")
    p.set_defaults(func=cmd_board)

    p = sub.add_parser("log-bet", parents=[net], help="record a bet you placed")
    p.add_argument("--season", type=int, required=True)
    p.add_argument("--week", type=int, required=True)
    p.add_argument("--game", required=True, help="the label on the board, e.g. \"Auburn @ Georgia\", or the game id")
    p.add_argument("--market", required=True, choices=["spread", "total", "moneyline"])
    p.add_argument("--side", required=True, choices=["home", "away", "over", "under"])
    p.add_argument("--line", type=float, required=True,
                   help="the number you SAW for your side (Auburn +6.5 -> 6.5, side away); the price for a moneyline")
    p.add_argument("--price", type=float, required=True, help="American odds, e.g. -110")
    p.add_argument("--book", required=True)
    p.add_argument("--stake", type=float, required=True, help="amount staked (units or dollars; be consistent)")
    p.add_argument("--notes", default="")
    p.set_defaults(func=cmd_log_bet)

    p = sub.add_parser("clv", parents=[net], help="closing line value, results and ROI of your logged bets")
    p.set_defaults(func=cmd_clv)

    p = sub.add_parser("props-board", parents=[net], help="price the prop lines you typed into props_lines.csv")
    p.add_argument("--season", type=int, required=True)
    p.add_argument("--week", type=int, required=True)
    p.add_argument("--lines", default=None, help="path to your props lines CSV (default: output/props_lines.csv)")
    p.set_defaults(func=cmd_props_board)

    p = sub.add_parser("models", help="list parameter versions")
    p.set_defaults(func=cmd_models)

    p = sub.add_parser("rollback", help="make an earlier parameter version current again")
    p.add_argument("version")
    p.set_defaults(func=cmd_rollback)

    p = sub.add_parser("postmortem", help="what drove the biggest misses of a week")
    p.add_argument("--season", type=int, required=True)
    p.add_argument("--week", type=int, required=True)
    p.set_defaults(func=cmd_postmortem)

    p = sub.add_parser("update", parents=[net], help="after week W: refit team ratings, update players")
    p.add_argument("--season", type=int, required=True)
    p.add_argument("--week", type=int, required=True, help="the week that just finished")
    p.set_defaults(func=cmd_update)

    p = sub.add_parser("slate", parents=[net])
    p.add_argument("--season", type=int, required=True)
    p.add_argument("--week", type=int, required=True)
    p.add_argument("--w-model", type=float, default=None,
                   help="experiments only; default reads output/validation_status.json (0 if absent)")
    p.add_argument("--no-weather", action="store_true")
    p.set_defaults(func=cmd_slate)

    p = sub.add_parser("props", parents=[net])
    p.add_argument("--season", type=int, required=True)
    p.add_argument("--week", type=int, required=True)
    p.add_argument("--home", required=True)
    p.add_argument("--away", required=True)
    p.add_argument("--w-model", type=float, default=None,
                   help="experiments only; default reads output/validation_status.json (0 if absent)")
    p.add_argument("--p-play", type=float, default=1.0,
                   help="probability the player suits up (CFB has no injury report)")
    p.add_argument("--home-qb-out", action="store_true")
    p.add_argument("--away-qb-out", action="store_true")
    p.set_defaults(func=cmd_props)

    p = sub.add_parser("weather", parents=[net])
    p.add_argument("--season", type=int, required=True)
    p.add_argument("--week", type=int, required=True)
    p.set_defaults(func=cmd_weather)

    p = sub.add_parser("qb", parents=[net])
    p.add_argument("--season", type=int, required=True)
    p.add_argument("--week", type=int, required=True)
    p.set_defaults(func=cmd_qb)

    p = sub.add_parser("docs-data", parents=[net], help="regenerate docs/DATA.md from the schema")
    p.set_defaults(func=cmd_docs_data)

    p = sub.add_parser("week0", parents=[net], help="pull -> preseason ratings -> bet card, one command")
    p.add_argument("--season", type=int, default=2026)
    p.add_argument("--week", type=int, default=0)
    p.add_argument("--w-model", type=float, default=None,
                   help="experiments only; default reads output/validation_status.json (0 if absent)")
    p.add_argument("--no-weather", action="store_true")
    p.set_defaults(func=week0.run)

    args = ap.parse_args(argv)
    ingest.configure(force=getattr(args, "force", False), refresh=getattr(args, "refresh", False))
    try:
        return args.func(args) or 0
    except (ingest.CfbdError, budget.BudgetExceeded, schema.SchemaError, learn.LearnError,
            params.ParamsError, tracking.TrackingError, propsboard.PropsError) as e:
        print(f"\n{type(e).__name__}: {e}", file=sys.stderr)
        return 2
    except MissingKeyError as e:
        # A missing key is a setup problem, not a bug: say what to do, exit 2.
        print(f"\n{e}", file=sys.stderr)
        return 2
