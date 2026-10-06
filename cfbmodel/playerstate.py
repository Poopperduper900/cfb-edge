"""
Player ratings as carried-forward state (BUILD_PLAN Phase 3).

Each player in each group (rushing, receiving) has two posteriors that are UPDATED every week
from that week's box scores, never recomputed from scratch: last week's posterior is this
week's prior.

  usage share        Beta(a, b).  Each game adds the player's touches to `a` and the rest of
                     the team's touches to `b`.  mean = a/(a+b);  more games => larger a+b =>
                     smaller uncertainty.
  per-touch yards    Normal(mean, 1/precision).  Each game adds touches/sigma^2 to the
                     precision, so uncertainty shrinks with every touch.

Role changes WIDEN the posteriors on purpose. A role change is (a) a new depth rank inside the
team's group (RB2 becomes RB1) or (b) a player coming back after missing team games. The
old evidence was collected in a different job, so a fraction of it is forgotten:
a, b and the precision are multiplied by ROLE_CHANGE_RETENTION (< 1).

New players start from a position prior. Players returning from last season carry their final
posterior forward, discounted. Transfers (same athlete_id, different team than last season)
start from a prior built from their prior-school production, pulled half-way toward the
position average and widened, because the new offense is a different environment.

Everything here is a pure function of the box scores up to the as-of week: it never looks at a
later week (tests/test_player_state.py checks this).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# group -> (box-score category, touch stat, yards stat)
GROUPS = {"rush": ("rushing", "CAR", "YDS"), "rec": ("receiving", "REC", "YDS")}

# Position priors. Per-touch mean, between-player variance and per-touch sd come from
# props.POS_PRIORS (receiving converted from per-target to per-reception by the 0.635 catch
# rate). Usage prior strength is in team touches (about one game).
EFF_PRIOR = {
    "rush": dict(mean=4.72, tau2=0.55, sigma=6.4),
    "rec": dict(mean=7.95 / 0.635, tau2=1.85 / 0.635 ** 2, sigma=9.1 / 0.635),
}
NEW_PLAYER_SHARE = 0.10          # prior usage share for someone we have never seen
NEW_PLAYER_STRENGTH = 30.0       # prior strength (team touches) for new players
ROLE_CHANGE_RETENTION = 0.4      # share of old evidence kept after a role change
SEASON_CARRY = 0.5               # share of last season's evidence kept into a new season
TRANSFER_SHRINK = 0.5            # how far a transfer's old efficiency is pulled to the average
TRANSFER_WIDEN = 1.5             # prior variance multiplier for transfers

STATE_COLUMNS = ["pid", "team", "player", "group", "season", "week", "share_a", "share_b",
                 "share_mean", "share_sd", "eff_mean", "eff_prec", "eff_sd", "games",
                 "touches", "rank", "missed", "role_change", "prior_source"]


def _pid(athlete_id, team, player) -> str:
    return str(athlete_id) if pd.notna(athlete_id) and str(athlete_id) not in ("", "None") \
        else f"{team}|{player}"


def _share_sd(a: float, b: float) -> float:
    n = a + b
    return float(np.sqrt(a * b / (n ** 2 * (n + 1))))


def week_observations(box_week: pd.DataFrame) -> pd.DataFrame:
    """One row per (pid, group) for ONE week of box scores: touches, yards, and the team's
    total touches that week (the denominator of usage share)."""
    rows = []
    b = box_week.copy()
    b["value"] = pd.to_numeric(b["value"], errors="coerce")
    for group, (cat, touch_stat, yds_stat) in GROUPS.items():
        for stat, col in ((touch_stat, "touches"), (yds_stat, "yards")):
            s = b[(b["category"] == cat) & (b["stat_type"] == stat)]
            if s.empty:
                continue
            s = s.assign(pid=[_pid(a, t, p) for a, t, p in zip(s["athlete_id"], s["team"], s["player"])])
            g = s.groupby(["team", "pid", "player"], as_index=False)["value"].sum().rename(columns={"value": col})
            g["group"] = group
            rows.append(g)
    if not rows:
        return pd.DataFrame(columns=["team", "pid", "player", "group", "touches", "yards", "team_touches"])
    out = None
    for group in GROUPS:
        parts = [r for r in rows if len(r) and r["group"].iloc[0] == group]
        if not parts:
            continue
        m = parts[0]
        for extra in parts[1:]:
            m = m.merge(extra, on=["team", "pid", "player", "group"], how="outer")
        for c in ("touches", "yards"):
            if c not in m:
                m[c] = 0.0
        m[["touches", "yards"]] = m[["touches", "yards"]].fillna(0.0)
        m["team_touches"] = m.groupby("team")["touches"].transform("sum")
        out = m if out is None else pd.concat([out, m], ignore_index=True)
    return out


def _new_row(pid, team, player, group, season, week, prev_final: pd.DataFrame | None):
    """Prior for a player with no state yet this season."""
    ep = EFF_PRIOR[group]
    a0 = NEW_PLAYER_SHARE * NEW_PLAYER_STRENGTH
    b0 = (1 - NEW_PLAYER_SHARE) * NEW_PLAYER_STRENGTH
    eff_mean, eff_prec, source = ep["mean"], 1.0 / ep["tau2"], "position"
    if prev_final is not None:
        hit = prev_final[(prev_final["pid"] == pid) & (prev_final["group"] == group)]
        if len(hit):
            h = hit.iloc[0]
            if h["team"] == team:                                    # same school: carry forward
                a0, b0 = h["share_a"] * SEASON_CARRY, h["share_b"] * SEASON_CARRY
                eff_mean, eff_prec = h["eff_mean"], h["eff_prec"] * SEASON_CARRY
                source = "carried"
            else:                                                    # transfer: prior-school production
                m = h["share_mean"] * (1 - TRANSFER_SHRINK) + NEW_PLAYER_SHARE * TRANSFER_SHRINK
                a0, b0 = m * NEW_PLAYER_STRENGTH, (1 - m) * NEW_PLAYER_STRENGTH
                eff_mean = h["eff_mean"] * (1 - TRANSFER_SHRINK) + ep["mean"] * TRANSFER_SHRINK
                eff_prec = 1.0 / (ep["tau2"] * TRANSFER_WIDEN)
                source = "prior_school"
    return dict(pid=pid, team=team, player=player, group=group, season=season, week=week,
                share_a=a0, share_b=b0, eff_mean=eff_mean, eff_prec=eff_prec, games=0, touches=0.0,
                rank=0, missed=0, role_change=False, prior_source=source)


def update_player_state(prev: pd.DataFrame | None, box_week: pd.DataFrame, season: int, week: int,
                        prev_final: pd.DataFrame | None = None) -> pd.DataFrame:
    """Apply ONE completed week of box scores to the state and return the new state.

    `prev` is last week's state (None at the start of a season). `prev_final` is last season's
    final state, used only to build priors for players who have none yet.
    """
    obs = week_observations(box_week)
    state = {} if prev is None or prev.empty else {
        (r["pid"], r["group"]): r for r in prev.to_dict("records")}
    records = obs.to_dict("records")
    team_touches = {(r["team"], r["group"]): float(r["team_touches"]) for r in records}
    teams_played = {r["team"] for r in records}

    # create state for newly seen players
    for r in records:
        key = (r["pid"], r["group"])
        if key not in state:
            state[key] = _new_row(r["pid"], r["team"], r["player"], r["group"], season, week, prev_final)
    seen = {(r["pid"], r["group"]): r for r in records}

    for key, row in state.items():
        row = dict(row)
        o = seen.get(key)
        group = key[1]
        ep = EFF_PRIOR[group]
        row["season"], row["week"] = season, week
        returning = False
        if o is not None:
            T, c = float(o["team_touches"]), float(o["touches"])
            returning = row["missed"] > 0 and c > 0
            row["share_a"] += c
            row["share_b"] += max(T - c, 0.0)
            if c > 0:
                prec_obs = c / ep["sigma"] ** 2
                new_prec = row["eff_prec"] + prec_obs
                row["eff_mean"] = (row["eff_prec"] * row["eff_mean"] + prec_obs * (o["yards"] / c)) / new_prec
                row["eff_prec"] = new_prec
                row["games"] += 1
                row["touches"] += c
                row["missed"] = 0
            else:
                row["missed"] += 1
        elif row["team"] in teams_played:
            # the team played but this player has no line: no usage this week, one more missed game
            row["share_b"] += team_touches.get((row["team"], group), 0.0)
            row["missed"] += 1
        row["role_change"] = bool(returning)
        state[key] = row

    out = pd.DataFrame(state.values())
    if out.empty:
        return pd.DataFrame(columns=STATE_COLUMNS)

    # depth rank inside (team, group) by posterior mean; a changed rank is a role change
    out["share_mean"] = out["share_a"] / (out["share_a"] + out["share_b"])
    out = out.sort_values(["team", "group", "share_mean", "pid"], ascending=[True, True, False, True])
    new_rank = out.groupby(["team", "group"]).cumcount() + 1
    changed = (out["rank"] > 0) & (new_rank != out["rank"]) & (out["games"] > 0)
    out["role_change"] = out["role_change"].astype(bool) | changed
    out["rank"] = new_rank

    # widen: forget part of the old evidence for players whose role changed
    rc = out["role_change"].to_numpy()
    for col in ("share_a", "share_b", "eff_prec"):
        out.loc[rc, col] = out.loc[rc, col] * ROLE_CHANGE_RETENTION

    out["share_mean"] = out["share_a"] / (out["share_a"] + out["share_b"])
    out["share_sd"] = [_share_sd(a, b) for a, b in zip(out["share_a"], out["share_b"])]
    out["eff_sd"] = 1.0 / np.sqrt(out["eff_prec"])
    return out.sort_values(["team", "group", "rank", "pid"]).reset_index(drop=True)[STATE_COLUMNS]


def build_player_state(box: pd.DataFrame, season: int, through_week: int,
                       prev_final: pd.DataFrame | None = None) -> pd.DataFrame:
    """State after every completed week 0..through_week of `season`, built week by week.

    Only box-score rows of this season with week <= through_week are read, so rows from later
    weeks cannot change the answer.
    """
    b = box[(box["season"] == season) & (box["week"] <= through_week)]
    state = None
    for w in sorted(b["week"].unique()):
        state = update_player_state(state, b[b["week"] == w], season, int(w), prev_final)
    return state if state is not None else pd.DataFrame(columns=STATE_COLUMNS)
