"""
Quarterback value and injury dropoff.

Gemini's suggestion — compute the starter's EPA against the specific backup's
EPA — is the right target and unimplementable as stated, for one reason it
didn't account for: **the backup usually has no data.** A true freshman QB2 has
zero college snaps. That is precisely why the shipped version used a team-
strength proxy: not because it was better, but because the obvious thing has no
denominator in the majority of cases.

So the fix is hierarchical rather than direct. Three tiers, in order:

  1. Backup has 150+ dropbacks (~4 games) -> shrunk observed EPA/dropback.
  2. Backup has a few snaps -> heavy shrinkage toward tier 3.
  3. Backup has none -> a prior from recruiting rating, class year, and the
     team's non-QB offensive strength (OL and skill talent survive the QB
     change and set a floor).

One subtlety that bites here: the garbage-time filter in `ratings.py` deletes
exactly the plays backups take. If you compute backup EPA from filtered plays
you will have almost no sample and what remains is biased — it is prevent
defense against a losing team. `backup_epa` therefore runs on unfiltered plays
with an explicit garbage-time *correction* rather than a filter.

Second subtlety: the point estimate matters less than people think. The dropoff
distribution is wide and right-skewed, and the honest conclusion in most QB-out
spots is that variance has gone up more than the mean has moved. The market
usually reprices these within minutes of the news; if you are reading it on
Twitter, so is the trader.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .params import P
from .ratings import as_of, clean_plays, drop_non_fbs

# QB_EPA_MEAN and GARBAGE_TIME_EPA_INFLATION used to be constants here; they are registry
# parameters now (P.qb.epa_mean, P.qb.garbage_inflation).


def qb_epa(pbp: pd.DataFrame, asof_season: int, asof_week: int,
           filter_garbage: bool = True) -> pd.DataFrame:
    """Per-QB EPA per dropback, with dropback counts."""
    df = clean_plays(pbp) if filter_garbage else drop_non_fbs(pbp).copy()
    df = as_of(df, asof_season, asof_week)
    df["epa"] = pd.to_numeric(df.get("epa", df.get("ppa")), errors="coerce")

    pt = df.get("playType", pd.Series("", index=df.index)).fillna("")
    df = df[pt.str.contains("Pass|Sack", regex=True) & df["epa"].notna()]
    if df.empty or "passer" not in df.columns:
        return pd.DataFrame(columns=["qb", "team", "epa_db", "dropbacks"])

    g = df.groupby(["passer", "offense"])["epa"].agg(["mean", "size"])
    g.columns = ["epa_db", "dropbacks"]
    return g.reset_index().rename(columns={"passer": "qb", "offense": "team"})


def shrunk_qb_value(epa_db: float, dropbacks: float, k: float | None = None) -> float:
    """
    Empirical-Bayes shrinkage toward the FBS mean. k=180 dropbacks is roughly
    where QB EPA stabilises — about five games. Anything under 100 dropbacks is
    mostly noise and gets pulled hard toward the mean, which is correct.
    """
    Q = P.qb
    k = Q.shrink_k if k is None else k
    if dropbacks < 1:
        return Q.epa_mean
    w = dropbacks / (dropbacks + k)
    return float(w * epa_db + (1 - w) * Q.epa_mean)


def backup_prior(
    recruit_rating: float | None,
    class_year: str | None,
    team_off_rating_ex_qb: float,
) -> float:
    """
    Tier 3: what to expect from a QB who has never played.

    Recruiting rating is a weak but real signal for QBs specifically — weaker
    than for OL, stronger than for RB. Class year matters because redshirt
    juniors sitting behind a starter are usually there for a reason, while
    a true freshman five-star is unproven rather than bad.

    The team's non-QB offensive strength sets the floor: a backup handing off to
    a top-10 run game behind a veteran line is a fundamentally different problem
    from a backup on a bad roster.
    """
    Q = P.qb
    r = Q.recruit_default if recruit_rating is None else float(recruit_rating)
    rec_term = np.clip((r - Q.recruit_center) / Q.recruit_scale, Q.recruit_lo, Q.recruit_hi) * Q.recruit_weight

    year_term = {"FR": Q.year_fr, "SO": Q.year_so, "JR": Q.year_jr, "SR": Q.year_sr}.get(
        (class_year or "SO")[:2].upper(), Q.year_default
    )
    supporting = Q.supporting_weight * np.clip(team_off_rating_ex_qb, -Q.supporting_clip, Q.supporting_clip) \
        * Q.supporting_scale

    return float(Q.epa_mean + Q.backup_offset + rec_term + year_term + supporting)


def dropoff_points(
    starter_epa: float,
    starter_dropbacks: float,
    backup_epa: float | None,
    backup_dropbacks: float,
    backup_recruit: float | None = None,
    backup_class: str | None = None,
    team_off_rating_ex_qb: float = 0.0,
    dropbacks_per_game: float | None = None,
) -> dict:
    """
    Points of line movement from a QB change, plus the uncertainty around it.

    Returns both because the second number is the one that should govern your
    behaviour. A dropoff of 6.5 +/- 4.0 points means the line is now a coin flip
    with wider tails, and the correct action in most such spots is no action.
    """
    Q = P.qb
    dropbacks_per_game = Q.dropbacks_per_game if dropbacks_per_game is None else dropbacks_per_game
    s = shrunk_qb_value(starter_epa, starter_dropbacks)

    if backup_epa is not None and backup_dropbacks >= Q.tier1_min_dropbacks:
        b = shrunk_qb_value(backup_epa, backup_dropbacks)
        b -= Q.garbage_inflation * np.clip(1.0 - backup_dropbacks / Q.inflation_fade_dropbacks, 0, 1)
        tier, sd = 1, Q.tier1_sd
    elif backup_epa is not None and backup_dropbacks > 0:
        obs = shrunk_qb_value(backup_epa, backup_dropbacks) - Q.garbage_inflation
        pri = backup_prior(backup_recruit, backup_class, team_off_rating_ex_qb)
        w = backup_dropbacks / (backup_dropbacks + Q.tier2_k)
        b, tier, sd = w * obs + (1 - w) * pri, 2, Q.tier2_sd
    else:
        b = backup_prior(backup_recruit, backup_class, team_off_rating_ex_qb)
        tier, sd = 3, Q.tier3_sd

    pts = float(np.clip((s - b) * dropbacks_per_game, Q.points_min, Q.points_max))
    return {
        "points": round(pts, 2),
        "sd_points": sd,
        "tier": tier,
        "starter_epa_shrunk": round(s, 4),
        "backup_epa_est": round(b, 4),
        "note": {
            1: "backup has real sample",
            2: "backup has limited sample, blended with prior",
            3: "backup unplayed — this is a prior, not a measurement",
        }[tier],
    }


def build_qb_table(pbp: pd.DataFrame, asof_season: int, asof_week: int) -> pd.DataFrame:
    """
    Per-team QB1/QB2 with values and a precomputed dropoff, ready to feed
    `game_model.project_game(home_qb_out=...)`.
    """
    filt = qb_epa(pbp, asof_season, asof_week, filter_garbage=True)
    unfilt = qb_epa(pbp, asof_season, asof_week, filter_garbage=False)

    rows = []
    for team, grp in unfilt.groupby("team"):
        grp = grp.sort_values("dropbacks", ascending=False)
        if grp.empty:
            continue
        qb1 = grp.iloc[0]
        s_row = filt[(filt["team"] == team) & (filt["qb"] == qb1["qb"])]
        s_epa = float(s_row["epa_db"].iloc[0]) if len(s_row) else float(qb1["epa_db"])
        s_db = float(s_row["dropbacks"].iloc[0]) if len(s_row) else float(qb1["dropbacks"])

        qb2 = grp.iloc[1] if len(grp) > 1 else None
        d = dropoff_points(
            s_epa, s_db,
            float(qb2["epa_db"]) if qb2 is not None else None,
            float(qb2["dropbacks"]) if qb2 is not None else 0.0,
        )
        rows.append({
            "team": team, "qb1": qb1["qb"], "qb1_dropbacks": s_db,
            "qb1_epa": round(s_epa, 4),
            "qb2": qb2["qb"] if qb2 is not None else None,
            "qb2_dropbacks": float(qb2["dropbacks"]) if qb2 is not None else 0.0,
            **{f"dropoff_{k}": v for k, v in d.items()},
        })
    return pd.DataFrame(rows).set_index("team")
