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

from .ratings import as_of, clean_plays

# EPA/dropback, FBS. Sample-weighted; QB play has widened since 2019.
QB_EPA_MEAN = 0.115
QB_EPA_SD = 0.145
GARBAGE_TIME_EPA_INFLATION = 0.055   # EPA/dropback is inflated in blowouts


def qb_epa(pbp: pd.DataFrame, asof_season: int, asof_week: int,
           filter_garbage: bool = True) -> pd.DataFrame:
    """Per-QB EPA per dropback, with dropback counts."""
    df = clean_plays(pbp) if filter_garbage else pbp.copy()
    df = as_of(df, asof_season, asof_week)
    df["epa"] = pd.to_numeric(df.get("epa", df.get("ppa")), errors="coerce")

    pt = df.get("playType", pd.Series("", index=df.index)).fillna("")
    df = df[pt.str.contains("Pass|Sack", regex=True) & df["epa"].notna()]
    if df.empty or "passer" not in df.columns:
        return pd.DataFrame(columns=["qb", "team", "epa_db", "dropbacks"])

    g = df.groupby(["passer", "offense"])["epa"].agg(["mean", "size"])
    g.columns = ["epa_db", "dropbacks"]
    return g.reset_index().rename(columns={"passer": "qb", "offense": "team"})


def shrunk_qb_value(epa_db: float, dropbacks: float, k: float = 180.0) -> float:
    """
    Empirical-Bayes shrinkage toward the FBS mean. k=180 dropbacks is roughly
    where QB EPA stabilises — about five games. Anything under 100 dropbacks is
    mostly noise and gets pulled hard toward the mean, which is correct.
    """
    if dropbacks < 1:
        return QB_EPA_MEAN
    w = dropbacks / (dropbacks + k)
    return float(w * epa_db + (1 - w) * QB_EPA_MEAN)


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
    r = 0.85 if recruit_rating is None else float(recruit_rating)
    rec_term = np.clip((r - 0.85) / 0.13, -1.2, 1.6) * 0.045

    year_term = {"FR": -0.045, "SO": -0.020, "JR": -0.005, "SR": 0.005}.get(
        (class_year or "SO")[:2].upper(), -0.020
    )
    supporting = 0.35 * np.clip(team_off_rating_ex_qb, -2.0, 2.0) * 0.03

    return float(QB_EPA_MEAN - 0.075 + rec_term + year_term + supporting)


def dropoff_points(
    starter_epa: float,
    starter_dropbacks: float,
    backup_epa: float | None,
    backup_dropbacks: float,
    backup_recruit: float | None = None,
    backup_class: str | None = None,
    team_off_rating_ex_qb: float = 0.0,
    dropbacks_per_game: float = 33.0,
) -> dict:
    """
    Points of line movement from a QB change, plus the uncertainty around it.

    Returns both because the second number is the one that should govern your
    behaviour. A dropoff of 6.5 +/- 4.0 points means the line is now a coin flip
    with wider tails, and the correct action in most such spots is no action.
    """
    s = shrunk_qb_value(starter_epa, starter_dropbacks)

    if backup_epa is not None and backup_dropbacks >= 150:
        b = shrunk_qb_value(backup_epa, backup_dropbacks)
        b -= GARBAGE_TIME_EPA_INFLATION * np.clip(1.0 - backup_dropbacks / 250.0, 0, 1)
        tier, sd = 1, 2.4
    elif backup_epa is not None and backup_dropbacks > 0:
        obs = shrunk_qb_value(backup_epa, backup_dropbacks) - GARBAGE_TIME_EPA_INFLATION
        pri = backup_prior(backup_recruit, backup_class, team_off_rating_ex_qb)
        w = backup_dropbacks / (backup_dropbacks + 120.0)
        b, tier, sd = w * obs + (1 - w) * pri, 2, 3.4
    else:
        b = backup_prior(backup_recruit, backup_class, team_off_rating_ex_qb)
        tier, sd = 3, 4.6

    pts = float(np.clip((s - b) * dropbacks_per_game, -2.0, 16.0))
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
