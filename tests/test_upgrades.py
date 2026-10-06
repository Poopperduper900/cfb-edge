"""Validation of the four changes made in response to the Gemini review."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cfbmodel import priors, props, qb, script, weather  # noqa: E402

ok = True
print("=" * 68)

# ---- 1. joint game script vs the old static haircut
print("\n[1] JOINT GAME SCRIPT vs STATIC HAIRCUT")
for label, exp_margin in (("even game", 0.0), ("-24 favourite", 24.0)):
    sc = script.simulate_game_script(exp_margin, 55.0, base_pace=68.0, n_sims=60_000)
    joint = script.simulate_rush_yards_joint(0.55, sc, 5.1, 6.4, shock_sigma=0.18)
    old = props.simulate_rush_yards(0.55, 68.0, 0.55, 5.1, 6.4, exp_margin, 55.0,
                                    n_sims=60_000)
    print(f"  {label:<15} joint mean={joint.mean():6.1f} sd={joint.std():5.1f} "
          f"p10={np.percentile(joint,10):5.1f} p90={np.percentile(joint,90):6.1f} "
          f"| static mean={old.mean():6.1f} sd={old.std():5.1f}")
    if exp_margin > 20:
        # the point of the rewrite: blowout spots must be WIDER and more
        # left-skewed, not just shifted down
        sk = float(pd.Series(joint).skew())
        print(f"                  pulled in {sc['p_pulled']*100:4.1f}% of sims, "
              f"joint/static sd ratio={joint.std()/old.std():.2f}, skew={sk:+.2f}")
        ok &= joint.std() > old.std()

# the backup RB whose over gets LIVE in the same blowout
sc = script.simulate_game_script(24.0, 55.0, 68.0, n_sims=60_000)
starter = script.simulate_rush_yards_joint(0.55, sc, 5.1, 6.4, 0.18,
                                           share_pull_sensitivity=1.0)
backup = script.simulate_rush_yards_joint(0.20, sc, 4.6, 6.4, 0.30,
                                          share_pull_sensitivity=-1.4)
corr = np.corrcoef(starter, backup)[0, 1]
print(f"  starter/backup yardage correlation = {corr:+.3f}  "
      f"(static model forces this to ~0)")
ok &= corr < 0.0

# the shape that a static haircut cannot produce: conditional on how the game
# actually goes, the same starter is two different players
close = np.abs(sc["margin"]) < 14
blow = np.abs(sc["margin"]) >= 28
print(f"  starter RB | game stays close (<14): mean={starter[close].mean():6.1f}")
print(f"  starter RB | blowout (>=28):         mean={starter[blow].mean():6.1f}")
print(f"  backup  RB | blowout (>=28):         mean={backup[blow].mean():6.1f} "
      f"(vs {backup[close].mean():.1f} in close games)")
ok &= starter[close].mean() > starter[blow].mean()
ok &= backup[blow].mean() > backup[close].mean()

# receiving is where a favourite's script hurts unambiguously: fewer pass
# attempts AND benching, hitting the same player twice
wr_even = script.simulate_rec_yards_joint(
    0.24, script.simulate_game_script(0.0, 55.0, 68.0, n_sims=60_000),
    8.0, 0.635, 0.22)["yards"]
wr_fav = script.simulate_rec_yards_joint(
    0.24, script.simulate_game_script(24.0, 55.0, 68.0, n_sims=60_000),
    8.0, 0.635, 0.22)["yards"]
print(f"  WR1 rec yards: neutral={wr_even.mean():.1f}  "
      f"-24 favourite={wr_fav.mean():.1f}  ({wr_fav.mean()/wr_even.mean()-1:+.1%})")
ok &= wr_fav.mean() < wr_even.mean() * 0.92

# ---- 2. role instability replaces the static shock constant
print("\n[2] ROLE INSTABILITY -> VOLUME SHOCK SIGMA")
cases = {
    "bell cow, 9 games": (np.array([.62,.58,.65,.60,.63,.59,.61,.64,.60]), [.61,.22,.17]),
    "committee, 4 games": (np.array([.38,.22,.45,.30]), [.34,.33,.33]),
    "true FR, 2 games":   (np.array([.18,.41]), [.30,.30,.40]),
}
for name, (hist, mates) in cases.items():
    s = script.role_instability(hist, mates)
    print(f"  {name:<20} sigma={s:.3f}   (shipped constant was 0.210)")
sig_bell = script.role_instability(*cases["bell cow, 9 games"])
sig_fr = script.role_instability(*cases["true FR, 2 games"])
ok &= sig_bell < 0.21 < sig_fr

# ---- 3. QB dropoff across the three evidence tiers
print("\n[3] QB DROPOFF — HIERARCHICAL")
for name, kw in (
    ("QB2 has 400 dropbacks", dict(backup_epa=0.06, backup_dropbacks=400)),
    ("QB2 has 90 dropbacks",  dict(backup_epa=0.02, backup_dropbacks=90)),
    ("QB2 never played (4*)", dict(backup_epa=None, backup_dropbacks=0,
                                   backup_recruit=0.93, backup_class="FR")),
    ("QB2 never played (3*)", dict(backup_epa=None, backup_dropbacks=0,
                                   backup_recruit=0.83, backup_class="SO")),
):
    d = qb.dropoff_points(starter_epa=0.26, starter_dropbacks=420, **kw)
    print(f"  {name:<24} {d['points']:5.2f} +/- {d['sd_points']:.1f} pts  "
          f"tier {d['tier']} — {d['note']}")
d1 = qb.dropoff_points(0.26, 420, 0.06, 400)
d3 = qb.dropoff_points(0.26, 420, None, 0, 0.83, "SO")
ok &= d3["points"] > d1["points"] and d3["sd_points"] > d1["sd_points"]

# ---- 4. portal: position weights + production join
print("\n[4] PORTAL VALUE — POSITION + PRODUCTION")
pf = pd.DataFrame({
    "firstName": ["Ace", "Bo", "Cy", "Dex"],
    "lastName": ["Quarterback", "Longsnap", "Corner", "Runner"],
    "position": ["QB", "LS", "CB", "RB"],
    "rating": [0.86, 0.97, 0.88, 0.91],
    "origin": ["Old A", "Old B", "Old C", "Old D"],
    "destination": ["New U", "New U", "New U", "New U"],
})
prod = pd.DataFrame({
    "player": ["Ace Quarterback", "Bo Longsnap", "Cy Corner", "Dex Runner"],
    "ppa": [0.42, 0.00, 0.10, 0.05], "snaps": [780, 60, 400, 300],
})
naive = pf.assign(v=np.where(pf.rating > 0.95, 3.0,
                             np.where(pf.rating > 0.85, 1.5, 0.5)))
print("  star-only ranking:  " + ", ".join(
    naive.sort_values("v", ascending=False)["position"]))
df = pf.copy()
df["pos_mult"] = df["position"].map(priors.POSITION_VALUE)
vals = {}
for _, r in df.iterrows():
    nm = f"{r.firstName} {r.lastName}"
    pr = prod[prod.player == nm].iloc[0]
    cred = pr.snaps / (pr.snaps + 200)
    pz = (pr.ppa - prod.ppa.mean()) / prod.ppa.std()
    vals[r.position] = (cred * (2.0 + 2.2 * pz) + (1 - cred) * 1.5) * r.pos_mult
print("  position+production: " + ", ".join(
    k for k, _ in sorted(vals.items(), key=lambda x: -x[1])))
print(f"  -> a 3* transfer QB with a real season now outranks a 5* long snapper")
ok &= max(vals, key=vals.get) == "QB"

# ---- 5. weather: dome guard + forecast-horizon shrinkage
print("\n[5] WEATHER")
dome = weather.total_adjustment({"dome": True, "wind_mph": 30})
print(f"  dome with 30mph outside:        {dome[0]:+.2f} pts  (must be 0.00)")
ok &= dome[0] == 0.0
for h, lbl in ((6, "Saturday morning"), (48, "Thursday"), (120, "Monday")):
    wx = {"dome": False, "wind_mph": 24.0, "gust_mph": 34.0, "precip_in": 0.0,
          "temp_f": 48.0, "wx_confidence": weather.forecast_confidence(h)}
    pts, sd = weather.total_adjustment(wx)
    print(f"  24mph wind, {lbl:<17} {pts:+.2f} pts  "
          f"(conf {wx['wx_confidence']:.2f}, +{sd:.1f} sd)")
near = weather.total_adjustment({"dome": False, "wind_mph": 24.0, "gust_mph": 34.0,
                                 "precip_in": 0, "temp_f": 48,
                                 "wx_confidence": weather.forecast_confidence(6)})[0]
far = weather.total_adjustment({"dome": False, "wind_mph": 24.0, "gust_mph": 34.0,
                                "precip_in": 0, "temp_f": 48,
                                "wx_confidence": weather.forecast_confidence(120)})[0]
ok &= abs(near) > abs(far)

print("\n" + "=" * 68)
print("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED")
sys.exit(0 if ok else 1)
