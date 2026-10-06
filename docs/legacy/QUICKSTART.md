# Quickstart

## Google Colab

Open `cfbmodel_colab.ipynb` in Colab and run it top to bottom. There is no import
order to manage — Python handles that. Two things do matter:

- **Upload the zip, not loose files.** `cfbmodel/` must stay a directory or
  `from cfbmodel import ...` won't resolve.
- **Point the cache at Drive (cell 2).** Colab wipes local disk on disconnect,
  and re-pulling five seasons costs ~175 calls against a 1,000/month budget.
  Set `CFBMODEL_CACHE` *before* importing cfbmodel; paths resolve at import time.

## Local

```bash
pip install -r requirements.txt
export CFBD_API_KEY=...          # free: https://collegefootballdata.com/key

python tests/smoke_test.py       # verifies the plumbing — no key needed
python tests/test_upgrades.py    # verifies script/qb/weather/portal
```

## Commands

| command | what it does | when |
|---|---|---|
| `pull --seasons 2021 2022 2023 2024 2025` | cache games, lines, plays, box scores, recruiting | once, then weekly |
| `validate --seasons 2022 2023 2024 2025` | **the test that decides everything** — does the model beat the closing line? | before betting anything |
| `preseason --season 2026` | recruiting + returning production + portal ratings | weeks 0–2 |
| `slate --season 2026 --week 5` | spread/total bet card, weather applied | each week |
| `props --season 2026 --week 5 --home Auburn --away Georgia` | player projections under joint game script | each week |
| `weather --season 2026 --week 5` | totals-relevant forecasts, horizon-shrunk | Friday |
| `qb --season 2026 --week 5` | QB1/QB2 values and injury dropoffs by team | as needed |

## Order of operations for Week 0 (Aug 29)

```bash
python run_slate.py pull --seasons 2021 2022 2023 2024 2025
python run_slate.py validate --seasons 2022 2023 2024 2025   # ← read this before anything else
python run_slate.py preseason --season 2026
python run_slate.py weather --season 2026 --week 0
python run_slate.py slate --season 2026 --week 0 --w-model <coef from validate>
```

`--w-model` should come from the `model_margin` coefficient that `validate`
prints, not from how confident you feel. If that coefficient's p-value is above
0.10, set it to 0 and don't bet sides or totals this season.

## Flags worth knowing

- `--p-play 0.5` on props — for a genuine game-time decision. CFB has no injury
  report; betting a questionable starter at face value is the most common way to
  lose money here.
- `--home-qb-out` / `--away-qb-out` — routes through the hierarchical QB model.
  Check the `dropoff_sd_points` column before acting; tier 3 is a prior, not a
  measurement.
- `--no-weather` — skips the Open-Meteo call if you're offline or backtesting.

## Reading the output

`props` prints `close_game` and `blowout` columns: the same player's projection
under two scripts. When those straddle the book's line, the prop is a bet on the
game, not the player.

`slate` prints `wx_adj` and `softness`. A 2-point disagreement on a Tuesday MAC
total with softness 0.80 is worth more than the same disagreement on an SEC
game at 0.25 — that's already in the bet sizing, but it's worth seeing.
