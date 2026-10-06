# CFB Edge Finder

A program that looks for **edges** in college football betting lines: games where a model that has
been *proven* to work says the price is wrong by more than it costs to bet. It does **not** place bets
and it is **not** a pick list. Most weeks, most games will say **no bet**, and that is a correct answer.

## What the board means, and what it does not

Each week `board` prices every game (spread, total, moneyline) and gives every row a status:

| status | plain meaning |
|---|---|
| **BET** | The model clears the bar **and** it has been proven out-of-sample for this kind of bet. The only rows you might act on. |
| **INFO** | The model sees something, but nobody has proven it may be trusted here. Interesting, not a bet. |
| **PASS** | No edge. The usual answer. |
| **FCS** | Big school vs a small school. Priced, never a bet. |
| **NO_LINE** | The books have not posted a number yet. |

What it does **not** mean:
- A BET is not a prediction that you will win. Over one season, wins and losses are mostly luck; the number
  that matters is **closing line value** (did you get a better number than the market ended at?). Track it with `log-bet` and `clv`.
- Until the model passes validation, there are **zero BET rows**. That is the design. If `validate` says
  "nothing passed", the program prices off the betting line and shows you only INFO and PASS. (Your NFL model
  got exactly that answer, so expect it here until the evidence says otherwise.)
- CollegeFootballData has **no prices** for spreads and totals, so the board assumes -110 and says so at the top.
  Check your book's real price.
- College football has **no injury report**. For player props, `p_play` is your own number.

## First-time setup (Windows, PowerShell, from the project folder)

```
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m pytest -q
```
The last line should end with `passed`. It needs no key.

Then get a free key at https://collegefootballdata.com/key, copy `.env.example` to `.env`, and put the key after
`CFBD_API_KEY=`. **Never paste the key into a chat.**

## Weekly routine

```
python -m cfbmodel pull --seasons 2025 2026 --week N      # data (uses CFBD calls; it tells you how many are left)
python -m cfbmodel update --season 2026 --week N          # after week N finishes: ratings + player state
python -m cfbmodel board --season 2026 --week N+1         # the board: output\board_2026_wN+1.html (open on your phone)
python -m cfbmodel log-bet ...                            # every bet you actually place
python -m cfbmodel clv                                    # after the games: CLV, results, ROI
```
Every second week: `python -m cfbmodel learn --season 2026 --week N --seasons 2021 2022 2023 2024 2025 2026`
(it will usually answer "no change", which is healthy). Once, before the season: `fit-priors` and `validate`.
Free tier is 1,000 CFBD calls a month; the first full pull of 2021-2026 is roughly a quarter of that.

## Where things are

| path | what |
|---|---|
| `cfbmodel/` | the program (`python -m cfbmodel --help` lists every command) |
| `params/` | every tunable number, versioned (`v0001.json`, ...); `current.json` says which is active |
| `output/` | everything the commands write: boards, validation, reports, your bet log (not saved to git) |
| `data/cache/` | saved CFBD answers, so re-running costs nothing (not saved to git) |
| `docs/AUDIT.md` | what was wrong in the first version, and how each item was fixed and tested |
| `docs/DATA.md` | every data table and its columns |
| `docs/BUILD_PLAN.md`, `CLAUDE.md` | the plan and the rules the code is built under |

## What is proven, and what is not

Proven by the 350+ tests (all synthetic data with a known right answer): no lookahead, the validation gate, BET
being unreachable without it, the CLV signs, the learning loop declining to change on pure noise, rollback
reproducing an earlier board exactly, secrets never printed.

**Not yet proven: anything about real data.** The build environment could not reach CFBD, so the first real
`pull` is the real test. If CFBD names a field differently than expected, the program stops and names the field
instead of guessing; send the message (never the key).

Not built: per-venue home-field numbers, passing-yards props (need QB state), parquet files (CSV is used instead).
The dashboard (`streamlit run app/dashboard.py`) needs `pip install streamlit`, which is not in `requirements.txt` until you approve it.

*A research tool, not betting advice.*
