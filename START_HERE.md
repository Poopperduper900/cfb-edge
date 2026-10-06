# Start here (for you, not for Claude Code)

## Setup

1. Install Claude Code using the official guide: https://code.claude.com/docs
2. Unzip this folder somewhere like `Documents\cfb-edge`.
3. Copy `.env.example` to `.env` and paste your CFBD key after `CFBD_API_KEY=`.
   Free key: https://collegefootballdata.com/key — if you still have the key you pasted
   into a chat earlier, request a fresh one and use that instead.
4. Open a terminal in the folder and run `claude`.

Claude Code reads `CLAUDE.md` automatically every time it starts in this folder, so the
rules, pitfalls, and domain facts are always loaded. `docs/BUILD_PLAN.md` holds the
phases; Claude Code reads it when you point it there.

## Prompts to paste

**First session**
```
Read CLAUDE.md and docs/BUILD_PLAN.md. Do Phase 0 only. Show me your plan before
changing anything, then implement it and report against the acceptance criteria.
```

**Each following phase** (one phase per session works best)
```
Do Phase N from docs/BUILD_PLAN.md. Plan first, then implement. Stop when the
acceptance criteria pass and summarize what changed and what proves it.
```

**Weekly, once Phase 7 is done**
```
Pull this week's data, run update, and build the board for week N. Summarize the BET
and INFO rows, any sanity banners, and how old the validation status is.
```

**Every second week (the learning step)**
```
Run learn. Tell me whether the challenger was promoted, and walk me through diff.md:
what changed, by how much, and the evidence.
```

**When you get outside feedback**
```
Here's a review from [source]. Test each claim empirically against the code. Adopt or
reject each one with evidence, and show me the results.
```

## What to watch for

- If it says a test "needed to be updated," ask why. Rule 6 in CLAUDE.md forbids
  weakening tests to make them pass.
- If the first board shows a lot of `BET` rows, something is broken — not a great week.
- The first `validate` run may say nothing passed. That's a real answer, and the same
  one your NFL model gave. The key-number values and the CLV tracker still work either way.
- If `learn` promotes a new version almost every time it runs, it's chasing noise.
  A healthy learner mostly says "champion stays."
- Phases 0–6 are the important ones. Don't skip ahead to the dashboard.
