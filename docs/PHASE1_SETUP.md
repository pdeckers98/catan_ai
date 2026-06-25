# Phase 1 — Setup & Catanatron wiring

**Goal:** a runnable 1v1 Catan environment with a legal-move agent loop, before any learning.
**Status:** ✅ complete.

## Environment

- Developed on the **`catan` conda env** (Python 3.14).
  Interpreter: `C:\Users\pdeck\.conda\envs\catan\python.exe`.
- Install: `pip install -r requirements.txt`

## Installed stack (verified working)

| Package | Version | Notes |
|---|---|---|
| catanatron | 3.2.1 | core engine (GPL-3.0) |
| catanatron-gym | 4.0.0 | Gymnasium env (separate package; **not** bundled in core) |
| gymnasium | 0.29.1 | pinned by catanatron-gym 4.0.0 |
| stable-baselines3 | 2.9.0 | imports fine under gymnasium 0.29.1 |
| sb3-contrib | 2.9.0 | `MaskablePPO` + `ActionMasker` |
| torch | 2.12.1 | cp314 wheel exists — Python 3.14 is fine |

### Gotchas discovered (read before touching deps)

- The gym env is in **`catanatron-gym`**, a *separate* PyPI package. Core `catanatron` 3.2.1 has
  no `catanatron.gym` submodule (the docs site describes a newer, differently-structured release).
- Use **`catanatron-gym==4.0.0`**, not 3.2.1: the 3.2.1 gym package depends on the legacy `gym`
  library, which fails to build on Python 3.14. 4.0.0 uses `gymnasium`.
- catanatron-gym 4.0.0 **pins `gymnasium==0.29.1`** (downgrades it). SB3/sb3-contrib 2.9.0 still
  import and run under 0.29.1 — verified — so no action needed, but don't blindly upgrade gymnasium.
- Importing `catanatron_gym` is what **registers the env id**; `src/env/catan_env.py` does this.

## The real env API (catanatron-gym 4.0.0)

- **Env id:** `catanatron-v1` — created via `gym.make("catanatron-v1", config={...})`.
- **Inherently 1v1:** the agent is **P0 = `Color.BLUE`**; `config["enemies"]` is a list of opponent
  players. One enemy ⇒ 2-player game. Enemies must not be BLUE.
- **Action space:** `Discrete(290)`. Most actions are illegal each turn.
- **Observation:** default `"vector"` representation → flat `Box` of shape **(614,)**
  (alternative `"mixed"` gives a board tensor + numeric dict).
- **Valid actions / masking:** `env.unwrapped.get_valid_actions()` returns the legal action ints;
  also exposed as `info["valid_actions"]` after `reset()`/`step()`.
- **`config` keys** (with defaults): `enemies` (`[RandomPlayer(RED)]`), `map_type` (`"BASE"`),
  `vps_to_win` (`10`), `representation` (`"vector"`), `reward_function` (built-in win/loss/draw),
  `invalid_action_reward` (`-1`).
- **Reward:** built-in `simple_reward` → `+1` win / `-1` loss / `0` otherwise.

## Available opponent bots (important limitation)

The pip release only ships:
- `catanatron.RandomPlayer`
- `catanatron.players.weighted_random.WeightedRandomPlayer` (slightly smarter than random)
- `catanatron.players.search.VictoryPointPlayer` (greedy VP maximizer)

The strong **AlphaBeta / MCTS / ValueFunction** bots referenced in older write-ups are **not** in
this packaged version. The Phase 2 benchmark ladder is therefore
`Random → WeightedRandom → VictoryPoint`. Reaching a stronger search benchmark later would mean
installing from Catanatron source/experimental or implementing one ourselves.

## What was built

- `src/env/catan_env.py`
  - `make_1v1_env(enemy=None, map_type="BASE", vps_to_win=10, representation="vector",
    reward_function=None)` — constructs the 1v1 env (default enemy: `WeightedRandomPlayer(RED)`).
  - `valid_action_mask(env)` — boolean mask of shape `(action_space.n,)` for SB3-Contrib's
    `ActionMasker`.
- `src/env/smoke_test.py` — plays N full games choosing uniformly among `info["valid_actions"]`.

## Verification (reproduce)

```bash
python -m src.env.smoke_test --games 20
flake8 src/
```

Last run: 20 full games completed, P0 (legal-random) 7 wins / 13 losses vs `WeightedRandomPlayer`,
0 truncations, ~399 steps/game. flake8 clean. The win-rate < 50% is expected — a uniform-random
legal agent should lose to the weighted-random opponent. This is the baseline Phase 2 must beat.

## Next

→ `PHASE2_AI.md`: train a `MaskablePPO` agent that beats `WeightedRandomPlayer`, then push toward
`VictoryPointPlayer` and self-play.
