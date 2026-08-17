# 1v1 Catan AI for Colonist.io

A Python AI that learns to play 1v1 Settlers of Catan, built on the
[Catanatron](https://github.com/bcollazo/catanatron) engine. The agent is trained via deep
reinforcement learning in the cloud and (eventually) plays live games on colonist.io.

## Project Purpose

Hobby project: train a strong 1v1 Catan agent through self-play deep RL, then bridge it to play
real games on colonist.io. The game engine is a solved problem (Catanatron) — the effort goes into
the agent and, later, the web integration.

## Tech Stack

- **Language**: Python 3.10+ (developed on the `catan` conda env, Python 3.14)
- **Game engine**: [Catanatron](https://github.com/bcollazo/catanatron) (GPL-3.0) — fast pure-Python
  Catan simulator with a Gymnasium env, action masking, and strong baseline bots
- **Learning algorithm**: **`MaskablePPO` (`src/agent/train.py`) is the only trainer.** Under
  12 VP with Longest Road it reaches 98–99% vs weighted-random and 96–98% vs greedy; at 15 VP it
  wins 15.0 VP in ~150 turns. **The deployed agent is that checkpoint plus 50-sim PUCT search plus
  both placement models** — three artifacts, never one.
- **What actually helps** (measured 2026-08-16, all at 12/15 VP with Longest Road):
  **search at inference dominates further training.** In a mirror match — identical weights,
  the only difference being search — 50 sims/move is worth **+9.8 points** (59.8% over 200
  games). Over the same day, a shared policy/value trunk bought +2.9 points (not significant
  over 800 games) and 800k steps of 15 VP fine-tuning bought −1.5 (not significant).
  100 sims scores 60.8%, i.e. **search saturates by 50 sims** — the ceiling is critic quality,
  not search budget, since PPO's value head estimates a discounted return rather than a win
  probability and never trained on the positions search explores. Every PPO run so far plateaus
  once it beats its references, so **reach for search, or harder opponents, before another
  training run.**
- **An AlphaZero track existed and was removed** (`train_az.py`, `selfplay.py`, `net.py`). It never
  got past feasibility at 70.8% vs weighted-random. Recoverable from git history; the argument for
  it was always the value head, and that argument is still open. Don't rebuild it casually.
- **Game mode**: 1v1 (`enemies=[one bot]`, `map_type="BASE"`). The VP target, the
  Longest Road award and the turn cap are **per-run**, selected via `src/env/ruleset.py`
  (`--vps-to-win` / `--longest-road` / `--max-turns`, or `CATAN_*` env vars). Defaults
  reproduce the historical setup: 8 VP, no Longest Road. **The target is the colonist.io
  1v1 ruleset: 15 VP with Longest Road enabled**, and agents have now trained under it —
  see `checkpoints/archive/` below. Raise `--max-turns` to 1500 at 15 VP; the default 1000
  is already binding there, and a truncated episode pays 0, so capped games teach nothing.
- **Custom rules**: `src/env/rules.py` monkeypatches Catanatron at import time. Applied
  automatically via `src/env/catan_env.py`. Eight patches:
  1. discard on a 7 only above **9** cards (`discard_limit=9`, vs. stock 7)
  2. per-resource, one-card-at-a-time discard the policy actually chooses
     (expands the action space 290 → **294**)
  3. correct multi-discarder sequencing (fixes an upstream `> 7` hardcode)
  4. Colonist.io 1v1 robber placement restrictions
  5. **Longest Road awards no VP** by default (length still tracked, `HAS_ROAD` never
     set); `--longest-road` restores stock scoring
  6. a dev card **cannot be played the turn it was bought** (one-per-turn is already
     enforced upstream)
  7. `_discard_remaining` survives `State.copy()` — required for MCTS
  8. **Road Building playable while broke** — upstream gates the card behind
     affording a road, though its two roads are free. Found via the colonist
     bridge; **every agent trained before this could not play it when short**
- **Opening placement**: a separate self-trained specialist (`src/placement/`). Initial settlement
  choice gets ~2 of ~300 gradient samples per episode, so the main policy learned "settle where
  three tiles meet" but never learned that an 8 beats a 3. The specialist trains on random
  openings labelled with actual game outcomes, using duplicate-board pairs for variance reduction.
  Two models, used together: a per-corner scorer (`scorer_ppo.pt`) shortlists candidates, and a
  **bundle scorer** (`bundle_noroads.pt`) ranks whole corner *pairs*, so the first settlement is
  chosen knowing what the second could be. Pair-search beat greedy corner selection 53.5% over
  1200 games; three attempts at improving the labels moved strength not at all.
  **No hardcoded placement knowledge**: features are mechanical board facts only. The hand-written
  pip scorer that used to serve as an evaluation yardstick (`heuristic.py`, `evaluate.py`) has been
  removed, so **games are now the only check on a refit** — budget 800+ of them.
  Opening *roads* are deliberately unmodelled: a bundle over (settlement, road) × 2 scored 44.1%
  over 1200 games and was cut. The road replay in data generation stays; it is a correct control.
- **Reward shaping**: none, and the machinery is gone rather than merely
  switched off. `EpisodeStatsWrapper` keeps the end-of-episode telemetry the old
  `RewardShapingWrapper` also carried (VPs, settlements, cities, roads) without touching
  the reward.
- **Action masking**: mandatory — most of the 294 actions are illegal each turn; always respect
  `info["valid_actions"]` / `env.unwrapped.get_valid_actions()`, or
  `src/agent/encoding.py:legal_action_mask` off a raw `Game`.
- **Training hardware**: high-core CPU cloud instance (RL here is CPU-bound — parallel game
  rollouts dominate; the policy is a small MLP)
- **Inference hardware**: local GTX 1660 Super
- **Web integration (Phase 3)**: colonist.io WebSocket interception (read state) + Playwright (clicks)

## Project Structure

```
src/
├── env/
│   ├── ruleset.py       # per-run rules (VP target, longest road, turn cap) via env vars
│   ├── rules.py         # custom-rule monkeypatches (applied at import)
│   ├── catan_env.py     # gym env + raw-Game factory, shared constants
│   ├── lookahead.py     # two-roll dice features (P(afford), discard risk)
│   ├── dice.py          # fixed_dice: a dedicated roll stream for paired games
│   └── render.py        # matplotlib board renderer (used by eval/play.py)
├── agent/
│   ├── encoding.py      # obs vector + action mask off a live Game
│   ├── evaluator.py     # leaf evaluators: PPO adapter, uniform control
│   ├── mcts.py          # PUCT search with chance nodes; MCTSPlayer  <-- the +10 points
│   ├── train.py         # MaskablePPO loop  <-- the only trainer
│   ├── trunk.py         # shared policy/value trunk; REQUIRED to load the shipped model
│   ├── arena.py         # head-to-head match play + agent-by-name registry
│   ├── pool.py / elo.py / checkpoint_manager.py   # self-play ladder for train.py
│   └── opponent.py      # PolicyPlayer (frozen PPO checkpoint as a Player)
├── placement/           # opening-settlement specialist (self-trained, no heuristics)
│   ├── features.py      # 45 mechanical per-node board facts
│   ├── dataset.py       # random openings + duplicate-board outcome labels
│   ├── model.py         # PlacementNet (scores one node) + BundleNet (scores a pair)
│   ├── chooser.py       # OpeningChooser: shortlist corners, then search pairs
│   ├── train.py         # supervised fit on the outcome labels
│   ├── env_wrapper.py   # gym-side equivalent; opens inside reset()
│   └── player.py        # wraps any agent; takes over only the opening
├── eval/
│   ├── benchmark.py     # any agent vs any agent
│   └── play.py          # human vs AI (matplotlib)
└── bridge/              # (Phase 3) colonist.io bridge, one Playwright session for read+click
    ├── capture.py       # hand-driven browser; records WS frames + --summarize
    ├── protocol.py      # colonist messages -> BoardSpec + catanatron Actions
    ├── board.py         # exact board reconstruction (BoardSpec -> CatanMap)
    ├── replay.py        # GameReplay: a Game advanced by observed actions; DesyncError
    └── player.py        # the deployed agent, all three artifacts or nothing
docs/          # Per-phase guides (see below)
checkpoints/   # Saved models (git-ignored)
tests/         # Unit & integration tests
```

## Essential Commands

**Setup**: `pip install -r requirements.txt`

**Smoke test**: `python -m src.env.smoke_test`

**Train under the target ruleset** (15 VP, Longest Road on, 1500-turn cap). Raise `--gamma` with
the VP target -- games run ~2.4x longer at 15 VP than at 8 (median 417 turns vs 172, measured over
30 WeightedRandom mirror games), and the terminal win/loss is the only reward there is:

```bash
python -m src.agent.train --vps-to-win 15 --longest-road --max-turns 1500 --gamma 0.999     --total-steps 3000000 --lookahead --opponent pool     --placement-model checkpoints/placement/scorer_ppo.pt     --bundle-model checkpoints/placement/bundle_noroads.pt     --eval-games 200 --eval-workers 8 --run-name <name>
```

But read "What actually helps" above first: **another training run is usually the wrong move.**

**How the ruleset travels.** `src/env/rules.py` decides at *import* time whether to suppress
the Longest Road award, and `SubprocVecEnv` / arena workers are **spawned** on Windows, so
they re-import everything from a fresh interpreter. A function argument therefore cannot
carry the ruleset. It lives in environment variables instead (`CATAN_VPS_TO_WIN`,
`CATAN_LONGEST_ROAD`, `CATAN_MAX_TURNS`), which children inherit for free -- the same
mechanism `determinism.py` uses for `PYTHONHASHSEED`. Entry points call
`ruleset.apply_cli_overrides()` as their **first statement, above the engine imports**;
`train.py` then hard-fails if argparse disagrees with the ruleset the engine imported under,
so a run can never quietly train under the wrong rules. This is why `.flake8` grants E402 to
`train.py` and `benchmark.py`.

**Why 15 VP is a different game.** Buildings cap at **9 VP** (5 settlements, 4 of them
upgraded to cities). So 12 VP is unreachable without Largest Army or Longest Road, and 15 is
unreachable without VP cards on top. Road-building stops being the pathology the old rules
made it and becomes mandatory.

**Train the placement scorer**: generate, then fit each target off the same data:

```bash
python -m src.placement.dataset --pairs 8000 --workers 8 \
    --rollout ppo --rollout-model checkpoints/ppo-pool-placement-lookahead/best.zip
python -m src.placement.train --data data/placement/samples.npz \
    --target corner --out checkpoints/placement/scorer_ppo.pt
python -m src.placement.train --data data/placement/samples.npz \
    --target bundle --out checkpoints/placement/bundle_noroads.pt
```

**Benchmark**: `python -m src.eval.benchmark --agent ppo-mcts --model checkpoints/<run>/best.zip
--opponent value --games 200`. To hand the opening to the placement specialist, pass **both**
models — the corner scorer shortlists, the bundle scorer picks the pair:

```
--placement-model checkpoints/placement/scorer_ppo.pt \
--bundle-model    checkpoints/placement/bundle_noroads.pt
```

`--placement-model` alone still works and falls back to greedy corner-at-a-time selection, which
is what every measurement before `51a3097` used.

**Play with search** (`--agent ppo-mcts`), which is how a trained checkpoint should actually be
deployed — worth ~10 points over the same weights playing directly:

```bash
python -m src.eval.benchmark --vps-to-win 15 --longest-road --max-turns 1500 \
    --agent ppo-mcts --model checkpoints/archive/ppo-15vp-lr-step400000.zip --simulations 50 \
    --opponent ppo --opponent-model checkpoints/archive/ppo-15vp-lr-step400000.zip --games 200 \
    --placement-model checkpoints/placement/scorer_ppo.pt \
    --bundle-model    checkpoints/placement/bundle_noroads.pt \
    --opponent-placement-model checkpoints/placement/scorer_ppo.pt \
    --opponent-bundle-model    checkpoints/placement/bundle_noroads.pt
```

**Any search measurement recorded before `c54ea46` is void.** MCTS built 614-value
observations and handed them to nets expecting 642, so `ppo-mcts` crashed against every
checkpoint trained with `--lookahead`. The older "80% → 95% vs weighted-random" and the
52.8/55.0/56.0% mirror figures at 50/100/200 sims were measured on pre-lookahead models and
do not describe the current agents. The evaluator now declares `wants_lookahead` (inferred
from its own checkpoint) and the search reads it once at construction.

**Sample sizes.** A 300-game head-to-head reported a 3-point edge that a 500-game run at a
different seed did not reproduce (56.0% then 51.0%; pooled 52.9% ± 1.8% over 800). Budget
**800+ games** before believing any difference under ~5 points, and treat 200 games as
resolving nothing finer than ~7 points.

**Play vs the AI** (matplotlib window; you are RED, the trained agent is BLUE; type a move
number, click "Next turn" to let the AI play). Give it the same ruleset, search and placement
models a benchmark would, or you are not playing the agent that was measured:

```bash
python -m src.eval.play --vps-to-win 15 --longest-road --max-turns 1500 \
    --agent ppo-mcts --model checkpoints/archive/ppo-15vp-lr-step400000.zip \
    --simulations 50 \
    --placement-model checkpoints/placement/scorer_ppo.pt \
    --bundle-model    checkpoints/placement/bundle_noroads.pt
```

**Test**: `python -m pytest tests/ -q`

**Lint**: `flake8 src/ tests/`

## Checkpoints worth keeping

`checkpoints/` is git-ignored, so `checkpoints/archive/` is the convention for models that
should outlive their run directory (a rerun with the same `--run-name` overwrites everything
else). **None of these can place their own opening** — see the Caveats below.

| file | rules | notes |
| --- | --- | --- |
| `ppo-15vp-lr-step400000.zip` | 15 VP, LR | **current agent.** Fine-tuned from the 12 VP trunk model; run it with `--agent ppo-mcts --simulations 50` |
| `ppo-12vp-trunk-step2000000.zip` | 12 VP, LR | shared trunk, 932k params; its parent |
| `ppo-12vp-lr-step1800000.zip` | 12 VP, LR | two-tower `[256,256]`, 537k params; a statistical tie with the trunk model |

The first two are **trunk models**: their `policy_kwargs` names `src.agent.trunk.SharedTrunk` by
module path, so `src/agent/trunk.py` cannot be moved or deleted without breaking them.

The 12 VP models play 15 VP without retraining (80–0 vs weighted-random out of the box,
15.0 VP in 145 turns) — the win condition changes, the mechanics do not. 800k steps of 15 VP
fine-tuning on top measured 48.5% ± 2.5% against its own starting point, i.e. nothing.

## Reproducibility

Every entry point calls `src/env/determinism.py:ensure_hash_seed()`, which **relaunches the
process once under `PYTHONHASHSEED=0`** when the variable is unset. Without it a seed fixes the
board but not the game: Catanatron builds `playable_actions` off sets of enum members, and
`enum.Enum.__hash__` hashes the member *name string*, so randomised string hashing reshuffles
action order every process. The same seed produced five different games across five runs before
this landed. Set `PYTHONHASHSEED=random` to opt out, or any other value to pin a different one.

Two consequences: results are now identical regardless of `--workers`, and **any measurement
recorded before `c34ab25` is not reproducible** — its statistics are still valid, but the exact
games cannot be recovered.

## Code Quality

This project uses **flake8** for linting (config in `.flake8`, max line length 100). Before
presenting any code change, verify it passes flake8.

## Phased Roadmap & Docs

See `docs/` for per-phase guides:

- **`PHASE1_SETUP.md`** — Catanatron install, 1v1 env wiring, smoke test
- **`PHASE2_AI.md`** — the shipped agent: PPO training, PUCT search at inference, the opening
  specialist, benchmarking, and a list of measured dead ends so they are not re-run
- **`PHASE3_WEB.md`** — colonist.io bridge (WebSocket read + Playwright clicks)

## Caveats

**No trained checkpoint here is a standalone agent — every one must ship with a placement
scorer**, including all three in `checkpoints/archive/`. Any run trained through
`PlacementWrapper` plays the
opening inside `reset()`, so those decisions never enter the rollout buffer and the policy head
receives **zero gradient on placement**. Deprived of the scorer it places with an untrained head.
Measured over 200 games each: 100.0% -> 88.2% vs weighted-random, and head-to-head against
`ppo-8vp-scratch` it goes from 53.0% (both scored) to 42.2% (neither) -- i.e. bare it is *worse*
than the older agent that at least learned placement badly. This applies to the Phase 3
colonist.io bridge too: the scorer has to be part of the deployed agent, not an eval-time extra.

**The placement specialist was fitted under the old rules**, on rollouts that never ran to 12
or 15 VP. Whether corner values shift when the game runs longer is untested, and it is the
most likely place a stale assumption is still costing points. Refitting it against
`ppo-15vp-lr-step400000.zip` rollouts is the clearest open piece of work — and note the
rank/rho diagnostic is gone, so validate a refit with 800+ benchmark games, not with the loss.
