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
- **Learning algorithm**: **AlphaZero** — one PyTorch net (policy + value heads) trained by
  self-play, where PUCT tree search supplies the improved policy target. `src/agent/train_az.py`.
  **`MaskablePPO` (`src/agent/train.py`) is the active track** -- one forward per decision vs
  AlphaZero's ~200, and ~92% vs weighted-random at 3M steps vs AlphaZero's 70.8%. The old "PPO only
  builds roads" verdict predates the Longest-Road rule change and is confounded by it.
- **Game mode**: 1v1 (`enemies=[one bot]`, `map_type="BASE"`, `vps_to_win=8`)
- **Custom rules**: `src/env/rules.py` monkeypatches Catanatron at import time. Applied
  automatically via `src/env/catan_env.py`. Seven patches:
  1. discard on a 7 only above **9** cards (`discard_limit=9`, vs. stock 7)
  2. per-resource, one-card-at-a-time discard the policy actually chooses
     (expands the action space 290 → **294**)
  3. correct multi-discarder sequencing (fixes an upstream `> 7` hardcode)
  4. Colonist.io 1v1 robber placement restrictions
  5. **Longest Road awards no VP** (length still tracked, `HAS_ROAD` never set)
  6. a dev card **cannot be played the turn it was bought** (one-per-turn is already
     enforced upstream)
  7. `_discard_remaining` survives `State.copy()` — required for MCTS
- **Opening placement**: a separate self-trained specialist (`src/placement/`). Initial settlement
  choice gets ~2 of ~300 gradient samples per episode, so the main policy learned "settle where
  three tiles meet" but never learned that an 8 beats a 3. The specialist trains on random
  openings labelled with actual game outcomes, using duplicate-board pairs for variance reduction.
  Two models, used together: a per-corner scorer (`scorer_ppo.pt`) shortlists candidates, and a
  **bundle scorer** (`bundle_noroads.pt`) ranks whole corner *pairs*, so the first settlement is
  chosen knowing what the second could be. Pair-search beat greedy corner selection 53.5% over
  1200 games; three attempts at improving the labels moved strength not at all.
  **No hardcoded placement knowledge**: features are mechanical board facts only, and the
  hand-written scorer in `heuristic.py` is an evaluation yardstick that never plays.
- **Reward shaping**: none. AlphaZero trains on the sparse win/loss outcome; search provides the
  dense signal that milestone bonuses used to stand in for.
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
│   ├── rules.py         # custom-rule monkeypatches (applied at import)
│   ├── catan_env.py     # gym env + raw-Game factory, shared constants
│   └── lookahead.py     # two-roll dice features (P(afford), discard risk)
├── agent/
│   ├── encoding.py      # obs vector + action mask off a live Game
│   ├── net.py           # AlphaZeroNet (residual MLP, policy + value heads)
│   ├── evaluator.py     # leaf evaluators: net, PPO adapter, uniform control
│   ├── mcts.py          # PUCT search with chance nodes; MCTSPlayer
│   ├── selfplay.py      # self-play game generation + value targets
│   ├── train_az.py      # AlphaZero training loop  <-- the main track
│   ├── arena.py         # head-to-head match play + agent-by-name registry
│   ├── train.py         # MaskablePPO loop  <-- the active track
│   ├── pool.py / elo.py / checkpoint_manager.py   # self-play ladder for train.py
│   └── opponent.py      # PolicyPlayer (frozen PPO checkpoint as a Player)
├── placement/           # opening-settlement specialist (self-trained, no heuristics)
│   ├── features.py      # 45 mechanical per-node board facts
│   ├── dataset.py       # random openings + duplicate-board outcome labels
│   ├── model.py         # PlacementNet (scores one node) + BundleNet (scores a pair)
│   ├── chooser.py       # OpeningChooser: shortlist corners, then search pairs
│   ├── train.py         # supervised fit on the outcome labels
│   ├── heuristic.py     # hand-written scorer -- EVALUATION YARDSTICK ONLY
│   ├── evaluate.py      # rank/rho diagnostics on held-out boards
│   ├── env_wrapper.py   # gym-side equivalent; opens inside reset()
│   └── player.py        # wraps any agent; takes over only the opening
├── eval/
│   ├── benchmark.py     # any agent vs any agent
│   ├── bench_mcts.py    # search compute-budget profiler
│   └── play.py          # human vs AI (matplotlib)
└── bridge/              # (Phase 3) colonist.io WebSocket reader + Playwright clicker
docs/          # Per-phase guides (see below)
checkpoints/   # Saved models (git-ignored)
tests/         # Unit & integration tests
```

## Essential Commands

**Setup**: `pip install -r requirements.txt`

**Smoke test (Phase 1)**: `python -m src.env.smoke_test`

**Train (Phase 2)**: `python -m src.agent.train_az --iterations 200 --games-per-iter 64 --workers 16`

**Train (PPO, sparse + placement + lookahead)**: `python -m src.agent.train --total-steps 3000000
--no-shaping --placement-model checkpoints/placement/scorer_ppo.pt --lookahead --eval-games 200
--eval-workers 8 --run-name <name>`

`src/agent/train.py` has **no `--bundle-model` flag**, so a training run opens with greedy
per-corner selection even though pair-search is the stronger rule at eval time. `env_wrapper.py`
and `arena.py` both accept a bundle; only the train CLI does not plumb it through.

**Train the placement scorer**: generate, then fit each target off the same data:

```bash
python -m src.placement.dataset --pairs 8000 --workers 8 \
    --rollout ppo --rollout-model checkpoints/ppo-pool-placement-lookahead/best.zip
python -m src.placement.train --data data/placement/samples.npz \
    --target corner --out checkpoints/placement/scorer_ppo.pt
python -m src.placement.train --data data/placement/samples.npz \
    --target bundle --out checkpoints/placement/bundle_noroads.pt
```

**Benchmark**: `python -m src.eval.benchmark --agent az --model checkpoints/<run>/best.pt
--opponent value --games 200`. To hand the opening to the placement specialist, pass **both**
models — the corner scorer shortlists, the bundle scorer picks the pair:

```
--placement-model checkpoints/placement/scorer_ppo.pt \
--bundle-model    checkpoints/placement/bundle_noroads.pt
```

`--placement-model` alone still works and falls back to greedy corner-at-a-time selection, which
is what every measurement before `51a3097` used.

**Search budget**: `python -m src.eval.bench_mcts --simulations 100`

**Play vs the AI**: `python -m src.eval.play --agent az --model checkpoints/<run>/best.pt`
(matplotlib window; you are RED, the trained agent is BLUE; type a move number, click
"Next turn" to let the AI play)

**Test**: `python -m pytest tests/ -q`

**Lint**: `flake8 src/ tests/`

## Reproducibility

Every entry point calls `src/env/determinism.py:ensure_hash_seed()`, which **relaunches the
process once under `PYTHONHASHSEED=0`** when the variable is unset. Without it a seed fixes the
board but not the game: Catanatron builds `playable_actions` off sets of enum members, and
`enum.Enum.__hash__` hashes the member *name string*, so randomised string hashing reshuffles
action order every process. The same seed produced five different games across five runs before
this landed. Set `PYTHONHASHSEED=random` to opt out, or any other value to pin a different one.

Two consequences: results are now identical regardless of `--workers`, and **any measurement
recorded before `1b6a3f2` is not reproducible** — its statistics are still valid, but the exact
games cannot be recovered.

## Code Quality

This project uses **flake8** for linting (config in `.flake8`, max line length 100). Before
presenting any code change, verify it passes flake8.

## Phased Roadmap & Docs

See `docs/` for per-phase guides:

- **`PHASE1_SETUP.md`** — Catanatron install, 1v1 env wiring, smoke test
- **`PHASE2_AI.md`** — the AlphaZero agent (MCTS + self-play), the PPO track it replaced,
  cloud training, eval
- **`PHASE3_WEB.md`** — colonist.io bridge (WebSocket read + Playwright clicks)

## Caveats

**`checkpoints/ppo-pool-placement-lookahead/best.zip` is not a standalone agent — it must ship
with a placement scorer.** It was *trained* against `scorer.pt` (weighted-random labels), so that
checkpoint is its native opening; `scorer_ppo.pt` measured as a tie and is the better default for
new work, but swapping it in gives this agent openings it never trained under.
Any run trained through `PlacementWrapper` plays the
opening inside `reset()`, so those decisions never enter the rollout buffer and the policy head
receives **zero gradient on placement**. Deprived of the scorer it places with an untrained head.
Measured over 200 games each: 100.0% -> 88.2% vs weighted-random, and head-to-head against
`ppo-8vp-scratch` it goes from 53.0% (both scored) to 42.2% (neither) -- i.e. bare it is *worse*
than the older agent that at least learned placement badly. This applies to the Phase 3
colonist.io bridge too: the scorer has to be part of the deployed agent, not an eval-time extra.
