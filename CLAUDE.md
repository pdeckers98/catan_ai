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
  The `MaskablePPO` training loop was removed; its strongest checkpoint
  (`checkpoints/ppo-8vp-scratch`) is retained as a benchmark opponent, loadable via the `ppo` and
  `ppo-mcts` agent specs.
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
│   └── catan_env.py     # gym env + raw-Game factory, shared constants
├── agent/
│   ├── encoding.py      # obs vector + action mask off a live Game
│   ├── net.py           # AlphaZeroNet (residual MLP, policy + value heads)
│   ├── evaluator.py     # leaf evaluators: net, PPO adapter, uniform control
│   ├── mcts.py          # PUCT search with chance nodes; MCTSPlayer
│   ├── selfplay.py      # self-play game generation + value targets
│   ├── train_az.py      # AlphaZero training loop  <-- the main track
│   ├── arena.py         # head-to-head match play + agent-by-name registry
│   └── opponent.py      # PolicyPlayer (frozen PPO checkpoint as a Player)
├── placement/           # opening-settlement specialist (self-trained, no heuristics)
│   ├── features.py      # 45 mechanical per-node board facts
│   ├── dataset.py       # random openings + duplicate-board outcome labels
│   ├── model.py         # PlacementNet (small MLP, scores one node)
│   ├── train.py         # supervised fit on the outcome labels
│   ├── heuristic.py     # hand-written scorer -- EVALUATION YARDSTICK ONLY
│   ├── evaluate.py      # rank/rho diagnostics on held-out boards
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

**Train the placement scorer**: `python -m src.placement.dataset --pairs 8000 --workers 8` then
`python -m src.placement.train --data data/placement/samples.npz`

**Benchmark**: `python -m src.eval.benchmark --agent az --model checkpoints/<run>/best.pt
--opponent value --games 200` (add `--placement-model checkpoints/placement/scorer.pt` to hand
the opening to the placement specialist)

**Search budget**: `python -m src.eval.bench_mcts --simulations 100`

**Play vs the AI**: `python -m src.eval.play --agent az --model checkpoints/<run>/best.pt`
(matplotlib window; you are RED, the trained agent is BLUE; type a move number, click
"Next turn" to let the AI play)

**Test**: `python -m pytest tests/ -q`

**Lint**: `flake8 src/ tests/`

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