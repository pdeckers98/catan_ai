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
- **RL framework**: PyTorch via `stable-baselines3` + `sb3-contrib` (`MaskablePPO`); optional
  raw-PyTorch PPO later
- **Game mode**: 1v1 (`enemies=[one bot]`, `map_type="BASE"`, `vps_to_win=15`)
- **Action masking**: mandatory — most actions are illegal each turn; always respect
  `info["valid_actions"]` / `env.unwrapped.get_valid_actions()`
- **Training hardware**: high-core CPU cloud instance (RL here is CPU-bound — parallel game
  rollouts dominate; the policy is a small MLP)
- **Inference hardware**: local GTX 1660 Super
- **Web integration (Phase 3)**: colonist.io WebSocket interception (read state) + Playwright (clicks)

## Project Structure

```
src/
├── env/       # Gym env construction, opponent config, reward shaping
├── agent/     # PyTorch policy/value net + training (deep RL)
├── eval/      # Benchmark harness vs Catanatron's built-in bots
└── bridge/    # (Phase 3) colonist.io WebSocket reader + Playwright clicker
docs/          # Per-phase guides (see below)
checkpoints/   # Saved models (git-ignored)
tests/         # Unit & integration tests
```

## Essential Commands

**Setup**: `pip install -r requirements.txt`

**Smoke test (Phase 1)**: `python -m src.env.smoke_test`

**Train (Phase 2)**: `python -m src.agent.train`

**Benchmark (Phase 2)**: `python -m src.eval.benchmark --games 500`

**Lint**: `flake8 src/ tests/`

## Code Quality

This project uses **flake8** for linting (config in `.flake8`, max line length 100). Before
presenting any code change, verify it passes flake8.

## Phased Roadmap & Docs

See `docs/` for per-phase guides:

- **`PHASE1_SETUP.md`** — Catanatron install, 1v1 env wiring, smoke test
- **`PHASE2_AI.md`** — deep RL agent (MaskablePPO → custom PyTorch), self-play, cloud training, eval
- **`PHASE3_WEB.md`** — colonist.io bridge (WebSocket read + Playwright clicks)

## Caveats