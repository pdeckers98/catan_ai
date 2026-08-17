# 1v1 Catan AI for Colonist.io

A Python AI that learns to play **1v1 Settlers of Catan** via deep reinforcement learning, built on
the [Catanatron](https://github.com/bcollazo/catanatron) engine, with the eventual goal of playing
live games on [colonist.io](https://colonist.io).

The game engine is a solved problem (Catanatron) — the work is in the agent and the web bridge.

## Roadmap

| Phase | Goal | Status |
|---|---|---|
| **1 — Setup** | Catanatron 1v1 env wired up + smoke test | ✅ done — [docs/PHASE1_SETUP.md](docs/PHASE1_SETUP.md) |
| **2 — AI** | MaskablePPO + search at inference + a learned opening | ✅ agent shipped — [docs/PHASE2_AI.md](docs/PHASE2_AI.md) |
| **3 — Web** | colonist.io bridge (WebSocket read + Playwright clicks) | ⏳ next — [docs/PHASE3_WEB.md](docs/PHASE3_WEB.md) |

## Quickstart

```bash
# Uses the `catan` conda env (Python 3.14)
pip install -r requirements.txt
python -m src.env.smoke_test --games 20   # env sanity check
python -m pytest tests/ -q
flake8 src/ tests/
```

## Play against it

You are RED, the agent is BLUE. **The agent is three artifacts, not one** — the checkpoint, the
search, and both placement models. Leave any of them out and you are not playing the agent that was
measured.

```bash
python -m src.eval.play --vps-to-win 15 --longest-road --max-turns 1500     --agent ppo-mcts --model checkpoints/archive/ppo-15vp-lr-step400000.zip     --simulations 50     --placement-model checkpoints/placement/scorer_ppo.pt     --bundle-model    checkpoints/placement/bundle_noroads.pt
```

## Layout

```
src/env/        # 1v1 Gymnasium env, house rules, per-run ruleset, lookahead features
src/agent/      # MaskablePPO training, PUCT search, arena/benchmark harness
src/placement/  # the opening-settlement specialist (self-trained)
src/eval/       # benchmark, human-vs-AI play
src/bridge/     # (Phase 3) colonist.io WebSocket reader + Playwright clicker
docs/           # per-phase guides
```

## Caveats

- **License:** Catanatron is GPL-3.0 (copyleft). Fine for personal use; distributing this code
  would require GPL-3.0.
- **ToS:** Automating colonist.io likely violates its Terms of Service and risks account bans.
  Phase 3 is opt-in and uses a throwaway account.

See [CLAUDE.md](CLAUDE.md) for the full project guide.
