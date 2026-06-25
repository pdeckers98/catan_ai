# 1v1 Catan AI for Colonist.io

A Python AI that learns to play **1v1 Settlers of Catan** via deep reinforcement learning, built on
the [Catanatron](https://github.com/bcollazo/catanatron) engine, with the eventual goal of playing
live games on [colonist.io](https://colonist.io).

The game engine is a solved problem (Catanatron) — the work is in the agent and the web bridge.

## Roadmap

| Phase | Goal | Status |
|---|---|---|
| **1 — Setup** | Catanatron 1v1 env wired up + smoke test | ✅ done — [docs/PHASE1_SETUP.md](docs/PHASE1_SETUP.md) |
| **2 — AI** | Deep RL agent (MaskablePPO → custom), self-play, cloud training | ⏳ [docs/PHASE2_AI.md](docs/PHASE2_AI.md) |
| **3 — Web** | colonist.io bridge (WebSocket read + Playwright clicks) | ⏳ [docs/PHASE3_WEB.md](docs/PHASE3_WEB.md) |

## Quickstart

```bash
# Uses the `catan` conda env (Python 3.14)
pip install -r requirements.txt
python -m src.env.smoke_test --games 20   # Phase 1 sanity check
flake8 src/
```

## Layout

```
src/env/     # 1v1 Gymnasium env helpers + smoke test
src/agent/   # (Phase 2) PyTorch RL agent + training
src/eval/    # (Phase 2) benchmark vs built-in bots
src/bridge/  # (Phase 3) colonist.io WebSocket reader + Playwright clicker
docs/        # per-phase guides
```

## Caveats

- **License:** Catanatron is GPL-3.0 (copyleft). Fine for personal use; distributing this code
  would require GPL-3.0.
- **ToS:** Automating colonist.io likely violates its Terms of Service and risks account bans.
  Phase 3 is opt-in and uses a throwaway account.

See [CLAUDE.md](CLAUDE.md) for the full project guide.
