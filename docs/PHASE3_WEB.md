# Phase 3 — Web integration (colonist.io)

**Goal:** bridge the trained agent so it can read and play real 1v1 games on colonist.io.
**Status:** ⏳ deferred until a strong agent exists (Phase 2 done). Opt-in.

> ⚠️ **ToS / bans:** Automating play on colonist.io likely violates its Terms of Service and can
> get accounts banned. Use a **throwaway account**, run supervised, and never automate ranked play
> on a real account. This phase is opt-in.

## Architecture

Keep the agent untouched. Add an adapter in `src/bridge/` that translates between colonist.io and
our observation/action representation. Chosen approach: **WebSocket read + browser-automation
clicks.**

```
colonist.io (browser)
   │  WebSocket JSON (game state)
   ▼
Tampermonkey userscript ──► local Python server ──► state translator ──► (614,) observation
                                                                              │
                                                          agent.act(obs, mask)│
                                                                              ▼
                                                   action → Catanatron action → UI click plan
                                                                              │
                                                                              ▼
                                                              Playwright performs clicks
```

## Components

1. **State reader** — Tampermonkey userscript hooks the WebSocket and forwards colonist.io's JSON
   messages to a local Python server. Reconstruct a `catanatron` `Game`/`State`, then run it
   through the same feature pipeline (`catanatron_gym.features.create_sample`) the agent trained on
   to produce the `(614,)` observation. Prior art:
   [robottler](https://github.com/meesg/robottler),
   [this writeup](https://medium.com/@alberttheblacksheep/abusing-my-computer-science-knowledge-to-cheat-at-catan-a0f72fa30309).
2. **Action sender** — map the agent's chosen Catanatron action (one of the 290) to a colonist.io
   UI click sequence via Playwright.
3. **The hard part — coordinate translation.** colonist.io's tile/node/edge IDs and pixel
   coordinates must be mapped to Catanatron's node/edge/tile indexing (and back). Scope this as its
   own mini-project; it is the main source of risk and effort here.

## Verification ladder

1. **Spectator dry-run:** read state only, log the move the agent *would* make each turn. No clicks.
   Confirms the state translator and observation match the agent's training distribution.
2. **Single supervised live game** on a throwaway account, human ready to intervene.
3. Only then consider unattended runs.

## Deps (add when starting)

Uncomment `playwright` in `requirements.txt`, then `playwright install chromium`.
