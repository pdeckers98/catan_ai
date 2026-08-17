# Phase 3 — Web integration (colonist.io)

**Goal:** bridge the trained agent so it can read and play real 1v1 games on colonist.io.
**Status:** ⏳ next up — a strong agent now exists. Opt-in.

> ⚠️ **ToS / bans:** Automating play on colonist.io likely violates its Terms of Service and can
> get accounts banned. Use a **throwaway account**, run supervised, and never automate ranked play
> on a real account. This phase is opt-in.

## Architecture

Keep the agent untouched. Add an adapter in `src/bridge/` that translates between colonist.io and
our observation/action representation. Chosen approach: **WebSocket read + browser-automation
clicks.**

**What "the agent" means here is three artifacts, not one.** The deployed player is the PPO
checkpoint *plus* 50-sim PUCT search *plus* both placement models — see the Caveats in
`PHASE2_AI.md`. Shipping the checkpoint alone gives up ~10 points to the missing search and places
its opening with an untrained head.

```
colonist.io (browser)
   │  WebSocket JSON (game state)
   ▼
Tampermonkey userscript ──► local Python server ──► state translator ──► catanatron Game
                                                                              │
                                          MCTSPlayer(PPOEvaluator).decide(...)│
                                                                              ▼
                                                   action → Catanatron action → UI click plan
                                                                              │
                                                                              ▼
                                                              Playwright performs clicks
```

## Components

1. **State reader** — Tampermonkey userscript hooks the WebSocket and forwards colonist.io's JSON
   messages to a local Python server. Reconstruct a `catanatron` `Game`/`State`. Reconstructing the
   *`Game`*, not just the observation vector, is what the search needs: `MCTSPlayer` copies and
   advances a real `Game` as it descends. `src/agent/encoding.py` then produces the observation
   (**642** values with `--lookahead`, 614 without) and the action mask off that same object. Prior
   art:
   [robottler](https://github.com/meesg/robottler),
   [this writeup](https://medium.com/@alberttheblacksheep/abusing-my-computer-science-knowledge-to-cheat-at-catan-a0f72fa30309).
2. **Action sender** — map the agent's chosen Catanatron action (one of the 294) to a colonist.io
   UI click sequence via Playwright.
3. **The hard part — coordinate translation.** colonist.io's tile/node/edge IDs and pixel
   coordinates must be mapped to Catanatron's node/edge/tile indexing (and back). Scope this as its
   own mini-project; it is the main source of risk and effort here.
4. **Rule reconciliation.** The agent trained under house rules that are *not* stock Catan:
   discard above 9 cards rather than 7, per-resource discard, no dev card the turn it was bought,
   and the 15 VP / Longest Road target. Where colonist.io's ruleset differs, the bridge is playing
   a different game than the one the policy was fitted to. Check each of the seven patches in
   `src/env/rules.py` against the actual lobby settings before the first live game.
5. **Public information only.** The observation was built so every feature is derivable from what a
   human player can see — no opponent hand composition anywhere. That constraint exists precisely so
   this phase is possible; do not relax it in the translator.

## Verification ladder

0. **Translator round-trip:** reconstruct a `Game` from captured WebSocket traffic and assert the
   observation it produces matches one built by `make_1v1_game` on an equivalent position. A silent
   mismatch here looks like a weak agent, not like a bug.
1. **Spectator dry-run:** read state only, log the move the agent *would* make each turn. No clicks.
   Confirms the state translator and observation match the agent's training distribution.
2. **Single supervised live game** on a throwaway account, human ready to intervene.
3. Only then consider unattended runs.

## Deps (add when starting)

Uncomment `playwright` in `requirements.txt`, then `playwright install chromium`.
