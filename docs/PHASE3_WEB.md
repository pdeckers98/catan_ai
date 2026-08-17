# Phase 3 — Web integration (colonist.io)

**Goal:** bridge the trained agent so it can read and play real 1v1 games on colonist.io.
**Status:** 🚧 started 2026-08-17. The offline core (board reconstruction, replay, player
assembly) is in `src/bridge/` and tested; everything colonist-specific is blocked on captured
WebSocket traffic. Opt-in.

> ⚠️ **ToS / bans:** Automating play on colonist.io likely violates its Terms of Service and can
> get accounts banned. Use a **throwaway account**, run supervised, and never automate ranked play
> on a real account. This phase is opt-in.

## Architecture

Keep the agent untouched. Add an adapter in `src/bridge/` that translates between colonist.io and
our observation/action representation. Chosen approach: **WebSocket read + browser-automation
clicks**, both through **one Playwright session**. The read side attaches a CDP session and
listens to `Network.webSocketFrameReceived`/`Sent`; there is no Tampermonkey userscript and no
local HTTP server. Playwright is required for the clicks regardless, so routing the read through
it too costs one process instead of three and keeps the page handle shared between reading and
clicking. (An earlier draft of this doc specified a userscript forwarding frames to a local
server; it was dropped before any of it was written.)

**What "the agent" means here is three artifacts, not one.** The deployed player is the PPO
checkpoint *plus* 50-sim PUCT search *plus* both placement models — see the Caveats in
`PHASE2_AI.md`. Shipping the checkpoint alone gives up ~10 points to the missing search and places
its opening with an untrained head.

```
colonist.io (browser, driven by Playwright)
   │  CDP: Network.webSocketFrameReceived  (game state as JSON)
   ▼
protocol translator ──► BoardSpec + observed Actions ──► GameReplay ──► catanatron Game
                                                                              │
                                          MCTSPlayer(PPOEvaluator).decide(...)│
                                                                              ▼
                                                   action → Catanatron action → UI click plan
                                                                              │
                                                                              ▼
                                                    Playwright clicks, same page handle
```

## What exists (`src/bridge/`)

- **`board.py`** — `BoardSpec` → `CatanMap`. `build_map("BASE")` shuffles, so it cannot express
  the board colonist dealt; `initialize_tiles` takes the three shuffled lists as parameters, and
  feeding it explicit ones yields the live layout while keeping catanatron's node/edge/tile
  numbering — the numbering the action space and placement features are written against. The spec
  validates itself against the BASE multisets, so a mistranslated board fails here rather than
  looking like a weak agent later.
- **`replay.py`** — `GameReplay`: a `Game` advanced by *observed* actions instead of player
  decisions. `apply_action` accepts realized values for every stochastic action (`ROLL` takes the
  dice, `BUY_DEVELOPMENT_CARD` the card, `MOVE_ROBBER` the stolen resource), so a game can be
  re-applied move for move and the resulting `State` is consistent by construction. Every action is
  checked against `playable_actions` first — modulo the chance outcome — and a mismatch raises
  `DesyncError` immediately, because silent drift is the failure mode that looks like a weak agent.
  It also **forces the seating order**: `State.__init__` shuffles the players, and in 1v1 the first
  seat settles first.
- **`player.py`** — `build_bridge_player`, which refuses to build anything less than all three
  artifacts. Every other entry point makes search and the placement models optional flags, which is
  right for benchmarking and wrong for live play.
- **`tests/test_bridge.py`** — rung 0 below, offline: a full self-play game replayed through a
  reconstructed board must produce identical observations, VPs and legal-move sets.

Writing this surfaced a real bug in `src/env/rules.py`: the patched per-resource discard replaced
upstream's `apply_action` for `DISCARD` and never logged the action, so **every game's action log
was silently missing its discards**. Replay desynced on the first 7 where a hand went over the
limit. Fixed, with a regression test in `tests/test_rules.py`.

## Components still to build

1. **Protocol translator** — turn colonist.io's WebSocket JSON into a `BoardSpec` and a stream of
   fully-specified catanatron `Action`s. Blocked on captured traffic; the protocol is undocumented.
   Prior art:
   [robottler](https://github.com/meesg/robottler),
   [this writeup](https://medium.com/@alberttheblacksheep/abusing-my-computer-science-knowledge-to-cheat-at-catan-a0f72fa30309).
2. **Action sender** — map the agent's chosen Catanatron action (one of the 294) to a colonist.io
   UI click sequence via Playwright.
3. **The hard part — coordinate translation.** colonist.io's tile/node/edge IDs and pixel
   coordinates must be mapped to Catanatron's node/edge/tile indexing (and back). Scope this as its
   own mini-project; it is the main source of risk and effort here.
4. **Hidden information.** "Public information only" holds for the *observation*, and the
   constraint was worth keeping — but search is the tighter requirement: `MCTSPlayer` rolls a real
   `Game` forward, so it needs the opponent's **actual hand**, not just its size. In 1v1 nearly all
   of that is publicly derivable: roll payouts are deterministic from the board, bank and port
   trades are public, monopoly and year-of-plenty resolve in the open, and both directions of a
   robber steal involve us, so we always see the card. Exactly two leaks remain — **the opponent's
   discards on a 7** and **its dev cards before they are played**. So the belief state stays exact
   until the first opponent discard, after which the bridge needs a small determinization
   (sample a hand consistent with the counts and the deck; resample per search). Contained, but
   design it rather than discover it live.
5. **Rule reconciliation.** The agent trained under house rules that are *not* stock Catan:
   discard above 9 cards rather than 7, per-resource discard, no dev card the turn it was bought,
   and the 15 VP / Longest Road target. **Decision: assume the lobby matches those patches**, and
   let `GameReplay.check_legal` fail loudly on the first divergent event rather than quietly
   playing a different game than the policy was fitted to. Still worth eyeballing the seven patches
   in `src/env/rules.py` against the lobby settings before the first live game — a rule that
   differs without ever producing an illegal action (a different discard limit, say) will not trip
   the assertion, it will just cost points.

## Verification ladder

0. **Replay round-trip.** ✅ done offline in `tests/test_bridge.py`: a self-play game replayed
   through a reconstructed board produces identical observations. Repeat it on *captured* traffic
   once the translator exists — same assertion, real input.
1. **Spectator dry-run:** read state only, log the move the agent *would* make each turn. No clicks.
   Confirms the state translator and observation match the agent's training distribution.
2. **Single supervised live game** on a throwaway account, human ready to intervene.
3. Only then consider unattended runs.

## What is blocked on captured traffic

The protocol is undocumented, so nothing colonist-specific can be written without recordings:

- **2–4 complete 1v1 games** as raw WebSocket frames, both directions, ideally covering a 7 with a
  discard, a dev card bought and later played, a port trade, and a robber steal each way.
- **The lobby settings**, to check against the seven patches.
- **A DOM dump of a live board**, for the click layer.

## Deps

`playwright` is uncommented in `requirements.txt`; run `playwright install chromium` once.
