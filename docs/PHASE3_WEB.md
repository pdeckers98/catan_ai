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
- **`capture.py`** — the read half of the transport, run standalone and by hand. It opens a real
  browser (persistent profile under `data/bridge/profile`, so you log in once), you play or
  spectate, and every WebSocket frame in both directions lands in a JSONL file. It automates
  nothing and never clicks. Frames are decoded best-effort — socket.io's integer prefix is
  unwrapped, binary frames are tried as msgpack then JSON — and anything that resists is kept as
  base64 rather than dropped, because an unparsed frame is evidence too.
  `--summarize` reads a capture back as a histogram of message types with one example each, which
  is the form a protocol is actually readable in; a full game is megabytes.

  ```bash
  python -m src.bridge.capture --label game1
  python -m src.bridge.capture --summarize data/bridge/game1-<stamp>.jsonl
  python -m src.bridge.capture --summarize <file> --only sent --chars 2000   # drill in
  ```
- **`protocol.py`** — colonist's wire messages to catanatron `Action`s: the enums, the coordinate
  solution, the board rebuild, and a decoder that walks the server's state diffs. `decode_capture`
  turns a recording into `(board, seating, our colour, actions)`. It says nothing about *sending* a
  move, on purpose — see component 2 below.
- **`player.py`** — `build_bridge_player`, which refuses to build anything less than all three
  artifacts. Every other entry point makes search and the placement models optional flags, which is
  right for benchmarking and wrong for live play.
- **`tests/test_bridge.py`** — rung 0 below, offline: a full self-play game replayed through a
  reconstructed board must produce identical observations, VPs and legal-move sets.

Writing this surfaced a real bug in `src/env/rules.py`: the patched per-resource discard replaced
upstream's `apply_action` for `DISCARD` and never logged the action, so **every game's action log
was silently missing its discards**. Replay desynced on the first 7 where a hand went over the
limit. Fixed, with a regression test in `tests/test_rules.py`.

## The protocol (decoded from two complete 1v1 games, 2026-08-17)

Two full 1v1 games (87 and 94 turns) were captured and read back — one won, one lost. **Every
action in both replays legally through `GameReplay`**, which is the strongest offline statement
the bridge can make. What they settle:

The second game was recorded as a held-out test and earned its keep immediately:

- **Dev card 14 is Road Building**, played twice and each time followed by two free roads. It had
  been left unmapped rather than inferred; now it is observed.
- Two new log entries, both consequences rather than decisions: **68** an achievement changing
  hands, **139** a player-count notice.
- **An upstream catanatron bug**: `road_building_possibilities` gates `PLAY_ROAD_BUILDING` behind
  being able to *afford* a road, though the card's two roads are free. The opponent played it with
  an empty hand, won longest road and won the game; the reconstruction refused the move. Fixed as
  patch 8 in `src/env/rules.py` — and note it means **every agent trained before this could not
  play Road Building when short of resources**, which is when it is worth the most.

What still does not reconstruct exactly: the opponent's **unplayed** dev cards. Game 2's opponent
bought twenty and played sixteen; the four never revealed are drawn at random by the engine, so the
final tally came out at 14 VP instead of 15. Legality is unaffected — this is the residual hidden
information, not a decoding error.

**The lobby matches our house rules.** The settings frame carries `victoryPointsToWin: 15`,
`cardDiscardLimit: 9`, `maxPlayers: 2`, `friendlyRobber: true` — the VP target, the discard limit
and the robber restriction the agent was trained under, confirmed rather than assumed. Also
present and not yet checked against anything: `diceSetting: 1`, `eloType`, `gameSpeed`.

**The server sends one full state, then diffs.** Message `type: 4` is the entire initial
`gameState` (~8 KB): `tileHexStates` (19 hexes, `{x, y, type, diceNumber}`; type 0 is the desert),
`tileCornerStates` (54), `tileEdgeStates` (72), `portEdgeStates` (9, with a `type` per port),
`bankState`, `playerStates`, and per-mechanic blocks for settlements/cities/roads/dev
cards/longest road/largest army/robber. `playerColor` is us; `playOrder` is the seating the
bridge must force. 54/72/9 are exactly catanatron's counts, and each corner and edge carries an
`{x, y, z}` coordinate — so the id mapping is a geometry problem with a deterministic answer, not
a guess.

Thereafter `type: 91` carries a `diff` of that same tree plus `gameLogState` entries, which is the
action stream in readable form. Log entry types seen: 10 roll (`firstDice`, `secondDice`),
47 resource distribution, 44 turn marker, 4 free/initial placement and 5 built-or-bought (both
with `pieceEnum`: 0 road, 2 settlement, 3 city, 5 robber), 11 robber moved, 116 bank trade,
20 dev card played, 86 monopoly, 21 year of plenty, 55 **discard** (`cardEnums`, so the opponent's
discards are visible after all), 14/15 card gained/lost with `specificRecipients`, 66 achievement,
45 game won.

**The client sends actions as frames, not clicks.** Every move is
`{"action": <int>, "payload": ..., "sequence": <int>}` behind the routing header, with the room
name being the game id (`045804`). Correlating each sent frame against the log entry it produced
gives the table:

| action | meaning | payload |
| --- | --- | --- |
| 2 | roll dice | `true` |
| 6 | end turn | `true` |
| 15 / 11 | initial settlement / initial road | corner id / edge id |
| 16 / 19 / 12 | build settlement / city / road | corner id / corner id / edge id |
| 3 | move robber | tile index |
| 49 | trade | `{creator, isBankTrade, offeredResources, wantedResources, ...}` |
| 48 | play dev card | card enum |
| 7 / 8 | resolve a played card (monopoly's resource) | `[resource]` |

`sequence` increments per action. Codes 47, 53, 64, 66 produced no log entry and look like UI
chatter (66 mostly carries a null payload); they are not needed to play.

**This removes the hard part.** `docs/` previously scoped pixel/coordinate translation as its own
mini-project. If the server accepts a synthesized frame — and the client is doing nothing more
than emitting these — the action sender is a msgpack write, and Playwright's role shrinks to
hosting the authenticated session. Unproven until we send one.

## What the wire looks like (from the first capture, login + lobby only)

- **Socket:** `wss://socket.svr.colonist.io/?version=2`. Ignore everything else the browser
  opens — a Discord login gateway and a dozen `127.0.0.1` RPC ports were most of the first
  recording.
- **Encoding: msgpack, not JSON**, in both directions.
- **Server → client:** bare msgpack. Two envelope shapes so far — `{"type": "Connected",
  "userSessionId": ...}` / `{"type": "SessionEstablished"}`, and `{"id": "139", "data": {"type":
  1, "payload": {...}}}` where `id` looks like a subscription and `type` a message code.
- **Client → server:** a routing header first — `0x02 <id> <len> <room-name>` — then msgpack
  `{"action": <int>, "payload": ...}`. `room-name` was `"lobby"`; in a game it is presumably the
  game id. The second header byte varies (`02`, `07`, `0b`) while the room does not, and one
  capture does not say what it counts.
- So the action sender is likely **frames, not clicks**, if the server accepts them — which would
  remove the entire coordinate-translation problem below. Not yet established; a game capture
  showing our own moves is what settles it.

## Components

1. **Protocol translator** — ✅ `src/bridge/protocol.py`. Colonist's messages in, a `BoardSpec` and
   fully-specified catanatron `Action`s out. Every one of the 278 actions in the captured game
   replays legally through `GameReplay`, and under the lobby's ruleset the reconstruction ends on
   the same winner at 15 VP. Unknown messages raise `ProtocolError` rather than being skipped.
   Prior art, no longer needed but worth keeping:
   [robottler](https://github.com/meesg/robottler),
   [this writeup](https://medium.com/@alberttheblacksheep/abusing-my-computer-science-knowledge-to-cheat-at-catan-a0f72fa30309).
2. **Action sender** — the one real unknown left. The client's own frames are legible and could in
   principle be synthesized, but the server may well require a genuine click, with whatever else the
   page attaches to it. **Assume nothing here until a frame has actually been sent and accepted.**
   `protocol.py` deliberately says nothing about sending, so either strategy can be built on it: a
   msgpack write, or a Playwright click sequence driven by the same corner/edge ids.
3. **Coordinate translation** — ✅ solved, and it was never the pixel problem this doc feared.
   Colonist addresses corners and edges as `(hex, z)`; a hex owns its north and south corners and
   its three western edges. Laying both boards on one integer grid and matching positions gives the
   bijection, which is then *validated* per board rather than trusted.
4. **Hidden information.** "Public information only" holds for the *observation*, and the constraint
   was worth keeping — but search is the tighter requirement: `MCTSPlayer` rolls a real `Game`
   forward, so it needs the opponent's **actual hand**, not just its size. In 1v1 nearly all of that
   is publicly derivable. The capture shrank this further than expected: **discards are broadcast**
   (log entry 55 carries the card enums), and both directions of a robber steal are reported to us
   with the card. So exactly **one** leak remains — **dev cards between being bought and being
   played** — and the bridge needs a determinization only for those: sample a deck-consistent
   assignment, resample per search. `protocol.reveal_purchases` handles the *recorded* case by
   back-filling from what was later played; live, that information does not exist yet.
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

The protocol is undocumented, so nothing colonist-specific can be written without recordings.
`capture.py` collects them; what is still needed is the *playing*:

- **2–4 complete 1v1 games** as raw WebSocket frames, both directions, ideally covering a 7 with a
  discard, a dev card bought and later played, a port trade, and a robber steal each way.
- **The lobby settings**, to check against the seven patches. Capture the lobby screen too — the
  settings are sent over the same socket when a game is created.
- **A DOM dump of a live board**, for the click layer.

Frames the agent's own moves generate are as valuable as the ones it receives: the `sent`
direction is the entire specification of the action sender, and it can only be learned by
playing the moves by hand and reading what the page emitted.

## Deps

`playwright` is uncommented in `requirements.txt`; run `playwright install chromium` once.
