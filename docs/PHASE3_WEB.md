# Phase 3 — Web integration (colonist.io)

**Goal:** bridge the trained agent so it can read and play real 1v1 games on colonist.io.
**Status:** 🚧 started 2026-08-17. The whole read side is built and tested: board
reconstruction, replay, the protocol translator, and a live session that follows a game in real
time and says what the agent would play. **Rung 1 of the ladder below has now been run live** --
one full 1v1 game watched end to end. It found one real translation bug (2:1 ports), which is
fixed; the captured game now reconstructs all 88 turns and the agent answered 141 positions
legally. **The action sender works, and that was the last real unknown.** On 2026-08-19 a
synthesized `end turn` frame was accepted by a live server: the frame went out at capture line 658
and the server's turn marker came back at 662, followed by the opponent's roll. Colonist takes
frames the page's own client never authored, so the sender is a msgpack write and the
coordinate/click project this doc once scoped is not needed.

**Rung 2 has now been run twice: the agent plays its own games.** `--auto-play` arms the live
session, and on 2026-08-19 it played two 1v1 casual matchmaking games start to finish, opening
placement included, with no human input after the game began. 59 moves sent in the first, 0
determinization repairs in either. Both ended on a bug rather than a result, and **both bugs were
found by playing, not by testing** — the offline round-trip agreed with three captured games and
still missed them. See rung 2 below. Opt-in.

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
- **`moves.py`** — the mirror: a catanatron `Action` to the frames colonist's client would have
  sent. Pure, and no browser anywhere near it, so it round-trips offline against the frames a real
  capture recorded. Every code was correlated against the log entry it produced rather than read
  off its number.
- **`sender.py`** — those frames on the wire, over colonist's own socket. `FrameCodec` learns the
  routing header by watching the client and falls back to the `serverId` the server announces, so
  a game with nobody clicking can still be routed.
- **`session.py`** — the live read loop, and rung 1. `LiveGame` keeps a `GameReplay` in step
  with the message stream one message at a time; `DryRun` puts the agent around it, logs the move
  it *would* play, and clicks nothing. Two things make it worth more than a `decode_capture` in a
  `while` loop:

  - **`--replay` drives it off a recording**, so the whole live path runs with no browser, no
    account and no ToS exposure. It is also the only setting where the agent's choice can be
    scored against a *human's* on the same position -- live, you would have to make the move to
    find out. On `game2` at 50 sims: 128 decisions, 74.2% agreeing with the move actually played.
  - **It checks the lobby instead of trusting it.** `victoryPointsToWin`, `cardDiscardLimit`,
    `maxPlayers` and `friendlyRobber` ride on the full-state message, and a mismatch against
    `src/env/rules.py` raises `LobbyMismatch` before a single action is replayed. Item 5 below
    asked for a human to eyeball this; there was no reason for that to be a human's job.

  `protocol.py` was refactored to make it possible: the state machine that used to live inside
  `decode_capture` is now `MessageDecoder`, fed a message at a time, and `decode_capture` is a
  loop over it. So the offline tests exercise the live path.
- **`sender.py`** — the write half, and the only part of the bridge that talks *to* colonist.
  Two pieces on purpose. `FrameCodec` turns a move into the bytes the client would have emitted
  and is pure, so it can be held to the strictest standard available offline: re-encoded against
  a real capture, **all 188 in-game frames come out byte-identical**, which is a stronger claim
  than "it decodes the same" — identical bytes mean the server has nothing to tell our frame from
  the client's. Nothing about the room is hardcoded: the codec learns the routing header and the
  `sequence` counter by *watching* the client use them, the same stance the read side takes.
  `PageSender` is the transport: CDP has no command for writing a websocket frame, so an init
  script wraps `window.WebSocket` before colonist opens it and exposes a send hook, and we write
  on **colonist's own authenticated socket** rather than opening a second one.
  `session.py --allow-send` puts a console on stdin around it and sends **only what you type**.
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

**One signal is on the wire and still undecoded: `playerStates[pid].victoryPointsState`.**
It arrives as a diff of `{source: count}` — keys 0..4, with 4 visibly changing hands the way
longest road does. It would be the strongest desync check available, because it is the server's
own scoreboard and would catch a wrong determinization *silently* corrupting the position, which
legality checks cannot. But the obvious reading is wrong: summed over sources it disagrees with
catanatron's public VP on 324/550 and 437/588 comparisons across the two captures. So it is left
alone rather than half-understood — the same stance the rest of `protocol.py` takes. Cracking it
is the cheapest remaining win in the read half.

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

**And the routing header is no longer a mystery.** It is `<kind> <channel> <len> <room>`: byte 0
is the room kind (2 lobby, 3 game), byte 1 a channel id constant for the room's life, byte 2 the
name's length, then the room name as ASCII — `"lobby"`, or the game id in play. Across the whole
live game the in-game header was byte-identical on all 188 frames (`03 01 06 "02F707"`) while the
lobby used `02 07 05 "lobby"`, which is what "the second byte varies while the room does not" in
the older note below was actually seeing: different rooms, not different frames.

`sequence` is the one thing a synthesized frame cannot get right by copying. It counts the
client's actions, and the client has no idea we spent one — so after we send, its next frame
reuses the number. Whether the server minds is part of what the first live send is for;
`FrameCodec.sent` records every number we consumed so it can be read back against what happened.

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
2. **Action sender** — ✅ `src/bridge/sender.py`, and **proven against a live server on
   2026-08-19**. The encoder rebuilds every in-game client frame of a real capture byte for byte;
   the transport is JS injection, an init script wrapping `window.WebSocket` before colonist opens
   it so the frame leaves on the page's own authenticated socket. That was chosen over Playwright
   clicks (which reintroduce the DOM/coordinate problem component 3 was relieved to have avoided)
   and over a second socket of our own (re-authenticates, looks like a duplicate session, and is
   the version most likely to read as a bot).

   **What the live send settled.** A hand-triggered `end turn` on our own turn:

   ```
   line 658   SENT   action 6, sequence 21          <- ours; PageSender reported ok, 37 bytes
   line 662   RECV   turn marker (log 44)           <- the server acted on it
   line 663+  RECV   opponent rolled 6+2 and took resources
   ```

   So the click layer is not needed and stays unwritten. Two findings ride along:

   - **The server does not enforce `sequence`.** Ours consumed 21; the client, which had no idea,
     later sent its own `action 6, sequence 21` (line 824) and the server accepted that too (turn
     marker at 842). The predicted collision is real and harmless, so there is no need to
     shadow-correct the client's counter.
   - **A synthesized frame is indistinguishable by construction**, which is a verification problem
     as much as a feature: ours is identified in the capture by its timing and sequence, not its
     bytes. `FrameCodec.sent` is the record of what we spent.

   **Which frame to send is a separate module**, `src/bridge/moves.py` — the mirror of
   `protocol.py`, and pure. Every code in its table was correlated against the log entry it
   produced across three captured games, which is why several disagree with a plausible reading
   of the numbers: `7` is not "discard", it is *confirm the current selection*, preceded by `8`
   frames carrying the selection as it grows.

   Three shapes do not line up one-to-one, and each is a real difference between the two games:

   - `PLAY_MONOPOLY` and `PLAY_YEAR_OF_PLENTY` name their resource in the action; colonist plays
     the card first and resolves it after.
   - **`DISCARD` runs the other way.** `src/env/rules.py` made it one action per card, because
     that is the decision the policy has a slot for; colonist wants the finished hand. So the
     agent is asked repeatedly against a *private copy* of the game — the live replay cannot
     advance in between, since it only moves on what the server says happened — and the whole
     discard goes out as one frame.
   - colonist **batches** several bank trades into a single frame. Catanatron has no such action,
     so the read side decodes one frame as N trades and the write side re-emits N frames.

   The offline proof is the same shape as the replay test: decode a captured game, hand every
   move *we* made back to the translator, and compare against what the browser actually put on
   the wire. `game1` regenerates all 141 in-game frames, in order, exactly.

   **That proof was not sufficient, and the way it failed is the lesson.** It can only check
   mappings a human's clicks happened to exercise; the two live games each died on one it did
   not. Codes now proven *live* are roll, end turn, opening settlement and road, road, city, buy
   dev, move robber, knight, monopoly, road building, and bank trade. Still unproven: **Year of
   Plenty (`48 15`)** and the **discard** selection — a live YoP has not yet been played, and no 7
   has been rolled against a big enough hand.
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
   played**.

   ✅ **Sample-and-repair**, in `session.py`. A purchase nobody has revealed is drawn from what is
   left of the deck, weighted as the deck is, and the replay carries on. It is repaired rather
   than defended because the replay is a pure function of the action list: re-derive the
   purchases, rebuild from move one. Two mechanisms, and the order between them is the finding:

   - `reveal_purchases` run over the actions observed **so far**. The moment a card is played,
     the purchase it came from stops being a guess. Mid-game this reads the past, not the future.
   - failing that, a whole fresh determinization after a `DesyncError`, bounded at
     `MAX_REPAIRS = 8`.

   **The second mechanism never fires.** Not on either capture, and not even when the
   determinizer is rigged to guess victory point every single time — the worst draw available.
   Attribution always gets there first, because a card can only contradict a guess by being
   played, which is exactly the event that reveals it. That is a stronger guarantee than expected
   and it has a sharp edge: **a wrong guess is silent, not loud.** A sampled victory point hands
   the opponent a VP they do not have and nothing about it is ever illegal. Measured on the two
   captures, the live feed and the fully-revealed offline replay agree exactly on turn count,
   winner and public VP, and differ by at most one VP per never-revealed purchase.

   Which is why the repair loop is now **narrow**: it fires only when the action that failed is a
   *dev-card play* by a player with an outstanding guess — the one move a wrong guess can make
   illegal. The first live game showed what the wide version costs. A mistranslated port trade
   corrupted the hand, 177 later actions desynced, and each one burned all 8 redraws rebuilding
   the game from move one to re-guess dev cards that had nothing to do with it: **1416 rebuilds,
   zero useful repairs, and the real bug buried under them.** Everything else raises the first
   time, so a translation bug arrives as one clean error at the turn it happened.

   Still open: search resamples nothing. `MCTSPlayer` rolls forward from whatever the current
   determinization says, so it explores one sampled world rather than averaging over the
   deck. That is the cheap version and it is what ships; per-rollout resampling is the correct
   one.
5. **Rule reconciliation.** The agent trained under house rules that are *not* stock Catan:
   discard above 9 cards rather than 7, per-resource discard, no dev card the turn it was bought,
   and the 15 VP / Longest Road target. **Decision: assume the lobby matches those patches**, and
   let `GameReplay.check_legal` fail loudly on the first divergent event rather than quietly
   playing a different game than the policy was fitted to. ✅ And the silent half is now checked
   rather than eyeballed: `LiveGame._check_lobby` compares `victoryPointsToWin`,
   `cardDiscardLimit`, `maxPlayers` and `friendlyRobber` against our patches on the full-state
   message and raises `LobbyMismatch` before any action is replayed. A rule that differs without
   ever producing an illegal action would not trip `check_legal`; it would just cost points.

## Verification ladder

0. **Replay round-trip.** ✅ done offline in `tests/test_bridge.py`: a self-play game replayed
   through a reconstructed board produces identical observations, and both captures replay move
   for move.

   Building rung 1 found this rung had been quietly flaky since it was written — about one run
   in six. `reveal_purchases` leaves never-revealed purchases as `None`, and the engine fills
   those from a deck it shuffled independently; those draws can consume the very card a later
   *revealed* purchase needs, and it dies inside catanatron's `draw_from_listdeck` with a deck
   error that looks nothing like a translation bug. **`reveal_purchases` alone is not enough to
   replay a game** — `session.determinize_purchases` accounts for the whole deck, and both
   captures now replay soundly and reproducibly.
1. **Spectator dry-run:** read state only, log the move the agent *would* make each turn. No
   clicks. ✅ **offline**, `python -m src.bridge.session --replay <capture>`: over both captures
   the agent was asked 266 times and answered legally every time, with the placement specialist
   taking the opening and 50-sim search taking the rest. Agreement with the move a human actually
   played was 74.2% on `game2` and 65.2% on `game1`, by action type. ✅ **live**, 2026-08-18, one
   full 1v1 game on a throwaway account: the lobby check passed, we were BLUE, and after the fix
   below all 88 turns reconstruct with 141 legal decisions at 67.4% agreement, 0 repairs, median
   0.16s and worst case 0.39s per decision — comfortably inside a live turn timer.

   **The live game found what two captures could not: a 2:1 port.** `_bank_trade` split a run of
   identical given cards assuming one received card per run, which cannot express a 2:1 trade at
   all (six bricks for three cards is *one* run). Colonist puts the answer on the wire —
   `playerStates[pid].bankTradeRatiosState`, 4 everywhere at the start and lowered by ports — so
   the decoder now tracks it and divides each run by that player's actual rate. Without it the
   error was not even loud: the exception escaped mid-diff, the rest of that diff was lost, and
   the agent went on answering confidently off a hand that was wrong for the next 177 actions.
   Hence the second change: **a decode error now stops the session deciding** (`DryRun.broken`)
   while the capture keeps being written, so the game can be finished by hand and diagnosed
   afterwards. "Loud but not fatal" turned out to be neither.

   Two things that run measured out of it. Search is doing real work but not uniformly: between
   1 and 200 simulations, 9 of 128 decisions changed on `game2` and **0 of 138 on `game1`**. And
   agreement with a human is a weak yardstick in the direction that flatters the agent — most
   positions offer one sensible move, and the disagreements cluster exactly where they should
   (whether to trade, whether to buy a dev card or end the turn).
2. **Single supervised live game** on a throwaway account, human ready to intervene. 🚧 **Run
   nine times on 2026-08-19, all through 1v1 casual matchmaking with no human input after the
   game started.** Add `--auto-play` to the live command. The opening is the agent's too — the
   placement specialist places it, and the free-placement codes (`15`/`11`) are visibly different
   from the paid ones in the log.

   The first game: **61 decisions, 59 moves sent, 0 repairs, 90.6% agreement** with what the
   position offered. The routing header came from the server's `serverId` with no human click,
   which is what makes an autonomous game possible at all — the codec used to learn routing by
   watching the client, and a client nobody clicks never speaks.

   **Both games ended on a bug, and neither bug was in the sender.** That is the substantive
   finding of this rung:

   - **Our own dev cards went unattributed once we played one.** Colonist sends our whole hand in
     the diff that carries the buy, so the new card is whatever the hand gained — but the
     remembered hand was only updated *on a purchase*, so playing a card never removed it. Buy a
     knight, play it, buy another, and the new hand is a subset of the stale one: nothing looks
     new, and the purchase decodes as `None`. `None` means "the opponent's, unknown", so the
     determinizer drew from the deck and handed the agent a Year of Plenty it did not own. It
     tried to play it three turns running; each turn ended with nothing done. **No `DesyncError`
     fires**, because the server never reports the move we could not make — the one failure this
     module is built to make loud was silent, since the illegal move never happened. Fixed:
     `absorb_dev_cards` runs on every diff.
   - **Road Building's roads are placed, not bought.** The card itself worked (`48 14` → log 20),
     then the road went out as `12` and colonist *ignored* it — no refusal, no entry, just a turn
     that would not advance. Colonist splits its build codes by who pays rather than by when: an
     opening road and a Road Building road both log as entry 4, "placed for free", while every
     bought road logs as entry 5. Four Road Building plays across the captures agree. So the two
     free roads go out as `11`, and `translate()` takes `state.is_road_building` as an argument —
     catanatron cannot say so through the prompt, since it stays in `PLAY_TURN`.

   Both were invisible to three games of offline round-tripping, because a round-trip can only
   check the mappings a human's clicks happened to exercise.

   **Nine games in, the failures have changed category twice.** They are worth reading in that
   order, because each category needed a different kind of fix and only the first was about the
   translation table.

   *Mistranslations* — a mapping we got wrong. Road Building above; the 2:1 port before it.
   Fixed by learning the code.

   *Timing and identity* — the frames were right and arrived wrong.

   - **`sequence` is one counter per connection, not one per writer.** Both the page's client and
     ours were numbering their own frames, so every move we sent forked the count and the server
     resynced. Over a human-played game 162 consecutive client frames step by exactly 1 with no
     exceptions. `FrameCodec` now takes the client's value as authoritative *even when it goes
     backwards*, since the client is the one the server will keep agreeing with. This also
     retires "a session is not a game" as a separate problem: the second full state was never a
     second game, it was the server resyncing a forked counter.
     **Operationally: do not click while the agent is playing.**
   - **A monopoly or a year of plenty is acknowledged by state, not by log.** The card play, the
     selection and the confirm went out as one burst and the server dropped the tail, so the card
     was played and its choice never was. Gating the tail on log entry 20 then *deadlocked* —
     colonist does not log either card until the choice arrives, so it waits for something the
     choice itself causes. The gate is `currentState.actionState` ∈ {32, 33}, which comes back
     about 120ms after the bare card play. `split_after_card_play` holds the tail until then.
   - **A mid-game resync is not a new game.** Every agent-played game provokes at least one, one
     of them seven. Rebuilding on it fed an empty board a mid-game Year of Plenty. `_is_resync`
     tells them apart, and since a resync is the server stating the whole position outright,
     `LiveGame._audit` compares it against ours rather than discarding it. **All 8 resyncs across
     three captures pass** — the reconstruction matched the server's own account of the board
     exactly.

   *Narration* — colonist talking about things that are not the game. Four games died this way
   (karma vote 36/26/33, resignation 112, trade offer 118, opponent disconnect 24/130), so the
   fifth was fixed by a stance rather than a fifth constant: an unknown log entry raises only when
   the diff it rode in **moves something no entry we do decode accounts for**. Both conditions
   matter — colonist announces "must discard" (64) in the same diff as the discard (55), which
   moves cards but is fully explained. Unknowns that move nothing are counted and printed once.
   Trade offers are also answered: an offer awaiting us gets a decline, because ignoring one
   stalls the opponent's turn on our clock.

   *A guess that ended a game* — the newest category, and the quietest.

   - **The opponent's hidden hand is drawn, and a bad draw can win.** Late in a long game the
     opponent has bought several dev cards nobody ever revealed. Draw enough of them as victory
     points and their *reconstructed* score crosses the target: `replay.winning_color()` returns
     a winner, `our_turn()` returned `False`, and the agent stopped deciding for the rest of a
     game it was still in. No exception, no frame, no log line — the ninth game burned its last
     four turns in silence and had to be finished by hand. On that capture **1 seed in 4** hits
     it; two captures from 2026-08-17 carry it latently.

     The fix is conditioning, not tolerance. "The game is still running" is evidence: a hand that
     wins is one the server would already have paid out on, so the sample is *impossible* and is
     rejected and redrawn (`_phantom_win`, up to `MAX_PHANTOM_REDRAWS`). This matters for strength
     and not only for liveness — an agent that believes the game is already lost chooses
     meaninglessly, and near-misses degrade its play the same way without freezing it.
     Only a guessed hand can produce one: our own cards are on the wire, so a win of *ours* the
     server has not announced is a real bug and is left alone to be found as one.
   - **The watchdog that would have caught it in game one.** `DryRun._check_idle` compares the
     server's own `currentTurnPlayerColor` against ours and says so, aloud, when the server has
     been waiting on us for 30s and the agent has made no move. It does not retry — what to do
     depends entirely on why — but the failure now has a symptom. Both halves of a stall are
     covered: a move sent and lost, and a move never decided.
   - **A guess must not spend a card a later reveal needs.** Found by the same sweep:
     `determinize_purchases` drew guesses greedily against a deck it had not yet subtracted the
     *revealed* purchases from, so the last victory point could be guessed away at move 30 and
     then genuinely revealed at move 70 — dying inside catanatron's `draw_from_listdeck` with an
     error that says nothing about what went wrong. Three seeds in forty on `game2`, and
     pre-existing. Reveals now have first claim on the deck.

   **Still unproven live:** the discard selection (`8`/`7`), which needs a 7 rolled against a hand
   over 9 cards, and whether an auto-declined trade offer really reads as a decline.

   Playing a live game, which is what this rung is:

   ```bash
   python -m src.bridge.session --vps-to-win 15 --longest-road --max-turns 1500 \
       --auto-play --send-delay 1.5 --simulations 50 \
       --send-file data/bridge/send.txt --log data/bridge/decisions.jsonl \
       --model checkpoints/archive/ppo-15vp-lr-step400000.zip \
       --placement-model checkpoints/placement/scorer_ppo.pt \
       --bundle-model    checkpoints/placement/bundle_noroads.pt
   ```

   Pick 1v1 casual matchmaking and stop touching it when the board appears. Read the `==` lines,
   not the `-> sent` ones: `-> sent` only means the bytes left the page, while `==` means the
   *server* reported the move back. Two failures announce themselves that way and no other —
   frames that go out and never return, and a second `game started:` line. A broken
   reconstruction stops the agent sending at all, and an unanswered move is reported but never
   resent: resending a settlement plays it twice, somewhere the search never looked.

   The hand probe, for sending one frame without arming the agent:

   ```bash
   python -m src.bridge.session --allow-send --vps-to-win 15 --longest-road --max-turns 1500        --simulations 50 --model checkpoints/archive/ppo-15vp-lr-step400000.zip        --placement-model checkpoints/placement/scorer_ppo.pt        --bundle-model    checkpoints/placement/bundle_noroads.pt
   ```

   Join a 1v1, play normally, and on **your own turn** type `roll` or `end` instead of clicking —
   or, when the session was started detached and stdin is not a keyboard,
   `echo end >> data/bridge/send.txt`. Read the game log rather than the console: the console only
   reports that the bytes left the page, and a synthesized frame is byte-identical to the client's
   by design.
3. Only then consider unattended runs.

## What is blocked on captured traffic

The protocol is undocumented, so nothing colonist-specific can be written without recordings.
`capture.py` collects them; what is still needed is the *playing*:

- ✅ **2–4 complete 1v1 games** — two captured, both decoded, both replaying move for move.
- ✅ **The lobby settings** — they ride on the full-state message and are now checked
  automatically against `src/env/rules.py`.
- ✅ **Our own client frames** — the live dry run records both directions, so
  `data/bridge/dryrun-*.jsonl` contains 1237 `sent` frames. They were the entire specification of
  the action sender, and they have now been spent on it: the routing header is decoded, and
  `tests/test_bridge.py` re-encodes every in-game frame byte for byte against them.
- ⬜ **A DOM dump of a live board**, for the click layer — only if frames turn out not to work.

Frames the agent's own moves generate are as valuable as the ones it receives: the `sent`
direction is the entire specification of the action sender, and it can only be learned by
playing the moves by hand and reading what the page emitted.

## Deps

`playwright` is uncommented in `requirements.txt`; run `playwright install chromium` once.
