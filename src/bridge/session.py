"""Follow a live colonist.io game and say what the agent would play.

This is rung 1 of the verification ladder in ``docs/PHASE3_WEB.md``: read the
state, log the move the agent *would* make each turn, and click nothing. It is
the last step that carries no Terms-of-Service risk and the first that runs the
real agent against real traffic, so it is what the whole read side was built
for.

Two halves, deliberately separable:

- :class:`LiveGame` keeps a :class:`~src.bridge.replay.GameReplay` in step with
  the message stream. No browser, no agent, no I/O -- which is why it can be
  driven straight off a recorded capture (``--replay``), and is.
- :class:`DryRun` puts an agent and a browser around it.

**Sample-and-repair.** One thing in a 1v1 game is genuinely hidden: a dev card
between being bought and being played. The engine needs a concrete card to
advance, so :class:`LiveGame` draws one from what is left of the deck and
carries on -- and repairs when the guess turns out wrong. Repair is cheap and
exact because the replay is a pure function of the action list: re-derive the
purchases, rebuild the game from move one, done. Two things drive it:

1. :func:`~src.bridge.protocol.reveal_purchases`, run over the actions observed
   *so far*. Once the opponent plays a card, the purchase it came from is no
   longer a guess, and this attributes it. Mid-game that reads the past, not the
   future, so it is honest here in a way it would not be at the moment of the
   buy.
2. Failing that, a fresh determinization. A wrong guess can wreck the position
   in ways attribution cannot untangle -- a sampled victory point hands the
   opponent a VP they do not have, and the replay can decide the game is over --
   so a :class:`~src.bridge.replay.DesyncError` with guesses outstanding redraws
   them and tries again, up to :data:`MAX_REPAIRS` times.

A desync with *no* guesses outstanding is never repaired. Nothing was being
guessed, so the reconstruction is simply wrong, and that is the failure this
whole module exists to make loud.

Usage::

    # offline, against a capture -- no browser, no account, no risk
    python -m src.bridge.session --replay data/bridge/game2-<stamp>.jsonl \\
        --vps-to-win 15 --longest-road --max-turns 1500 \\
        --model checkpoints/archive/ppo-15vp-lr-step400000.zip \\
        --placement-model checkpoints/placement/scorer_ppo.pt \\
        --bundle-model    checkpoints/placement/bundle_noroads.pt

    # live, watching a game you play by hand
    python -m src.bridge.session --vps-to-win 15 --longest-road --max-turns 1500 \\
        --model ... --placement-model ... --bundle-model ...

Add ``--auto-play`` to the live form and the agent sends its own decisions
instead of only reporting them; everything else about the run is identical.

⚠️ Use a throwaway account. Without ``--auto-play`` this module sends nothing;
with it, the agent plays a real game against a real person.
"""

import src.env.ruleset as ruleset  # the engine reads the rules at import time

ruleset.apply_cli_overrides()

import argparse
import json
import random
import sys
import threading
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from queue import Queue
from typing import Dict, List, Optional

from catanatron import Color
from catanatron.models.actions import Action
from catanatron.models.decks import starting_devcard_bank
from catanatron.models.enums import ActionPrompt, ActionType

from src.bridge import moves, protocol, sender
from src.bridge.capture import CAPTURE_DIR, PROFILE_DIR, decode_payload
from src.bridge.player import DEFAULT_SIMULATIONS, build_bridge_player
from src.bridge.replay import DesyncError, GameReplay
from src.env.rules import DISCARD_LIMIT

#: How many times a determinization may be redrawn before a desync is believed.
#: Each attempt is a whole fresh sample of the opponent's unseen cards, so a few
#: are plenty; the point of the bound is that a genuine translation bug must not
#: spin here forever pretending to be bad luck.
MAX_REPAIRS = 8

#: The only moves a wrong guess about a hidden purchase can make illegal.
DEV_CARD_PLAYS = frozenset({
    ActionType.PLAY_KNIGHT_CARD, ActionType.PLAY_MONOPOLY,
    ActionType.PLAY_YEAR_OF_PLENTY, ActionType.PLAY_ROAD_BUILDING,
})


def draw_card(remaining: Counter, rng: random.Random) -> str:
    """One card from what is left of the deck, weighted as the deck is.

    Unweighted would be wrong in the direction that matters: the deck is 14
    knights against 5 victory points and two each of the rest, and the
    opponent's likely holding is exactly what the search is about to reason
    over.
    """
    pool = [card for card, count in remaining.items() for _ in range(count)]
    if not pool:
        raise DesyncError("the development deck is empty but a card was bought")
    return rng.choice(pool)


def determinize_purchases(actions: List[Action], rng: random.Random,
                          guesses: Optional[Dict[int, str]] = None,
                          draw=draw_card) -> List[Action]:
    """Give every dev-card purchase a concrete card, tracking the deck as it goes.

    Three sources, in descending order of how much they are worth: the card was
    ours and we saw it; the opponent later played it, so
    :func:`~src.bridge.protocol.reveal_purchases` can attribute it; or nobody
    ever revealed it and it is drawn from what remains of the deck.

    **``reveal_purchases`` alone is not enough to replay a game**, which is what
    this exists to fix. It leaves the never-revealed purchases as ``None``, and
    the engine then fills those from a deck it shuffled by itself -- draws that
    can consume the very card a later *revealed* purchase needs, blowing up
    inside ``draw_from_listdeck`` with a deck error that has nothing to do with
    the translation. Rare (about one game in six) and pure luck of the shuffle.
    Accounting for the whole deck here makes the replay sound *and*
    reproducible, given the ``rng``.

    Args:
        actions: observed actions, purchases possibly ``None``.
        rng: draws the unknowns. Seed it and the replay repeats exactly.
        guesses: index -> card, read *and written*. Passing the same dict back
            keeps a guess stable across calls, so a position does not churn
            under a search that is reading it; entries are dropped as the
            purchases behind them get revealed.
        draw: the draw itself, overridable so a test can force a bad one.
    """
    revealed = protocol.reveal_purchases(actions)
    remaining = Counter(starting_devcard_bank())
    guesses = {} if guesses is None else guesses
    filled: List[Action] = []
    for index, action in enumerate(revealed):
        if action.action_type != ActionType.BUY_DEVELOPMENT_CARD:
            filled.append(action)
            continue
        card = action.value
        if card is None:
            card = guesses.get(index)
            if card is None or remaining[card] <= 0:
                card = draw(remaining)
            guesses[index] = card
        else:
            guesses.pop(index, None)  # revealed; no longer a guess
        remaining[card] -= 1
        filled.append(Action(action.color, action.action_type, card))
    return filled


class LobbyMismatch(RuntimeError):
    """The lobby is not playing by the rules the agent was trained under.

    Raised before a single action is replayed. A rule that differs *and*
    produces an illegal move would trip ``DesyncError`` on its own, but the
    dangerous ones are silent: a different discard limit or victory target
    never makes anything illegal, it just quietly makes the policy wrong. So
    the settings colonist sends are checked rather than assumed.
    """


@dataclass
class Progress:
    """What one fed message did, for a caller that wants to narrate it."""

    observed: List[Action]      # actions decoded from this message
    repaired: int               # determinizations redrawn while applying them
    started: bool               # this message began a new game


class LiveGame:
    """A ``GameReplay`` held in step with colonist's message stream.

    Args:
        colors: catanatron colors handed out in colonist's play order.
        vps_to_win: victory target. Defaults to the ruleset the process started
            under, and is checked against the lobby's own setting.
        seed: fixes the determinization draw, so a dry run is reproducible.
        check_lobby: verify the lobby's settings against our patches. Off only
            for tests that construct a game by hand.

    Feed it with :meth:`feed` and read :attr:`replay`. Nothing here talks to a
    browser or an agent.
    """

    def __init__(self, colors=(Color.BLUE, Color.RED),
                 vps_to_win: Optional[int] = None, seed: Optional[int] = None,
                 check_lobby: bool = True):
        self.decoder = protocol.MessageDecoder(colors)
        self.vps_to_win = ruleset.VPS_TO_WIN if vps_to_win is None else vps_to_win
        self.check_lobby = check_lobby
        self.rng = random.Random(seed)
        self.replay: Optional[GameReplay] = None
        #: Determinizations redrawn wholesale after a desync. Should stay at 0:
        #: attribution normally gets there first, so a nonzero count means the
        #: guessing is doing real work and is worth reading the log over.
        self.repairs = 0
        #: Games replayed from move one because something already applied
        #: changed underneath. Routine -- a steal or a revealed purchase does it.
        self.rebuilds = 0
        #: Full states that re-described the game already in progress. Routine
        #: for an agent -- injected frames provoke them -- and each one is
        #: audited rather than acted on.
        self.resyncs = 0

        # Purchases we had to guess, keyed by their index in the action list.
        # Sticky: a guess is redrawn only when it is contradicted, so the
        # position does not churn under a search that is reading it.
        self._guesses: Dict[int, str] = {}
        # The action list with every purchase filled in -- what actually gets
        # applied, as opposed to what was observed.
        self._filled: List[Action] = []
        self._applied = 0
        self._game_id = 0

    # -- reading ------------------------------------------------------------
    @property
    def started(self) -> bool:
        return self.replay is not None

    @property
    def our_color(self) -> Optional[Color]:
        return self.decoder.our_color

    @property
    def game(self):
        return self.replay.game if self.replay is not None else None

    def our_turn(self) -> bool:
        """Whether the live game is waiting on a decision from us."""
        return (self.replay is not None
                and self.replay.winning_color() is None
                and self.replay.current_color == self.decoder.our_color)

    # -- writing ------------------------------------------------------------
    def feed(self, kind: int, payload) -> Progress:
        """Consume one server message and bring the replay up to date."""
        before = len(self.decoder.actions)
        revised = self.decoder.feed(kind, payload)
        if not self.decoder.started:
            return Progress(observed=[], repaired=0, started=False)

        started = self.decoder.game_id != self._game_id
        if started:
            self._restart()
            before = 0
        elif revised is not None and revised < self._applied:
            # A steal rewrote a robber move we had already applied.
            self._rebuild()

        repairs_before = self.repairs
        resyncs_before = self.resyncs
        self._sync()
        self.resyncs = self.decoder.resyncs
        if self.resyncs != resyncs_before and self.decoder.last_resync:
            # Free audit: see _audit. Runs after _sync so it compares against a
            # replay that has absorbed everything the resync's diffs carried.
            self._audit(self.decoder.last_resync)
        return Progress(observed=list(self.decoder.actions[before:]),
                        repaired=self.repairs - repairs_before, started=started)

    def feed_frame(self, record: dict) -> Optional[Progress]:
        """Consume one decoded WebSocket frame; ``None`` if it holds no game message."""
        message = protocol.server_message(record)
        if message is None:
            return None
        return self.feed(*message)

    # -- internals ----------------------------------------------------------
    def _restart(self) -> None:
        """A full state arrived: new board, new game, nothing carries over."""
        if self.check_lobby:
            self._check_lobby()
        self._guesses.clear()
        self._game_id = self.decoder.game_id
        self._rebuild()

    def _audit(self, payload: dict) -> None:
        """Check our reconstruction against the position the server just stated.

        A resync is the one moment mid-game when colonist describes the whole
        board outright instead of a diff, so it is the only chance to find out
        whether the replay has drifted -- and drift is the failure this module
        exists to catch. Ignoring the message is right; ignoring the *evidence*
        in it would be waste.

        Ownership only, of corners and edges. Building type is deliberately not
        compared: the enum for a city has never been observed, and an audit that
        can be wrong about what it is auditing is worse than a narrower one.
        """
        map_state = (payload.get("gameState") or {}).get("mapState") or {}
        coords = self.decoder.coords
        by_colonist = self.decoder.color_by_colonist

        theirs_nodes, theirs_edges = {}, {}
        for corner_id, corner in (map_state.get("tileCornerStates") or {}).items():
            owner = (corner or {}).get("owner")
            if owner is not None:
                theirs_nodes[coords.node_by_corner[int(corner_id)]] = by_colonist[owner]
        for edge_id, edge in (map_state.get("tileEdgeStates") or {}).items():
            owner = (edge or {}).get("owner")
            if owner is not None:
                theirs_edges[coords.edge_by_edge[int(edge_id)]] = by_colonist[owner]

        board = self.replay.state.board
        ours_nodes = {node: colour for node, (colour, _) in board.buildings.items()}
        ours_edges = {tuple(sorted(edge)): colour for edge, colour in board.roads.items()}

        if ours_nodes != theirs_nodes or ours_edges != theirs_edges:
            raise DesyncError(
                "the server re-sent the position and it is not the one we "
                f"reconstructed: {len(ours_nodes)} buildings and "
                f"{len(ours_edges)} roads here against "
                f"{len(theirs_nodes)} and {len(theirs_edges)} there"
            )

    def _rebuild(self) -> None:
        self.rebuilds += 1
        self.replay = GameReplay(self.decoder.board, colors=self.decoder.seating,
                                 vps_to_win=self.vps_to_win)
        self._applied = 0

    def _check_lobby(self) -> None:
        """Compare colonist's own settings against the rules we trained under.

        Only settings actually on the wire are checked, and a missing one is not
        an error -- colonist owes us no schema. Item 5 of ``docs/PHASE3_WEB.md``
        called for eyeballing this before the first live game; there is no
        reason for a human to do it.
        """
        settings = self.decoder.settings
        expected = {
            "victoryPointsToWin": self.vps_to_win,
            "cardDiscardLimit": DISCARD_LIMIT,
            "maxPlayers": 2,
            "friendlyRobber": True,
        }
        wrong = {key: (settings[key], want) for key, want in expected.items()
                 if key in settings and settings[key] != want}
        if not ruleset.LONGEST_ROAD_VP:
            wrong["longestRoad"] = ("awarded by colonist",
                                    "suppressed; pass --longest-road")
        if wrong:
            detail = "; ".join(f"{key}: lobby says {got!r}, we play {want!r}"
                               for key, (got, want) in sorted(wrong.items()))
            raise LobbyMismatch(
                f"the lobby is not our ruleset ({detail}). The agent was fitted "
                "under the rules in src/env/rules.py, and playing a different "
                "game costs points silently."
            )

    def _repairable(self, failed: Optional[Action]) -> bool:
        """Whether redrawing the hidden cards could plausibly explain this desync.

        Exactly one thing in the reconstruction is a guess: which card an
        unrevealed purchase was. That can make exactly one kind of move
        illegal -- playing a dev card the guessed hand does not hold. Anything
        else (a resource count, a placement, a rule) is a real bug, and
        redrawing the deck only buries it under rebuilds. The first live game
        did precisely that: 177 desyncs from one mistranslated 2:1 port trade
        cost 1416 pointless rebuilds and not a single useful repair.
        """
        if failed is None or failed.action_type not in DEV_CARD_PLAYS:
            return False
        return any(self._filled[index].color == failed.color
                   for index in self._guesses if index < len(self._filled))

    def _sync(self) -> None:
        """Fill in the hidden cards, then apply everything not yet applied."""
        filled = self._resolve()
        if filled[:self._applied] != self._filled[:self._applied]:
            # A purchase we had guessed just got attributed for real. Everything
            # downstream of it was played out of the wrong hand.
            self._rebuild()
        self._filled = filled

        for attempt in range(MAX_REPAIRS + 1):
            try:
                self._apply_pending()
                return
            except DesyncError:
                failed = (self._filled[self._applied]
                          if self._applied < len(self._filled) else None)
                if attempt == MAX_REPAIRS or not self._repairable(failed):
                    raise
                # Attribution has nothing left to offer; draw the opponent's
                # unseen cards again and replay from move one.
                self.repairs += 1
                self._guesses.clear()
                self._filled = self._resolve()
                self._rebuild()

    def _apply_pending(self) -> None:
        while self._applied < len(self._filled):
            self.replay.apply(self._filled[self._applied])
            self._applied += 1

    def _resolve(self) -> List[Action]:
        """The observed actions with every purchase determinized, guesses sticky."""
        return determinize_purchases(self.decoder.actions, self.rng,
                                     guesses=self._guesses, draw=self._draw)

    def _draw(self, remaining: Counter) -> str:
        """Overridable hook, so a test can force the worst possible draw."""
        return draw_card(remaining, self.rng)


class DryRun:
    """The agent, deciding on every one of our turns.

    By default it clicks nothing -- that is rung 1, and it is what the name
    means. Give it a :class:`~src.bridge.sender.PageSender` through
    :meth:`arm` and the same decisions go out on colonist's socket instead of
    only to the console; nothing else about the class changes, which is the
    point. The board it plays from is the same reconstruction a dry run scores,
    so a dry run that agrees with a human is the evidence that armed play is
    worth allowing at all.

    **The safety property is that a broken reconstruction cannot send.**
    :attr:`broken` already stops the agent being asked; armed, that is also what
    stops it moving. A stalled game is a bad outcome a human can fix in one
    click, and a confident move off a fictional board is not.

    Args:
        model_path, placement_path, bundle_path, simulations: the three
            artifacts the deployed agent is made of; see
            :func:`~src.bridge.player.build_bridge_player`.
        log: where to write one JSON line per decision, or ``None``.
        seed: fixes the determinization; see :class:`LiveGame`.
        check_lobby: forwarded to :class:`LiveGame`.
    """

    def __init__(self, model_path, placement_path, bundle_path,
                 simulations: int = DEFAULT_SIMULATIONS,
                 log: Optional[Path] = None, seed: Optional[int] = None,
                 check_lobby: bool = True):
        self.live = LiveGame(seed=seed, check_lobby=check_lobby)
        self.simulations = simulations
        self._artifacts = (model_path, placement_path, bundle_path)
        self.player = None
        self.log = log
        self.decisions = 0
        self.agreements = 0
        self.comparable = 0
        self._pending: Optional[Action] = None  # our move, awaiting the real one
        self._asked_at = -1
        #: The error that stopped the reconstruction, or ``None``. Once set, the
        #: agent is never asked again: the position it would be reading is known
        #: to be wrong, and a confident answer off a wrong board is the one
        #: output worse than no answer.
        self.broken: Optional[Exception] = None
        #: Set by :meth:`arm`. While ``None`` this is rung 1 and nothing is sent.
        self.sender = None
        #: Seconds to wait before putting a decision on the wire. The search
        #: answers in a fifth of a second and a human does not; a move that
        #: lands the instant the server finishes speaking is the single loudest
        #: thing about an agent playing.
        self.send_delay = 0.0
        self.sent_moves = 0
        self._sent_at: Optional[float] = None
        self._stalled = False
        #: Frames held back until the server acknowledges the card they resolve.
        #: See :func:`~src.bridge.moves.split_after_card_play`.
        self._deferred: List[tuple] = []
        #: Offers already answered, so a repeated diff does not answer twice.
        self._declined: set = set()

    def arm(self, page_sender, delay: float = 0.0) -> None:
        """Let the agent play its decisions rather than only report them."""
        self.sender = page_sender
        self.send_delay = delay

    def _build_player(self, color):
        model_path, placement_path, bundle_path = self._artifacts
        return build_bridge_player(color, model_path, placement_path, bundle_path,
                                   simulations=self.simulations)

    # -- driving ------------------------------------------------------------
    def feed_frame(self, record: dict) -> None:
        if self.broken is not None:
            return
        try:
            progress = self.live.feed_frame(record)
        except (protocol.ProtocolError, DesyncError, LobbyMismatch) as exc:
            self._break(exc)
            return
        if progress is None:
            return
        if progress.started:
            self.player = self._build_player(self.live.our_color)
            self._pending = None
            self._asked_at = -1
            print(f"game started: we are {self.live.our_color.value}, "
                  f"seats {[c.value for c in self.live.decoder.seating]}")
        if progress.repaired:
            print(f"  [repair] redrew the opponent's unseen cards "
                  f"{progress.repaired}x at turn {self.live.game.state.num_turns}")
        for action in progress.observed:
            self._score(action)
        self._flush_deferred()
        self._answer_offers()
        self._maybe_decide()

    def _score(self, observed: Action) -> None:
        """Compare a move we actually made against the one the agent chose.

        This is the substance of a dry run. The replay proves the translation is
        legal; only this says whether the agent, reading it, wants to play the
        game at all.
        """
        if self._pending is None or observed.color != self.live.our_color:
            return
        self.comparable += 1
        same_kind = observed.action_type == self._pending.action_type
        self.agreements += same_kind
        mark = "==" if observed == self._pending else ("~=" if same_kind else "!=")
        print(f"  agent {self._pending}  {mark}  played {observed}")
        self._record({"kind": "comparison", "agent": str(self._pending),
                      "played": str(observed), "match": mark})
        self._pending = None

    def _maybe_decide(self) -> None:
        """Ask the agent, once per position it is actually facing."""
        if not self.live.our_turn():
            return
        state = self.live.game.state
        if len(state.actions) == self._asked_at:
            return  # same position; the agent has already answered it
        self._asked_at = len(state.actions)

        if state.current_prompt == ActionPrompt.DISCARD:
            self._decide_discard()
            return

        legal = self.live.replay.playable_actions
        started = time.time()
        action = self.player.decide(self.live.game, legal)
        elapsed = time.time() - started
        self.decisions += 1
        self._pending = action
        verb = "plays" if self.sender is not None else "would play"
        print(f"turn {state.num_turns} [{state.current_prompt}] "
              f"agent {verb} {action}  ({elapsed:.2f}s, {len(legal)} legal)")
        self._record({"kind": "decision", "turn": state.num_turns,
                      "prompt": str(state.current_prompt), "action": str(action),
                      "legal": len(legal), "seconds": round(elapsed, 3)})
        self._play(action, lambda: moves.translate(
            action, self.live.decoder.coords,
            self.live.decoder.our_colonist_color, str(state.current_prompt),
            free_road=getattr(state, "is_road_building", False)))

    def _decide_discard(self) -> None:
        """Choose a whole discard, then send it as the one frame colonist wants.

        The mismatch is structural rather than incidental. ``src/env/rules.py``
        made discarding one action per card, because that is the decision the
        policy has a slot for; colonist's client picks cards in the UI and posts
        the finished hand. So the agent has to be asked several times before
        anything can be sent, and the live replay cannot advance in between --
        it only moves on what the *server* says happened.

        The way out is a private copy of the game. Each pick is applied to the
        copy so the next question is asked from the right position, and the copy
        is thrown away; the real replay still learns the discard the ordinary
        way, from log entry 55. Nothing here is speculative about the opponent,
        because a discard is a decision about our own hand alone.
        """
        game = self.live.game.copy()
        state = self.live.game.state
        chosen: List[str] = []
        started = time.time()
        while (game.state.current_prompt == ActionPrompt.DISCARD
               and game.state.current_color() == self.live.our_color):
            action = self.player.decide(game, game.state.playable_actions)
            if action.action_type != ActionType.DISCARD:
                break
            chosen.append(action.value)
            game.execute(action)
        elapsed = time.time() - started
        self.decisions += 1
        # Not scored against the human's move: _score compares one action, and
        # a discard is several. The frame is what matters here.
        self._pending = None
        print(f"turn {state.num_turns} [DISCARD] agent discards "
              f"{', '.join(chosen)}  ({elapsed:.2f}s)")
        self._record({"kind": "decision", "turn": state.num_turns,
                      "prompt": "DISCARD", "action": f"DISCARD {chosen}",
                      "legal": len(self.live.replay.playable_actions),
                      "seconds": round(elapsed, 3)})
        self._play(f"DISCARD {chosen}", lambda: moves.discard_frames(chosen))

    def _play(self, action, build) -> None:
        """Put a decision on colonist's socket, if this run is armed for it.

        ``build`` is a thunk so the translation only runs when it is going to be
        used -- a dry run should not be able to fail on a frame it was never
        going to send.
        """
        if self.sender is None:
            return
        try:
            frames = build()
        except moves.TranslationError as exc:
            self._break(exc)
            return
        if self.send_delay:
            time.sleep(self.send_delay)
        lead, self._deferred = moves.split_after_card_play(frames)
        if not self._send_frames(lead):
            return
        self.sent_moves += 1
        self._sent_at = time.time()
        self._stalled = False
        self._record({"kind": "sent", "action": str(action),
                      "frames": [code for code, _ in lead],
                      "deferred": [code for code, _ in self._deferred]})

    def _send_frames(self, frames) -> bool:
        """Put frames on the wire in order. False if one could not go."""
        for code, payload in frames:
            try:
                result = self.sender.send(code, payload)
            except sender.SendError as exc:
                print(f"  !! could not send action {code}: {exc}", file=sys.stderr)
                self._record({"kind": "send-failed", "action": code,
                              "detail": str(exc)})
                return False
            print(f"  -> sent action {code} {json.dumps(payload, default=str)} "
                  f"(seq {result.get('sequence')})")
        return True

    def _answer_offers(self) -> None:
        """Decline any trade the opponent has put to us.

        The offer itself moves nothing, so the reconstruction ignores it -- but
        the *game* does not: an unanswered offer holds the turn until it times
        out, and a player who never answers one is conspicuous. Declining is
        also the only honest answer, since the engine has no player-trade action
        and the agent therefore cannot weigh the offer at all.

        A real evaluation would mean modelling player trades end to end, in the
        engine and in the action space. That is a Phase 2 question, not a bridge
        one.
        """
        if self.sender is None:
            return
        for offer_id in self.live.decoder.offers_awaiting_us:
            if offer_id in self._declined:
                continue
            self._declined.add(offer_id)
            print(f"  declining trade offer {offer_id}")
            self._send_frames([moves.decline_offer(offer_id)])

    def _flush_deferred(self) -> None:
        """Send a card's resolution, once the server says the card is down.

        Colonist enters the state that asks "which resource?" only after it has
        processed the card, and a choice that arrives first is discarded rather
        than queued -- a live monopoly was lost exactly that way, the whole
        ``48``/``8``/``7`` burst leaving before the server had acknowledged
        anything. A human never trips it because clicking is slower than the
        round trip.

        **The acknowledgement is a state change, not a log entry**, and getting
        that wrong cost a second game. Colonist does not log a monopoly or a
        year of plenty until the choice arrives, so gating the choice on the log
        waits for something the choice itself causes. ``actionState`` comes back
        about 120ms after the bare card play; the log never comes at all.
        """
        if not self._deferred or not self.live.decoder.awaiting_card_selection:
            return
        deferred, self._deferred = self._deferred, []
        self._send_frames(deferred)

    def check_stall(self, seconds: float = 30.0) -> None:
        """Say so, once, when a sent move has gone unanswered.

        Deliberately not a retry. A frame the server ignored and a frame whose
        answer is merely slow look identical from here, and resending the second
        one plays the move twice -- which in Catan can mean a settlement in a
        place the search never evaluated. A human with the window open fixes
        either case in one click, so the useful thing is to be told.
        """
        if self._sent_at is None or self._stalled or self._pending is None:
            return
        waited = time.time() - self._sent_at
        if waited < seconds:
            return
        self._stalled = True
        print(f"\n!! the move we sent {waited:.0f}s ago has not come back from "
              f"the server.\n   Nothing will be resent -- play it by hand in the "
              f"browser and the agent picks up again.\n", file=sys.stderr)

    def _break(self, exc: Exception) -> None:
        """Stop deciding; the caller keeps recording.

        A decode error means the board we are holding no longer matches the one
        on screen, and every later answer is off a fiction. Live, the browser
        stays open and the capture keeps being written, so the game can be
        finished by hand and the frames kept for diagnosis -- but the agent
        says nothing more.
        """
        self.broken = exc
        turn = self.live.game.state.num_turns if self.live.started else -1
        print(f"\n!! {type(exc).__name__} at turn {turn}: {exc}\n"
              "   the reconstruction is broken; no further decisions will be "
              "made. Recording continues.\n", file=sys.stderr)
        self._record({"kind": "broken", "turn": turn,
                      "error": type(exc).__name__, "detail": str(exc)})

    def _record(self, entry: dict) -> None:
        if self.log is None:
            return
        with self.log.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, default=str) + "\n")

    def summary(self) -> str:
        rate = (f"{100 * self.agreements / self.comparable:.1f}%"
                if self.comparable else "n/a")
        broken = ("" if self.broken is None
                  else f", BROKEN by {type(self.broken).__name__}: {self.broken}")
        sent = f", {self.sent_moves} moves sent" if self.sender is not None else ""
        return (f"{self.decisions} decisions, {self.comparable} comparable to a move "
                f"actually played, {rate} agreed on the action type, "
                f"{self.live.repairs} repairs{sent}{broken}")


# --------------------------------------------------------------------------
# Drivers
# --------------------------------------------------------------------------


def run_replay(dry: DryRun, path: Path) -> None:
    """Drive a dry run off a recorded capture. No browser, no account, no risk.

    Worth more than it sounds: the capture contains our own moves, so this is
    the only setting where the agent's choice can be scored against a human's on
    the same position -- live, we would have to make the move to find out.
    """
    for record in protocol.iter_colonist_frames(path):
        dry.feed_frame(record)


SEND_HELP = """
send console -- rung 2's first question, asked by hand.

  roll            send action 2 (roll dice)
  end             send action 6 (end turn)
  send <n> <json> send action <n> with a JSON payload, e.g. `send 12 53`
  status          what the codec has learned from the client so far
  help            this
  (blank line)    nothing

Nothing is sent until you type it. Send only on your own turn, and read the
game log rather than the console to find out whether the server took it.
"""


def _console_reader(queue: "Queue[str]") -> None:
    """Read commands off stdin forever, on a thread of their own.

    Playwright's sync API is not thread-safe, so this thread only *collects*;
    the browser loop drains the queue and does the sending itself.
    """
    try:
        for line in sys.stdin:
            queue.put(line.strip())
    except (OSError, ValueError):
        pass  # no terminal attached; the file channel is the way in


def _file_reader(path: Path, queue: "Queue[str]") -> None:
    """Watch a file for commands, so the console survives having no terminal.

    The live session is often started detached -- from a tool, or in the
    background -- and then stdin is not the keyboard and typing `roll` is not
    possible. Appending that line to a file is the same instruction given a
    different way: still a human asking for one specific frame, which is the
    property that matters here, not which fd it arrived on.
    """
    seen = 0
    while True:
        try:
            if path.exists():
                lines = path.read_text(encoding="utf-8").splitlines()
                for line in lines[seen:]:
                    queue.put(line.strip())
                seen = len(lines)
        except OSError:
            pass  # being written to as we read; try again next tick
        time.sleep(0.5)


def _run_command(command: str, page_sender: "sender.PageSender") -> None:
    """Execute one console line. Never raises: a typo must not end the game."""
    parts = command.split(maxsplit=2)
    verb = parts[0].lower()
    try:
        if verb == "help":
            print(SEND_HELP)
        elif verb == "status":
            codec = page_sender.codec
            print(f"  header={codec.header} last_sequence={codec.last_sequence} "
                  f"sent={len(codec.sent)}")
        elif verb in sender.PROBE_ACTIONS:
            print(f"  -> {page_sender.send_probe(verb)}")
        elif verb == "send" and len(parts) >= 2:
            payload = parts[2] if len(parts) > 2 else "true"
            print(f"  -> {page_sender.send_raw(int(parts[1]), payload)}")
        else:
            print(f"  ? {command!r} -- type `help`")
    except (sender.SendError, ValueError) as exc:
        print(f"  !! {exc}")


def run_live(dry: DryRun, url: str, channel: str, record_to: Optional[Path],
             allow_send: bool = False, send_file: Optional[Path] = None,
             auto_play: bool = False, send_delay: float = 0.0) -> None:
    """Attach to a browser you drive by hand and follow the game in real time.

    The same CDP plumbing :mod:`src.bridge.capture` uses, writing a capture in
    the same format -- a dry run is a capture that happens to have an agent
    attached, and you only get to play each game once.

    **Both directions are recorded, and ``sent`` is now read too.** The client's
    own frames say nothing the server does not repeat back, so the agent has no
    use for them -- but they are the entire specification of the action sender,
    and with ``allow_send`` they are also its calibration:
    :class:`~src.bridge.sender.FrameCodec` learns this room's routing header and
    sequence counter by watching the client use them.

    ``allow_send`` adds a console and nothing else: no move goes out that you
    did not type. That was rung 2's first question -- whether the server accepts
    a synthesized frame at all -- and it is answered. Commands arrive on stdin,
    or, when the session was started detached and stdin is not a keyboard, by
    appending a line to ``send_file``.

    ``auto_play`` is the rung after: the agent sends its own decisions and the
    game plays itself. The console stays available on top of it, which is the
    intended way to intervene -- there is no pause button, but there is a
    browser window you can click in, and a reconstruction that breaks stops the
    agent dead rather than letting it guess.
    """
    from playwright.sync_api import sync_playwright

    handle = record_to.open("w", encoding="utf-8") if record_to else None
    sockets: Dict[str, str] = {}
    codec = sender.FrameCodec()
    commands: "Queue[str]" = Queue()

    started = time.time()

    def write(entry: dict) -> None:
        if handle is None:
            return
        # Stamped like capture.py's, for a reason a live game supplied: the
        # monopoly race could be *inferred* from frame order but never measured,
        # because this writer was the one recording without a clock.
        entry["t"] = round(time.time() - started, 3)
        handle.write(json.dumps(entry, default=str) + "\n")
        handle.flush()  # so the capture survives a crash and can be read mid-game

    def attach(page) -> None:
        cdp = page.context.new_cdp_session(page)
        cdp.send("Network.enable")

        def on_created(event):
            # Recorded because iter_colonist_frames identifies the game socket by
            # url; without these lines the capture cannot be replayed at all.
            sockets[event.get("requestId")] = event.get("url", "")
            write({"kind": "socket", "id": event.get("requestId"),
                   "url": event.get("url")})
            print(f"  websocket opened: {event.get('url')}")

        def on_frame(direction):
            def handler(event):
                response = event.get("response", {})
                opcode = response.get("opcode", 1)
                entry = {"kind": "frame", "dir": direction,
                         "id": event.get("requestId"), "opcode": opcode}
                entry.update(decode_payload(response.get("payloadData", ""), opcode))
                write(entry)
                # A browser opens a dozen other sockets -- a Discord login
                # gateway, a pile of localhost RPC. Only colonist's has a game on
                # it, and only the server's half of that describes one.
                if direction != "recv":
                    # ...but our own half says how to speak, so the codec reads
                    # the routing header and sequence counter off the client.
                    codec.observe(entry)
                    return
                if not codec.ready:
                    # A game the agent plays start to finish has no human
                    # clicking, so the client may never send a frame to copy.
                    # The server names the room itself when it starts the game.
                    room = sender.room_from_server_frame(entry.get("payload"))
                    if room:
                        codec.bootstrap(room)
                        print(f"  routing: game room {room!r}")
                if "colonist" not in sockets.get(event.get("requestId"), ""):
                    return
                try:
                    dry.feed_frame(entry)
                except (protocol.ProtocolError, DesyncError, LobbyMismatch) as exc:
                    # Loud but not fatal: the browser stays open so the game can
                    # be finished by hand and the capture kept for diagnosis.
                    print(f"\n!! {type(exc).__name__}: {exc}\n", file=sys.stderr)
            return handler

        cdp.on("Network.webSocketCreated", on_created)
        cdp.on("Network.webSocketFrameReceived", on_frame("recv"))
        cdp.on("Network.webSocketFrameSent", on_frame("sent"))

    with sync_playwright() as driver:
        context = driver.chromium.launch_persistent_context(
            user_data_dir=str(PROFILE_DIR), channel=channel or None,
            headless=False, viewport=None, args=["--start-maximized"],
        )
        if allow_send:
            # Before any navigation, or colonist's socket is already open and
            # the hook has nothing left to wrap.
            sender.install(context)
        context.on("page", attach)
        page = context.pages[0] if context.pages else context.new_page()
        attach(page)
        page.goto(url)
        page_sender = sender.PageSender(page, codec) if allow_send else None
        if page_sender is not None and auto_play:
            dry.arm(page_sender, send_delay)
            print(f"AUTO-PLAY ARMED: the agent will send its own moves "
                  f"({send_delay:.1f}s before each). Close the window to stop.")
        if page_sender is not None:
            threading.Thread(target=_console_reader, args=(commands,),
                             daemon=True).start()
            if send_file is not None:
                send_file.parent.mkdir(parents=True, exist_ok=True)
                send_file.write_text("", encoding="utf-8")
                threading.Thread(target=_file_reader, args=(send_file, commands),
                                 daemon=True).start()
                print(f"send channel: append a command to {send_file}")
            print(SEND_HELP)
        print("watching. play a game by hand; close the window when done.")
        try:
            while context.pages and not context.pages[0].is_closed():
                context.pages[0].wait_for_timeout(500)
                # Drained here rather than on the reader thread: Playwright's
                # sync API belongs to the thread that created the browser.
                dry.check_stall()
                while page_sender is not None and not commands.empty():
                    command = commands.get()
                    if command:
                        _run_command(command, page_sender)
        except KeyboardInterrupt:
            print("\ninterrupted")
        except Exception as exc:  # the window was closed mid-wait
            if "closed" not in str(exc).lower():
                raise
        finally:
            if handle is not None:
                handle.close()
            try:
                context.close()
            except Exception:
                pass


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--replay", type=Path,
                        help="drive off a recorded capture instead of a browser")
    parser.add_argument("--model", required=True, help="PPO checkpoint")
    parser.add_argument("--placement-model", required=True)
    parser.add_argument("--bundle-model", required=True)
    parser.add_argument("--simulations", type=int, default=DEFAULT_SIMULATIONS)
    parser.add_argument("--seed", type=int, default=0,
                        help="fixes the determinization draw")
    parser.add_argument("--log", type=Path, help="write one JSON line per decision")
    parser.add_argument("--url", default="https://colonist.io")
    parser.add_argument("--channel", default="chrome")
    parser.add_argument("--no-record", action="store_true",
                        help="do not also write a capture of the live session")
    parser.add_argument("--allow-send", action="store_true",
                        help="open a console that can put frames on colonist's "
                             "socket. Nothing is sent that you do not ask for.")
    parser.add_argument("--send-file", type=Path,
                        help="also read send commands from this file, one per "
                             "line, for when stdin is not a keyboard")
    parser.add_argument("--auto-play", action="store_true",
                        help="the agent sends its own moves. Implies "
                             "--allow-send. Throwaway account, supervised.")
    parser.add_argument("--send-delay", type=float, default=1.5,
                        help="seconds to pause before each move (default 1.5)")
    # Declared so --help lists them; they were already read at import time by
    # ruleset.apply_cli_overrides(), which has to run above the engine imports.
    parser.add_argument("--vps-to-win", type=int)
    parser.add_argument("--longest-road", action="store_true")
    parser.add_argument("--max-turns", type=int)
    args = parser.parse_args()

    print(f"ruleset: {ruleset.describe()}")
    dry = DryRun(args.model, args.placement_model, args.bundle_model,
                 simulations=args.simulations, log=args.log, seed=args.seed)

    if args.replay:
        run_replay(dry, args.replay)
    else:
        record_to = None
        if not args.no_record:
            CAPTURE_DIR.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
            record_to = CAPTURE_DIR / f"dryrun-{stamp}.jsonl"
            print(f"recording to {record_to}")
        run_live(dry, args.url, args.channel, record_to,
                 args.allow_send or args.auto_play, args.send_file,
                 args.auto_play, args.send_delay)

    print(dry.summary())
    return 1 if dry.broken is not None else 0


if __name__ == "__main__":
    sys.exit(main())
