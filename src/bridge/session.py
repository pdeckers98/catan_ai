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

⚠️ Use a throwaway account. This module never sends a move, but the session it
attaches to is the one that later will.
"""

import src.env.ruleset as ruleset  # the engine reads the rules at import time

ruleset.apply_cli_overrides()

import argparse
import json
import random
import sys
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from catanatron import Color
from catanatron.models.actions import Action
from catanatron.models.decks import starting_devcard_bank
from catanatron.models.enums import ActionType

from src.bridge import protocol
from src.bridge.capture import CAPTURE_DIR, PROFILE_DIR, decode_payload
from src.bridge.player import DEFAULT_SIMULATIONS, build_bridge_player
from src.bridge.replay import DesyncError, GameReplay
from src.env.rules import DISCARD_LIMIT

#: How many times a determinization may be redrawn before a desync is believed.
#: Each attempt is a whole fresh sample of the opponent's unseen cards, so a few
#: are plenty; the point of the bound is that a genuine translation bug must not
#: spin here forever pretending to be bad luck.
MAX_REPAIRS = 8


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
        self._sync()
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
                if not self._guesses or attempt == MAX_REPAIRS:
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
    """The agent, watching. It decides on every one of our turns and clicks nothing.

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

    def _build_player(self, color):
        model_path, placement_path, bundle_path = self._artifacts
        return build_bridge_player(color, model_path, placement_path, bundle_path,
                                   simulations=self.simulations)

    # -- driving ------------------------------------------------------------
    def feed_frame(self, record: dict) -> None:
        progress = self.live.feed_frame(record)
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

        legal = self.live.replay.playable_actions
        started = time.time()
        action = self.player.decide(self.live.game, legal)
        elapsed = time.time() - started
        self.decisions += 1
        self._pending = action
        print(f"turn {state.num_turns} [{state.current_prompt}] "
              f"agent would play {action}  ({elapsed:.2f}s, {len(legal)} legal)")
        self._record({"kind": "decision", "turn": state.num_turns,
                      "prompt": str(state.current_prompt), "action": str(action),
                      "legal": len(legal), "seconds": round(elapsed, 3)})

    def _record(self, entry: dict) -> None:
        if self.log is None:
            return
        with self.log.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, default=str) + "\n")

    def summary(self) -> str:
        rate = (f"{100 * self.agreements / self.comparable:.1f}%"
                if self.comparable else "n/a")
        return (f"{self.decisions} decisions, {self.comparable} comparable to a move "
                f"actually played, {rate} agreed on the action type, "
                f"{self.live.repairs} repairs")


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


def run_live(dry: DryRun, url: str, channel: str, record_to: Optional[Path]) -> None:
    """Attach to a browser you drive by hand and follow the game in real time.

    The same CDP plumbing :mod:`src.bridge.capture` uses, writing a capture in
    the same format -- a dry run is a capture that happens to have an agent
    attached, and you only get to play each game once.

    **Both directions are recorded, only ``recv`` is decoded.** The client's own
    frames say nothing the server does not repeat back, so the agent has no use
    for them; but they are the entire specification of the action sender, which
    is the next piece of work. Dropping them would mean playing another game by
    hand to get them back.
    """
    from playwright.sync_api import sync_playwright

    handle = record_to.open("w", encoding="utf-8") if record_to else None
    sockets: Dict[str, str] = {}

    def write(entry: dict) -> None:
        if handle is None:
            return
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
                    return
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
        context.on("page", attach)
        page = context.pages[0] if context.pages else context.new_page()
        attach(page)
        page.goto(url)
        print("watching. play a game by hand; close the window when done.")
        try:
            while context.pages and not context.pages[0].is_closed():
                context.pages[0].wait_for_timeout(500)
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
        run_live(dry, args.url, args.channel, record_to)

    print(dry.summary())
    return 0


if __name__ == "__main__":
    sys.exit(main())
