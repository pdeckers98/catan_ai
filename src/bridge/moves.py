"""Turn a catanatron ``Action`` into the frames colonist's client would send.

The other half of :mod:`src.bridge.protocol`, and deliberately its mirror image:
that module reads colonist's messages into actions, this one writes actions back
out as colonist's messages. Keeping them apart means the read side stays usable
on its own -- a dry run imports nothing from here.

**Every mapping below was correlated, not guessed.** For each client frame in
three captured games, the server log entry it produced was read off the next
diff; the action code is whatever frame preceded ``built`` (5), ``rolled`` (10),
``moved the robber`` (11), and so on. That is why the codes here disagree with a
plausible reading of the numbers: ``7`` is not "discard", it is *confirm the
current card selection*, and it is preceded by ``8`` frames carrying the
selection as it grows. Both are sent, because both are what the client sent.

**Three catanatron actions are more than one frame**, and the reason is always
that colonist splits a decision the engine keeps whole:

- ``PLAY_MONOPOLY`` and ``PLAY_YEAR_OF_PLENTY`` name their resource in the
  action; colonist plays the card first (``48``) and resolves it after
  (``8``/``7``).
- ``MARITIME_TRADE`` is preceded by a bare ``47``. Its meaning is not settled --
  it appears before a dev-card buy too -- but the client sent one before every
  single trade in the captures, so we do too. It is cheap, and a trade the
  server silently drops would be an expensive thing to debug.

**Discard runs the other way: many actions, one frame.** ``src/env/rules.py``
replaces the stock all-at-once discard with one action per card, which is the
discard the agent learned; colonist wants the finished hand. So the caller
collects the agent's picks and hands them to :func:`discard_frames` together.
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple

from catanatron.models.enums import ActionType

from src.bridge.protocol import (
    DEV_KNIGHT, DEV_MONOPOLY, DEV_ROAD_BUILDING, DEV_YEAR_OF_PLENTY,
    ProtocolError, RESOURCE_BY_CARD,
)

#: Client action codes, each one read off the log entry it produced.
SEND_ROLL = 2                 # payload: true          -> log 10, rolled
SEND_MOVE_ROBBER = 3          # payload: hex id        -> log 11, robber moved
SEND_END_TURN = 6             # payload: true          -> log 44, turn started
SEND_CONFIRM_CARDS = 7        # payload: [card, ...]   -> log 55/86/21
SEND_SELECT_CARDS = 8         # payload: [card, ...]   the selection so far
SEND_BUY_DEV_CARD = 9         # payload: true          -> log 1, bought
SEND_INITIAL_ROAD = 11        # payload: edge id       -> log 4, free placement
SEND_ROAD = 12                # payload: edge id       -> log 5, built
SEND_INITIAL_SETTLEMENT = 15  # payload: corner id     -> log 4, free placement
SEND_SETTLEMENT = 16          # payload: corner id     -> log 5, built
SEND_CITY = 19                # payload: corner id     -> log 5, built
SEND_OPEN_PANEL = 47          # payload: true          no log of its own
SEND_PLAY_DEV_CARD = 48       # payload: card enum     -> log 20, played
SEND_TRADE = 49               # payload: trade object  -> log 116, bank trade

#: Frames that are safe to send unprompted: they either succeed at a moment we
#: chose or are refused, and neither outcome costs a position.
PROBE_ACTIONS = {"roll": (SEND_ROLL, True), "end": (SEND_END_TURN, True)}

#: catanatron resource name -> colonist card enum. The inverse of the map the
#: read side uses, built from it so the two can never drift apart.
CARD_BY_RESOURCE: Dict[str, int] = {
    resource: card for card, resource in RESOURCE_BY_CARD.items()
}

_DEV_CARD_BY_ACTION = {
    ActionType.PLAY_KNIGHT_CARD: DEV_KNIGHT,
    ActionType.PLAY_MONOPOLY: DEV_MONOPOLY,
    ActionType.PLAY_ROAD_BUILDING: DEV_ROAD_BUILDING,
    ActionType.PLAY_YEAR_OF_PLENTY: DEV_YEAR_OF_PLENTY,
}

#: The prompts colonist places for free. Both use a different action code than
#: the paid version of the same build, and nothing else distinguishes them.
INITIAL_PROMPTS = ("BUILD_INITIAL_SETTLEMENT", "BUILD_INITIAL_ROAD")

#: One frame: the action code and its msgpack payload.
Frame = Tuple[int, Any]


class TranslationError(RuntimeError):
    """The agent chose a move we cannot express as colonist frames.

    Loud on purpose. Every legal action has a mapping, so this means either the
    ruleset drifted or the agent found a corner the captures never showed -- and
    guessing a frame code live is how a game gets thrown away.
    """


def _cards(resources: Sequence[str]) -> List[int]:
    out = []
    for resource in resources:
        if resource not in CARD_BY_RESOURCE:
            raise TranslationError(f"{resource!r} is not a tradeable resource")
        out.append(CARD_BY_RESOURCE[resource])
    return out


def _selection(cards: Sequence[int]) -> List[Frame]:
    """The client's card picker: one ``8`` per click, then ``7`` to confirm.

    The growing prefixes are how the captures read -- ``[2]``, ``[2,2]``,
    ``[2,2,2]``, then ``7`` with the finished list -- and reproducing them costs
    nothing. Whether the server needs anything but the ``7`` is unknown, and
    finding out by omission would cost a real game.
    """
    frames: List[Frame] = [(SEND_SELECT_CARDS, list(cards[:i + 1]))
                           for i in range(len(cards))]
    frames.append((SEND_CONFIRM_CARDS, list(cards)))
    return frames


def discard_frames(resources: Sequence[str]) -> List[Frame]:
    """The whole discard, from the per-card actions the agent produced."""
    if not resources:
        raise TranslationError("a discard of no cards")
    return _selection(_cards(resources))


def translate(action, coords, colonist_color: int,
              prompt: Optional[str] = None,
              free_road: bool = False) -> List[Frame]:
    """The frames colonist's client would have sent for ``action``.

    Args:
        action: a catanatron ``Action``, fully specified.
        coords: this game's :class:`~src.bridge.protocol.CoordinateMap`.
        colonist_color: our seat in colonist's own numbering -- the ``creator``
            field of a trade, and the only place the read side's colour mapping
            has to be inverted.
        prompt: ``str(state.current_prompt)`` if known. Only the opening needs
            it: a settlement placed for free is action ``15`` and a bought one
            is ``16``, and the frame is the only difference.
        free_road: whether this road is one of Road Building's two, i.e.
            ``state.is_road_building``. The prompt cannot say so -- catanatron
            stays in ``PLAY_TURN`` and tracks the card in a counter -- and the
            code differs, so it has to be passed.

    Raises:
        TranslationError: for anything with no mapping.
    """
    kind = action.action_type
    value = action.value
    initial = bool(prompt) and any(name in prompt for name in INITIAL_PROMPTS)

    if kind == ActionType.ROLL:
        return [(SEND_ROLL, True)]
    if kind == ActionType.END_TURN:
        return [(SEND_END_TURN, True)]
    if kind == ActionType.BUY_DEVELOPMENT_CARD:
        return [(SEND_BUY_DEV_CARD, True)]

    if kind == ActionType.BUILD_SETTLEMENT:
        code = SEND_INITIAL_SETTLEMENT if initial else SEND_SETTLEMENT
        return [(code, _corner(coords, value))]
    if kind == ActionType.BUILD_CITY:
        return [(SEND_CITY, _corner(coords, value))]
    if kind == ActionType.BUILD_ROAD:
        # Road Building's two roads are free, and colonist splits its codes by
        # who pays rather than by when: they log as entry 4, "placed for free",
        # exactly like an opening road, and never as entry 5, "built". Sending
        # 12 for one is silently ignored -- it cost a live game, which is the
        # only reason this distinction is here at all.
        code = SEND_INITIAL_ROAD if (initial or free_road) else SEND_ROAD
        return [(code, _edge(coords, value))]

    if kind == ActionType.MOVE_ROBBER:
        # (coordinate, victim, stolen card). Only the tile is ours to send: in
        # 1v1 there is one candidate victim, so colonist never asks.
        return [(SEND_MOVE_ROBBER, _hex(coords, value[0]))]

    if kind in (ActionType.PLAY_KNIGHT_CARD, ActionType.PLAY_ROAD_BUILDING):
        return [(SEND_PLAY_DEV_CARD, _DEV_CARD_BY_ACTION[kind])]
    if kind == ActionType.PLAY_MONOPOLY:
        # catanatron's value is the single resource; colonist resolves it after
        # the card is down.
        return ([(SEND_PLAY_DEV_CARD, DEV_MONOPOLY)]
                + _selection(_cards([value])))
    if kind == ActionType.PLAY_YEAR_OF_PLENTY:
        # One card when the bank is nearly empty, otherwise two.
        return ([(SEND_PLAY_DEV_CARD, DEV_YEAR_OF_PLENTY)]
                + _selection(_cards(list(value))))

    if kind == ActionType.MARITIME_TRADE:
        # Five slots: up to four given, padded with None, then the one wanted.
        given = [resource for resource in value[:4] if resource is not None]
        wanted = value[4]
        return [
            (SEND_OPEN_PANEL, True),
            (SEND_TRADE, {
                "creator": colonist_color,
                "isBankTrade": True,
                "counterOfferInResponseToTradeId": None,
                "offeredResources": _cards(given),
                "wantedResources": _cards([wanted]),
            }),
        ]

    if kind == ActionType.DISCARD:
        raise TranslationError(
            "a discard is one frame carrying every card, so the whole discard "
            "has to be collected first; call discard_frames()"
        )

    raise TranslationError(f"no colonist frame is known for {action}")


def _corner(coords, node_id: int) -> int:
    try:
        return coords.corner_by_node[node_id]
    except KeyError:
        raise TranslationError(f"node {node_id} is not on this board") from None


def _edge(coords, edge) -> int:
    key = tuple(sorted(edge))
    try:
        return coords.edge_by_catanatron[key]
    except KeyError:
        raise TranslationError(f"edge {key} is not on this board") from None


def _hex(coords, coordinate) -> int:
    for hex_id, cube in coords.coordinate_by_hex.items():
        if tuple(cube) == tuple(coordinate):
            return hex_id
    raise TranslationError(f"no colonist hex at coordinate {coordinate}")


def frames_for_client_action(payload: dict) -> Optional[Frame]:
    """One recorded client frame as a ``Frame``, or ``None`` if it is not one.

    Used by the round-trip test: every frame our own client sent during a
    captured game is replayed through :func:`translate` and compared here.
    """
    if not isinstance(payload, dict) or "action" not in payload:
        return None
    return (payload["action"], payload.get("payload"))


__all__ = [
    "CARD_BY_RESOURCE", "Frame", "PROBE_ACTIONS", "ProtocolError",
    "TranslationError", "discard_frames", "frames_for_client_action",
    "translate",
]
