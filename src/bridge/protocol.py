"""Translate colonist.io's wire messages into catanatron actions.

This is the read half of the bridge and nothing else. It takes what the server
says and produces a :class:`~src.bridge.board.BoardSpec` plus a stream of
fully-specified ``Action``s that :class:`~src.bridge.replay.GameReplay` can
apply. **It deliberately says nothing about how a move is sent.** The client's
own frames are legible -- ``{"action": <int>, "payload": ..., "sequence": N}``
-- but whether the server accepts a frame we synthesize, or insists on a real
click with whatever else the page attaches to it, is unknown and not this
module's problem. Reading is useful on its own: it is the entire spectator
dry-run, and it is what any sending strategy would have to be built on.

Everything here was derived from captured traffic (`src/bridge/capture.py`), and
the parts that are *inferred* rather than observed are marked as such. Where a
message has never been seen, the decoder raises :class:`ProtocolError` rather
than guessing -- the same stance `GameReplay` takes: a wrong guess looks like a
weak agent, an exception looks like a bug.

What the server sends:

- one ``type: 4`` message carrying the full initial ``gameState``, and
- a stream of ``type: 91`` diffs of that same tree, each carrying
  ``gameLogState`` entries that name what happened.

The log says *what*, the diff says *where*: a road build is a log entry with a
piece enum plus a ``tileEdgeStates`` change holding the edge id. Both halves are
needed, which is why this decodes diffs rather than logs alone.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from catanatron import Color
from catanatron.models.actions import Action
from catanatron.models.enums import ActionType
from catanatron.models.map import (
    PORT_DIRECTION_TO_NODEREFS,
    EdgeRef,
    LandTile,
    NodeRef,
    Port,
    build_map,
)

from src.bridge.board import BoardSpec
from src.bridge.capture import decode_payload

# --------------------------------------------------------------------------
# Colonist enums
#
# Confirmed by cross-checking against the game they came from; where a value has
# never been observed it is absent, and hitting it raises rather than guessing.
# --------------------------------------------------------------------------

# Resource cards. The counts confirm the split -- enums 1/3/4 appear on four
# tiles each and 2/5 on three, matching wood/sheep/wheat against brick/ore -- and
# the specific assignment is colonist's documented order. A swap within a group
# would survive that check but not a replay, which is where it would surface.
RESOURCE_BY_CARD = {1: "WOOD", 2: "BRICK", 3: "SHEEP", 4: "WHEAT", 5: "ORE"}
HIDDEN_CARD = 0  # what the opponent's hand looks like to us

# Ports: type 1 is the 3:1 generic, and the five specific ports are the resource
# enum plus one.
PORT_GENERIC = 1

# Development cards, each identified by what followed it being played: 11 moves
# the robber, 13 takes every card of one resource, 15 draws two, and 14 is
# followed by two free roads. 12 was only ever held, but the end-of-game deck
# statistics show it drawn three times, and the deck holds just two each of
# monopoly, year of plenty and road building -- so 12 is a knight or a victory
# point, and 11 is already the knight.
DEV_KNIGHT = 11
DEV_VICTORY_POINT = 12
DEV_MONOPOLY = 13
DEV_ROAD_BUILDING = 14
DEV_YEAR_OF_PLENTY = 15
DEV_HIDDEN = 10

# Pieces, as they appear in placement log entries.
PIECE_ROAD = 0
PIECE_SETTLEMENT = 2
PIECE_CITY = 3
PIECE_ROBBER = 5

# Server message types.
MSG_FULL_STATE = 4
MSG_DIFF = 91

# Game-log entry types.
LOG_BOUGHT_DEV_CARD = 1
LOG_FREE_PLACEMENT = 4      # opening settlements and roads
LOG_BUILT = 5               # built or bought during play
LOG_ROLL = 10
LOG_ROBBER_MOVED = 11
LOG_STOLE_FROM_THEM = 14    # we gained a card; playerColor is the victim
LOG_ROBBED_BY_THEM = 15     # we lost a card; playerColor is the thief
LOG_PLAYED_DEV_CARD = 20
LOG_YEAR_OF_PLENTY = 21
LOG_TURN_STARTED = 44
LOG_DISCARDED = 55
LOG_MONOPOLY = 86
LOG_BANK_TRADE = 116

# Log entries that describe consequences rather than decisions. The engine
# derives all of these itself, so they produce no action: resource payouts,
# achievements being won (66) and changing hands (68), a blocked tile, the win,
# and a player-count notice (139) that has nothing to do with the position.
LOG_IGNORED = frozenset({
    2, 22, 45, 47, 49, 60, 66, 68, 74, 139,
})


class ProtocolError(RuntimeError):
    """A message the decoder does not understand.

    Raised rather than skipped. An unknown message is either a rule we do not
    model or a protocol change, and both corrupt the reconstruction silently if
    they are ignored.
    """


# --------------------------------------------------------------------------
# Geometry
#
# Colonist addresses corners and edges as (hex, z): the hex owns two of the six
# corners and three of the six edges, and z picks which. Catanatron addresses
# them by id, assigned by a topology walk. The bridge between the two is
# position: lay both boards out on the same integer grid and the ids that land
# on the same point are the same place.
#
# The assignments below were solved for, not assumed -- searching all rotations,
# reflections and ref-orderings against a captured board left exactly one
# consistent answer for edges and, for corners, the natural one (a hex owns its
# north and south corners). They are still validated on every board, because a
# convention that changed silently would be indistinguishable from a weak agent.
# --------------------------------------------------------------------------

CORNER_REFS = {0: NodeRef.NORTH, 1: NodeRef.SOUTH}
EDGE_REFS = {0: EdgeRef.NORTHWEST, 1: EdgeRef.WEST, 2: EdgeRef.SOUTHWEST}

# Doubled integer coordinates for a pointy-top hex: exact, so positions compare
# with ``==`` instead of a tolerance.
_NODE_OFFSET = {
    NodeRef.NORTH: (0, 2), NodeRef.NORTHEAST: (1, 1), NodeRef.SOUTHEAST: (1, -1),
    NodeRef.SOUTH: (0, -2), NodeRef.SOUTHWEST: (-1, -1), NodeRef.NORTHWEST: (-1, 1),
}
_EDGE_OFFSET = {
    EdgeRef.EAST: (2, 0), EdgeRef.SOUTHEAST: (1, -3), EdgeRef.SOUTHWEST: (-1, -3),
    EdgeRef.WEST: (-2, 0), EdgeRef.NORTHWEST: (-1, 3), EdgeRef.NORTHEAST: (1, 3),
}


def _hex_center(x: int, y: int) -> Tuple[int, int]:
    """Colonist's axial hex coordinate as a point on the doubled grid."""
    return (2 * x + y, -3 * y)


@dataclass(frozen=True)
class CoordinateMap:
    """Colonist's ids for hexes, corners and edges, in catanatron's terms."""

    tile_by_hex: Dict[int, int]                  # colonist hex id -> tile id
    coordinate_by_hex: Dict[int, Tuple[int, int, int]]
    node_by_corner: Dict[int, int]
    edge_by_edge: Dict[int, Tuple[int, int]]

    @property
    def hex_by_tile(self) -> Dict[int, int]:
        return {tile: hex_id for hex_id, tile in self.tile_by_hex.items()}

    @property
    def corner_by_node(self) -> Dict[int, int]:
        return {node: corner for corner, node in self.node_by_corner.items()}

    @property
    def edge_by_catanatron(self) -> Dict[Tuple[int, int], int]:
        return {edge: eid for eid, edge in self.edge_by_edge.items()}


def _reference_geometry():
    """Positions of every node and edge on a stock BASE board.

    Any BASE map serves: ``initialize_tiles`` assigns ids from the template's
    topology walk, so the *numbering* is fixed even though the resources dealt
    onto it are not.
    """
    reference = build_map("BASE")
    land = {c: t for c, t in reference.tiles.items() if isinstance(t, LandTile)}

    node_at, position_of, tile_at = {}, {}, {}
    for coordinate, tile in land.items():
        cx, cy = _hex_center(coordinate[0], coordinate[2])
        tile_at[(cx, cy)] = (tile.id, coordinate)
        for ref, node_id in tile.nodes.items():
            offset = _NODE_OFFSET[ref]
            node_at[(cx + offset[0], cy + offset[1])] = node_id
            position_of[node_id] = (cx + offset[0], cy + offset[1])

    edge_at = {}
    for coordinate, tile in land.items():
        for edge in tile.edges.values():
            first, second = position_of[edge[0]], position_of[edge[1]]
            edge_at[(first[0] + second[0], first[1] + second[1])] = tuple(sorted(edge))

    return reference, tile_at, node_at, edge_at


def build_coordinate_map(map_state: dict) -> CoordinateMap:
    """Solve colonist's ids against catanatron's, and prove the solution."""
    _, tile_at, node_at, edge_at = _reference_geometry()

    tile_by_hex, coordinate_by_hex = {}, {}
    for hex_id, hex_state in map_state["tileHexStates"].items():
        position = _hex_center(hex_state["x"], hex_state["y"])
        if position not in tile_at:
            raise ProtocolError(
                f"hex {hex_id} at {(hex_state['x'], hex_state['y'])} is not a land tile "
                "on a BASE board; the coordinate convention has changed"
            )
        tile_id, coordinate = tile_at[position]
        tile_by_hex[int(hex_id)] = tile_id
        coordinate_by_hex[int(hex_id)] = coordinate

    node_by_corner = {}
    for corner_id, corner in map_state["tileCornerStates"].items():
        cx, cy = _hex_center(corner["x"], corner["y"])
        offset = _NODE_OFFSET[CORNER_REFS[corner["z"]]]
        position = (cx + offset[0], cy + offset[1])
        if position not in node_at:
            raise ProtocolError(f"corner {corner_id} at {corner} is off the board")
        node_by_corner[int(corner_id)] = node_at[position]

    edge_by_edge = {}
    for edge_id, edge in map_state["tileEdgeStates"].items():
        cx, cy = _hex_center(edge["x"], edge["y"])
        offset = _EDGE_OFFSET[EDGE_REFS[edge["z"]]]
        position = (2 * cx + offset[0], 2 * cy + offset[1])
        if position not in edge_at:
            raise ProtocolError(f"edge {edge_id} at {edge} is off the board")
        edge_by_edge[int(edge_id)] = edge_at[position]

    # Bijections, or the mapping is not a translation. Sizes alone would pass a
    # mapping that sent two corners to one node.
    for name, mapping, expected in (("hexes", tile_by_hex, 19),
                                    ("corners", node_by_corner, 54),
                                    ("edges", edge_by_edge, 72)):
        if len(mapping) != expected or len(set(mapping.values())) != expected:
            raise ProtocolError(
                f"{name}: expected {expected} distinct ids, got {len(mapping)} "
                f"colonist ids covering {len(set(mapping.values()))} catanatron ones"
            )

    return CoordinateMap(tile_by_hex, coordinate_by_hex, node_by_corner, edge_by_edge)


# --------------------------------------------------------------------------
# Board
# --------------------------------------------------------------------------


def board_spec_from_state(map_state: dict, coords: CoordinateMap) -> BoardSpec:
    """The dealt board, in the tile and port order ``BoardSpec`` expects."""
    tiles: Dict[int, Tuple[Optional[str], Optional[int]]] = {}
    for hex_id, hex_state in map_state["tileHexStates"].items():
        resource = RESOURCE_BY_CARD.get(hex_state["type"])
        number = hex_state["diceNumber"] or None
        tiles[coords.tile_by_hex[int(hex_id)]] = (resource, number)

    reference, _, _, edge_at = _reference_geometry()
    ports_by_nodes = {}
    for port_state in map_state["portEdgeStates"].values():
        cx, cy = _hex_center(port_state["x"], port_state["y"])
        offset = _EDGE_OFFSET[EDGE_REFS[port_state["z"]]]
        edge = edge_at.get((2 * cx + offset[0], 2 * cy + offset[1]))
        if edge is None:
            raise ProtocolError(f"port at {port_state} is not on a board edge")
        kind = port_state["type"]
        ports_by_nodes[edge] = (None if kind == PORT_GENERIC
                                else RESOURCE_BY_CARD.get(kind - 1))

    # ``BoardSpec.ports`` is indexed by port id, and a port's id likewise comes
    # from the template walk, so the reference map supplies the ordering. A port
    # tile is a water hex with six corners; only the two facing the island are
    # the port, and its direction names which.
    ports: Dict[int, Optional[str]] = {}
    for tile in reference.tiles.values():
        if not isinstance(tile, Port):
            continue
        nodes = tuple(sorted(tile.nodes[ref]
                             for ref in PORT_DIRECTION_TO_NODEREFS[tile.direction]))
        if nodes not in ports_by_nodes:
            raise ProtocolError(f"port {tile.id} has no counterpart in the observed board")
        ports[tile.id] = ports_by_nodes[nodes]

    return BoardSpec(
        tiles=tuple(tiles[i] for i in sorted(tiles)),
        ports=tuple(ports[i] for i in sorted(ports)),
    )


# --------------------------------------------------------------------------
# Frames
# --------------------------------------------------------------------------


def iter_colonist_frames(path: Path):
    """Decoded frames on colonist's own socket, in order, both directions."""
    sockets = {}
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if record.get("kind") == "socket":
                sockets[record.get("id")] = record.get("url", "")
                continue
            if record.get("kind") != "frame":
                continue
            if "colonist" not in sockets.get(record.get("id"), ""):
                continue
            if record.get("encoding") == "base64":  # captured before we could decode it
                record = {**record, **decode_payload(record["payload"], 2)}
            yield record


def server_message(record: dict) -> Optional[Tuple[int, object]]:
    """``(type, payload)`` for one decoded frame, or ``None`` if it holds no game message.

    Split out from the file reader because the live session gets the same
    records straight off CDP and must unwrap them identically. Client frames and
    transport chatter both return ``None``.
    """
    payload = record.get("payload")
    if record.get("dir") != "recv" or not isinstance(payload, dict):
        return None
    data = payload.get("data")
    if isinstance(data, dict) and "type" in data:
        return data["type"], data.get("payload")
    return None


def iter_server_messages(path: Path):
    """``(type, payload)`` for each game message in a capture, in order."""
    for record in iter_colonist_frames(path):
        message = server_message(record)
        if message is not None:
            yield message


# --------------------------------------------------------------------------
# Actions
# --------------------------------------------------------------------------


@dataclass
class DecodedGame:
    """A captured game, in the terms the agent and the engine speak."""

    board: BoardSpec
    coords: CoordinateMap
    seating: Tuple[Color, ...]
    our_color: Color
    actions: List[Action]


class _ActionDecoder:
    """Walks the diffs, turning log entries plus state changes into actions.

    Kept as a class only because the translation is stateful: a log entry names
    the piece but not the place, so the decoder pairs it with the map changes
    riding in the same diff, and a played dev card is resolved by the entry that
    follows it.
    """

    def __init__(self, coords: CoordinateMap, colors: Dict[int, Color],
                 our_color: Color):
        self.coords = coords
        self.colors = colors
        self.our_color = our_color
        self.actions: List[Action] = []
        #: Lowest index the last :meth:`feed` *rewrote* (as opposed to appended).
        #: Only a steal does this, and only to the robber move before it -- but a
        #: caller replaying incrementally has to know, because an action it has
        #: already applied just changed underneath it.
        self.revised_from: Optional[int] = None
        self.rolled_this_turn = False
        self.pending_dev_card: Optional[int] = None
        self.our_dev_cards: List[int] = []

    # -- helpers ------------------------------------------------------------
    def _color(self, colonist_color: int) -> Color:
        if colonist_color not in self.colors:
            raise ProtocolError(f"unknown player color {colonist_color}")
        return self.colors[colonist_color]

    @staticmethod
    def _changed(diff: dict, section: str) -> Dict[int, dict]:
        return {int(k): v for k, v in
                (diff.get("mapState", {}).get(section) or {}).items()}

    def _resources(self, card_enums) -> List[str]:
        out = []
        for card in card_enums or ():
            resource = RESOURCE_BY_CARD.get(card)
            if resource is None:
                raise ProtocolError(f"card enum {card} is not a resource")
            out.append(resource)
        return out

    # -- entry point --------------------------------------------------------
    def feed(self, diff: dict) -> Optional[int]:
        """Decode one diff. Returns the lowest index it rewrote, or ``None``."""
        self.revised_from = None
        entries = sorted((diff.get("gameLogState") or {}).items(), key=lambda kv: int(kv[0]))
        for _, entry in entries:
            self._entry((entry or {}).get("text") or {}, diff)
        return self.revised_from

    def _revise(self, index: int, action: Action) -> None:
        """Rewrite an already-emitted action, recording that it happened."""
        self.actions[index] = action
        self.revised_from = (index if self.revised_from is None
                             else min(self.revised_from, index))

    def _entry(self, text: dict, diff: dict) -> None:
        kind = text.get("type")
        if kind in LOG_IGNORED or kind is None:
            return

        handler = {
            LOG_TURN_STARTED: self._turn_started,
            LOG_ROLL: self._roll,
            LOG_FREE_PLACEMENT: self._placement,
            LOG_BUILT: self._placement,
            LOG_BOUGHT_DEV_CARD: self._bought_dev_card,
            LOG_ROBBER_MOVED: self._robber,
            LOG_STOLE_FROM_THEM: self._steal,
            LOG_ROBBED_BY_THEM: self._steal,
            LOG_BANK_TRADE: self._bank_trade,
            LOG_PLAYED_DEV_CARD: self._played_dev_card,
            LOG_MONOPOLY: self._monopoly,
            LOG_YEAR_OF_PLENTY: self._year_of_plenty,
            LOG_DISCARDED: self._discard,
        }.get(kind)
        if handler is None:
            raise ProtocolError(f"unhandled game-log entry {kind}: {text}")
        handler(text, diff)

    # -- handlers -----------------------------------------------------------
    def _turn_started(self, text: dict, diff: dict) -> None:
        """A new turn begins, so the previous player's turn ended.

        The engine wants that ``END_TURN`` explicitly, but only for turns that
        were really played: the marker also fires for the first turn after the
        opening, where nobody has yet had a turn to end. A roll is the reliable
        witness -- every played turn contains exactly one, and the opening
        contains none.
        """
        if self.rolled_this_turn:
            previous = self.actions[-1].color
            self.actions.append(Action(previous, ActionType.END_TURN, None))
        self.rolled_this_turn = False

    def _roll(self, text: dict, diff: dict) -> None:
        color = self._color(text["playerColor"])
        dice = (text["firstDice"], text["secondDice"])
        self.actions.append(Action(color, ActionType.ROLL, dice))
        self.rolled_this_turn = True

    def _placement(self, text: dict, diff: dict) -> None:
        """A settlement, city or road; the diff says where."""
        color = self._color(text["playerColor"])
        piece = text.get("pieceEnum")
        if piece == PIECE_ROAD:
            changed = self._changed(diff, "tileEdgeStates")
            if not changed:
                raise ProtocolError("a road was built but no edge changed hands")
            edge_id = next(iter(changed))
            self.actions.append(
                Action(color, ActionType.BUILD_ROAD, self.coords.edge_by_edge[edge_id]))
        elif piece in (PIECE_SETTLEMENT, PIECE_CITY):
            changed = self._changed(diff, "tileCornerStates")
            if not changed:
                raise ProtocolError("a building went up but no corner changed hands")
            corner_id = next(iter(changed))
            action_type = (ActionType.BUILD_SETTLEMENT if piece == PIECE_SETTLEMENT
                           else ActionType.BUILD_CITY)
            self.actions.append(
                Action(color, action_type, self.coords.node_by_corner[corner_id]))
        else:
            raise ProtocolError(f"unknown piece enum {piece} in {text}")

    def _bought_dev_card(self, text: dict, diff: dict) -> None:
        """Which card was drawn is visible only when it is ours.

        ``None`` is the honest value for the opponent's purchase: the engine
        then draws from its own deck, which is exactly the uncertainty a
        determinization step has to model. This is the last hidden-information
        leak in 1v1 -- discards turned out to be broadcast.
        """
        color = self._color(text["playerColor"])
        drawn = None
        cards = (((diff.get("mechanicDevelopmentCardsState") or {}).get("players") or {})
                 .get(str(text["playerColor"])) or {}).get("developmentCards")
        if isinstance(cards, dict):
            visible = [c for c in cards.get("cards", []) if c != DEV_HIDDEN]
            new = _first_new(self.our_dev_cards, visible)
            if new is not None:
                self.our_dev_cards = visible
                drawn = _CATANATRON_DEV_CARD.get(new)
                if drawn is None:
                    raise ProtocolError(f"unknown development card {new}")
        self.actions.append(Action(color, ActionType.BUY_DEVELOPMENT_CARD, drawn))

    def _robber(self, text: dict, diff: dict) -> None:
        """Where the robber went. Whom it robbed arrives in the next entry."""
        location = (diff.get("mechanicRobberState") or {}).get("locationTileIndex")
        if location is None:
            raise ProtocolError("the robber moved without a destination")
        color = self._color(text["playerColor"])
        coordinate = self.coords.coordinate_by_hex[location]
        self.actions.append(
            Action(color, ActionType.MOVE_ROBBER, (coordinate, None, None)))

    def _steal(self, text: dict, diff: dict) -> None:
        """Fill in the victim and card on the robber move just emitted.

        Both directions are visible: colonist tells each client what it gained
        and what it lost, so in 1v1 the stolen card is never a guess. Entry 14
        is a card coming to us and 15 one leaving, with ``playerColor`` naming
        the *other* player either way -- established from hand-size deltas,
        since the field itself does not say which side it means. So the victim
        is that other player when we did the stealing, and ourselves when we
        did not.
        """
        if not self.actions or self.actions[-1].action_type != ActionType.MOVE_ROBBER:
            raise ProtocolError("a steal arrived without a robber move before it")
        move = self.actions[-1]
        coordinate, _, _ = move.value
        other = self._color(text["playerColor"])
        stole = text["type"] == LOG_STOLE_FROM_THEM
        thief, victim = (self.our_color, other) if stole else (other, self.our_color)
        if move.color != thief:
            raise ProtocolError(
                f"{thief} stole the card but {move.color} moved the robber")
        resources = self._resources(text.get("cardEnums"))
        if len(resources) != 1:
            raise ProtocolError(f"a steal moved {len(resources)} cards")
        self._revise(len(self.actions) - 1, Action(
            move.color, ActionType.MOVE_ROBBER, (coordinate, victim, resources[0])))

    def _bank_trade(self, text: dict, diff: dict) -> None:
        """One log entry can hold several trades, so it is split back apart.

        Colonist reports a player's consecutive bank trades as a single entry --
        six cards out, two in, being two separate 3:1 trades. The engine wants
        them one at a time, and the split is unambiguous because each trade
        gives up one resource: consecutive runs of the same card are the trades,
        in step with the cards received.
        """
        color = self._color(text["playerColor"])
        given = self._resources(text["givenCardEnums"])
        received = self._resources(text["receivedCardEnums"])

        runs: List[List[str]] = []
        for resource in given:
            if runs and runs[-1][0] == resource:
                runs[-1].append(resource)
            else:
                runs.append([resource])
        if len(runs) != len(received) or any(len(run) not in (2, 3, 4) for run in runs):
            raise ProtocolError(f"unexpected bank trade shape: {text}")

        for run, got in zip(runs, received):
            # catanatron pads the give side to four slots with None.
            value = tuple(run + [None] * (4 - len(run)) + [got])
            self.actions.append(Action(color, ActionType.MARITIME_TRADE, value))

    def _played_dev_card(self, text: dict, diff: dict) -> None:
        """A knight or road building resolves here; the others need their result.

        Knight and road building are complete decisions on their own -- what
        follows (the robber move, the two free roads) arrives as its own log
        entry and becomes its own action, which is exactly how the engine
        prompts for them. Monopoly and year of plenty carry their choice in a
        later entry, so they wait.
        """
        card = text.get("cardEnum")
        color = self._color(text["playerColor"])
        if card in (DEV_KNIGHT, DEV_ROAD_BUILDING):
            action_type = (ActionType.PLAY_KNIGHT_CARD if card == DEV_KNIGHT
                           else ActionType.PLAY_ROAD_BUILDING)
            self.actions.append(Action(color, action_type, None))
            self.pending_dev_card = None
            return
        if card in (DEV_MONOPOLY, DEV_YEAR_OF_PLENTY):
            self.pending_dev_card = card
            return
        raise ProtocolError(
            f"development card {card} has never been captured being played; "
            "road building and victory-point cards are still unobserved"
        )

    def _monopoly(self, text: dict, diff: dict) -> None:
        if self.pending_dev_card != DEV_MONOPOLY:
            raise ProtocolError("a monopoly resolved without being played")
        color = self._color(text["playerColor"])
        resource = self._resources([text["cardEnum"]])[0]
        self.actions.append(Action(color, ActionType.PLAY_MONOPOLY, resource))
        self.pending_dev_card = None

    def _year_of_plenty(self, text: dict, diff: dict) -> None:
        if self.pending_dev_card != DEV_YEAR_OF_PLENTY:
            raise ProtocolError("a year of plenty resolved without being played")
        color = self._color(text["playerColor"])
        cards = tuple(self._resources(text["cardEnums"]))
        self.actions.append(Action(color, ActionType.PLAY_YEAR_OF_PLENTY, cards))
        self.pending_dev_card = None

    def _discard(self, text: dict, diff: dict) -> None:
        """One action per card, because that is the discard the agent learned.

        ``src/env/rules.py`` replaces the stock all-at-once discard with a
        per-resource one, which is why the action space is 294 and not 290. The
        log hands over the whole discarded hand at once, so it is expanded.
        """
        if not text.get("areResourceCards", True):
            raise ProtocolError(f"a non-resource discard: {text}")
        color = self._color(text["playerColor"])
        for resource in self._resources(text["cardEnums"]):
            self.actions.append(Action(color, ActionType.DISCARD, resource))


_CATANATRON_DEV_CARD = {
    DEV_KNIGHT: "KNIGHT",
    DEV_VICTORY_POINT: "VICTORY_POINT",
    DEV_MONOPOLY: "MONOPOLY",
    DEV_ROAD_BUILDING: "ROAD_BUILDING",
    DEV_YEAR_OF_PLENTY: "YEAR_OF_PLENTY",
}


_PLAYED_DEV_CARD = {
    ActionType.PLAY_KNIGHT_CARD: "KNIGHT",
    ActionType.PLAY_MONOPOLY: "MONOPOLY",
    ActionType.PLAY_YEAR_OF_PLENTY: "YEAR_OF_PLENTY",
    ActionType.PLAY_ROAD_BUILDING: "ROAD_BUILDING",
}


def reveal_purchases(actions: List[Action]) -> List[Action]:
    """Back-fill the opponent's dev-card purchases from what they later played.

    A purchase we did not make decodes as ``None``, because at the moment it
    happens the card is genuinely unknown. The engine then draws one at random,
    and the replay desyncs the first time the opponent plays a card the draw
    did not give them.

    In a *finished* capture the answer is already on the tape: a card that was
    played was bought earlier, so each revealed card can be attributed to one of
    that player's still-unexplained purchases. Purchases never revealed stay
    ``None``, which is honest: they were victory points, or were never used.

    "Earlier" has to mean *on an earlier turn*, not merely earlier in the
    sequence. Neither colonist nor `src/env/rules.py` lets a card be played the
    turn it was bought, so attributing a card to a purchase from the same turn
    produces a hand the engine will refuse to play from -- which is precisely
    how this was found, on a player who bought twenty cards and played sixteen.

    **This is for replaying recorded games, not for live play.** Live, the card
    stays unknown until it is played, and that is a determinization problem, not
    a decoding one.
    """
    revealed = list(actions)
    pending: Dict[Color, List[Tuple[int, int]]] = {}  # color -> [(turn, index)]
    turn = 0
    for index, action in enumerate(revealed):
        if action.action_type == ActionType.END_TURN:
            turn += 1
            continue
        if (action.action_type == ActionType.BUY_DEVELOPMENT_CARD
                and action.value is None):
            pending.setdefault(action.color, []).append((turn, index))
            continue
        card = _PLAYED_DEV_CARD.get(action.action_type)
        if card is None:
            continue
        bought = pending.get(action.color, [])
        playable = next((entry for entry in bought if entry[0] < turn), None)
        if playable is None:
            continue  # nothing it could have come from; leave the replay to complain
        bought.remove(playable)
        revealed[playable[1]] = Action(
            action.color, ActionType.BUY_DEVELOPMENT_CARD, card)
    return revealed


def _first_new(before: List[int], after: List[int]) -> Optional[int]:
    """The card that appeared, comparing two hands as multisets."""
    remaining = list(before)
    for card in after:
        if card in remaining:
            remaining.remove(card)
        else:
            return card
    return None


class MessageDecoder:
    """The whole read side, driven one server message at a time.

    :func:`decode_capture` used to hold this state machine inline, which made it
    a file reader by construction. A live session needs the identical logic fed
    off the wire, so it lives here and the file reader became a loop over it --
    the same code path, so the offline tests still cover the live one.

    Two things a caller replaying incrementally has to watch, because both mean
    actions it already applied are no longer what the decoder says happened:

    - :attr:`game_id` changes. A second full state is a reconnect or a new game;
      the board itself is different, so nothing survives.
    - :meth:`feed` returns an index. A steal rewrites the robber move emitted
      just before it (that is where the victim and the stolen card arrive), so
      an action can change after it has been handed out.

    Both are reported rather than smoothed over: a replay that quietly drifts
    looks exactly like a weak agent, which is the failure mode this whole module
    is built to avoid.
    """

    def __init__(self, colors=(Color.BLUE, Color.RED)):
        self.colors = colors
        self.board: Optional[BoardSpec] = None
        self.coords: Optional[CoordinateMap] = None
        self.seating: Tuple[Color, ...] = ()
        self.our_color: Optional[Color] = None
        #: The lobby's own settings, as the full state reported them. This is how
        #: the house rules stop being an assumption: ``victoryPointsToWin`` and
        #: ``cardDiscardLimit`` are on the wire, so they can be checked against
        #: the patches in :mod:`src.env.rules` instead of eyeballed.
        self.settings: dict = {}
        #: Bumped on every full state, so a caller can tell one game from the next.
        self.game_id = 0
        self._decoder: Optional[_ActionDecoder] = None

    @property
    def started(self) -> bool:
        """Whether a full state has arrived and the board is known."""
        return self._decoder is not None

    @property
    def actions(self) -> List[Action]:
        """Every action decoded so far, oldest first. Live, so do not mutate it."""
        return self._decoder.actions if self._decoder is not None else []

    def feed(self, kind: int, payload) -> Optional[int]:
        """Consume one server message.

        Returns the lowest action index that is no longer valid -- ``0`` when a
        full state resets the game, the rewritten index when a steal revises the
        robber move, and ``None`` when the message only appended (the usual
        case) or was not one we decode.
        """
        if kind == MSG_FULL_STATE and isinstance(payload, dict):
            self._begin(payload)
            return 0
        if kind == MSG_DIFF and self._decoder is not None and isinstance(payload, dict):
            diff = payload.get("diff")
            if isinstance(diff, dict):
                return self._decoder.feed(diff)
        return None

    def _begin(self, payload: dict) -> None:
        state = payload["gameState"]
        self.settings = payload.get("gameSettings") or {}
        self.coords = build_coordinate_map(state["mapState"])
        self.board = board_spec_from_state(state["mapState"], self.coords)
        order = payload["playOrder"]
        by_colonist = {c: self.colors[i] for i, c in enumerate(order)}
        self.seating = tuple(by_colonist[c] for c in order)
        self.our_color = by_colonist[payload["playerColor"]]
        self._decoder = _ActionDecoder(self.coords, by_colonist, self.our_color)
        self.game_id += 1

    def decoded(self, reveal: bool = False) -> "DecodedGame":
        """Everything decoded so far, as a :class:`DecodedGame`.

        ``reveal`` back-fills the opponent's purchases from cards they later
        played (:func:`reveal_purchases`). It defaults off here because the live
        caller is the one that has to ask for it deliberately -- mid-game it
        reads the past, not the future, but it is still a guess being made on
        the caller's behalf.
        """
        if not self.started:
            raise ProtocolError("no game: no full-state message has arrived")
        actions = reveal_purchases(self.actions) if reveal else list(self.actions)
        return DecodedGame(board=self.board, coords=self.coords,
                           seating=self.seating, our_color=self.our_color,
                           actions=actions)


def decode_capture(path: Path, colors=(Color.BLUE, Color.RED),
                   reveal=True) -> DecodedGame:
    """Decode one captured game into a board and a stream of actions.

    Args:
        path: a JSONL capture from :mod:`src.bridge.capture`.
        colors: catanatron colors to hand out in colonist's play order. The
            first seat settles first, which is the part that matters; the colors
            themselves are arbitrary labels.
        reveal: back-fill the opponent's dev-card purchases from cards they
            later played (see :func:`reveal_purchases`). Right for a recorded
            game, wrong for a live one.
    """
    decoder = MessageDecoder(colors)
    for kind, payload in iter_server_messages(Path(path)):
        if kind == MSG_FULL_STATE and decoder.started:
            break  # a second game in one capture; stop at the first
        decoder.feed(kind, payload)

    if not decoder.started:
        raise ProtocolError(f"{path} contains no game: no full-state message was sent")
    return decoder.decoded(reveal=reveal)


__all__ = [
    "CoordinateMap", "DecodedGame", "MessageDecoder", "ProtocolError",
    "board_spec_from_state", "build_coordinate_map", "decode_capture",
    "iter_colonist_frames", "iter_server_messages", "reveal_purchases",
    "server_message",
]
