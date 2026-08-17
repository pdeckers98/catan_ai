"""Tests for the colonist.io bridge core -- rung 0 of the verification ladder.

Nothing here touches colonist.io. The claim under test is the one everything
else in Phase 3 rests on: an observed stream of actions, replayed into a
reconstructed board, produces a game the agent's own encoders cannot tell from
the original. A silent mismatch there looks like a weak agent, not like a bug,
so it is pinned before any protocol work starts.
"""

import base64
import pathlib

import numpy as np
import pytest

from catanatron import Color
from catanatron.models.actions import Action
from catanatron.models.enums import ActionType
from catanatron.models.map import (
    PORT_DIRECTION_TO_NODEREFS,
    LandTile,
    Port,
    build_map,
)
from catanatron.players.weighted_random import WeightedRandomPlayer

from src.agent.encoding import encode_observation
from src.bridge.board import BoardSpec, build_map_from_spec, spec_from_map
from src.bridge import protocol
from src.bridge.capture import decode_payload, message_type, shape
from src.bridge.player import build_bridge_player
from src.bridge.replay import DesyncError, GameReplay, blank_outcome
from src.env.catan_env import make_1v1_game

COLORS = (Color.BLUE, Color.RED)
# Captures are personal browser sessions, so none is committed; the test that
# replays real traffic runs only where one has been recorded.
CAPTURES = sorted(pathlib.Path("data/bridge").glob("*.jsonl"))


def play_reference_game(seed: int):
    """A full self-play game, as (board spec, seating, action log, game).

    The seating is returned rather than assumed: ``State`` shuffles the players,
    so which color moves first is a property of the played game -- exactly the
    thing the bridge has to observe rather than guess.
    """
    players = [WeightedRandomPlayer(color) for color in COLORS]
    game = make_1v1_game(players=players, seed=seed)
    while game.winning_color() is None and game.state.num_turns < 400:
        game.play_tick()
    return (spec_from_map(game.state.board.map), tuple(game.state.colors),
            list(game.state.actions), game)


def vps(state):
    return {
        color: state.player_state[
            f"P{state.color_to_index[color]}_ACTUAL_VICTORY_POINTS"
        ]
        for color in COLORS
    }


# --------------------------------------------------------------------------
# Board reconstruction
# --------------------------------------------------------------------------
def test_board_spec_round_trips_through_a_rebuilt_map():
    """A spec read off a map must rebuild that map, tile for tile.

    Node and edge ids come from the topology walk, not from the spec, so this
    also pins that a rebuilt board keeps catanatron's numbering -- which is what
    the action space and the placement features are written against.
    """
    original = build_map("BASE")
    rebuilt = build_map_from_spec(spec_from_map(original))

    assert spec_from_map(rebuilt) == spec_from_map(original)
    for tile_id, tile in original.tiles_by_id.items():
        other = rebuilt.tiles_by_id[tile_id]
        assert (other.resource, other.number) == (tile.resource, tile.number)
        assert other.nodes == tile.nodes
        assert other.edges == tile.edges
    assert rebuilt.port_nodes == original.port_nodes
    assert rebuilt.node_production == original.node_production


def test_board_spec_rejects_a_board_that_is_not_a_base_board():
    spec = spec_from_map(build_map("BASE"))
    resources = [r for r, _ in spec.tiles]
    swap = resources.index("ORE")
    broken = list(spec.tiles)
    broken[swap] = ("WOOD", spec.tiles[swap][1])

    with pytest.raises(ValueError, match="tile resources"):
        BoardSpec(tiles=tuple(broken), ports=spec.ports)


def test_board_spec_rejects_a_numbered_desert():
    spec = spec_from_map(build_map("BASE"))
    desert = next(i for i, (r, _) in enumerate(spec.tiles) if r is None)
    broken = list(spec.tiles)
    broken[desert] = (None, 6)

    with pytest.raises(ValueError, match="desert"):
        BoardSpec(tiles=tuple(broken), ports=spec.ports)


# --------------------------------------------------------------------------
# Replay
# --------------------------------------------------------------------------
@pytest.mark.parametrize("seed", [7, 11])
def test_replay_reproduces_the_observed_game_exactly(seed):
    """The whole bridge stands on this: same actions in, same position out."""
    spec, seating, actions, reference = play_reference_game(seed)

    replay = GameReplay(spec, colors=seating)
    replay.apply_many(actions)

    assert replay.winning_color() == reference.winning_color()
    assert replay.state.num_turns == reference.state.num_turns
    assert vps(replay.state) == vps(reference.state)
    for color in COLORS:
        for lookahead in (False, True):
            assert np.array_equal(
                encode_observation(replay.game, color, lookahead=lookahead),
                encode_observation(reference, color, lookahead=lookahead),
            ), f"observation mismatch for {color}, lookahead={lookahead}"


def test_replay_agrees_on_the_legal_moves_at_every_step():
    """Not just the final position: the *decision* offered has to match too.

    An agent asked to move from a position whose legal set has drifted will pick
    an action the live game rejects, which is the desync the click layer would
    discover the hard way.
    """
    spec, seating, actions, _ = play_reference_game(seed=13)
    reference = GameReplay(spec, colors=seating)
    replay = GameReplay(spec, colors=seating)

    for action in actions:
        assert ({blank_outcome(a) for a in replay.playable_actions}
                == {blank_outcome(a) for a in reference.playable_actions})
        assert replay.current_color == reference.current_color
        reference.apply(action)
        replay.apply(action)


def test_replay_forces_the_observed_dice_rather_than_rolling_its_own():
    spec, seating, actions, _ = play_reference_game(seed=5)
    replay = GameReplay(spec, colors=seating)

    rolls = []
    for action in actions:
        logged = replay.apply(action)
        if action.action_type == ActionType.ROLL:
            rolls.append((action.value, logged.value))

    assert rolls, "the reference game contained no rolls"
    assert all(observed == applied for observed, applied in rolls)


def test_replay_raises_on_an_action_the_reconstructed_game_does_not_allow():
    spec, seating, actions, _ = play_reference_game(seed=3)
    replay = GameReplay(spec, colors=seating)
    replay.apply(actions[0])

    # An end-of-turn before anyone has finished settling is never legal.
    with pytest.raises(DesyncError):
        replay.apply(Action(seating[0], ActionType.END_TURN, None))


# --------------------------------------------------------------------------
# Capture decoding
# --------------------------------------------------------------------------
def test_text_frame_unwraps_the_socket_io_prefix():
    """``42[...]`` is engine.io bookkeeping around the JSON we actually want."""
    decoded = decode_payload('42["game",{"n":1}]', opcode=1)

    assert decoded["encoding"] == "json"
    assert decoded["payload"] == ["game", {"n": 1}]
    assert decoded["socketio_prefix"] == "42"


def test_binary_frame_decodes_as_msgpack():
    msgpack = pytest.importorskip("msgpack")
    payload = base64.b64encode(msgpack.packb({"type": 7})).decode()

    assert decode_payload(payload, opcode=2) == {"encoding": "msgpack", "payload": {"type": 7}}


def test_a_framed_client_message_is_split_from_its_routing_header():
    """Every frame colonist's client sends is msgpack behind ``02 <id> <len><room>``.

    The offset is found by scanning rather than hardcoded: msgpack is
    self-delimiting, so the first offset that consumes the rest exactly is the
    body. The header is kept because its second byte is still unexplained.
    """
    msgpack = pytest.importorskip("msgpack")
    raw = b"\x02\x07\x05lobby" + msgpack.packb({"action": 1, "payload": {}})

    decoded = decode_payload(base64.b64encode(raw).decode(), opcode=2)

    assert decoded["encoding"] == "msgpack+header"
    assert decoded["payload"] == {"action": 1, "payload": {}}
    assert decoded["channel"] == "lobby"
    assert decoded["header"] == b"\x02\x07\x05lobby".hex()


def test_an_undecodable_frame_is_kept_rather_than_dropped():
    """A frame nobody can parse is still evidence about the protocol."""
    decoded = decode_payload("not json at all", opcode=1)

    assert decoded == {"encoding": "text", "payload": "not json at all"}


def test_message_type_descends_into_a_nested_envelope():
    """The label has to be a scalar, or every message becomes its own group."""
    label = message_type({"action": {"type": 2, "payload": {"anything": [1, 2, 3]}}})

    assert label == "action.type=2"


def test_shape_keeps_both_halves_of_a_socket_io_frame():
    """Collapsing a 2-list to "first element x2" would hide the whole payload."""
    assert shape(["game_event", {"type": 8}]) == ["str", {"type": "int"}]


# --------------------------------------------------------------------------
# Player assembly
# --------------------------------------------------------------------------
@pytest.mark.parametrize("missing", ["model_path", "placement_path", "bundle_path"])
def test_bridge_player_refuses_to_ship_a_partial_agent(missing):
    paths = {
        "model_path": "checkpoints/archive/ppo-15vp-lr-step400000.zip",
        "placement_path": "checkpoints/placement/scorer_ppo.pt",
        "bundle_path": "checkpoints/placement/bundle_noroads.pt",
    }
    paths[missing] = None

    with pytest.raises(ValueError, match=missing):
        build_bridge_player(Color.BLUE, **paths)


# --------------------------------------------------------------------------
# Protocol translation
# --------------------------------------------------------------------------
def colonist_state_from_map(catan_map):
    """A colonist-shaped ``mapState`` describing a catanatron board.

    Built from catanatron's geometry rather than from `protocol`'s tables, so
    the round-trip below exercises the translation instead of agreeing with
    itself. Colonist's axial hex coordinate is catanatron's ``(x, z)``; a hex
    owns its north and south corners and its three western edges, and corners on
    the sea ring belong to hexes off the island -- hence the search over a wider
    range of coordinates than the nineteen land tiles.
    """
    land = {c: t for c, t in catan_map.tiles.items() if isinstance(t, LandTile)}
    span = range(-3, 4)
    centers = {(x, y): protocol._hex_center(x, y) for x in span for y in span}

    def owner(point, refs, offsets, doubled):
        for (x, y), (cx, cy) in centers.items():
            for z, ref in refs.items():
                dx, dy = offsets[ref]
                base = (2 * cx, 2 * cy) if doubled else (cx, cy)
                if (base[0] + dx, base[1] + dy) == point:
                    return {"x": x, "y": y, "z": z}
        raise AssertionError(f"no hex owns {point}")

    card_of = {resource: card for card, resource in protocol.RESOURCE_BY_CARD.items()}
    hexes, position_of = {}, {}
    for coordinate, tile in land.items():
        cx, cy = protocol._hex_center(coordinate[0], coordinate[2])
        hexes[str(tile.id)] = {"x": coordinate[0], "y": coordinate[2],
                               "type": card_of.get(tile.resource, 0),
                               "diceNumber": tile.number or 0}
        for ref, node_id in tile.nodes.items():
            dx, dy = protocol._NODE_OFFSET[ref]
            position_of[node_id] = (cx + dx, cy + dy)

    def midpoint(edge):
        first, second = position_of[edge[0]], position_of[edge[1]]
        return (first[0] + second[0], first[1] + second[1])

    corners = {str(node): owner(point, protocol.CORNER_REFS,
                                protocol._NODE_OFFSET, doubled=False)
               for node, point in position_of.items()}

    edges, seen = {}, set()
    for tile in land.values():
        for edge in tile.edges.values():
            key = tuple(sorted(edge))
            if key not in seen:
                seen.add(key)
                edges[str(len(edges))] = owner(midpoint(key), protocol.EDGE_REFS,
                                               protocol._EDGE_OFFSET, doubled=True)

    ports = {}
    for tile in catan_map.tiles.values():
        if not isinstance(tile, Port):
            continue
        nodes = [tile.nodes[ref] for ref in PORT_DIRECTION_TO_NODEREFS[tile.direction]]
        spec = owner(midpoint(tuple(nodes)), protocol.EDGE_REFS,
                     protocol._EDGE_OFFSET, doubled=True)
        spec["type"] = (protocol.PORT_GENERIC if tile.resource is None
                        else card_of[tile.resource] + 1)
        ports[str(tile.id)] = spec

    return {"tileHexStates": hexes, "tileCornerStates": corners,
            "tileEdgeStates": edges, "portEdgeStates": ports}


def make_decoder():
    coords = protocol.build_coordinate_map(colonist_state_from_map(build_map("BASE")))
    return protocol._ActionDecoder(coords, {1: Color.BLUE, 2: Color.RED},
                                   our_color=Color.BLUE)


def test_a_colonist_board_translates_back_to_the_board_it_describes():
    """Tiles *and* ports, in catanatron's own ordering.

    Getting that ordering wrong is the failure that would not announce itself:
    the spec still validates as a BASE board, and the agent simply plays a
    different one than colonist dealt.
    """
    original = build_map("BASE")
    state = colonist_state_from_map(original)

    coords = protocol.build_coordinate_map(state)
    spec = protocol.board_spec_from_state(state, coords)

    assert spec == spec_from_map(original)
    assert spec_from_map(build_map_from_spec(spec)) == spec_from_map(original)


def test_a_board_whose_corners_collide_is_rejected():
    """Bijectivity is the property under test; matching counts is not enough."""
    state = colonist_state_from_map(build_map("BASE"))
    for corner in state["tileCornerStates"].values():
        corner["z"] = 0

    with pytest.raises(protocol.ProtocolError, match="corner"):
        protocol.build_coordinate_map(state)


def test_one_log_entry_can_hold_several_bank_trades():
    """Colonist merges a player's consecutive trades; the engine wants them apart."""
    decoder = make_decoder()
    decoder.feed({"gameLogState": {"0": {"text": {
        "type": protocol.LOG_BANK_TRADE, "playerColor": 1,
        "givenCardEnums": [2, 2, 2, 1, 1, 1], "receivedCardEnums": [4, 4]}}}})

    assert [a.value for a in decoder.actions] == [
        ("BRICK", "BRICK", "BRICK", None, "WHEAT"),
        ("WOOD", "WOOD", "WOOD", None, "WHEAT"),
    ]


def test_a_steal_against_us_names_us_as_the_victim():
    """``playerColor`` is always the *other* player, so the sides swap by entry."""
    decoder = make_decoder()
    decoder.feed({"mechanicRobberState": {"locationTileIndex": 0},
                  "gameLogState": {"0": {"text": {
                      "type": protocol.LOG_ROBBER_MOVED, "playerColor": 2,
                      "pieceEnum": protocol.PIECE_ROBBER}}}})
    decoder.feed({"gameLogState": {"0": {"text": {
        "type": protocol.LOG_ROBBED_BY_THEM, "playerColor": 2, "cardEnums": [3]}}}})

    action = decoder.actions[-1]
    assert action.color == Color.RED         # the thief moved the robber
    assert action.value[1] == Color.BLUE     # and robbed us
    assert action.value[2] == "SHEEP"


def test_a_played_card_reveals_the_purchase_that_bought_it():
    """Offline only: live, an opponent's card is unknown until it is played."""
    actions = [
        Action(Color.RED, ActionType.BUY_DEVELOPMENT_CARD, None),
        Action(Color.RED, ActionType.BUY_DEVELOPMENT_CARD, None),
        Action(Color.RED, ActionType.END_TURN, None),
        Action(Color.RED, ActionType.PLAY_KNIGHT_CARD, None),
    ]

    revealed = protocol.reveal_purchases(actions)

    assert revealed[0].value == "KNIGHT"
    assert revealed[1].value is None  # never revealed, so still unknown


def test_a_card_played_the_turn_it_was_bought_reveals_nothing():
    """Neither ruleset allows it, so such a purchase cannot be the source.

    Attributing it anyway builds a hand the engine refuses to play from, which
    is how this surfaced: an opponent who bought twenty cards and played sixteen
    desynced on a road building it demonstrably held.
    """
    actions = [
        Action(Color.RED, ActionType.BUY_DEVELOPMENT_CARD, None),
        Action(Color.RED, ActionType.PLAY_KNIGHT_CARD, None),
    ]

    assert protocol.reveal_purchases(actions)[0].value is None


@pytest.mark.skipif(not CAPTURES, reason="no colonist capture recorded locally")
def test_a_captured_game_replays_move_for_move():
    """Rung 0 of the verification ladder, on real traffic instead of self-play.

    Captures are personal browser sessions and live under the git-ignored
    ``data/``, so this runs only where one has been recorded. It is the
    strongest statement the bridge can make offline: every move colonist
    reported was legal in the reconstruction, at the moment it was made.
    """
    decoded = None
    for path in CAPTURES:
        try:
            decoded = protocol.decode_capture(path)
            break
        except protocol.ProtocolError:
            continue  # a capture of the lobby only, with no game in it
    if decoded is None:
        pytest.skip("recorded captures contain no complete game")

    replay = GameReplay(decoded.board, colors=decoded.seating, vps_to_win=15)

    replay.apply_many(decoded.actions)  # DesyncError if any move was mistranslated

    assert decoded.our_color in decoded.seating
    assert replay.state.num_turns > 20
    assert encode_observation(replay.game, decoded.our_color,
                              lookahead=True).shape == (642,)
