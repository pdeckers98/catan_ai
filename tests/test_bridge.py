"""Tests for the colonist.io bridge core -- rung 0 of the verification ladder.

Nothing here touches colonist.io. The claim under test is the one everything
else in Phase 3 rests on: an observed stream of actions, replayed into a
reconstructed board, produces a game the agent's own encoders cannot tell from
the original. A silent mismatch there looks like a weak agent, not like a bug,
so it is pinned before any protocol work starts.
"""

import base64
import json
import pathlib
import random

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
from src.bridge import protocol, sender
from src.bridge.capture import decode_payload, message_type, shape
from src.bridge.player import build_bridge_player
from src.bridge.replay import DesyncError, GameReplay, blank_outcome
from src.bridge.session import LiveGame, LobbyMismatch, determinize_purchases
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


def public_vps(state):
    """Victory points from buildings and awards -- everything that is on the board."""
    return {
        color: state.player_state[f"P{state.color_to_index[color]}_VICTORY_POINTS"]
        for color in COLORS
    }


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
    three_to_one = {str(card): 3 for card in range(1, 6)}
    decoder.feed({
        "playerStates": {"1": {"bankTradeRatiosState": three_to_one}},
        "gameLogState": {"0": {"text": {
            "type": protocol.LOG_BANK_TRADE, "playerColor": 1,
            "givenCardEnums": [2, 2, 2, 1, 1, 1], "receivedCardEnums": [4, 4]}}}})

    assert [a.value for a in decoder.actions] == [
        ("BRICK", "BRICK", "BRICK", None, "WHEAT"),
        ("WOOD", "WOOD", "WOOD", None, "WHEAT"),
    ]


def test_a_port_ratio_decides_how_many_trades_a_run_of_cards_is():
    """Six bricks is two 3:1 trades or three 2:1 ones; only the ratio says which.

    This is the bug the first live game died of. The decoder assumed one
    received card per run of given cards, which cannot express a 2:1 port at
    all, and the resulting hand was wrong for the remaining 177 actions.
    """
    decoder = make_decoder()
    decoder.feed({"playerStates": {"1": {"bankTradeRatiosState": {"2": 2}}},
                  "gameLogState": {"0": {"text": {
                      "type": protocol.LOG_BANK_TRADE, "playerColor": 1,
                      "givenCardEnums": [2] * 6, "receivedCardEnums": [4, 4, 5]}}}})

    assert [a.value for a in decoder.actions] == [
        ("BRICK", "BRICK", None, None, "WHEAT"),
        ("BRICK", "BRICK", None, None, "WHEAT"),
        ("BRICK", "BRICK", None, None, "ORE"),
    ]


def test_a_ratio_diff_only_names_the_resource_that_changed():
    """A new port arrives as ``{'2': 2}``; the other four rates must survive."""
    decoder = make_decoder()
    three_to_one = {str(card): 3 for card in range(1, 6)}
    decoder.feed({"playerStates": {"1": {"bankTradeRatiosState": three_to_one}}})
    decoder.feed({"playerStates": {"1": {"bankTradeRatiosState": {"2": 2}}},
                  "gameLogState": {"0": {"text": {
                      "type": protocol.LOG_BANK_TRADE, "playerColor": 1,
                      "givenCardEnums": [2, 2, 1, 1, 1], "receivedCardEnums": [4, 5]}}}})

    assert [a.value for a in decoder.actions] == [
        ("BRICK", "BRICK", None, None, "WHEAT"),
        ("WOOD", "WOOD", "WOOD", None, "ORE"),
    ]


def test_a_trade_that_does_not_divide_by_the_ratio_is_refused():
    """Five cards at 3:1 is not a shape the log can mean; better loud than guessed."""
    decoder = make_decoder()
    with pytest.raises(protocol.ProtocolError):
        decoder.feed({"playerStates": {"1": {"bankTradeRatiosState": {"2": 3}}},
                      "gameLogState": {"0": {"text": {
                          "type": protocol.LOG_BANK_TRADE, "playerColor": 1,
                          "givenCardEnums": [2] * 5, "receivedCardEnums": [4]}}}})


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


def test_a_steal_reports_that_it_rewrote_the_robber_move():
    """The one place the decoder edits an action it already handed out.

    Harmless when a whole capture is decoded at once and fatal when it is not:
    a live replay has already *applied* that robber move, so unless the rewrite
    is announced it plays on from a position where the wrong card changed hands.
    """
    decoder = make_decoder()
    appended = decoder.feed({"mechanicRobberState": {"locationTileIndex": 0},
                             "gameLogState": {"0": {"text": {
                                 "type": protocol.LOG_ROBBER_MOVED, "playerColor": 2,
                                 "pieceEnum": protocol.PIECE_ROBBER}}}})
    revised = decoder.feed({"gameLogState": {"0": {"text": {
        "type": protocol.LOG_ROBBED_BY_THEM, "playerColor": 2, "cardEnums": [3]}}}})

    assert appended is None      # the robber move only appended
    assert revised == 0          # the steal rewrote it in place


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
    _, decoded = first_capture_with_a_game()
    replay = GameReplay(decoded.board, colors=decoded.seating, vps_to_win=15)

    # Determinized rather than fed raw: a purchase left as None is drawn by the
    # engine from its own shuffled deck, and about one shuffle in six starves a
    # later revealed card and dies inside draw_from_listdeck. That failure is
    # the deck's luck, not a mistranslation, and this test is about the latter.
    actions = determinize_purchases(decoded.actions, random.Random(0))

    replay.apply_many(actions)  # DesyncError if any move was mistranslated

    assert decoded.our_color in decoded.seating
    assert replay.state.num_turns > 20
    assert encode_observation(replay.game, decoded.our_color,
                              lookahead=True).shape == (642,)


# --------------------------------------------------------------------------
# Live session -- rung 1's engine, driven offline
# --------------------------------------------------------------------------
def first_capture_with_a_game():
    """The first recording that holds a complete game, or a skip."""
    for path in CAPTURES:
        try:
            return path, protocol.decode_capture(path)
        except protocol.ProtocolError:
            continue  # a capture of the lobby only
    pytest.skip("recorded captures contain no complete game")


def feed_capture(live, path):
    for record in protocol.iter_colonist_frames(path):
        live.feed_frame(record)
    return live


def test_the_message_decoder_reads_a_capture_the_way_the_file_reader_does():
    """The refactor's whole claim: one state machine, two ways of feeding it.

    ``decode_capture`` is now a loop over ``MessageDecoder``, so this would be
    trivially true -- except that it is what lets the offline tests stand in for
    the live path, and that is worth stating rather than assuming.
    """
    path, expected = first_capture_with_a_game()

    decoder = protocol.MessageDecoder()
    for kind, payload in protocol.iter_server_messages(path):
        if kind == protocol.MSG_FULL_STATE and decoder.started:
            break
        decoder.feed(kind, payload)

    assert decoder.decoded(reveal=True) == expected
    assert decoder.settings["maxPlayers"] == 2


@pytest.mark.skipif(not CAPTURES, reason="no colonist capture recorded locally")
def test_a_live_feed_reaches_the_state_a_whole_capture_replays_to():
    """Rung 1's core claim: fed one message at a time, the game comes out the same.

    The offline replay gets the whole tape and back-fills every dev card from
    what was later played. The live feed gets the same messages in order and has
    to guess the ones nobody has revealed yet.

    They must agree on everything *public* -- turn count, winner, and the
    victory points that come from buildings and awards. They are allowed to
    disagree on the hidden ones, and do: a purchase nobody ever revealed is a
    card both sides are guessing at, and the engine's own random draw is no
    better a guess than ours. The gap is bounded by the number of such
    purchases, which is the honest statement of what determinization costs.
    """
    path, expected = first_capture_with_a_game()
    offline = GameReplay(expected.board, colors=expected.seating, vps_to_win=15)
    offline.apply_many(determinize_purchases(expected.actions, random.Random(1)))

    live = feed_capture(LiveGame(vps_to_win=15, seed=0, check_lobby=False), path)

    assert live.replay.state.num_turns == offline.state.num_turns
    assert live.replay.winning_color() == offline.winning_color()
    assert public_vps(live.replay.state) == public_vps(offline.state)
    assert live.repairs == 0  # attribution should get there first, every time

    hidden = sum(abs(vps(live.replay.state)[c] - vps(offline.state)[c])
                 for c in COLORS)
    assert hidden <= len(live._guesses)


@pytest.mark.skipif(not CAPTURES, reason="no colonist capture recorded locally")
def test_every_purchase_is_given_a_card_even_when_nobody_revealed_it():
    """The engine cannot advance on ``None``, so the guess is not optional."""
    path, _ = first_capture_with_a_game()

    live = feed_capture(LiveGame(vps_to_win=15, seed=0, check_lobby=False), path)

    buys = [a for a in live._filled
            if a.action_type == ActionType.BUY_DEVELOPMENT_CARD]
    assert buys and all(a.value is not None for a in buys)
    # And some of them really were guesses -- otherwise this proves nothing.
    assert live._guesses
    assert live.rebuilds > 1  # a revealed purchase invalidated an applied one


@pytest.mark.skipif(not CAPTURES, reason="no colonist capture recorded locally")
def test_the_worst_possible_guess_still_replays_the_whole_game():
    """Every unknown card guessed as a victory point -- the least likely draw.

    Not a curiosity: it is the bound on how badly the determinization can go
    wrong. Attribution re-derives each purchase the moment it is played, so the
    guessing never accumulates, and the replay survives a determinizer that is
    wrong on purpose.
    """
    path, _ = first_capture_with_a_game()

    class Poisoned(LiveGame):
        def _draw(self, remaining):
            if remaining.get("VICTORY_POINT", 0) > 0:
                return "VICTORY_POINT"
            return super()._draw(remaining)

    live = feed_capture(Poisoned(vps_to_win=15, seed=0, check_lobby=False), path)

    assert live.replay.state.num_turns > 20


@pytest.mark.skipif(not CAPTURES, reason="no colonist capture recorded locally")
def test_a_desync_that_guessing_cannot_explain_is_not_repaired():
    """Only a dev-card play can be illegal because of a wrong guess.

    The first live game desynced 177 times on a mistranslated port trade and
    answered each one by redrawing the deck eight times: 1416 rebuilds, none of
    which could have helped, all of which hid the real bug. A desync on
    anything but a dev-card play now raises the first time it happens.
    """
    path, _ = first_capture_with_a_game()

    class Broken(LiveGame):
        """Refuses every action, as a mistranslation eventually would."""

        def _apply_pending(self):
            if self._applied < len(self._filled):
                raise DesyncError("pretend the translation is wrong")

    live = Broken(vps_to_win=15, seed=0, check_lobby=False)
    with pytest.raises(DesyncError):
        feed_capture(live, path)

    assert live.repairs == 0


def test_a_lobby_that_is_not_our_ruleset_is_refused():
    """The silent divergences are the dangerous ones, so they are checked.

    A different victory target never makes a single move illegal; it just means
    the policy is playing to the wrong finish line for the whole game.
    """
    live = LiveGame(vps_to_win=15)
    live.decoder.settings = {"victoryPointsToWin": 10, "cardDiscardLimit": 9,
                             "maxPlayers": 2, "friendlyRobber": True}

    with pytest.raises(LobbyMismatch, match="victoryPointsToWin"):
        live._check_lobby()


# --------------------------------------------------------------------------
# The action sender. The codec half is pure and can be held to the strictest
# standard available offline: the bytes a real client actually put on the wire.
# --------------------------------------------------------------------------


def client_frames():
    """Every in-game frame our own client sent, across all local captures."""
    frames = []
    for path in CAPTURES:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                entry = json.loads(line)
                if entry.get("dir") == "sent" and "header" in entry:
                    frames.append(entry)
    return frames


def test_a_routing_header_round_trips_through_its_parts():
    header = sender.RoutingHeader.parse(bytes.fromhex("030106303246373037"))

    assert (header.kind, header.channel, header.room) == (3, 1, "02F707")
    assert header.encode() == bytes.fromhex("030106303246373037")


def test_a_header_whose_length_disagrees_with_its_name_is_refused():
    """The length byte is the only self-check the header carries; use it."""
    with pytest.raises(sender.SendError, match="length"):
        sender.RoutingHeader.parse(b"	" + b"02F707")


@pytest.mark.skipif(not CAPTURES, reason="no colonist capture recorded locally")
def test_synthesized_frames_are_byte_identical_to_the_client_s_own():
    """The strongest offline statement the sender can make.

    A frame that merely *decodes* the same proves our reading is consistent.
    Identical bytes prove the server has nothing to distinguish our frame from
    the client's -- which is the entire question rung 2 asks.
    """
    in_game = [f for f in client_frames()
               if bytes.fromhex(f["header"])[0] == sender.ROOM_GAME]
    assert in_game, "no in-game client frames in the local captures"

    # Lobby frames are excluded rather than fixed: one of them carries a msgpack
    # Timestamp extension that the *capture* stringifies on its way to JSON, so
    # it cannot round-trip through a recording -- and we never send one.
    mismatches = sender.frames_round_trip(in_game)

    assert mismatches == [], mismatches[:3]


@pytest.mark.skipif(not CAPTURES, reason="no colonist capture recorded locally")
def test_the_codec_learns_its_routing_by_watching_the_client():
    """Nothing about the room is hardcoded; it is observed, like the rest."""
    codec = sender.FrameCodec()
    assert not codec.ready

    for entry in client_frames():
        codec.observe(entry)

    assert codec.ready
    assert codec.header is not None and codec.header.kind == sender.ROOM_GAME
    assert codec.last_sequence and codec.last_sequence > 0


def test_the_codec_will_not_speak_before_it_has_listened():
    """A guessed header would route a move nowhere, silently."""
    with pytest.raises(sender.SendError, match="no game frame observed"):
        sender.FrameCodec().build(sender.SEND_ROLL, True)


def test_the_lobby_is_not_mistaken_for_a_game_room():
    codec = sender.FrameCodec()
    codec.observe({"dir": "sent", "header": bytes(b"lobby").hex(),
                   "payload": {"action": 1, "payload": {}, "sequence": 3}})

    assert not codec.ready


def test_each_frame_takes_the_next_sequence_number():
    """The client has no idea we consumed one, which is worth being able to see."""
    codec = sender.FrameCodec()
    codec.observe({"dir": "sent", "header": bytes(b"" + b"02F707").hex(),
                   "payload": {"action": 6, "payload": True, "sequence": 41}})

    codec.build(sender.SEND_ROLL, True)
    codec.build(sender.SEND_END_TURN, True)

    assert codec.last_sequence == 43
    assert [record["sequence"] for record in codec.sent] == [42, 43]
