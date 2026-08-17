"""Tests for the colonist.io bridge core -- rung 0 of the verification ladder.

Nothing here touches colonist.io. The claim under test is the one everything
else in Phase 3 rests on: an observed stream of actions, replayed into a
reconstructed board, produces a game the agent's own encoders cannot tell from
the original. A silent mismatch there looks like a weak agent, not like a bug,
so it is pinned before any protocol work starts.
"""

import numpy as np
import pytest

from catanatron import Color
from catanatron.models.actions import Action
from catanatron.models.enums import ActionType
from catanatron.models.map import build_map
from catanatron.players.weighted_random import WeightedRandomPlayer

from src.agent.encoding import encode_observation
from src.bridge.board import BoardSpec, build_map_from_spec, spec_from_map
from src.bridge.player import build_bridge_player
from src.bridge.replay import DesyncError, GameReplay, blank_outcome
from src.env.catan_env import make_1v1_game

COLORS = (Color.BLUE, Color.RED)


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
