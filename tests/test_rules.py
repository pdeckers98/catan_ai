"""Tests for the custom 1v1 rule patches in ``src.env.rules``."""

import numpy as np

from catanatron import Color
from catanatron.models.enums import Action, ActionType
from catanatron.state_functions import player_key

from src.agent.encoding import action_size
from src.env.catan_env import VPS_TO_WIN, make_1v1_game
from src.env.rules import DISCARD_LIMIT


def test_action_space_expanded_to_one_discard_per_resource():
    # Stock catanatron-gym has 290 slots with a single (DISCARD, None); rules.py
    # replaces it with five, one per resource.
    assert action_size() == 294


def test_games_are_played_to_seven_points():
    assert VPS_TO_WIN == 7
    assert make_1v1_game().vps_to_win == 7


def test_discard_limit_is_nine():
    assert make_1v1_game().state.discard_limit == DISCARD_LIMIT == 9


def test_longest_road_awards_no_victory_points():
    """Building a 5-road chain must not move VPs or set HAS_ROAD."""
    game = make_1v1_game(seed=11)
    state = game.state
    color = state.current_color()
    key = player_key(state, color)

    before = state.player_state[f"{key}_ACTUAL_VICTORY_POINTS"]

    # Drive the initial build phase, which lays settlements and roads for free.
    while state.is_initial_build_phase:
        game.execute(state.playable_actions[0], validate_action=False)

    assert state.player_state[f"{key}_HAS_ROAD"] is False
    # Every VP so far came from the two initial settlements, never from roads.
    gained = state.player_state[f"{key}_ACTUAL_VICTORY_POINTS"] - before
    assert gained == 2


def test_longest_road_length_is_still_tracked_over_a_full_game():
    """Only the VP award is suppressed; the length feature stays in the vector.

    Note the initial build phase never calls ``mantain_longest_road`` upstream, so
    lengths only start moving once roads are built in normal play -- hence a whole
    game rather than just the setup.
    """
    from catanatron.players.weighted_random import WeightedRandomPlayer

    game = make_1v1_game(
        players=[WeightedRandomPlayer(Color.BLUE), WeightedRandomPlayer(Color.RED)],
        seed=21,
    )
    game.play()
    state = game.state

    lengths = [
        state.player_state[f"P{i}_LONGEST_ROAD_LENGTH"]
        for i in range(len(state.colors))
    ]
    has_road = [
        state.player_state[f"P{i}_HAS_ROAD"] for i in range(len(state.colors))
    ]
    assert any(length > 0 for length in lengths)
    assert not any(has_road)


def test_discard_remaining_survives_state_copy():
    """The MCTS-critical fix: a mid-discard copy keeps its remaining quota."""
    game = make_1v1_game(seed=3)
    state = game.state
    color = Color.BLUE
    key = player_key(state, color)

    # Hand the player 12 cards so a 7 forces a 6-card discard.
    for resource in ("WOOD", "BRICK", "SHEEP", "WHEAT"):
        state.player_state[f"{key}_{resource}_IN_HAND"] = 3

    state._discard_remaining = {color: 4}
    copied = state.copy()

    assert getattr(copied, "_discard_remaining", None) == {color: 4}
    # And it must be a distinct dict, or a search branch would corrupt its parent.
    copied._discard_remaining[color] = 1
    assert state._discard_remaining[color] == 4


def test_sequential_discard_drops_exactly_half_the_hand():
    """A 7 with 12 cards must cost exactly 6, one chosen card at a time."""
    game = make_1v1_game(seed=5)
    state = game.state
    while state.is_initial_build_phase:
        game.execute(state.playable_actions[0], validate_action=False)

    color = state.current_color()
    key = player_key(state, color)
    for resource in ("WOOD", "BRICK", "SHEEP", "WHEAT"):
        state.player_state[f"{key}_{resource}_IN_HAND"] = 3
    total_before = sum(
        state.player_state[f"{key}_{r}_IN_HAND"]
        for r in ("WOOD", "BRICK", "SHEEP", "WHEAT", "ORE")
    )
    assert total_before == 12

    # Force a 7 by injecting the dice roll.
    game.execute(Action(color, ActionType.ROLL, (3, 4)), validate_action=False)
    assert state.is_discarding

    steps = 0
    while state.is_discarding and steps < 20:
        action = state.playable_actions[0]
        assert action.action_type == ActionType.DISCARD
        assert action.value is not None  # a specific resource, not "random half"
        game.execute(action, validate_action=False)
        steps += 1

    total_after = sum(
        state.player_state[f"{key}_{r}_IN_HAND"]
        for r in ("WOOD", "BRICK", "SHEEP", "WHEAT", "ORE")
    )
    assert steps == 6
    assert total_after == 6


def test_robber_cannot_camp_opponent_before_third_settlement():
    """Colonist.io 1v1 restriction: no robbing a freshly-set-up opponent."""
    from catanatron.models.actions import robber_possibilities

    game = make_1v1_game(seed=9)
    state = game.state
    while state.is_initial_build_phase:
        game.execute(state.playable_actions[0], validate_action=False)

    color = state.current_color()
    opponent = next(c for c in state.colors if c != color)
    actions = robber_possibilities(state, color)

    # Straight after setup neither player qualifies, so no action may target a
    # tile touching an opponent building.
    opponent_nodes = {
        node
        for node, building in state.board.buildings.items()
        if building[0] == opponent
    }
    for action in actions:
        tile = state.board.map.land_tiles[action.value[0]]
        assert not (set(tile.nodes.values()) & opponent_nodes)


def test_observation_and_mask_shapes_agree():
    from src.agent.encoding import (
        encode_observation, legal_action_mask, obs_size,
    )

    game = make_1v1_game(seed=1)
    obs = encode_observation(game, game.state.current_color())
    mask = legal_action_mask(game.state.playable_actions)

    assert obs.shape == (obs_size(),)
    assert mask.shape == (action_size(),)
    assert mask.sum() == len(set(
        a for a in np.arange(action_size())[mask]
    ))
    assert mask.any()


# --------------------------------------------------------------------------
# Development card rules
# --------------------------------------------------------------------------
def _start_turn_with_dev_card_money(seed):
    """Advance past initial placement and hand the mover a dev card's worth."""
    game = make_1v1_game(seed=seed)
    state = game.state
    while state.is_initial_build_phase:
        game.execute(state.playable_actions[0], validate_action=False)

    color = state.current_color()
    key = player_key(state, color)
    state.player_state[f"{key}_HAS_ROLLED"] = True
    for resource in ("SHEEP", "WHEAT", "ORE"):
        state.player_state[f"{key}_{resource}_IN_HAND"] = 1
    return game, state, color, key


def _buy_dev_card(game, state, color, key):
    """Buy one card; returns its name, or None if it was a Victory Point card.

    Victory Point cards are never *played* -- buying one scores it immediately --
    so they are not subject to the same-turn restriction and the caller skips
    them. Which card the deck yields is seed-dependent, hence the None return.
    """
    playable = ("KNIGHT", "MONOPOLY", "ROAD_BUILDING", "YEAR_OF_PLENTY")
    before = {c: state.player_state[f"{key}_{c}_IN_HAND"] for c in playable}
    game.execute(
        Action(color, ActionType.BUY_DEVELOPMENT_CARD, None), validate_action=False
    )
    for card in playable:
        if state.player_state[f"{key}_{card}_IN_HAND"] > before[card]:
            return card
    return None


def test_development_card_cannot_be_played_on_the_turn_it_was_bought():
    """Stock catanatron tracks no purchase time, so a knight was insta-playable.

    Asserted against ``playable_actions`` rather than the predicate, because that
    is what the agent actually sees -- and because ``player_can_play_dev`` is
    imported by name into two other modules, so a patch that missed a binding
    would still pass a predicate-level check.
    """
    import catanatron.state_functions as state_functions

    checked = 0
    for seed in range(12):
        game, state, color, key = _start_turn_with_dev_card_money(seed)
        card = _buy_dev_card(game, state, color, key)
        if card is None:
            continue

        assert state.player_state[f"{key}_{card}_IN_HAND"] >= 1
        offered = [
            a for a in state.playable_actions
            if a.action_type.name.startswith("PLAY_")
        ]
        assert offered == [], f"{card} was playable the turn it was bought"

        # ...and becomes available once the turn ends.
        state_functions.player_clean_turn(state, color)
        assert state_functions.player_can_play_dev(state, color, card)
        checked += 1
        if checked == 3:
            break

    assert checked, "no playable dev card was drawn in 12 seeds"


def test_only_one_development_card_may_be_played_per_turn():
    """Enforced upstream via HAS_PLAYED_DEVELOPMENT_CARD_IN_TURN; pin it."""
    import catanatron.state_functions as state_functions

    game, state, color, key = _start_turn_with_dev_card_money(seed=3)
    state.player_state[f"{key}_KNIGHT_IN_HAND"] = 1
    state.player_state[f"{key}_MONOPOLY_IN_HAND"] = 1

    assert state_functions.player_can_play_dev(state, color, "KNIGHT")
    state_functions.play_dev_card(state, color, "KNIGHT")
    assert not state_functions.player_can_play_dev(state, color, "MONOPOLY")

    state_functions.player_clean_turn(state, color)
    assert state_functions.player_can_play_dev(state, color, "MONOPOLY")


def test_largest_army_awards_no_victory_points():
    """Three knights must not move VPs or set HAS_ARMY; the count still rises."""
    import catanatron.state_functions as state_functions

    game, state, color, key = _start_turn_with_dev_card_money(seed=7)
    before = state.player_state[f"{key}_ACTUAL_VICTORY_POINTS"]

    for _ in range(3):
        state_functions.player_clean_turn(state, color)
        state.player_state[f"{key}_KNIGHT_IN_HAND"] = 1
        state_functions.play_dev_card(state, color, "KNIGHT")

    assert state.player_state[f"{key}_PLAYED_KNIGHT"] == 3
    assert state.player_state[f"{key}_HAS_ARMY"] is False
    assert state.player_state[f"{key}_ACTUAL_VICTORY_POINTS"] == before


def test_bought_this_turn_counter_survives_state_copy():
    """MCTS copies mid-turn; a dropped counter would resurrect the old bug."""
    game, state, color, key = _start_turn_with_dev_card_money(seed=1)
    card = _buy_dev_card(game, state, color, key)
    if card is None:
        return  # drew a Victory Point card; nothing to assert

    copied = state.copy()
    field = f"{key}_{card}_BOUGHT_THIS_TURN"
    assert copied.player_state[field] == state.player_state[field] == 1
