"""Tests for the custom 1v1 rule patches in ``src.env.rules``."""

import contextlib
import os
from unittest import mock

import numpy as np

from catanatron import Color
from catanatron.models.enums import Action, ActionType
import catanatron.state as catanatron_state
from catanatron.state_functions import player_key

from src.agent.encoding import action_size
from src.env.catan_env import VPS_TO_WIN, make_1v1_game
from src.env.rules import DISCARD_LIMIT
from src.env import ruleset


def test_action_space_expanded_to_one_discard_per_resource():
    # Stock catanatron-gym has 290 slots with a single (DISCARD, None); rules.py
    # replaces it with five, one per resource.
    assert action_size() == 294


def test_default_victory_point_target_is_eight():
    """The default ruleset, i.e. no CATAN_* variables set in the environment.

    The target is per-run now (``--vps-to-win`` / ``CATAN_VPS_TO_WIN``), so this
    pins the default rather than the only possible value.
    """
    assert VPS_TO_WIN == ruleset.VPS_TO_WIN == 8
    assert make_1v1_game().vps_to_win == 8


def test_victory_point_target_is_configurable():
    assert make_1v1_game(vps_to_win=15).vps_to_win == 15


def test_ruleset_reads_flags_off_the_command_line(monkeypatch):
    """``apply_cli_overrides`` is a pre-pass over argv, so test it as one.

    It sets environment variables rather than returning values, because the
    processes that need them are spawned later and inherit the environment.
    Both the ``--flag value`` and ``--flag=value`` spellings must work.
    """
    for form in (["--vps-to-win", "15", "--longest-road"],
                 ["--vps-to-win=15", "--longest-road"]):
        monkeypatch.delenv("CATAN_VPS_TO_WIN", raising=False)
        monkeypatch.delenv("CATAN_LONGEST_ROAD", raising=False)
        ruleset.apply_cli_overrides(form)
        assert os.environ["CATAN_VPS_TO_WIN"] == "15"
        assert os.environ["CATAN_LONGEST_ROAD"] == "1"

    ruleset.apply_cli_overrides(["--no-longest-road"])
    assert os.environ["CATAN_LONGEST_ROAD"] == "0"


def test_cli_overrides_rebind_the_module_constants(monkeypatch):
    """Setting the environment is not enough -- the parent must see it too.

    ``apply_cli_overrides`` is imported *from* this module, so the module's
    constants are read one statement before the override is applied. Without a
    re-read the new ruleset would reach spawned children (fresh interpreters,
    fresh import) but not the parent -- and the parent is the process that
    installs, or skips, the Longest Road patch.
    """
    original = (ruleset.VPS_TO_WIN, ruleset.MAX_TURNS, ruleset.LONGEST_ROAD_VP)
    try:
        monkeypatch.setenv("CATAN_VPS_TO_WIN", "8")
        ruleset.apply_cli_overrides(["--vps-to-win", "15", "--longest-road"])
        assert ruleset.VPS_TO_WIN == 15
        assert ruleset.LONGEST_ROAD_VP is True
        assert "15 VP" in ruleset.describe()
        assert ruleset.as_config()["vps_to_win"] == 15
    finally:
        (ruleset.VPS_TO_WIN, ruleset.MAX_TURNS,
         ruleset.LONGEST_ROAD_VP) = original


def test_discard_limit_is_nine():
    assert make_1v1_game().state.discard_limit == DISCARD_LIMIT == 9


def test_longest_road_patch_is_skipped_when_the_award_is_enabled():
    """``longest_road_vp=True`` must leave stock scoring alone.

    Every ``_patch_*`` is stubbed and the applied-once flag cleared, so this
    exercises the wiring without actually re-patching catanatron -- re-running
    the real patches inside a live test session would double-wrap
    ``Game.__init__`` and corrupt every test after it.
    """
    import src.env.rules as rules_mod

    applied = []
    names = [n for n in dir(rules_mod) if n.startswith("_patch_")]
    with mock.patch.object(rules_mod._game_mod, rules_mod._PATCH_FLAG, False,
                           create=True):
        with contextlib.ExitStack() as stack:
            for name in names:
                stack.enter_context(mock.patch.object(
                    rules_mod, name,
                    side_effect=lambda *a, _n=name, **k: applied.append(_n),
                ))
            rules_mod.apply_rule_patches(longest_road_vp=True)
    assert "_patch_no_longest_road" not in applied
    # The other six still go on -- only the road award is in question.
    assert "_patch_robber_placement" in applied
    assert "_patch_sequential_discard" in applied


def test_longest_road_awards_no_victory_points():
    """Building a 5-road chain must not move VPs or set HAS_ROAD.

    Asserts the *default* ruleset (``CATAN_LONGEST_ROAD`` unset).
    """
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


def test_every_discard_lands_in_the_action_log():
    """The patched discard has to log itself; upstream's ``apply_action`` did.

    The log is the record a game is reconstructed from -- the colonist.io bridge
    replays live positions off exactly this list. Discards that mutate five cards
    out of a hand without leaving a trace desync any replay the moment a 7 lands,
    and they do it silently.
    """
    game = make_1v1_game(seed=5)
    state = game.state
    while state.is_initial_build_phase:
        game.execute(state.playable_actions[0], validate_action=False)

    color = state.current_color()
    key = player_key(state, color)
    for resource in ("WOOD", "BRICK", "SHEEP", "WHEAT"):
        state.player_state[f"{key}_{resource}_IN_HAND"] = 3

    game.execute(Action(color, ActionType.ROLL, (3, 4)), validate_action=False)
    logged_before = len(state.actions)

    discarded = []
    while state.is_discarding:
        action = state.playable_actions[0]
        discarded.append(action.value)
        game.execute(action, validate_action=False)

    new = state.actions[logged_before:]
    assert [a.action_type for a in new] == [ActionType.DISCARD] * len(discarded)
    assert [a.value for a in new] == discarded
    assert all(a.color == color for a in new)


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


def test_the_third_point_lifts_the_protection_however_it_was_earned():
    """Points, not buildings — the distinction that cost a live game.

    An opponent still on their two opening settlements is protected right up
    until something gives them a third *visible* point, and Largest Army does
    that as surely as a settlement would. Reading the rule as a settlement count
    agrees everywhere colonist ever showed us its legal tiles and is still
    wrong: it protects a player colonist would let you rob.
    """
    from catanatron.models.actions import robber_possibilities
    from catanatron.state_functions import player_key

    game = make_1v1_game(seed=9)
    state = game.state
    while state.is_initial_build_phase:
        game.execute(state.playable_actions[0], validate_action=False)

    color = state.current_color()
    opponent = next(c for c in state.colors if c != color)
    opponent_nodes = {node for node, building in state.board.buildings.items()
                      if building[0] == opponent}

    # Largest Army, on the same two settlements they started with.
    state.player_state[f"{player_key(state, opponent)}_VICTORY_POINTS"] = 4

    reachable = [a for a in robber_possibilities(state, color)
                 if set(state.board.map.land_tiles[a.value[0]].nodes.values())
                 & opponent_nodes]
    assert reachable, "a four-point opponent is not protected"


def test_your_own_tiles_are_protected_on_the_same_terms():
    """The protection is the player's, not the mover's, so it covers you too."""
    from catanatron.models.actions import robber_possibilities
    from catanatron.state_functions import player_key

    game = make_1v1_game(seed=9)
    state = game.state
    while state.is_initial_build_phase:
        game.execute(state.playable_actions[0], validate_action=False)

    color = state.current_color()
    own_nodes = {node for node, building in state.board.buildings.items()
                 if building[0] == color}

    def can_reach_own():
        return [a for a in robber_possibilities(state, color)
                if set(state.board.map.land_tiles[a.value[0]].nodes.values())
                & own_nodes]

    assert not can_reach_own()  # two points, so off limits even to ourselves
    state.player_state[f"{player_key(state, color)}_VICTORY_POINTS"] = 3
    assert can_reach_own()


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


def test_largest_army_awards_two_victory_points():
    """Largest Army is deliberately left ON, unlike Longest Road.

    Worth pinning explicitly: this project patches the *other* +2 bonus out, so
    a reader could reasonably assume both go, and an over-eager patch would be
    caught here rather than silently changing what the agent is optimising.
    """
    import catanatron.state_functions as state_functions

    game, state, color, key = _start_turn_with_dev_card_money(seed=7)
    before = state.player_state[f"{key}_ACTUAL_VICTORY_POINTS"]

    for knights in range(1, 4):
        state_functions.player_clean_turn(state, color)
        state.player_state[f"{key}_KNIGHT_IN_HAND"] = 1
        state_functions.play_dev_card(state, color, "KNIGHT")
        # The award lands on the third knight, not before.
        expected = 2 if knights == 3 else 0
        gained = state.player_state[f"{key}_ACTUAL_VICTORY_POINTS"] - before
        assert gained == expected, f"after {knights} knight(s)"

    assert state.player_state[f"{key}_PLAYED_KNIGHT"] == 3
    assert state.player_state[f"{key}_HAS_ARMY"] is True


def test_bought_this_turn_counter_survives_state_copy():
    """MCTS copies mid-turn; a dropped counter would resurrect the old bug."""
    game, state, color, key = _start_turn_with_dev_card_money(seed=1)
    card = _buy_dev_card(game, state, color, key)
    if card is None:
        return  # drew a Victory Point card; nothing to assert

    copied = state.copy()
    field = f"{key}_{card}_BOUGHT_THIS_TURN"
    assert copied.player_state[field] == state.player_state[field] == 1


def _broke_player_holding_road_building(seed=3):
    """Past the opening, holding Road Building and nothing else."""
    game = make_1v1_game(seed=seed)
    state = game.state
    while state.is_initial_build_phase:
        game.execute(state.playable_actions[0], validate_action=False)

    color = state.current_color()
    key = player_key(state, color)
    state.player_state[f"{key}_HAS_ROLLED"] = True
    for resource in ("WOOD", "BRICK", "SHEEP", "WHEAT", "ORE"):
        state.player_state[f"{key}_{resource}_IN_HAND"] = 0
    state.player_state[f"{key}_ROAD_BUILDING_IN_HAND"] = 1
    # Looked up on the module, not imported by name: the rule patches rebind the
    # attribute, and this test file imports catanatron before src.env applies them.
    state.playable_actions = catanatron_state.generate_playable_actions(state)
    return game, state, color


def test_road_building_can_be_played_without_the_money_for_a_road():
    """The card gives *free* roads, so affording one cannot be the price of entry.

    Upstream gates ``PLAY_ROAD_BUILDING`` behind the same affordability check it
    uses to offer ordinary paid road builds, which locks the card away exactly
    when it is worth most. A colonist opponent played it with an empty hand and
    won on longest road; the reconstruction refused the move.
    """
    _, state, color = _broke_player_holding_road_building()

    assert Action(color, ActionType.PLAY_ROAD_BUILDING, None) in state.playable_actions


def test_road_building_places_both_roads_while_still_broke():
    """Getting in is not enough; the two free roads have to be placeable too."""
    game, state, color = _broke_player_holding_road_building()
    game.execute(Action(color, ActionType.PLAY_ROAD_BUILDING, None),
                 validate_action=False)

    roads = 0
    while state.is_road_building and state.free_roads_available > 0:
        road = next(a for a in state.playable_actions
                    if a.action_type == ActionType.BUILD_ROAD)
        game.execute(road, validate_action=False)
        roads += 1

    assert roads == 2
    key = player_key(state, color)
    assert state.player_state[f"{key}_WOOD_IN_HAND"] == 0  # and still free
    assert state.player_state[f"{key}_BRICK_IN_HAND"] == 0


def test_paid_road_building_still_requires_the_resources():
    """The affordability test is correct in its third use, so it must survive."""
    _, state, color = _broke_player_holding_road_building()

    assert not [a for a in state.playable_actions
                if a.action_type == ActionType.BUILD_ROAD]
