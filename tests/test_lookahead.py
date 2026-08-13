"""Tests for the two-roll dice lookahead features."""

import numpy as np
import pytest

from catanatron import Color
from catanatron.models.decks import SETTLEMENT_COST_FREQDECK

from src.env.catan_env import make_1v1_game
from src.env.lookahead import (
    DICE_PROBS,
    LOOKAHEAD_SIZE,
    RESOURCES,
    SEVEN,
    lookahead_features,
)
from src.env.rules import DISCARD_LIMIT

# Slice boundaries, mirroring the layout documented on lookahead_features.
MEAN_GAIN = slice(0, 5)
VAR_GAIN = slice(5, 10)
AFFORD_1 = slice(10, 14)
AFFORD_2 = slice(14, 18)
DISCARD_P, DISCARD_LOSS = 18, 19
HAND_1, HAND_2 = 20, 21
OPP_GAIN = slice(22, 27)
OPP_DISCARD_P = 27


def _advanced_game(seed=7, ticks=400):
    game = make_1v1_game(seed=seed)
    for _ in range(ticks):
        if game.winning_color() is not None:
            break
        game.play_tick()
    return game


def _set_hand(game, color, **amounts):
    key = f"P{game.state.color_to_index[color]}"
    for resource in RESOURCES:
        game.state.player_state[f"{key}_{resource}_IN_HAND"] = amounts.get(
            resource.lower(), 0
        )


def test_dice_probabilities_are_a_real_distribution():
    assert DICE_PROBS.sum() == pytest.approx(1.0)
    assert DICE_PROBS[SEVEN] == pytest.approx(6 / 36)


def test_vector_has_the_documented_size():
    vector = lookahead_features(_advanced_game(), Color.BLUE)
    assert vector.shape == (LOOKAHEAD_SIZE,)
    assert vector.dtype == np.float32
    assert np.isfinite(vector).all()


def test_probabilities_stay_in_range():
    vector = lookahead_features(_advanced_game(), Color.BLUE)
    for block in (AFFORD_1, AFFORD_2):
        assert (vector[block] >= 0.0).all()
        assert (vector[block] <= 1.0).all()
    assert 0.0 <= vector[DISCARD_P] <= 1.0
    assert 0.0 <= vector[OPP_DISCARD_P] <= 1.0


def test_only_a_second_roll_seven_can_cost_affordability():
    """Two rolls may dominate one by everything except the odds of a discard.

    An earlier form of this asserted plain dominance, on the reasoning that
    resources only accumulate. They do not: a 7 on the *second* roll discards,
    and that can drop a hand below a cost it could already pay. So the honest
    bound is dominance minus P(7) -- only that one branch of the grid subtracts,
    and it carries exactly that much probability. A discard leaking into any
    other branch puts more mass at risk and breaks this.
    """
    for seed in (7, 11, 23, 42):
        vector = lookahead_features(_advanced_game(seed=seed), Color.BLUE)
        floor = vector[AFFORD_1] - DICE_PROBS[SEVEN] - 1e-9
        assert (vector[AFFORD_2] >= floor).all()


def test_a_second_roll_seven_actually_discards():
    """The counterpart: prove the discard is applied, not merely bounded.

    The bound above still passes if the second-roll discard were dropped
    entirely, so this pins the other side with a case where it must bite. No
    buildings means no production, so neither roll can add anything and the
    discard is the only thing that moves the hand. 20 cards survive one halving
    with a road still affordable (1 wood, 1 brick) and fail the second, so the
    road odds must come out at exactly 1 - P(7)^2.
    """
    game = make_1v1_game(seed=7)   # untouched board: nobody has built yet
    _set_hand(game, Color.BLUE, wood=2, brick=2, sheep=16)
    vector = lookahead_features(game, Color.BLUE)

    road = 0  # index 0 of the cost block
    assert vector[AFFORD_1][road] == pytest.approx(1.0)
    assert vector[AFFORD_2][road] == pytest.approx(1.0 - DICE_PROBS[SEVEN] ** 2)


def test_already_affordable_reads_as_certain():
    """A hand that can already pay must show probability 1 at both horizons."""
    game = make_1v1_game(seed=7)
    _set_hand(game, Color.BLUE, wood=4, brick=4, sheep=4, wheat=4, ore=4)
    vector = lookahead_features(game, Color.BLUE)

    # Index 1 of the cost block is the settlement.
    assert vector[AFFORD_1][1] == pytest.approx(1.0)
    assert vector[AFFORD_2][1] == pytest.approx(1.0)
    assert list(SETTLEMENT_COST_FREQDECK) == [1, 1, 1, 1, 0]


def test_empty_hand_with_no_buildings_can_afford_nothing():
    game = make_1v1_game(seed=7)
    _set_hand(game, Color.BLUE)
    vector = lookahead_features(game, Color.BLUE)
    # No settlements yet, so no production can arrive either.
    assert (vector[AFFORD_1] == 0.0).all()
    assert (vector[AFFORD_2] == 0.0).all()
    assert vector[MEAN_GAIN].sum() == pytest.approx(0.0)


def test_discard_exposure_tracks_the_house_limit():
    """Below the limit there is no exposure; above it, exactly P(7)."""
    game = make_1v1_game(seed=7)

    _set_hand(game, Color.BLUE, wood=DISCARD_LIMIT)
    safe = lookahead_features(game, Color.BLUE)
    assert safe[DISCARD_P] == pytest.approx(0.0)
    assert safe[DISCARD_LOSS] == pytest.approx(0.0)

    _set_hand(game, Color.BLUE, wood=DISCARD_LIMIT + 3)
    exposed = lookahead_features(game, Color.BLUE)
    assert exposed[DISCARD_P] == pytest.approx(6 / 36)
    # Half the hand, rounded down, weighted by P(7).
    assert exposed[DISCARD_LOSS] == pytest.approx((6 / 36) * ((DISCARD_LIMIT + 3) // 2))


def test_opponent_block_uses_only_public_information():
    """Opponent features must respond to their board and card count, nothing else.

    Their hand *composition* is not in the observation and must not leak in here:
    changing which resources they hold, at a constant count, must not move any
    feature.
    """
    game = _advanced_game(seed=11)
    before = lookahead_features(game, Color.BLUE)

    key = f"P{game.state.color_to_index[Color.RED]}"
    total = sum(game.state.player_state[f"{key}_{r}_IN_HAND"] for r in RESOURCES)
    # Same number of cards, entirely different composition.
    for resource in RESOURCES:
        game.state.player_state[f"{key}_{resource}_IN_HAND"] = 0
    game.state.player_state[f"{key}_WOOD_IN_HAND"] = total

    after = lookahead_features(game, Color.BLUE)
    assert np.allclose(before, after)


def test_opponent_discard_flag_responds_to_their_card_count():
    game = make_1v1_game(seed=7)
    _set_hand(game, Color.RED, wood=DISCARD_LIMIT)
    assert lookahead_features(game, Color.BLUE)[OPP_DISCARD_P] == pytest.approx(0.0)

    _set_hand(game, Color.RED, wood=DISCARD_LIMIT + 1)
    assert lookahead_features(game, Color.BLUE)[OPP_DISCARD_P] == pytest.approx(6 / 36)


def test_variance_is_non_negative():
    for seed in (7, 11, 23):
        vector = lookahead_features(_advanced_game(seed=seed), Color.BLUE)
        assert (vector[VAR_GAIN] >= 0.0).all()


def test_expected_hand_grows_over_the_horizon():
    game = _advanced_game(seed=23)
    vector = lookahead_features(game, Color.BLUE)
    assert vector[HAND_2] >= vector[HAND_1] - 1e-6


def test_robber_suppresses_the_tile_it_sits_on():
    """Production must drop when the robber is moved onto a producing tile."""
    game = _advanced_game(seed=42)
    board = game.state.board

    producing = None
    for node_id, (owner, _) in board.buildings.items():
        if owner != Color.BLUE:
            continue
        for tile in board.map.adjacent_tiles[node_id]:
            if tile.resource is not None:
                producing = tile
                break
        if producing:
            break
    if producing is None:
        pytest.skip("no producing tile adjacent to a BLUE building")

    coordinate = next(c for c, t in board.map.land_tiles.items()
                      if t.id == producing.id)

    # Park the robber off BLUE's tiles first. Measuring the baseline wherever it
    # happened to start makes the test depend on the board layout: if it already
    # sat on a tile BLUE touches, moving it *frees* that tile and production can
    # rise instead of fall.
    blue_tiles = {
        tile.id
        for node_id, (owner, _) in board.buildings.items() if owner == Color.BLUE
        for tile in board.map.adjacent_tiles[node_id]
    }
    neutral = next((c for c, t in board.map.land_tiles.items()
                    if t.id not in blue_tiles), None)
    if neutral is None:
        pytest.skip("every land tile touches a BLUE building")

    board.robber_coordinate = neutral
    before = lookahead_features(game, Color.BLUE)[MEAN_GAIN].sum()
    board.robber_coordinate = coordinate
    after = lookahead_features(game, Color.BLUE)[MEAN_GAIN].sum()

    assert after < before


def test_wrapper_extends_the_observation():
    from src.env.catan_env import make_1v1_env
    from src.env.lookahead import LookaheadWrapper

    base = make_1v1_env()
    base_dim = base.observation_space.shape[0]

    env = LookaheadWrapper(base)
    assert env.observation_space.shape[0] == base_dim + LOOKAHEAD_SIZE

    obs, _ = env.reset(seed=3)
    assert obs.shape == (base_dim + LOOKAHEAD_SIZE,)

    valid = env.unwrapped.get_valid_actions()
    obs, _, _, _, _ = env.step(int(valid[0]))
    assert obs.shape == (base_dim + LOOKAHEAD_SIZE,)
    assert np.isfinite(obs).all()
