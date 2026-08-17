"""Tests for the opening-placement specialist."""

from pathlib import Path

import numpy as np
import pytest

from catanatron import Color
from catanatron.models.board import STATIC_GRAPH
from catanatron.models.enums import ActionType

from src.env.catan_env import make_1v1_game
from src.placement.dataset import flatten_pairs, generate_pair
from src.placement.features import (
    FEATURE_SLICES,
    NUMBERS,
    RESOURCES,
    candidate_nodes,
    encode_candidates,
    feature_size,
    node_features,
)
from src.placement.model import PlacementNet
from src.placement.player import PlacementPlayer


def test_feature_size_matches_encoder():
    game = make_1v1_game(seed=7)
    vector = node_features(game, Color.BLUE, candidate_nodes(
        game.state.playable_actions)[0])
    assert vector.shape == (feature_size(),)
    assert vector.dtype == np.float32


def test_features_encode_the_actual_tiles():
    """The production and histogram blocks must match the board, not a summary."""
    game = make_1v1_game(seed=11)
    catan_map = game.state.board.map
    node = candidate_nodes(game.state.playable_actions)[0]
    vector = node_features(game, Color.BLUE, node)

    start, stop = FEATURE_SLICES["production"]
    expected = [
        catan_map.node_production[node].get(r, 0.0) for r in RESOURCES
    ]
    assert vector[start:stop] == pytest.approx(expected, abs=1e-6)

    start, stop = FEATURE_SLICES["number_hist"]
    histogram = vector[start:stop]
    tiles = [t for t in catan_map.adjacent_tiles[node] if t.resource is not None]
    assert histogram.sum() == len(tiles)
    for tile in tiles:
        assert histogram[NUMBERS.index(tile.number)] >= 1.0


def test_placement_index_flips_on_the_second_settlement():
    """The context block must show the model which of the two picks this is."""
    game = make_1v1_game(seed=23)
    color = game.state.current_color()
    nodes = candidate_nodes(game.state.playable_actions)
    first = node_features(game, color, nodes[0])

    settle = next(a for a in game.state.playable_actions
                  if a.action_type == ActionType.BUILD_SETTLEMENT
                  and a.value == nodes[0])
    game.execute(settle, validate_action=False)

    later = node_features(game, color, nodes[-1])
    flag, _ = FEATURE_SLICES["placement_index"]
    assert first[flag] == 0.0
    assert later[flag] == 1.0


def test_encode_candidates_handles_the_empty_case():
    empty = encode_candidates(make_1v1_game(seed=7), Color.BLUE, [])
    assert empty.shape == (0, feature_size())


def test_generate_pair_shape_and_delta_range():
    pairs, deltas, bundles = generate_pair(7)
    assert pairs.shape == (1, 4, feature_size())
    assert deltas.shape == (1,)
    # Same two openings, re-encoded as bundles: [seat][corner][settlement].
    assert bundles.shape == (1, 2, 2, feature_size())
    # A bundle corner is exactly the settlement vector from the pair.
    assert np.array_equal(bundles[0, 0, 0], pairs[0, 0])
    assert np.array_equal(bundles[0, 1, 0], pairs[0, 2])
    # Its second corner is encoded with the first assumed built, so it carries
    # the second-settlement flag even though nothing is on the board yet.
    first, last = FEATURE_SLICES["placement_index"]
    assert bundles[0, 0, 1, first] == 1.0
    # One comparison per pair: how much better the first seat's two corners did
    # than the second seat's, on the same board with the same dice.
    assert abs(deltas[0]) in (0.0, 0.5, 1.0)


def test_flatten_pairs_opposes_the_two_bundles():
    pairs, deltas, _ = generate_pair(7)
    features, labels = flatten_pairs(pairs, deltas)
    assert features.shape == (4, feature_size())
    assert labels[0] == labels[1]
    assert labels[2] == labels[3]
    assert labels[0] == pytest.approx(-labels[2])


def test_generate_pair_is_deterministic():
    first = generate_pair(23)
    second = generate_pair(23)
    assert np.array_equal(first[0], second[0])
    assert np.array_equal(first[1], second[1])


def test_fixed_dice_gives_paired_games_the_same_rolls():
    from src.env.dice import fixed_dice
    from catanatron import state as catanatron_state

    original = catanatron_state.roll_dice
    with fixed_dice(11):
        first = [catanatron_state.roll_dice() for _ in range(20)]
    with fixed_dice(11):
        second = [catanatron_state.roll_dice() for _ in range(20)]
    assert first == second
    # And the patch is fully undone, including the module-global rebind that
    # apply_action resolves through.
    assert catanatron_state.roll_dice is original


def test_training_learns_a_planted_ordering():
    from src.placement.train import train

    rng = np.random.default_rng(0)
    dim = feature_size()
    # The first seat's bundle wins exactly when its first feature is larger.
    pairs = rng.normal(size=(1500, 4, dim)).astype(np.float32)
    strength = pairs[:, :, 0]
    deltas = np.sign(strength[:, :2].sum(1) - strength[:, 2:].sum(1)).astype(np.float32)

    net, _ = train(pairs, deltas, epochs=120, patience=0, progress=False)

    probe = np.zeros((9, dim), dtype=np.float32)
    probe[:, 0] = np.linspace(-2, 2, 9)
    assert np.all(np.diff(net.score(probe)) > 0)


def test_assume_owned_encodes_a_hypothetical_first_settlement():
    from src.placement.features import node_features

    game = make_1v1_game(seed=5)
    color = game.state.colors[0]
    catan_map = game.state.board.map
    first, second = sorted(catan_map.land_nodes)[:2]

    bare = node_features(game, color, second)
    with_first = node_features(game, color, second, assume_owned=(first,))

    flag, _ = FEATURE_SLICES["placement_index"]
    assert bare[flag] == 0.0 and with_first[flag] == 1.0

    # The assumed corner's production shows up as production we already hold.
    start, stop = FEATURE_SLICES["own_production"]
    assert not bare[start:stop].any()
    expected = np.array(
        [catan_map.node_production[first].get(r, 0.0) for r in RESOURCES],
        dtype=np.float32,
    )
    assert np.allclose(with_first[start:stop], expected)

    # Blocks that describe the candidate itself must not move.
    for block in ("production", "tile_counts", "number_hist", "ports"):
        lo, hi = FEATURE_SLICES[block]
        assert np.array_equal(bare[lo:hi], with_first[lo:hi])


def test_opening_swap_replays_the_same_roads():
    """Roads travel with their settlement, or the pair is not a controlled test."""
    from src.placement.dataset import _draft_order, _draft_roads, _play

    game_a, explorers = _play(7, "weighted", 7, dice_seed=7)
    order, roads = _draft_order(game_a), _draft_roads(game_a)
    assert len(order) == 4 and len(roads) == 4

    swap = [1, 0, 3, 2]
    nodes = [n for _, n in order]
    first_seat, second_seat = order[0][0], order[1][0]
    forced = {first_seat: [nodes[swap[0]], nodes[swap[3]]],
              second_seat: [nodes[swap[1]], nodes[swap[2]]]}
    forced_roads = {first_seat: [roads[swap[0]], roads[swap[3]]],
                    second_seat: [roads[swap[1]], roads[swap[2]]]}

    game_b, _ = _play(7, "weighted", 7, forced_by_color=forced, dice_seed=7,
                      forced_roads_by_color=forced_roads)
    # Same four openings, same four roads, opposite seats.
    assert sorted(_draft_roads(game_b)) == sorted(roads)
    assert sorted(n for _, n in _draft_order(game_b)) == sorted(nodes)


def test_bundle_chooser_picks_a_legal_complementary_pair():
    """Pair search must commit to a second corner, and a legal one.

    Roads are deliberately not part of this search -- a bundle over
    (settlement, road) x 2 measured 44.1% against this -- so the chooser has an
    opinion about corners only.
    """
    from src.placement.chooser import OpeningChooser
    from src.placement.model import BundleNet

    game = make_1v1_game(seed=11)
    color = game.state.colors[0]
    chooser = OpeningChooser(PlacementNet(), BundleNet(feature_dim=feature_size()),
                             first_k=4, partner_k=8)
    nodes = sorted(game.state.board.map.land_nodes)

    first = chooser.choose(game, color, nodes)
    assert first in nodes

    second = chooser._plan["second"]
    assert second != first
    assert second not in STATIC_GRAPH.neighbors(first)

    # Losing the planned partner re-searches rather than failing.
    remaining = [n for n in nodes
                 if n != first and n not in STATIC_GRAPH.neighbors(first)
                 and n != second]
    assert chooser.choose(game, color, remaining) in remaining


def test_bundle_net_round_trip_and_shape(tmp_path):
    from src.placement.model import BundleNet

    rng = np.random.default_rng(0)
    dim = feature_size()
    openings = rng.normal(size=(32, 2, dim)).astype(np.float32)

    net = BundleNet()
    net.fit_normalizer(openings)
    before = net.score(openings)
    assert before.shape == (32,)

    # Flat (N, 2F) and stacked (N, 2, F) must mean the same thing.
    assert np.allclose(before, net.score(openings.reshape(32, -1)))

    after = BundleNet.load(net.save(tmp_path / "bundle.pt")).score(openings)
    assert np.allclose(before, after)


def test_model_round_trip_preserves_scores(tmp_path):
    rng = np.random.default_rng(0)
    features = rng.normal(size=(64, feature_size())).astype(np.float32)

    net = PlacementNet()
    net.fit_normalizer(features)
    before = net.score(features)

    path = net.save(tmp_path / "scorer.pt")
    after = PlacementNet.load(path).score(features)
    # The normalisation buffers must travel with the weights, or inference
    # quietly disagrees with training about what the inputs mean.
    assert np.allclose(before, after)


def test_fit_normalizer_survives_constant_columns():
    features = np.ones((16, feature_size()), dtype=np.float32)
    net = PlacementNet()
    net.fit_normalizer(features)
    assert np.isfinite(net.score(features)).all()


def test_placement_player_only_touches_the_opening():
    """Non-opening decisions must reach the inner agent untouched."""
    from catanatron.players.weighted_random import WeightedRandomPlayer

    class CountingPlayer(WeightedRandomPlayer):
        calls = 0

        def decide(self, game, playable_actions):
            CountingPlayer.calls += 1
            return super().decide(game, playable_actions)

    inner = CountingPlayer(Color.BLUE)
    net = PlacementNet()
    player = PlacementPlayer(Color.BLUE, inner, net)

    game = make_1v1_game(players=[player, WeightedRandomPlayer(Color.RED)], seed=7)
    while game.state.is_initial_build_phase:
        game.play_tick()

    # Two settlements went to the scorer; the two initial roads went to inner.
    assert CountingPlayer.calls == 2
    settled = [
        node for node, (owner, _) in game.state.board.buildings.items()
        if owner == Color.BLUE
    ]
    assert len(settled) == 2


# --------------------------------------------------------------------------
# Gym-side wrapper
# --------------------------------------------------------------------------
def _scorer_env(tmp_path, **kwargs):
    """A placement env backed by a freshly-saved (untrained) scorer."""
    path = PlacementNet().save(tmp_path / "scorer.pt")
    from src.placement.env_wrapper import make_placement_env
    return make_placement_env(str(path), seed=0, **kwargs)


def test_wrapper_finishes_the_opening_during_reset(tmp_path):
    """The learner's first observation must be past the initial build phase."""
    env = _scorer_env(tmp_path)
    obs, info = env.reset(seed=1)

    game = env.unwrapped.game
    assert not game.state.is_initial_build_phase
    # Both seats placed two settlements each.
    assert len(game.state.board.buildings) == 4
    assert obs.shape == env.observation_space.shape
    assert len(info["valid_actions"]) > 0


def test_wrapper_records_only_its_own_two_picks(tmp_path):
    env = _scorer_env(tmp_path)
    env.reset(seed=2)

    color = env.unwrapped.p0.color
    owned = [
        node for node, (owner, _) in env.unwrapped.game.state.board.buildings.items()
        if owner == color
    ]
    assert len(env.opening_nodes) == 2
    assert sorted(env.opening_nodes) == sorted(owned)


def test_wrapper_clears_picks_between_episodes(tmp_path):
    env = _scorer_env(tmp_path)
    env.reset(seed=3)
    env.reset(seed=4)
    assert len(env.opening_nodes) == 2


def test_wrapper_leaves_a_playable_episode(tmp_path):
    """Post-reset the env must step normally -- no half-finished opening."""
    env = _scorer_env(tmp_path)
    env.reset(seed=5)

    rng = np.random.default_rng(0)
    for _ in range(20):
        valid = env.unwrapped.get_valid_actions()
        _, _, terminated, truncated, _ = env.step(int(rng.choice(valid)))
        if terminated or truncated:
            break


def test_opponent_scorer_flag_controls_the_enemy(tmp_path):
    from src.placement.player import PlacementPlayer as PP

    shared = _scorer_env(tmp_path)
    assert isinstance(shared.unwrapped.enemies[0], PP)

    solo = _scorer_env(tmp_path, opponent_scorer=False)
    assert not isinstance(solo.unwrapped.enemies[0], PP)


def test_trained_scorer_opens_far_better_than_chance():
    """End-to-end: the shipped checkpoint must pick genuinely good corners.

    Ranked by total node production -- a raw board fact that is already one of
    the input features, not a hand-written opinion about what makes a corner
    good. Ranked against the env's *own* board, because ``env.reset(seed=n)``
    does not control the layout (see the wrapper docstring).
    """
    checkpoint = Path("checkpoints/placement/scorer_ppo.pt")
    if not checkpoint.exists():
        pytest.skip("no trained placement scorer available")

    from src.placement.env_wrapper import make_placement_env

    env = make_placement_env(str(checkpoint), seed=0)
    ranks = []
    for seed in range(8):
        env.reset(seed=seed)
        catan_map = env.unwrapped.game.state.board.map
        # The first pick saw an empty board, so every land node was legal.
        nodes = sorted(catan_map.land_nodes)
        production = {
            n: sum(catan_map.node_production[n].values()) for n in nodes
        }
        chosen = env.opening_nodes[0]
        ranks.append(
            1 + sum(1 for n in nodes if production[n] > production[chosen])
        )

    # Chance is ~27/54. A scorer that learned anything lands in the top handful;
    # 12 leaves ample headroom for board variance.
    assert np.mean(ranks) < 12.0
