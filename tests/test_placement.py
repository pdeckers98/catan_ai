"""Tests for the opening-placement specialist."""

import numpy as np
import pytest

from catanatron import Color
from catanatron.models.enums import ActionType

from src.env.catan_env import make_1v1_game
from src.placement.dataset import generate_pair
from src.placement.evaluate import diagnose
from src.placement.features import (
    FEATURE_SLICES,
    NUMBERS,
    RESOURCES,
    candidate_nodes,
    encode_candidates,
    feature_size,
    node_features,
)
from src.placement.heuristic import pip_score, rank_of, spearman
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


def test_heuristic_ranks_a_richer_corner_higher():
    """Sanity check on the yardstick itself, not on any learned model."""
    game = make_1v1_game(seed=42)
    catan_map = game.state.board.map
    nodes = candidate_nodes(game.state.playable_actions)
    best = max(nodes, key=lambda n: pip_score(catan_map, n))
    assert rank_of(catan_map, best, nodes) == 1


def test_spearman_endpoints():
    assert spearman([1, 2, 3], [1, 2, 3]) == pytest.approx(1.0)
    assert spearman([1, 2, 3], [3, 2, 1]) == pytest.approx(-1.0)
    # A constant column has no ranking to correlate with.
    assert spearman([1, 2, 3], [5, 5, 5]) == 0.0


def test_generate_pair_labels_are_paired_and_opposed():
    features, labels = generate_pair(7)
    assert features.shape == (4, feature_size())
    # Two samples per seat, and the seats' labels are exact opposites: the pair
    # measures one opening against the other on the same board.
    assert labels[0] == labels[1]
    assert labels[2] == labels[3]
    assert labels[0] == pytest.approx(-labels[2])
    assert set(np.abs(labels)) <= {0.0, 0.5, 1.0}


def test_generate_pair_is_deterministic():
    first = generate_pair(23)
    second = generate_pair(23)
    assert np.array_equal(first[0], second[0])
    assert np.array_equal(first[1], second[1])


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


def test_diagnose_reports_a_rank_within_range():
    result = diagnose(PlacementNet(), seeds=range(900_000, 900_004))
    assert result["boards"] == 4
    assert 1.0 <= result["mean_rank"] <= result["mean_candidates"]
    assert -1.0 <= result["mean_rho"] <= 1.0
