"""Tests for the self-play opponent pool and the Elo ladder."""

import random

import pytest

from catanatron import Color
from catanatron.players.weighted_random import WeightedRandomPlayer

from src.agent import pool as opponent_pool
from src.agent.elo import Ladder, MAX_DELTA, elo_delta


def _fake_checkpoint(directory, step: int):
    """A stand-in for a saved model: the pool only ever copies bytes."""
    path = directory / f"agent_step_{step:08d}.zip"
    path.write_bytes(b"not-a-real-model")
    return path


# --------------------------------------------------------------------------
# Elo arithmetic
# --------------------------------------------------------------------------
def test_even_score_means_equal_strength():
    assert elo_delta(0.5, 100) == pytest.approx(0.0)


def test_known_elo_differences():
    # The textbook anchors: 76% ~ 200 points, 64% ~ 100.
    assert elo_delta(0.76, 1000) == pytest.approx(200, abs=2)
    assert elo_delta(0.64, 1000) == pytest.approx(100, abs=2)


def test_delta_is_antisymmetric():
    assert elo_delta(0.75, 200) == pytest.approx(-elo_delta(0.25, 200))


def test_clean_sweep_is_bounded_by_sample_size():
    """A 100-0 result is evidence of *at least* some gap, not an infinite one."""
    small = elo_delta(1.0, 20)
    large = elo_delta(1.0, 400)
    assert 0 < small < large <= MAX_DELTA
    assert elo_delta(0.0, 100) == -elo_delta(1.0, 100)


def test_zero_games_is_rejected():
    with pytest.raises(ValueError):
        elo_delta(0.5, 0)


# --------------------------------------------------------------------------
# Ladder
# --------------------------------------------------------------------------
def test_ladder_round_trips_through_disk(tmp_path):
    ladder = Ladder(tmp_path / "ladder.json")
    ladder.seed("pool/pool_step_00001000.zip", 1000)
    ladder.add("pool/pool_step_00002000.zip", 120.0, 2000)

    reloaded = Ladder.load(tmp_path / "ladder.json")
    assert len(reloaded) == 2
    assert reloaded.top().elo == 120.0
    assert reloaded.top().step == 2000


def test_seeding_is_idempotent(tmp_path):
    ladder = Ladder(tmp_path / "ladder.json")
    ladder.seed("a.zip", 100)
    ladder.seed("b.zip", 200)
    assert len(ladder) == 1
    assert ladder.top().elo == 0.0


def test_top_is_the_strongest_not_the_newest(tmp_path):
    """Ratings can go down; the reference must not follow them down."""
    ladder = Ladder(tmp_path / "ladder.json")
    ladder.seed("a.zip", 0)
    ladder.add("b.zip", 150.0, 100)
    ladder.add("c.zip", 80.0, 200)
    assert ladder.top().path == "b.zip"


def test_empty_ladder_has_no_reference(tmp_path):
    with pytest.raises(ValueError):
        Ladder(tmp_path / "ladder.json").top()


# --------------------------------------------------------------------------
# Pool
# --------------------------------------------------------------------------
def test_pool_entries_survive_checkpoint_pruning(tmp_path):
    """The whole point of a separate directory: pruning must not touch it."""
    from src.agent.checkpoint_manager import prune_checkpoints

    for step in (1000, 2000, 3000, 4000):
        source = _fake_checkpoint(tmp_path, step)
        opponent_pool.add_to_pool(source, step, tmp_path)

    prune_checkpoints(tmp_path, keep_n=1)
    assert len(opponent_pool.list_pool(tmp_path)) == 4


def test_thinning_keeps_the_oldest_and_the_newest(tmp_path):
    """Coverage of history is the point; a sliding window would defeat it."""
    for step in range(1000, 11000, 1000):
        opponent_pool.add_to_pool(
            _fake_checkpoint(tmp_path, step), step, tmp_path, max_size=4
        )

    steps = [int(p.stem.split("_")[-1]) for p in opponent_pool.list_pool(tmp_path)]
    assert len(steps) == 4
    assert steps[0] == 1000, "origin checkpoint was thinned away"
    assert steps[-1] == 10000, "newest checkpoint was thinned away"


def test_thinning_never_deletes_a_ladder_anchor(tmp_path):
    """A deleted anchor would leave the ladder pointing at nothing."""
    anchor = opponent_pool.add_to_pool(
        _fake_checkpoint(tmp_path, 1000), 1000, tmp_path
    )
    second = opponent_pool.add_to_pool(
        _fake_checkpoint(tmp_path, 2000), 2000, tmp_path
    )
    for step in range(3000, 9000, 1000):
        opponent_pool.add_to_pool(
            _fake_checkpoint(tmp_path, step), step, tmp_path, max_size=3,
            protected={str(anchor), str(second)},
        )

    survivors = {str(p) for p in opponent_pool.list_pool(tmp_path)}
    assert str(anchor) in survivors
    assert str(second) in survivors


def test_empty_pool_falls_back_to_the_scripted_bot(tmp_path):
    enemies = opponent_pool.sample_enemies(8, tmp_path, weighted_frac=0.1)
    assert len(enemies) == 8
    assert all(isinstance(e, WeightedRandomPlayer) for e in enemies)


def test_mixture_keeps_one_scripted_env_when_the_fraction_rounds_to_zero(tmp_path):
    """0.1 * 8 = 0.8 rounds to 0; the fixed reference must survive that."""
    from src.agent.opponent import PolicyPlayer

    for step in (1000, 2000):
        opponent_pool.add_to_pool(_fake_checkpoint(tmp_path, step), step, tmp_path)

    enemies = opponent_pool.sample_enemies(
        8, tmp_path, weighted_frac=0.1, rng=random.Random(0)
    )
    scripted = [e for e in enemies if isinstance(e, WeightedRandomPlayer)]
    policies = [e for e in enemies if isinstance(e, PolicyPlayer)]
    assert len(scripted) == 1
    assert len(policies) == 7
    assert all(e.color == Color.RED for e in enemies)


def test_zero_fraction_means_no_scripted_games(tmp_path):
    opponent_pool.add_to_pool(_fake_checkpoint(tmp_path, 1000), 1000, tmp_path)
    enemies = opponent_pool.sample_enemies(
        4, tmp_path, weighted_frac=0.0, rng=random.Random(0)
    )
    assert not any(isinstance(e, WeightedRandomPlayer) for e in enemies)


def test_pool_opponents_are_stochastic_by_default(tmp_path):
    """A greedy opponent plays one line per position and is easy to overfit to."""
    opponent_pool.add_to_pool(_fake_checkpoint(tmp_path, 1000), 1000, tmp_path)
    enemies = opponent_pool.sample_enemies(
        4, tmp_path, weighted_frac=0.0, rng=random.Random(0)
    )
    assert all(not e.deterministic for e in enemies)


def test_policy_player_pickles_by_path_not_by_weights(tmp_path):
    """SubprocVecEnv pickles each env's opponent; shipping torch weights per env
    would push ~6 MB per worker through the pipe on every opponent swap."""
    import pickle

    from src.agent.opponent import PolicyPlayer

    player = PolicyPlayer(Color.RED, model_path=tmp_path / "model.zip")
    blob = pickle.dumps(player)
    restored = pickle.loads(blob)

    assert restored.policy is None
    assert restored.model_path == str(tmp_path / "model.zip")
    # ~12 KB of that is the 614-entry feature-name list; what matters is that
    # it is three orders of magnitude short of a ~6 MB checkpoint.
    assert len(blob) < 50_000


def test_policy_player_needs_a_policy_or_a_path():
    from src.agent.opponent import PolicyPlayer

    with pytest.raises(ValueError):
        PolicyPlayer(Color.RED)
