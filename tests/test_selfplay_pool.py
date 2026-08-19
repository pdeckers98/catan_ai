"""Tests for the self-play opponent pool and the Elo ladder."""

import random

import pytest

from catanatron import Color
from catanatron.players.search import VictoryPointPlayer
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


def test_empty_pool_falls_back_to_the_scripted_bots(tmp_path):
    """Before the first checkpoint exists there is nothing else to play.

    The fallback splits the envs between the two scripted bots rather than
    handing them all to weighted-random, or the opening interval of a
    from-scratch run would train against a single opponent by accident.
    """
    enemies = opponent_pool.sample_enemies(8, tmp_path, weighted_frac=0.1,
                                           greedy_frac=0.1)
    assert len(enemies) == 8
    assert all(isinstance(e, (WeightedRandomPlayer, VictoryPointPlayer))
               for e in enemies)
    assert sum(isinstance(e, VictoryPointPlayer) for e in enemies) == 4


def test_mixture_keeps_one_env_per_scripted_bot_when_the_fraction_rounds_to_zero(
        tmp_path):
    """0.1 * 8 = 0.8 rounds to 0; both fixed references must survive that."""
    from src.agent.opponent import PolicyPlayer

    for step in (1000, 2000):
        opponent_pool.add_to_pool(_fake_checkpoint(tmp_path, step), step, tmp_path)

    enemies = opponent_pool.sample_enemies(
        8, tmp_path, weighted_frac=0.1, greedy_frac=0.1, rng=random.Random(0)
    )
    weighted = [e for e in enemies if isinstance(e, WeightedRandomPlayer)]
    greedy = [e for e in enemies if isinstance(e, VictoryPointPlayer)]
    policies = [e for e in enemies if isinstance(e, PolicyPlayer)]
    assert len(weighted) == 1
    assert len(greedy) == 1
    assert len(policies) == 6
    assert all(e.color == Color.RED for e in enemies)


def test_zero_fraction_means_no_scripted_games(tmp_path):
    opponent_pool.add_to_pool(_fake_checkpoint(tmp_path, 1000), 1000, tmp_path)
    enemies = opponent_pool.sample_enemies(
        4, tmp_path, weighted_frac=0.0, greedy_frac=0.0, rng=random.Random(0)
    )
    assert not any(
        isinstance(e, (WeightedRandomPlayer, VictoryPointPlayer)) for e in enemies
    )


def test_scripted_slices_never_crowd_out_the_whole_batch(tmp_path):
    """Over-subscribed fractions must still return exactly ``num_envs``."""
    opponent_pool.add_to_pool(_fake_checkpoint(tmp_path, 1000), 1000, tmp_path)
    enemies = opponent_pool.sample_enemies(
        2, tmp_path, weighted_frac=0.9, greedy_frac=0.9, rng=random.Random(0)
    )
    assert len(enemies) == 2


def test_pool_opponents_are_stochastic_by_default(tmp_path):
    """A greedy opponent plays one line per position and is easy to overfit to."""
    opponent_pool.add_to_pool(_fake_checkpoint(tmp_path, 1000), 1000, tmp_path)
    enemies = opponent_pool.sample_enemies(
        4, tmp_path, weighted_frac=0.0, greedy_frac=0.0, rng=random.Random(0)
    )
    assert all(not e.deterministic for e in enemies)


def test_only_trained_opponents_are_given_the_placement_scorer(tmp_path):
    """The scorer follows the opponent type, not the run configuration.

    A checkpoint opponent has an untrained placement head and would throw the
    opening away without it. A scripted bot is in the mixture as a fixed
    difficulty reference, and is no longer that reference if it opens like a
    specialist.
    """
    from src.agent.opponent import PolicyPlayer
    from src.agent.train import _opponent_opens_with_scorer

    assert not _opponent_opens_with_scorer(WeightedRandomPlayer(Color.RED))
    assert not _opponent_opens_with_scorer(VictoryPointPlayer(Color.RED))
    assert _opponent_opens_with_scorer(
        PolicyPlayer(Color.RED, model_path=str(_fake_checkpoint(tmp_path, 1000)))
    )


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


# --------------------------------------------------------------------------
# Match telemetry
# --------------------------------------------------------------------------
def test_knights_played_is_reported_per_match():
    """Knights are the Largest Army tell, and Largest Army is the only +2 left."""
    from src.agent.arena import AgentSpec, play_match

    result = play_match(
        AgentSpec(kind="weighted"), AgentSpec(kind="random"), 6, seed=4,
    )
    assert result.mean_knights >= 0
    assert "knights" in result.summary()


def test_knights_counts_played_cards_not_bought_ones():
    """An unplayed knight scores nothing and must not be counted."""
    from catanatron.state_functions import player_key
    from src.agent.arena import _player_stats
    from src.env.catan_env import make_1v1_game

    game = make_1v1_game(seed=2)
    state = game.state
    color = state.current_color()
    key = player_key(state, color)

    state.player_state[f"{key}_KNIGHT_IN_HAND"] = 3
    assert _player_stats(state, color)["knights"] == 0

    state.player_state[f"{key}_PLAYED_KNIGHT"] = 2
    assert _player_stats(state, color)["knights"] == 2


def test_knights_diff_is_the_margin_over_the_opponent(tmp_path):
    """Largest Army is a race: 4 knights means nothing if the opponent played 5."""
    from src.agent.arena import MatchResult

    ahead = MatchResult(games=10, mean_knights=4.0, mean_opp_knights=1.5)
    behind = MatchResult(games=10, mean_knights=4.0, mean_opp_knights=5.0)

    assert ahead.knights_diff == pytest.approx(2.5)
    assert behind.knights_diff == pytest.approx(-1.0)
    # Same absolute count, opposite standing -- which is the whole point.
    assert ahead.mean_knights == behind.mean_knights


def test_knights_diff_is_shown_with_a_sign(tmp_path):
    from src.agent.arena import MatchResult

    result = MatchResult(games=4, wins=2, mean_knights=3.0, mean_opp_knights=1.0)
    assert "3.0 knights (+2.0)" in result.summary()


def test_pool_draws_cover_distinct_checkpoints(tmp_path):
    """Six envs and a pool of eleven should mean six *different* opponents.

    Drawing independently per env is uniform but wastes slots: it averages 4.8
    distinct of 6, so a PPO batch sees less of the history the pool exists to
    keep.
    """
    from src.agent.opponent import PolicyPlayer

    for step in range(200000, 2400000, 200000):
        opponent_pool.add_to_pool(_fake_checkpoint(tmp_path, step), step, tmp_path)

    enemies = opponent_pool.sample_enemies(
        8, tmp_path, weighted_frac=0.05, greedy_frac=0.05, rng=random.Random(0)
    )
    policies = [e for e in enemies if isinstance(e, PolicyPlayer)]
    assert len(policies) == 6
    assert len({p.model_path for p in policies}) == 6


def test_a_pool_smaller_than_the_batch_still_fills_every_env(tmp_path):
    """Early in a run the pool cannot cover the slots; duplicates are correct."""
    from src.agent.opponent import PolicyPlayer

    for step in (1000, 2000):
        opponent_pool.add_to_pool(_fake_checkpoint(tmp_path, step), step, tmp_path)

    enemies = opponent_pool.sample_enemies(
        8, tmp_path, weighted_frac=0.0, greedy_frac=0.0, rng=random.Random(0)
    )
    assert len(enemies) == 8
    assert all(isinstance(e, PolicyPlayer) for e in enemies)
    assert len({e.model_path for e in enemies}) == 2


def test_draws_stay_uniform_over_the_whole_pool(tmp_path):
    """Sampling must not drift toward recent checkpoints -- that is what makes
    self-play cycle, and it is the failure this module was written to avoid."""
    from collections import Counter
    from pathlib import Path

    from src.agent.opponent import PolicyPlayer

    steps = list(range(200000, 2400000, 200000))
    for step in steps:
        opponent_pool.add_to_pool(_fake_checkpoint(tmp_path, step), step, tmp_path)

    rng = random.Random(0)
    seen = Counter()
    for _ in range(1000):
        enemies = opponent_pool.sample_enemies(
            8, tmp_path, weighted_frac=0.05, greedy_frac=0.05, rng=rng
        )
        seen.update(Path(e.model_path).stem for e in enemies
                    if isinstance(e, PolicyPlayer))

    assert len(seen) == len(steps), "some checkpoint was never drawn"
    total = sum(seen.values())
    expected = 1.0 / len(steps)
    for name, count in seen.items():
        assert abs(count / total - expected) < 0.02, f"{name} is over/under-drawn"


# --------------------------------------------------------------------------
# The searching opponent slice
# --------------------------------------------------------------------------
def test_search_is_off_unless_asked_for(tmp_path):
    """The default pool is what every archived checkpoint trained against."""
    from src.agent.opponent import SearchPlayer

    for step in (1000, 2000, 3000):
        opponent_pool.add_to_pool(_fake_checkpoint(tmp_path, step), step,
                                  tmp_path, 25)
    enemies = opponent_pool.sample_enemies(
        8, tmp_path, weighted_frac=0.1, rng=random.Random(0), greedy_frac=0.1
    )
    assert not any(isinstance(e, SearchPlayer) for e in enemies)


def test_a_searching_slice_only_eats_pool_envs(tmp_path):
    """The scripted bots are fixed difficulty references and stay scripted.

    Only the pool slice can search -- there is nothing to search *with* for a
    scripted bot, and converting one would remove the fixed reference the
    mixture keeps it for.
    """
    from src.agent.opponent import PolicyPlayer, SearchPlayer

    for step in (1000, 2000, 3000):
        opponent_pool.add_to_pool(_fake_checkpoint(tmp_path, step), step,
                                  tmp_path, 25)
    enemies = opponent_pool.sample_enemies(
        8, tmp_path, weighted_frac=0.1, rng=random.Random(0), greedy_frac=0.1,
        search_frac=0.5, search_simulations=7,
    )
    assert len(enemies) == 8
    assert sum(isinstance(e, WeightedRandomPlayer) for e in enemies) == 1
    assert sum(isinstance(e, VictoryPointPlayer) for e in enemies) == 1
    searching = [e for e in enemies if isinstance(e, SearchPlayer)]
    heads = [e for e in enemies if isinstance(e, PolicyPlayer)]
    assert len(searching) + len(heads) == 6
    assert len(searching) == 3
    assert all(e.simulations == 7 for e in searching)


def test_a_fraction_that_rounds_to_zero_still_gets_one_searching_env(tmp_path):
    """Same rule the scripted slices follow, for a sharper reason: this is the
    only opponent in the run stronger than the learner's own policy head, and
    rounding it away would silently return the pool to self-play."""
    from src.agent.opponent import SearchPlayer

    opponent_pool.add_to_pool(_fake_checkpoint(tmp_path, 1000), 1000,
                              tmp_path, 25)
    enemies = opponent_pool.sample_enemies(
        8, tmp_path, weighted_frac=0.1, rng=random.Random(0), greedy_frac=0.1,
        search_frac=0.05,
    )
    assert sum(isinstance(e, SearchPlayer) for e in enemies) == 1


def test_the_searching_opponent_pickles_by_path_too(tmp_path):
    """It holds an evaluator wrapping a live torch model once built, so it has
    to be built in the worker and not before."""
    import pickle

    from src.agent.opponent import SearchPlayer

    player = SearchPlayer(Color.RED, tmp_path / "model.zip", simulations=10)
    restored = pickle.loads(pickle.dumps(player))

    assert restored._player is None
    assert restored.model_path == str(tmp_path / "model.zip")
    assert restored.simulations == 10


def test_a_searching_opponent_is_given_the_placement_scorer(tmp_path):
    """It is a trained checkpoint, so the argument in
    ``test_only_trained_opponents_are_given_the_placement_scorer`` applies."""
    from src.agent.opponent import SearchPlayer
    from src.agent.train import _opponent_opens_with_scorer

    assert _opponent_opens_with_scorer(
        SearchPlayer(Color.RED, _fake_checkpoint(tmp_path, 1000))
    )


def test_the_log_line_says_how_many_are_searching(tmp_path):
    from src.agent.opponent import SearchPlayer

    enemies = [
        WeightedRandomPlayer(Color.RED),
        SearchPlayer(Color.RED, tmp_path / "pool_step_00001000.zip",
                     simulations=7),
    ]
    line = opponent_pool.describe(enemies)
    assert "1 of them searching at 7 sims" in line
    # The searching entry is a pool checkpoint and belongs in the step list.
    assert "00001000" in line
