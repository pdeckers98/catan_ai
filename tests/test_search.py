"""Tests for the PUCT search that ``--agent ppo-mcts`` runs on."""

import numpy as np

from catanatron import Color

from src.agent.encoding import action_size, obs_size
from src.agent.evaluator import Evaluator, UniformEvaluator
from src.agent.mcts import MCTS, MCTSPlayer, _outcome_key
from src.env.catan_env import make_1v1_game


class NoisyEvaluator(Evaluator):
    """A stand-in for a trained net: non-flat priors and non-zero values.

    UniformEvaluator is degenerate in exactly the ways some of these assertions
    care about -- every prior equal, every value 0 -- so the bookkeeping tests
    use this instead to make sure nothing is passing by accident.
    """

    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)

    def evaluate_batch(self, obs_batch, mask_batch):
        mask = np.asarray(mask_batch)
        priors = self.rng.random(mask.shape).astype(np.float32) * mask
        priors /= np.maximum(priors.sum(axis=-1, keepdims=True), 1e-9)
        values = self.rng.uniform(-1.0, 1.0, len(mask)).astype(np.float32)
        return priors, values


def test_search_returns_a_normalized_policy_over_legal_actions():
    game = make_1v1_game(seed=2)
    mcts = MCTS(NoisyEvaluator(), simulations=32)
    result = mcts.search(game)

    assert np.isclose(result.policy.sum(), 1.0)
    assert np.allclose(result.policy[~result.mask], 0.0)
    assert result.visits.sum() == 32
    assert -1.0 <= result.value <= 1.0
    assert result.best_action() in game.state.playable_actions


def test_batched_search_spends_exactly_the_simulation_budget():
    """Virtual loss must net out: every charge is undone by its own backup.

    If the undo were wrong the visit counts would drift, silently corrupting the
    policy target the network is trained on.
    """
    for batch_size in (1, 4, 16, 64):
        game = make_1v1_game(seed=2)
        result = MCTS(
            NoisyEvaluator(seed=1), simulations=48, batch_size=batch_size
        ).search(game, rng=np.random.default_rng(0))

        assert result.visits.sum() == 48, f"batch_size={batch_size}"
        assert np.isclose(result.policy.sum(), 1.0)
        assert np.allclose(result.policy[~result.mask], 0.0)
        assert -1.0 <= result.value <= 1.0


def test_batched_search_still_finds_a_spiked_prior():
    """Batching must not destroy the search's ability to concentrate."""
    class SpikedEvaluator(UniformEvaluator):
        def evaluate_batch(self, obs_batch, mask_batch):
            priors, values = super().evaluate_batch(obs_batch, mask_batch)
            spike = np.argmax(mask_batch, axis=-1)
            priors[np.arange(len(priors)), spike] += 5.0
            priors /= priors.sum(axis=-1, keepdims=True)
            return priors, values

    game = make_1v1_game(seed=2)
    result = MCTS(
        SpikedEvaluator(), simulations=200, batch_size=16
    ).search(game, rng=np.random.default_rng(0))

    assert result.visits.argmax() == 0
    assert result.visits[0] > result.visits.sum() / len(result.actions)


def test_search_does_not_mutate_the_game_it_searches():
    game = make_1v1_game(seed=4)
    before_turns = game.state.num_turns
    before_actions = len(game.state.actions)

    MCTS(UniformEvaluator(), simulations=24).search(game)

    assert game.state.num_turns == before_turns
    assert len(game.state.actions) == before_actions


def test_dice_outcomes_key_chance_children_by_sum():
    """(2, 5) and (3, 4) are the same event and must share a child."""
    from catanatron.models.enums import Action, ActionType

    first = Action(Color.BLUE, ActionType.ROLL, (2, 5))
    second = Action(Color.BLUE, ActionType.ROLL, (3, 4))
    assert _outcome_key(first) == _outcome_key(second) == 7


def test_deterministic_actions_have_no_chance_key():
    from catanatron.models.enums import Action, ActionType

    assert _outcome_key(Action(Color.BLUE, ActionType.END_TURN, None)) is None
    assert _outcome_key(Action(Color.BLUE, ActionType.BUILD_ROAD, (1, 2))) is None


def test_puct_concentrates_visits_on_a_spiked_prior():
    """Search must follow the prior, not spread uniformly over 50+ legal moves.

    An untrained network gives a nearly flat prior and a nearly constant value, so
    a real net proves nothing here -- the evaluator is spiked deliberately.
    """

    class SpikedEvaluator(UniformEvaluator):
        def evaluate_batch(self, obs_batch, mask_batch):
            priors, values = super().evaluate_batch(obs_batch, mask_batch)
            for row, mask in enumerate(mask_batch):
                legal = np.flatnonzero(mask)
                priors[row] *= 0.1
                priors[row, legal[0]] = 0.9
                priors[row] /= priors[row].sum()
            return priors, values

    game = make_1v1_game(seed=6)
    result = MCTS(SpikedEvaluator(), simulations=200).search(game)

    favoured = int(np.argmax(result.visits))
    assert favoured == 0
    assert result.visits[favoured] / result.visits.sum() > 3.0 / len(result.actions)


def test_mcts_player_skips_search_on_forced_moves():
    """A single legal action must be returned without touching the evaluator."""

    class ExplodingEvaluator(UniformEvaluator):
        def evaluate_batch(self, obs_batch, mask_batch):
            raise AssertionError("evaluator called on a forced move")

    game = make_1v1_game(seed=8)
    player = MCTSPlayer(Color.BLUE, ExplodingEvaluator(), simulations=4)
    only = game.state.playable_actions[:1]

    assert player.decide(game, only) is only[0]


# --------------------------------------------------------------------------
# Lookahead features across the search boundary
# --------------------------------------------------------------------------
def test_encode_observation_appends_the_lookahead_features():
    from src.agent.encoding import encode_observation
    from src.env.lookahead import LOOKAHEAD_SIZE

    game = make_1v1_game(seed=3)
    plain = encode_observation(game, Color.BLUE)
    wide = encode_observation(game, Color.BLUE, lookahead=True)

    assert len(plain) == obs_size()
    assert len(wide) == obs_size() + LOOKAHEAD_SIZE
    assert np.array_equal(wide[:obs_size()], plain), "base features shifted"


def test_search_feeds_the_evaluator_the_width_it_asks_for():
    """The search builds observations; only the evaluator knows the net's shape.

    A 642-trained checkpoint used to receive 614 values here and crash -- or
    worse, be silently misread -- because the requirement never travelled from
    the evaluator to the encoder.
    """
    from src.agent.evaluator import Evaluator
    from src.env.lookahead import LOOKAHEAD_SIZE

    class WidthRecorder(Evaluator):
        wants_lookahead = True

        def __init__(self):
            self.widths = []

        def evaluate_batch(self, obs_batch, mask_batch):
            self.widths.append(np.asarray(obs_batch).shape[-1])
            n = np.asarray(mask_batch).shape[0]
            return (np.ones((n, action_size()), dtype=np.float32),
                    np.zeros(n, dtype=np.float32))

    evaluator = WidthRecorder()
    game = make_1v1_game(seed=5)
    MCTSPlayer(Color.BLUE, evaluator, simulations=8).decide(
        game, game.state.playable_actions
    )

    assert evaluator.widths, "evaluator was never called"
    assert set(evaluator.widths) == {obs_size() + LOOKAHEAD_SIZE}


def test_an_evaluator_that_does_not_want_lookahead_still_gets_the_base_width():
    """The default must stay 614 -- an evaluator opts in, it is not assumed."""
    evaluator = UniformEvaluator()
    assert not getattr(evaluator, "wants_lookahead", False)
    assert MCTS(evaluator, simulations=2)._lookahead is False


# --------------------------------------------------------------------------
# Compressed roll outcomes
# --------------------------------------------------------------------------
def test_dead_dice_sums_share_one_chance_key():
    """Every sum that pays nobody must collapse into a single pooled outcome.

    This is the whole point of the compression: without it a board with five
    unproductive numbers scatters five identical positions across five children,
    each costing its own network call and each left with one visit.
    """
    from src.agent.mcts import DEAD_ROLL, _roll_distribution
    from src.env.lookahead import DICE_PROBS, DICE_SUMS, live_dice_sums

    game = make_1v1_game(seed=7)
    live = live_dice_sums(game)
    keys, pairs, probs = _roll_distribution(game)

    live_sums = [total for index, total in enumerate(DICE_SUMS) if live[index]]
    assert keys[:len(live_sums)] == live_sums
    assert keys.count(DEAD_ROLL) == (0 if live.all() else 1)
    assert len(keys) == len(live_sums) + (0 if live.all() else 1)

    # Probabilities come off the fixed table, and the pooled entry carries
    # exactly the mass of the sums it stands for.
    for index, total in enumerate(live_sums):
        assert np.isclose(probs[index], DICE_PROBS[DICE_SUMS.index(total)])
    if not live.all():
        assert np.isclose(probs[-1], DICE_PROBS[~live].sum())
    assert np.isclose(probs.sum(), 1.0)

    # Every pair must actually roll the sum it claims, and be a real pair of dice.
    for key, pair in zip(keys, pairs):
        assert 1 <= pair[0] <= 6 and 1 <= pair[1] <= 6
        if key != DEAD_ROLL:
            assert pair[0] + pair[1] == key


def test_seven_is_live_even_when_no_tile_pays():
    """A 7 discards and moves the robber; it is never poolable."""
    from src.agent.mcts import _roll_distribution

    game = make_1v1_game(seed=3)
    keys, _, _ = _roll_distribution(game)
    assert 7 in keys


def test_pooled_dead_rolls_leave_identical_positions():
    """The merge is exact, not an approximation.

    Two different dead sums must produce the same successor state, or pooling
    them silently averages over positions that are not the same.
    """
    from catanatron.models.enums import Action, ActionType

    from src.agent.encoding import encode_observation
    from src.env.lookahead import DICE_SUMS, live_dice_sums

    game = make_1v1_game(seed=11)
    while game.state.playable_actions[0].action_type != ActionType.ROLL:
        game.execute(game.state.playable_actions[0], validate_action=False)

    dead = [DICE_SUMS[i] for i, alive in enumerate(live_dice_sums(game))
            if not alive]
    assert len(dead) >= 2, "seed does not produce a board with two dead numbers"

    color = game.state.current_color()
    observations = []
    for total in dead:
        copy = game.copy()
        pair = (min(6, total - 1), total - min(6, total - 1))
        copy.execute(Action(color, ActionType.ROLL, pair), validate_action=False)
        observations.append(encode_observation(copy, color))
    for other in observations[1:]:
        assert np.array_equal(observations[0], other)


def test_roll_edges_never_branch_wider_than_the_live_set():
    """Walk a real tree: no ROLL edge may hold a key outside its live set."""
    from catanatron.models.enums import ActionType

    from src.agent.mcts import DEAD_ROLL
    from src.env.lookahead import DICE_SUMS, live_dice_sums

    game = make_1v1_game(seed=5)
    mcts = MCTS(NoisyEvaluator(), simulations=120)
    mcts._root_turns = game.state.num_turns
    root = mcts._make_node(game.copy())
    rng = np.random.default_rng(0)
    for _ in range(120):
        mcts._simulate(root, rng)

    roll_edges = 0

    def walk(node):
        nonlocal roll_edges
        if node.is_terminal:
            return
        live = None
        for index, action in enumerate(node.actions):
            children = node.children[index]
            if action.action_type == ActionType.ROLL:
                if live is None:
                    live = live_dice_sums(node.game)
                allowed = {total for total, ok in zip(DICE_SUMS, live) if ok}
                allowed.add(DEAD_ROLL)
                assert set(children) <= allowed
                roll_edges += 1
            for child in children.values():
                walk(child)

    walk(root)
    assert roll_edges > 0, "no ROLL edge was searched"


def test_sampled_rolls_keep_the_true_marginal_distribution():
    """Compression must not change the odds -- only which children share a node."""
    from src.agent.mcts import DEAD_ROLL, _roll_distribution
    from src.env.lookahead import DICE_PROBS, DICE_SUMS, live_dice_sums

    game = make_1v1_game(seed=9)
    live = live_dice_sums(game)
    keys, _, probs = _roll_distribution(game)

    # Unpooled marginal per live sum is unchanged; pooled mass equals the rest.
    recovered = np.zeros(len(DICE_SUMS))
    for key, prob in zip(keys, probs):
        if key == DEAD_ROLL:
            recovered[~live] = DICE_PROBS[~live]
        else:
            recovered[DICE_SUMS.index(key)] = prob
    assert np.allclose(recovered, DICE_PROBS)


# --------------------------------------------------------------------------
# Search horizon
# --------------------------------------------------------------------------
def test_horizon_stops_the_tree_at_the_requested_turn():
    game = make_1v1_game(seed=6)
    root_turns = game.state.num_turns

    mcts = MCTS(NoisyEvaluator(), simulations=150, horizon=1)
    mcts._root_turns = root_turns
    root = mcts._make_node(game.copy())
    for _ in range(150):
        mcts._simulate(root, np.random.default_rng(2))

    depth_seen = []

    def walk(node):
        depth_seen.append(node.game.state.num_turns - root_turns)
        assert node.game.state.num_turns - root_turns <= 1
        if node.is_terminal:
            return
        for children in node.children:
            for child in children.values():
                walk(child)

    walk(root)
    assert max(depth_seen) >= 1, "horizon=1 should still reach the next turn"


def test_horizon_none_is_the_default_and_searches_deeper():
    unlimited = MCTS(NoisyEvaluator(), simulations=64)
    assert unlimited.horizon is None

    game = make_1v1_game(seed=6)
    result = MCTS(NoisyEvaluator(), simulations=64, horizon=2).search(
        game, rng=np.random.default_rng(0)
    )
    assert result.visits.sum() == 64
    assert np.isclose(result.policy.sum(), 1.0)


def test_horizon_must_be_at_least_one_turn():
    import pytest

    with pytest.raises(ValueError):
        MCTS(UniformEvaluator(), simulations=8, horizon=0)


def test_horizon_reaches_the_player_through_mcts_kwargs():
    player = MCTSPlayer(Color.BLUE, UniformEvaluator(), simulations=8, horizon=3)
    assert player.mcts.horizon == 3
