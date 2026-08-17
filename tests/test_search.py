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
