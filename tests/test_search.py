"""Tests for MCTS, the network, and the self-play value targets."""

import numpy as np

from catanatron import Color

from src.agent.encoding import action_size, obs_size
from src.agent.evaluator import NetEvaluator, UniformEvaluator
from src.agent.mcts import MCTS, MCTSPlayer, _outcome_key
from src.agent.net import AlphaZeroNet
from src.agent.selfplay import SelfPlayConfig, compute_value_targets, play_game
from src.env.catan_env import make_1v1_game


def small_net():
    return AlphaZeroNet(width=32, blocks=1)


def test_net_infer_respects_the_action_mask():
    net = small_net()
    mask = np.zeros(action_size(), dtype=bool)
    mask[[0, 5, 17]] = True
    obs = np.zeros((1, obs_size()), dtype=np.float32)

    priors, values = net.infer(obs, mask[None, :])

    assert priors.shape == (1, action_size())
    assert np.isclose(priors.sum(), 1.0)
    assert np.allclose(priors[0, ~mask], 0.0)
    assert -1.0 <= values[0] <= 1.0


def test_search_returns_a_normalized_policy_over_legal_actions():
    game = make_1v1_game(seed=2)
    mcts = MCTS(NetEvaluator(small_net()), simulations=32)
    result = mcts.search(game)

    assert np.isclose(result.policy.sum(), 1.0)
    assert np.allclose(result.policy[~result.mask], 0.0)
    assert result.visits.sum() == 32
    assert -1.0 <= result.value <= 1.0
    assert result.best_action() in game.state.playable_actions


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


def test_value_targets_blend_outcome_with_bootstrap():
    colors = [Color.BLUE, Color.RED, Color.BLUE, Color.RED, Color.BLUE]
    root_values = [0.5, -0.2, 0.6, -0.4, 0.9]

    targets = compute_value_targets(
        colors, root_values, winner=Color.BLUE, nstep=2, mix=0.5
    )

    # t=0 is BLUE: outcome +1, bootstrap is root_values[2] (also BLUE, so no flip).
    assert np.isclose(targets[0], 0.5 * 1.0 + 0.5 * 0.6)
    # t=1 is RED: outcome -1, bootstrap is root_values[3] (also RED).
    assert np.isclose(targets[1], 0.5 * -1.0 + 0.5 * -0.4)
    # The tail has nothing to bootstrap from and falls back to the outcome.
    assert np.isclose(targets[4], 1.0)


def test_value_target_sign_flips_across_players():
    colors = [Color.BLUE, Color.RED]
    targets = compute_value_targets(
        colors, [0.0, 0.8], winner=None, nstep=1, mix=0.0
    )
    # Bootstrapping from RED's +0.8 into BLUE's slot must become -0.8.
    assert np.isclose(targets[0], -0.8)


def test_pure_outcome_mix_reproduces_textbook_alphazero():
    colors = [Color.BLUE, Color.RED, Color.BLUE]
    targets = compute_value_targets(
        colors, [0.9, 0.9, 0.9], winner=Color.RED, nstep=1, mix=1.0
    )
    assert np.allclose(targets, [-1.0, 1.0, -1.0])


def test_self_play_produces_aligned_training_arrays():
    config = SelfPlayConfig(simulations=8, temperature_moves=3)
    result = play_game(NetEvaluator(small_net()), config, seed=17)

    count = len(result.values)
    assert count > 0
    assert result.obs.shape == (count, obs_size())
    assert result.masks.shape == (count, action_size())
    assert result.policies.shape == (count, action_size())
    # Policies are zero wherever the position said the action was illegal.
    assert np.allclose(result.policies[~result.masks], 0.0)
    assert np.all(np.abs(result.values) <= 1.0)
    assert result.stats["decisions"] == count


def test_mcts_player_skips_search_on_forced_moves():
    """A single legal action must be returned without touching the evaluator."""

    class ExplodingEvaluator(UniformEvaluator):
        def evaluate_batch(self, obs_batch, mask_batch):
            raise AssertionError("evaluator called on a forced move")

    game = make_1v1_game(seed=8)
    player = MCTSPlayer(Color.BLUE, ExplodingEvaluator(), simulations=4)
    only = game.state.playable_actions[:1]

    assert player.decide(game, only) is only[0]
