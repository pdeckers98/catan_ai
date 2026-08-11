"""PUCT Monte-Carlo tree search over the Catanatron engine.

Design notes specific to Catan:

**Stochastic transitions.** Dice, dev-card draws and robber steals are chance
events. Rather than enumerating outcomes, each edge keeps a dict of children keyed
by the *realized* outcome and lets the engine's own RNG sample it. Repeated visits
to the same edge therefore land on children in proportion to the true transition
probabilities, so the backed-up value is an unbiased estimate of the expectation
without paying for an 11-way fan-out on every roll. Dice children are keyed by the
sum, since (2, 5) and (3, 4) are the same event as far as the game is concerned.

**Alternating perspective.** A Catan turn is many consecutive decisions by one
player, so the tree is not strictly alternating. Every node records who is to move
and values are negated on backup whenever the perspective flips, which handles
both the within-turn runs and the hand-offs uniformly.

**Forced moves are free.** A large fraction of Catan plies have exactly one legal
action. Those are played immediately without search and generate no training
sample -- there is nothing to learn and nothing to choose.

The search operates on a real ``Game`` and copies it as it descends. That copy is
the dominant cost; see ``src.eval.bench_mcts`` for the measured budget.
"""

import math

import numpy as np

from catanatron import Player
from catanatron.models.enums import ActionType

from src.agent.encoding import (
    action_indices, action_size, encode_observation, legal_action_mask,
)
from src.env.catan_env import MAX_TURNS


def _outcome_key(executed_action):
    """Chance-outcome identity of an executed action, or None if deterministic.

    ``apply_action`` rewrites stochastic actions with their realized value, so the
    concrete outcome is readable off the returned action.
    """
    action_type = executed_action.action_type
    if action_type == ActionType.ROLL:
        return executed_action.value[0] + executed_action.value[1]
    if action_type == ActionType.BUY_DEVELOPMENT_CARD:
        return executed_action.value
    if action_type == ActionType.MOVE_ROBBER:
        return executed_action.value[2]  # stolen resource; None when nobody robbed
    return None


class Node:
    """One decision point in the tree."""

    __slots__ = (
        "game", "to_play", "actions", "priors", "visits", "value_sum",
        "children", "total_visits", "value_pred", "is_terminal",
    )

    def __init__(self, game, to_play, is_terminal, value_pred, actions=(), priors=None):
        self.game = game
        self.to_play = to_play
        self.is_terminal = is_terminal
        self.value_pred = value_pred
        self.actions = list(actions)
        n = len(self.actions)
        self.priors = priors if priors is not None else np.zeros(n, dtype=np.float32)
        self.visits = np.zeros(n, dtype=np.int32)
        self.value_sum = np.zeros(n, dtype=np.float32)
        # children[i] maps chance-outcome key -> Node (key None for deterministic).
        self.children = [dict() for _ in range(n)]
        self.total_visits = 0


class SearchResult:
    """What one ``MCTS.search`` call produced at the root."""

    __slots__ = ("actions", "visits", "policy", "value", "color", "obs", "mask")

    def __init__(self, actions, visits, policy, value, color, obs, mask):
        self.actions = actions
        self.visits = visits
        self.policy = policy   # normalized visit counts over the full action space
        self.value = value     # root value estimate, from ``color``'s perspective
        self.color = color
        self.obs = obs
        self.mask = mask

    def best_action(self):
        return self.actions[int(np.argmax(self.visits))]

    def sample_action(self, temperature: float, rng=None):
        """Sample proportional to visits**(1/temperature)."""
        if temperature <= 1e-6:
            return self.best_action()
        rng = rng or np.random
        weights = self.visits.astype(np.float64) ** (1.0 / temperature)
        total = weights.sum()
        if total <= 0:
            return self.best_action()
        return self.actions[int(rng.choice(len(self.actions), p=weights / total))]


class MCTS:
    """PUCT search driven by a leaf :class:`~src.agent.evaluator.Evaluator`.

    Args:
        evaluator: supplies priors and leaf values.
        simulations: playouts per move.
        c_puct: exploration constant in the PUCT term.
        dirichlet_alpha: concentration of the root exploration noise.
        dirichlet_epsilon: weight of that noise (0 disables it; use 0 for
            evaluation, non-zero only for self-play).
        fpu_reduction: how much to pessimize unvisited children relative to the
            parent's own value estimate, which stops the search from fanning out
            uniformly over Catan's very wide action lists.
        max_turns: simulations reaching this turn count score as a draw.
    """

    def __init__(
        self,
        evaluator,
        simulations: int = 100,
        c_puct: float = 1.5,
        dirichlet_alpha: float = 0.3,
        dirichlet_epsilon: float = 0.0,
        fpu_reduction: float = 0.25,
        max_turns: int = MAX_TURNS,
    ):
        self.evaluator = evaluator
        self.simulations = simulations
        self.c_puct = c_puct
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon
        self.fpu_reduction = fpu_reduction
        self.max_turns = max_turns
        self.num_actions = action_size()

    # ---- node construction ----------------------------------------------
    def _terminal_value(self, game, to_play):
        """+1/-1/0 from ``to_play``'s perspective, or None if not terminal."""
        winner = game.winning_color()
        if winner is not None:
            return 1.0 if winner == to_play else -1.0
        if game.state.num_turns >= self.max_turns:
            return 0.0
        return None

    def _make_node(self, game):
        """Evaluate a position and wrap it in a Node."""
        to_play = game.state.current_color()
        terminal_value = self._terminal_value(game, to_play)
        if terminal_value is not None:
            return Node(game, to_play, True, terminal_value)

        actions = list(game.state.playable_actions)
        obs = encode_observation(game, to_play)
        mask = legal_action_mask(actions)
        priors_full, value = self.evaluator.evaluate(obs, mask)

        # Map the action-space priors onto the positional action list. Several
        # catanatron Actions can normalize to one slot, so renormalize afterwards.
        indices = action_indices(actions)
        priors = priors_full[indices].astype(np.float32)
        total = priors.sum()
        if total <= 0:
            priors = np.full(len(actions), 1.0 / len(actions), dtype=np.float32)
        else:
            priors /= total

        return Node(game, to_play, False, float(value), actions, priors)

    def _add_root_noise(self, node, rng):
        if self.dirichlet_epsilon <= 0 or len(node.actions) < 2:
            return
        noise = rng.dirichlet([self.dirichlet_alpha] * len(node.actions))
        eps = self.dirichlet_epsilon
        node.priors = ((1 - eps) * node.priors + eps * noise).astype(np.float32)

    # ---- selection -------------------------------------------------------
    def _select(self, node):
        """PUCT: argmax over Q + c * P * sqrt(N_total) / (1 + N)."""
        visits = node.visits
        visited = visits > 0

        # First-play urgency: unvisited children inherit the parent's value,
        # discounted by how much prior mass has already been explored.
        explored_mass = float(node.priors[visited].sum()) if visited.any() else 0.0
        fpu = node.value_pred - self.fpu_reduction * math.sqrt(explored_mass)

        q = np.where(visited, node.value_sum / np.maximum(visits, 1), fpu)
        u = (
            self.c_puct
            * node.priors
            * math.sqrt(node.total_visits + 1)
            / (1 + visits)
        )
        return int(np.argmax(q + u))

    # ---- the search loop -------------------------------------------------
    def search(self, game, rng=None) -> SearchResult:
        """Run ``simulations`` playouts from ``game`` and return root statistics."""
        rng = rng or np.random.default_rng()
        root = self._make_node(game.copy())
        if root.is_terminal:
            raise ValueError("Cannot search from a finished position")
        self._add_root_noise(root, rng)

        for _ in range(self.simulations):
            self._simulate(root)

        obs = encode_observation(game, root.to_play)
        mask = legal_action_mask(root.actions)

        policy = np.zeros(self.num_actions, dtype=np.float32)
        total = root.visits.sum()
        if total > 0:
            for idx, action_id in enumerate(action_indices(root.actions)):
                policy[action_id] += root.visits[idx] / total
            value = float(root.value_sum.sum() / total)
        else:  # degenerate: fall back to the raw prior
            for idx, action_id in enumerate(action_indices(root.actions)):
                policy[action_id] += root.priors[idx]
            value = root.value_pred

        return SearchResult(
            root.actions, root.visits.copy(), policy, value, root.to_play, obs, mask
        )

    def _simulate(self, root):
        """One playout: descend to a leaf, evaluate it, back the value up."""
        node = root
        path = []  # (node, action index) pairs taken on the way down

        while True:
            if node.is_terminal:
                leaf_value, leaf_player = node.value_pred, node.to_play
                break

            index = self._select(node)
            path.append((node, index))

            child_game = node.game.copy()
            executed = child_game.execute(node.actions[index], validate_action=False)
            key = _outcome_key(executed)

            child = node.children[index].get(key)
            if child is None:
                child = self._make_node(child_game)
                node.children[index][key] = child
                leaf_value, leaf_player = child.value_pred, child.to_play
                break
            node = child

        for parent, index in path:
            signed = leaf_value if parent.to_play == leaf_player else -leaf_value
            parent.visits[index] += 1
            parent.value_sum[index] += signed
            parent.total_visits += 1


class MCTSPlayer(Player):
    """A catanatron ``Player`` that picks moves by tree search.

    Args:
        color: the seat this player occupies.
        evaluator: leaf evaluator (network, PPO adapter, or uniform).
        simulations: playouts per decision.
        temperature: 0 plays the most-visited move; >0 samples proportional to
            visits, which is what self-play uses for opening diversity.
        temperature_moves: number of searched decisions to apply ``temperature``
            to before dropping to greedy play.
        **mcts_kwargs: forwarded to :class:`MCTS`.
    """

    def __init__(
        self,
        color,
        evaluator,
        simulations: int = 100,
        temperature: float = 0.0,
        temperature_moves: int = 0,
        seed=None,
        **mcts_kwargs,
    ):
        super().__init__(color)
        self.mcts = MCTS(evaluator, simulations=simulations, **mcts_kwargs)
        self.temperature = temperature
        self.temperature_moves = temperature_moves
        self.rng = np.random.default_rng(seed)
        self._decisions = 0

    def reset_state(self):
        self._decisions = 0

    def decide(self, game, playable_actions):
        if len(playable_actions) == 1:
            return playable_actions[0]

        result = self.mcts.search(game, rng=self.rng)
        self._decisions += 1
        if self._decisions <= self.temperature_moves and self.temperature > 0:
            return result.sample_action(self.temperature, self.rng)
        return result.best_action()
