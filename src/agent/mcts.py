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

**Batched leaf evaluation.** A batch-1 forward pass is the dominant cost of the
search, not the state copy. Setting ``batch_size > 1`` collects that many leaves
before calling the evaluator once, using *virtual loss* to stop every descent in a
batch converging on the same leaf: an in-flight edge is temporarily credited with
a loss, which depresses its Q until the real value arrives. This changes search
results for a given seed, so it is off by default -- ``batch_size=1`` reproduces
the serial search exactly.

The search operates on a real ``Game`` and copies it as it descends. Measured on
the dev box, the evaluator's forward pass dominates that copy roughly 20:1
(~604 us against ~32 us), which is why batching leaves is the lever and shrinking
the net is not.

**Search saturates by ~50 simulations** against a PPO critic: 50 sims is worth
+9.8 points in a mirror match and 100 sims only 60.8%. The ceiling is the value
estimate, not the budget -- see :class:`~src.agent.evaluator.PPOEvaluator`.
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


class _PendingLeaf:
    """A leaf awaiting evaluation, plus where to hang it once the value lands."""

    __slots__ = ("path", "game", "parent", "index", "key",
                 "to_play", "actions", "obs", "mask")

    def __init__(self, path, game, parent, index, key, to_play, actions, obs, mask):
        self.path = path
        self.game = game
        self.parent = parent
        self.index = index
        self.key = key
        self.to_play = to_play
        self.actions = actions
        self.obs = obs
        self.mask = mask


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
        batch_size: leaves to collect before one ``evaluate_batch`` call. 1 keeps
            the exact serial search; larger values amortize the forward pass at
            the cost of descending against slightly staler statistics.
        virtual_loss: visits temporarily charged to an in-flight edge, which is
            what keeps a batch from collecting the same leaf repeatedly.
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
        batch_size: int = 1,
        virtual_loss: int = 1,
    ):
        self.evaluator = evaluator
        self.simulations = simulations
        self.c_puct = c_puct
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon
        self.fpu_reduction = fpu_reduction
        self.max_turns = max_turns
        self.batch_size = max(1, int(batch_size))
        self.virtual_loss = max(0, int(virtual_loss))
        self.num_actions = action_size()
        # The search builds observations; only the evaluator knows what its
        # network was trained on. Read it once here rather than per leaf.
        self._lookahead = bool(getattr(evaluator, "wants_lookahead", False))

    # ---- node construction ----------------------------------------------
    def _terminal_value(self, game, to_play):
        """+1/-1/0 from ``to_play``'s perspective, or None if not terminal."""
        winner = game.winning_color()
        if winner is not None:
            return 1.0 if winner == to_play else -1.0
        if game.state.num_turns >= self.max_turns:
            return 0.0
        return None

    def _terminal_node(self, game):
        """A finished-position Node, or None if the position is still live."""
        to_play = game.state.current_color()
        terminal_value = self._terminal_value(game, to_play)
        if terminal_value is None:
            return None
        return Node(game, to_play, True, terminal_value)

    def _leaf_request(self, game):
        """The evaluator inputs for a live position."""
        to_play = game.state.current_color()
        actions = list(game.state.playable_actions)
        return to_play, actions, \
            encode_observation(game, to_play, lookahead=self._lookahead), \
            legal_action_mask(actions)

    def _expand(self, game, to_play, actions, priors_full, value):
        """Wrap an evaluated position in a Node."""
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

    def _make_node(self, game):
        """Evaluate a position and wrap it in a Node (single, unbatched)."""
        node = self._terminal_node(game)
        if node is not None:
            return node
        to_play, actions, obs, mask = self._leaf_request(game)
        priors_full, value = self.evaluator.evaluate(obs, mask)
        return self._expand(game, to_play, actions, priors_full, value)

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

        if self.batch_size > 1:
            self._simulate_batched(root)
        else:
            for _ in range(self.simulations):
                self._simulate(root)

        obs = encode_observation(game, root.to_play, lookahead=self._lookahead)
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

    def _backup(self, path, leaf_value, leaf_player, undo_virtual=False):
        """Credit a real visit along ``path``, optionally clearing virtual loss.

        Values are negated wherever the perspective differs from the leaf's, which
        is what makes the non-alternating turn structure work.
        """
        vl = self.virtual_loss if undo_virtual else 0
        for parent, index in path:
            signed = leaf_value if parent.to_play == leaf_player else -leaf_value
            parent.visits[index] += 1 - vl
            parent.value_sum[index] += signed + vl
            parent.total_visits += 1 - vl

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

        self._backup(path, leaf_value, leaf_player)

    # ---- batched search --------------------------------------------------
    def _charge_virtual_loss(self, path):
        """Temporarily score every edge on ``path`` as a loss.

        Applied once at the end of a descent rather than incrementally: a descent
        visits each node at most once, so the two are equivalent here.
        """
        vl = self.virtual_loss
        if not vl:
            return
        for parent, index in path:
            parent.visits[index] += vl
            parent.value_sum[index] -= vl
            parent.total_visits += vl

    def _descend(self, root):
        """Walk to a leaf, charging virtual loss on the way.

        Returns ``(path, kind, payload)`` where kind is ``"terminal"`` (payload is
        the node) or ``"leaf"`` (payload is the unexpanded child's game plus where
        to attach it).
        """
        node = root
        path = []

        while True:
            if node.is_terminal:
                self._charge_virtual_loss(path)
                return path, "terminal", node

            index = self._select(node)
            path.append((node, index))

            child_game = node.game.copy()
            executed = child_game.execute(node.actions[index], validate_action=False)
            key = _outcome_key(executed)

            child = node.children[index].get(key)
            if child is None:
                self._charge_virtual_loss(path)
                return path, "leaf", (child_game, node, index, key)
            node = child

    def _simulate_batched(self, root):
        """Run the simulation budget, evaluating leaves ``batch_size`` at a time.

        Terminal leaves need no network call, so they are backed up during
        collection and simply do not join the batch.
        """
        remaining = self.simulations
        while remaining > 0:
            pending = []
            for _ in range(min(self.batch_size, remaining)):
                path, kind, payload = self._descend(root)
                if kind == "terminal":
                    self._backup(path, payload.value_pred, payload.to_play, True)
                    continue

                child_game, parent, index, key = payload
                terminal = self._terminal_node(child_game)
                if terminal is not None:
                    parent.children[index][key] = terminal
                    self._backup(path, terminal.value_pred, terminal.to_play, True)
                    continue

                to_play, actions, obs, mask = self._leaf_request(child_game)
                pending.append(_PendingLeaf(
                    path, child_game, parent, index, key, to_play, actions, obs, mask
                ))

            remaining -= min(self.batch_size, remaining)
            if not pending:
                continue

            priors_batch, values = self.evaluator.evaluate_batch(
                np.stack([leaf.obs for leaf in pending]),
                np.stack([leaf.mask for leaf in pending]),
            )
            for leaf, priors_full, value in zip(pending, priors_batch, values):
                child = self._expand(
                    leaf.game, leaf.to_play, leaf.actions, priors_full, value
                )
                # A duplicate collection overwrites the earlier node; both descents
                # still back up, which is why virtual loss matters more than dedup.
                leaf.parent.children[leaf.index][leaf.key] = child
                self._backup(leaf.path, child.value_pred, child.to_play, True)


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
