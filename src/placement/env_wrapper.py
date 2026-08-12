"""Hand the opening to the placement scorer on the Gymnasium side.

The scorer is a catanatron ``Player`` (:class:`~src.placement.player.PlacementPlayer`),
which is all that search, self-play and benchmarking need. A gym-based learner has
no Player object to wrap: it emits an action index and the env advances the
opponent for it. This wrapper is the bridge.

**The opening happens inside ``reset()``.** By the time the learner sees its first
observation, both seats have placed and the initial build phase is over. That is
deliberate, and it is the whole reason to do it this way rather than intercepting
mid-episode:

- **No fake samples.** Overriding an action the learner chose would leave a
  transition in the rollout buffer that the learner is then trained on as if it
  had made the decision. Doing the opening before the episode starts means those
  transitions never exist.
- **Data starvation stops being a problem by construction.** Placement was ~2 of
  ~300 gradient samples per episode -- too few to learn from, enough to add
  variance to everything else. Now it is 0 of ~300, handled by a specialist that
  trains on placement samples exclusively.

**Give the opponent the scorer too** (:func:`make_placement_env` does). Training
against opponents with bad openings teaches the agent to exploit an edge it will
not have against anything competent.

**Known approximation: initial roads are chosen at random here.** The scorer
ranks corners, not roads. At play time :class:`PlacementPlayer` delegates the
initial road to the inner agent instead, so training and play differ slightly on
a decision with typically 2-3 legal options. Worth revisiting if it shows up in
the numbers; not worth a second model yet.

**Seeding trap, measured:** ``env.reset(seed=n)`` does *not* control the board
layout, and is not even repeatable across two resets with the same seed. The gym
env builds its map off the global ``random`` module at reset time, so the layout
depends on whatever consumed the RNG beforehand -- the same bug
:func:`~src.env.catan_env.make_1v1_game` was fixed for, still live on the gym
path. Any measurement taken through this wrapper has board layout as an
uncontrolled variable; compare ranks against ``env.unwrapped.game.state.board.map``
rather than against a board you built separately from the same seed.
"""

import random

from gymnasium import Wrapper

import catanatron_gym.envs.catanatron_env as cenv
from catanatron.models.enums import ActionType

from src.env.catan_env import make_1v1_env
from src.placement.chooser import OpeningChooser
from src.placement.model import BundleNet, PlacementNet
from src.placement.player import PlacementPlayer


class PlacementWrapper(Wrapper):
    """Plays the initial build phase during ``reset``, using the scorer.

    Args:
        env: a Catanatron gym env whose controlled seat is P0.
        model: a :class:`~src.placement.model.PlacementNet`.
        seed: seed for the initial-road choice.
    """

    def __init__(self, env, model, seed=None, bundle_model=None):
        super().__init__(env)
        self.model = model
        self.chooser = OpeningChooser(model, bundle_model)
        self.rng = random.Random(seed)
        # Nodes the scorer picked this episode, in order. Useful telemetry: if
        # these stop varying across resets, exploration has collapsed.
        self.opening_nodes = []

    def _choose(self, game, color):
        """The action to play at the current initial-phase decision."""
        actions = game.state.playable_actions
        settlements = [
            a for a in actions if a.action_type == ActionType.BUILD_SETTLEMENT
        ]
        if not settlements:
            roads = [a for a in actions if a.action_type == ActionType.BUILD_ROAD]
            if roads and self.chooser.bundle_model is not None:
                edge = self.chooser.choose_road(
                    game, color, [a.value for a in roads]
                )
                return next(
                    a for a in roads
                    if tuple(sorted(a.value)) == tuple(sorted(edge))
                )
            return self.rng.choice(list(actions))

        nodes = [a.value for a in settlements]
        node = self.chooser.choose(game, color, nodes)
        chosen = next(a for a in settlements if a.value == node)
        self.opening_nodes.append(chosen.value)
        return chosen

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.opening_nodes = []
        self.chooser.reset()

        game = self.env.unwrapped.game
        color = self.env.unwrapped.p0.color

        while game.state.is_initial_build_phase:
            action = self._choose(game, color)
            obs, reward, terminated, truncated, info = self.env.step(
                cenv.to_action_space(action)
            )
            # The opening pays nothing under a sparse win/loss reward, and a game
            # cannot end during it. Either would mean silently dropping a
            # transition the learner should have seen.
            if reward or terminated or truncated:
                raise RuntimeError(
                    "initial build phase ended the episode or paid a reward "
                    f"(reward={reward}, terminated={terminated}, "
                    f"truncated={truncated}); the wrapper would be hiding it"
                )

        return obs, info


def make_placement_env(model_path, enemy=None, seed=None, opponent_scorer=True,
                       bundle_path=None, **env_kwargs):
    """A 1v1 env where both seats open with the placement scorer.

    Args:
        model_path: PlacementNet checkpoint.
        bundle_path: optional BundleNet checkpoint; when given, openings are
            chosen as a pair of corners rather than greedily one at a time.
        enemy: opponent Player. Defaults to the env's WeightedRandomPlayer(RED).
        seed: seed for initial-road choices.
        opponent_scorer: give the opponent the scorer as well. Leave this on
            unless you are deliberately measuring against a weak opening.
        **env_kwargs: forwarded to :func:`~src.env.catan_env.make_1v1_env`.

    Returns:
        A wrapped env whose first observation is the position after both
        players have completed the initial build phase.
    """
    model = PlacementNet.load(model_path)
    bundle = BundleNet.load(bundle_path) if bundle_path else None

    if opponent_scorer:
        from catanatron import Color
        from catanatron.players.weighted_random import WeightedRandomPlayer

        inner = enemy if enemy is not None else WeightedRandomPlayer(Color.RED)
        enemy = PlacementPlayer(inner.color, inner, model, bundle)

    env = make_1v1_env(enemy=enemy, **env_kwargs)
    return PlacementWrapper(env, model, seed=seed, bundle_model=bundle)
