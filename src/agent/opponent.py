"""Opponent Player wrapper around a frozen trained policy.

Lets a trained MaskablePPO checkpoint act as a Catanatron ``Player`` so it can be
dropped into ``config["enemies"]`` for self-play. Works inside ``SubprocVecEnv``
workers: the player (and its policy) is pickled to each worker, and ``decide`` builds
the observation/mask itself from the live ``Game`` -- no env handle required.
"""

import numpy as np

from catanatron import Player
from catanatron_gym.features import create_sample_vector, get_feature_ordering
# Imported as a module so the (monkeypatched) expanded action space is seen at
# call time rather than frozen at import.
import catanatron_gym.envs.catanatron_env as cenv


class PolicyPlayer(Player):
    """A catanatron Player that acts via a frozen SB3 MaskablePPO policy."""

    def __init__(self, color, policy_model=None, map_type="BASE", num_players=2,
                 model_path=None, deterministic: bool = True):
        """Initialize.

        Args:
            color: catanatron.Color enum.
            policy_model: trained MaskablePPO with ``.predict(obs, action_masks=...)``.
            map_type: board type the policy was trained on (feature ordering depends
                on it).
            num_players: player count the policy was trained on (614-dim obs for 2p).
            model_path: load the policy from this checkpoint instead, lazily and
                inside whatever process ends up using it. Self-play draws a
                different opponent per env, so the alternative is pickling eight
                torch models through the SubprocVecEnv pipe on every swap.
            deterministic: take the argmax action. Evaluation wants this so a
                measurement is repeatable; self-play opponents generally do not,
                since a greedy opponent plays one fixed line per position and the
                learner can overfit to beating exactly that.
        """
        super().__init__(color)
        if policy_model is None and model_path is None:
            raise ValueError("PolicyPlayer needs policy_model or model_path")
        self.policy = policy_model
        self.model_path = str(model_path) if model_path is not None else None
        self.deterministic = deterministic
        # Feature ordering must match training (defaults to 4 players otherwise).
        self._features = get_feature_ordering(num_players, map_type)

    def __getstate__(self):
        """Ship the path, not the weights, when a path is available."""
        state = self.__dict__.copy()
        if self.model_path is not None:
            state["policy"] = None
        return state

    def _ensure_policy(self):
        if self.policy is None:
            import torch
            from sb3_contrib import MaskablePPO
            # One core per env worker; the vector env supplies the parallelism.
            torch.set_num_threads(1)
            self.policy = MaskablePPO.load(
                self.model_path, device="cpu", custom_objects={"n_steps": 1}
            )
        return self.policy

    def decide(self, game, playable_actions):
        """Choose an action via the trained policy.

        Args:
            game: catanatron.game.Game instance.
            playable_actions: list of legal catanatron Actions this turn.

        Returns:
            One of ``playable_actions`` (a catanatron Action), as the engine expects.
        """
        if len(playable_actions) == 1:
            return playable_actions[0]

        obs = np.array(
            create_sample_vector(game, self.color, self._features), dtype=float
        )
        mask = np.zeros(cenv.ACTION_SPACE_SIZE, dtype=bool)
        for action in playable_actions:
            mask[cenv.to_action_space(action)] = True

        action_int, _ = self._ensure_policy().predict(
            obs, action_masks=mask, deterministic=self.deterministic
        )
        return cenv.from_action_space(int(action_int), playable_actions)
