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

    def __init__(self, color, policy_model, map_type="BASE", num_players=2):
        """Initialize.

        Args:
            color: catanatron.Color enum.
            policy_model: trained MaskablePPO with ``.predict(obs, action_masks=...)``.
            map_type: board type the policy was trained on (feature ordering depends
                on it).
            num_players: player count the policy was trained on (614-dim obs for 2p).
        """
        super().__init__(color)
        self.policy = policy_model
        # Feature ordering must match training (defaults to 4 players otherwise).
        self._features = get_feature_ordering(num_players, map_type)

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

        action_int, _ = self.policy.predict(
            obs, action_masks=mask, deterministic=True
        )
        return cenv.from_action_space(int(action_int), playable_actions)
