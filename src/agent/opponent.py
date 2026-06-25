"""Opponent Player wrapper around a frozen trained policy.

Allows a trained agent to be plugged into config["enemies"] for self-play.
"""

from catanatron import Player
from src.env.catan_env import valid_action_mask


class PolicyPlayer(Player):
    """A catanatron Player that acts via a frozen SB3 policy."""

    def __init__(self, color, policy_model):
        """Initialize.

        Args:
            color: catanatron.Color enum.
            policy_model: trained SB3 model with .predict(obs, deterministic=True).
        """
        super().__init__(color)
        self.policy = policy_model

    def decide(self, game, playable_actions):
        """Choose an action via the trained policy.

        Args:
            game: catanatron.game.Game instance.
            playable_actions: list of valid ActionType values (unused; we read the mask).

        Returns:
            Chosen action (catanatron Action enum value).
        """
        from catanatron_gym import CatanEnv
        env = CatanEnv(config={"enemies": []})
        obs, info = env.reset()
        obs, info = env._sync_from_game(game)
        mask = valid_action_mask(env)
        action, _ = self.policy.predict(obs, action_masks=mask, deterministic=True)
        return action
