"""1v1 Catanatron Gymnasium environment helpers.

Thin wrappers around catanatron-gym's ``catanatron-v1`` env so the rest of the
project has a single place that knows the env id, the 1v1 config, and how to
turn ``get_valid_actions()`` into the boolean mask SB3-Contrib's ActionMasker
expects.

Key facts about the underlying env (catanatron-gym 4.0.0):
- Env id ``catanatron-v1``; the controlled agent is P0 (Color.BLUE).
- It is inherently 1v1: one entry in ``config["enemies"]`` => a 2-player game.
- Action space ``Discrete(290)``; most actions are illegal each turn, so the
  valid-action mask is mandatory for any learning agent.
- Default observation is the 614-dim ``"vector"`` representation.
"""

import gymnasium as gym
from gymnasium import Wrapper
import numpy as np

import catanatron_gym  # noqa: F401  -- registers the "catanatron-v1" env id
from catanatron import Color
from catanatron.players.weighted_random import WeightedRandomPlayer

from src.env.rules import apply_rule_patches

ENV_ID = "catanatron-v1"

# Install custom 1v1 rules (discard only on >9 cards) at import time. This module is
# imported by every env constructor, so the patch lands in SubprocVecEnv workers too.
apply_rule_patches()


def make_1v1_env(
    enemy=None,
    map_type="BASE",
    vps_to_win=15,
    representation="vector",
    reward_function=None,
):
    """Construct a 1v1 Catanatron env.

    Args:
        enemy: opponent Player instance (must not be Color.BLUE). Defaults to a
            WeightedRandomPlayer on RED -- a slightly stronger-than-random bot.
        map_type: "BASE" (full board) or "MINI" (faster iteration).
        vps_to_win: victory points to win; 15 for extended gameplay.
        representation: "vector" (flat Box) or "mixed" (board tensor + numeric).
        reward_function: optional callable(game, p0_color) -> float. Defaults to
            the env's built-in win/loss/draw reward.

    Returns:
        A gymnasium env wrapping a single 1v1 game.
    """
    if enemy is None:
        enemy = WeightedRandomPlayer(Color.RED)

    config = {
        "enemies": [enemy],
        "map_type": map_type,
        "vps_to_win": vps_to_win,
        "representation": representation,
    }
    if reward_function is not None:
        config["reward_function"] = reward_function

    return gym.make(ENV_ID, config=config)


class TurnLimitWrapper(Wrapper):
    """Enforce a maximum turn limit; truncates when exceeded.

    Args:
        env: the base environment.
        max_turns: maximum turns; game truncates (draw) if exceeded.
    """

    def __init__(self, env, max_turns):
        super().__init__(env)
        self.max_turns = max_turns

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        num_turns = self.env.unwrapped.game.state.num_turns
        # Truncate if turn limit exceeded
        if num_turns >= self.max_turns:
            truncated = True
        # Expose the game length when an episode ends, so a callback can track how
        # many turns games take (should fall as the agent gets more efficient).
        if terminated or truncated:
            info["game_turns"] = num_turns
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


# Milestone VP thresholds and their one-time bonus rewards.
_VP_MILESTONES = {6: 0.1, 8: 0.25, 12: 0.5}


class RewardShapingWrapper(Wrapper):
    """Milestone-based reward shaping on top of the sparse win/loss signal.

    One-time bonuses fire the first time the agent crosses each VP threshold:
      6 VP -> +0.10
      8 VP -> +0.25
     12 VP -> +0.50
    The base env still provides +1 on win and -1 on loss.

    On episode end, records final VPs and build counts in ``info`` for W&B.

    Place this OUTSIDE TurnLimitWrapper so it observes turn-limit truncations too.
    """

    def __init__(self, env, agent_color=Color.BLUE):
        super().__init__(env)
        self.agent_color = agent_color
        self._milestones_reached: set = set()
        self._settlements_built = 0
        self._prev_settlements_avail = 5

    def _actual_vp(self, color):
        state = self.env.unwrapped.game.state
        key = f"P{state.color_to_index[color]}"
        return state.player_state[f"{key}_ACTUAL_VICTORY_POINTS"]

    def _settlements_available(self, color):
        state = self.env.unwrapped.game.state
        key = f"P{state.color_to_index[color]}"
        return state.player_state[f"{key}_SETTLEMENTS_AVAILABLE"]

    def _roads_built(self, color):
        state = self.env.unwrapped.game.state
        key = f"P{state.color_to_index[color]}"
        return 15 - state.player_state[f"{key}_ROADS_AVAILABLE"]

    def _cities_built(self, color):
        state = self.env.unwrapped.game.state
        key = f"P{state.color_to_index[color]}"
        return 4 - state.player_state[f"{key}_CITIES_AVAILABLE"]

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._milestones_reached = set()
        self._settlements_built = 0
        self._prev_settlements_avail = self._settlements_available(self.agent_color)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        vp = self._actual_vp(self.agent_color)

        for threshold, bonus in _VP_MILESTONES.items():
            if vp >= threshold and threshold not in self._milestones_reached:
                reward += bonus
                self._milestones_reached.add(threshold)

        avail = self._settlements_available(self.agent_color)
        if avail < self._prev_settlements_avail:
            self._settlements_built += self._prev_settlements_avail - avail
        self._prev_settlements_avail = avail

        if terminated or truncated:
            opponent = next(
                c for c in self.env.unwrapped.game.state.colors
                if c != self.agent_color
            )
            info["final_vp"] = vp
            info["opp_vp"] = self._actual_vp(opponent)
            info["settlements_built"] = self._settlements_built
            info["roads_built"] = self._roads_built(self.agent_color)
            info["cities_built"] = self._cities_built(self.agent_color)
        return obs, reward, terminated, truncated, info


def valid_action_mask(env):
    """Boolean mask over the full action space for SB3-Contrib's ActionMasker.

    Args:
        env: a (possibly wrapped) Catanatron env.

    Returns:
        np.ndarray[bool] of shape (action_space.n,), True where legal.
    """
    n = env.action_space.n
    mask = np.zeros(n, dtype=bool)
    mask[env.unwrapped.get_valid_actions()] = True
    return mask
