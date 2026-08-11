"""1v1 Catanatron Gymnasium environment helpers.

Thin wrappers around catanatron-gym's ``catanatron-v1`` env so the rest of the
project has a single place that knows the env id, the 1v1 config, and how to
turn ``get_valid_actions()`` into the boolean mask SB3-Contrib's ActionMasker
expects.

Key facts about the underlying env (catanatron-gym 4.0.0):
- Env id ``catanatron-v1``; the controlled agent is P0 (Color.BLUE).
- It is inherently 1v1: one entry in ``config["enemies"]`` => a 2-player game.
- Action space ``Discrete(294)`` *after* ``src.env.rules`` expands the single
  DISCARD slot into one per resource; most actions are illegal each turn, so the
  valid-action mask is mandatory for any learning agent.
- Default observation is the 614-dim ``"vector"`` representation.
- Games are played to 8 VP with no Longest Road bonus (see ``src.env.rules``).
"""

import random

import gymnasium as gym
from gymnasium import Wrapper
import numpy as np

import catanatron_gym  # noqa: F401  -- registers the "catanatron-v1" env id
from catanatron import Color, Game
from catanatron.models.map import build_map
from catanatron.models.player import RandomPlayer
from catanatron.players.weighted_random import WeightedRandomPlayer

from src.env.rules import apply_rule_patches

ENV_ID = "catanatron-v1"

# Victory points needed to win. Short games keep the RL horizon (and the MCTS
# search depth) manageable; with Longest Road disabled, 8 VP is reached through
# settlements, cities, VP dev cards and Largest Army.
VPS_TO_WIN = 8

# Safety net so a degenerate policy cannot stall a game forever.
MAX_TURNS = 300

# Install custom 1v1 rules (discard only on >9 cards) at import time. This module is
# imported by every env constructor, so the patch lands in SubprocVecEnv workers too.
apply_rule_patches()


def make_1v1_game(players=None, seed=None, map_type="BASE", vps_to_win=VPS_TO_WIN):
    """Construct a raw Catanatron ``Game`` for tree search / self-play.

    The gym env is the wrong substrate for MCTS: it auto-advances the opponent and
    hides the ``Game`` behind a step interface. Search and AlphaZero self-play drive
    the engine directly, so they build games here instead.

    Args:
        players: list of catanatron Players. Defaults to two placeholder
            RandomPlayers (BLUE first, then RED) -- callers that drive the engine
            themselves never invoke ``decide``.
        seed: RNG seed, or None for a random one. Controls the board layout as
            well as the dice -- see below.
        map_type: "BASE" (full board) or "MINI" (faster iteration).
        vps_to_win: victory points to win.

    Returns:
        An initialized ``Game``.
    """
    if players is None:
        players = [RandomPlayer(Color.BLUE), RandomPlayer(Color.RED)]

    # ``Game.__init__`` reseeds the *global* random module, but ``build_map``
    # shuffles the board off that same module -- and as a plain argument it would
    # run first, leaving the layout at the mercy of whatever consumed the RNG
    # beforehand. Seeding here first is what makes a seed reproduce a board;
    # without it two games with the same seed get different maps.
    if seed is not None:
        random.seed(seed)
    catan_map = build_map(map_type)

    return Game(
        players=players,
        seed=seed,
        vps_to_win=vps_to_win,
        catan_map=catan_map,
    )


def make_1v1_env(
    enemy=None,
    map_type="BASE",
    vps_to_win=VPS_TO_WIN,
    representation="vector",
    reward_function=None,
):
    """Construct a 1v1 Catanatron env.

    Args:
        enemy: opponent Player instance (must not be Color.BLUE). Defaults to a
            WeightedRandomPlayer on RED -- a slightly stronger-than-random bot.
        map_type: "BASE" (full board) or "MINI" (faster iteration).
        vps_to_win: victory points to win.
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


# Milestone VP thresholds and their one-time bonus rewards, scaled to an 8-VP
# game. The top milestone tracks "one VP short of winning", so it moves with
# VPS_TO_WIN; leaving it at 6 would have paid the largest bonus two VP early.
_VP_MILESTONES = {3: 0.1, 5: 0.25, 7: 0.5}


class RewardShapingWrapper(Wrapper):
    """Milestone-based reward shaping on top of the sparse win/loss signal.

    Only the PPO training path (``src.agent.train``) uses this. The AlphaZero path
    (``src.agent.train_az``) trains on the sparse win/loss outcome alone, because
    MCTS supplies the dense signal that shaping was standing in for.

    One-time bonuses fire the first time the agent crosses each VP threshold:
      3 VP -> +0.10
      5 VP -> +0.25
      6 VP -> +0.50
    The base env still provides +1 on win and -1 on loss.

    Two additional one-time building bonuses:
      3rd settlement placed -> +0.05
      1st city built        -> +0.05

    On episode end, records final VPs and build counts in ``info`` for W&B.

    Place this OUTSIDE TurnLimitWrapper so it observes turn-limit truncations too.
    """

    def __init__(self, env, agent_color=Color.BLUE):
        super().__init__(env)
        self.agent_color = agent_color
        self._milestones_reached: set = set()
        self._settlements_built = 0
        self._prev_settlements_avail = 5
        self._3rd_settlement_bonus_given = False
        self._4th_settlement_bonus_given = False
        self._1st_city_bonus_given = False
        self._2nd_city_bonus_given = False

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
        self._3rd_settlement_bonus_given = False
        self._4th_settlement_bonus_given = False
        self._1st_city_bonus_given = False
        self._2nd_city_bonus_given = False
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

        if self._settlements_built >= 3 and not self._3rd_settlement_bonus_given:
            reward += 0.05
            self._3rd_settlement_bonus_given = True

        if self._settlements_built >= 4 and not self._4th_settlement_bonus_given:
            reward += 0.05
            self._4th_settlement_bonus_given = True

        cities = self._cities_built(self.agent_color)
        if cities >= 1 and not self._1st_city_bonus_given:
            reward += 0.05
            self._1st_city_bonus_given = True

        if cities >= 2 and not self._2nd_city_bonus_given:
            reward += 0.05
            self._2nd_city_bonus_given = True

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
