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
- Games are played to ``VPS_TO_WIN`` VP; whether Longest Road pays its +2 VP is
  part of the per-run ruleset (see ``src.env.ruleset`` and ``src.env.rules``).
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
from src.env.ruleset import MAX_TURNS, VPS_TO_WIN  # noqa: F401 -- re-exported

# ``VPS_TO_WIN`` and ``MAX_TURNS`` are re-exported from :mod:`src.env.ruleset`,
# where they are read from the environment so that spawned workers agree with the
# parent. They used to be literals here; the names are unchanged so every
# existing import still works.
#
# On VPS_TO_WIN: briefly 10 (the standard target) for the ppo-10vp run, then 8,
# because the shorter game keeps the RL horizon manageable and that mattered more
# than it looked. At 10 VP episodes ran ~250 agent steps, so with gamma=0.99 only
# ~8% of the terminal win/loss signal survived back to the opening placement; at
# 8 VP it is ~150 steps. Anything changing this must move ``--gamma`` in
# src/agent/train.py with it -- see the note there.
#
# On MAX_TURNS: a safety net so a degenerate policy cannot stall a game forever,
# but it has to clear honest games by a wide margin, because a truncated episode
# pays 0 -- neither win nor loss -- and the agent cannot learn from it. Measured
# over 30 WeightedRandom mirror games with Longest Road on: median 172 turns at 8
# VP (mean 214, max 517) rising to a median of 417 at 15 VP (mean 424, max 1000).
# The old cap of 300 was already truncating a meaningful share of 8-VP games and
# would have truncated the majority at 15 VP, so the default is now 1000.

ENV_ID = "catanatron-v1"

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


class EpisodeStatsWrapper(Wrapper):
    """Record end-of-episode VP and build counts in ``info``. Reward untouched.

    This used to be ``RewardShapingWrapper``, which did two unrelated jobs: it
    paid one-time bonuses for crossing VP milestones, *and* it recorded the
    telemetry below. The milestones are gone -- the sparse arm answered the
    question they existed for (~92% vs weighted-random in 3M steps), and at a
    15-VP target the thresholds would have needed retuning against a game where
    the whole difficulty is that the reward comes only at the end. Shaping that
    honestly is a research question, not a config change, so the crutch is
    removed rather than left lying around half-calibrated.

    The telemetry is worth keeping on its own and is now always on: build counts
    are the sharpest read on whether the agent is actually expanding, and at 15
    VP it *must* reach 9 VP of buildings, so ``cities_built`` and
    ``settlements_built`` are the numbers to watch during a run.

    Place this OUTSIDE TurnLimitWrapper so it observes turn-limit truncations too.
    """

    def __init__(self, env, agent_color=Color.BLUE):
        super().__init__(env)
        self.agent_color = agent_color
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
        self._settlements_built = 0
        self._prev_settlements_avail = self._settlements_available(self.agent_color)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Settlements are counted by watching the piece pool fall rather than
        # read off it directly: upgrading to a city *returns* the settlement
        # piece, so the pool alone undercounts everything ever built.
        avail = self._settlements_available(self.agent_color)
        if avail < self._prev_settlements_avail:
            self._settlements_built += self._prev_settlements_avail - avail
        self._prev_settlements_avail = avail

        if terminated or truncated:
            vp = self._actual_vp(self.agent_color)
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
