"""1v1 Catanatron Gymnasium environment helpers.

One place that knows the env id and the 1v1 config. Two entry points, for two
different substrates:

- :func:`make_1v1_game` -- a raw ``Game``. This is what search, self-play and
  the placement pipeline use; they drive the engine directly.
- :func:`make_1v1_env` -- the Gymnasium env, now used only by the Phase 1 smoke
  and visual tests. The gym step interface auto-advances the opponent and hides
  the ``Game``, which is why nothing in the training path goes through it.

Key facts about the underlying env (catanatron-gym 4.0.0):
- Env id ``catanatron-v1``; the controlled agent is P0 (Color.BLUE).
- It is inherently 1v1: one entry in ``config["enemies"]`` => a 2-player game.
- Action space ``Discrete(294)`` *after* ``src.env.rules`` expands the single
  DISCARD slot into one per resource; most actions are illegal each turn, so the
  valid-action mask is mandatory for any learning agent.
- Default observation is the 614-dim ``"vector"`` representation.
- Games are played to ``VPS_TO_WIN`` VP with no Longest Road bonus (see
  ``src.env.rules``).
"""

import random

import gymnasium as gym

import catanatron_gym  # noqa: F401  -- registers the "catanatron-v1" env id
from catanatron import Color, Game
from catanatron.models.map import build_map
from catanatron.models.player import RandomPlayer
from catanatron.players.weighted_random import WeightedRandomPlayer

from src.env.rules import apply_rule_patches

ENV_ID = "catanatron-v1"

# Victory points needed to win. With Longest Road disabled, VPs come from
# settlements, cities, VP dev cards and Largest Army. Briefly 10 (the standard
# target) for the ppo-10vp run; back to 8 because the shorter game keeps the RL
# horizon manageable, and that turned out to matter more than it looked. At 10 VP
# episodes ran ~250 agent steps, so with gamma=0.99 only ~8% of the terminal
# win/loss signal survived back to the opening placement; at 8 VP it is ~150
# steps. The AlphaZero track has no discount to keep in step with this, but the
# episode length still sets how far search has to look.
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
