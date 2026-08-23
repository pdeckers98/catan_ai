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

# The five development-card types, as they appear in ``player_state`` keys.
DEV_CARDS = ("KNIGHT", "MONOPOLY", "YEAR_OF_PLENTY", "ROAD_BUILDING",
             "VICTORY_POINT")

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


class PotentialShapingWrapper(Wrapper):
    """Potential-based reward shaping on the victory-point differential.

    ``F(s, s') = gamma * Phi(s') - Phi(s)`` with ``Phi(s) = weight * (my VP -
    their VP)``. This is the Ng/Harada/Russell form, and the reason it is
    allowed here when the old ``RewardShapingWrapper`` was not: the shaping
    telescopes over an episode to ``gamma^T Phi(s_T) - Phi(s_0)``, so with
    ``Phi`` forced to zero in the absorbing state the total added return is the
    constant ``-Phi(s_0)``. It cannot invent a new optimal policy. The deleted
    wrapper paid one-time bonuses for crossing VP milestones, which do not
    telescope and genuinely could move the optimum; that is a different object
    and it is still not coming back.

    What this is for. At 15 VP the agent takes ~600 decisions per episode for
    one bit of terminal reward, so a single wasteful action moves the return by
    far less than the noise in its own advantage estimate -- burning three cards
    on a maritime trade is *free in the loss*, which is why the agent does it on
    36% of its turns and two thirds of the time while it can already afford
    something (``python -m src.eval.waste``). Shaping does not punish the trade;
    it pays for the *city*, immediately, so building has a local advantage over
    trading that survives the credit-assignment distance.

    Two deliberate choices:

    - **The opponent's contribution is their VISIBLE VP**, not their actual. Our
      own uses actual, because the agent knows its own hidden cards; scoring
      theirs would leak a face-down victory-point card into the reward and teach
      the critic to expect a signal it cannot observe.
    - **Truncation is treated as absorbing too.** A turn-limited game pays 0 and
      teaches nothing either way, so leaving ``Phi`` un-zeroed there would let
      shaping pay out a return no terminal reward ever balances -- the one way
      this wrapper could stop being policy-invariant.

    Place it INSIDE :class:`EpisodeStatsWrapper` and OUTSIDE
    :class:`TurnLimitWrapper`, so it sees the truncation flag.

    Args:
        env: the base environment.
        weight: scale of the potential, in units of the terminal +/-1 reward
            per victory point of lead. Zero disables the wrapper's effect.
        gamma: the discount the trainer uses. Shaping is only policy-invariant
            at the gamma it is written for, so this must match ``--gamma``.
        agent_color: whose point of view.
    """

    def __init__(self, env, weight: float, gamma: float,
                 agent_color=Color.BLUE):
        super().__init__(env)
        self.weight = weight
        self.gamma = gamma
        self.agent_color = agent_color
        self._prev_potential = 0.0

    def _potential(self) -> float:
        from catanatron.state_functions import get_visible_victory_points
        state = self.env.unwrapped.game.state
        key = f"P{state.color_to_index[self.agent_color]}"
        mine = state.player_state[f"{key}_ACTUAL_VICTORY_POINTS"]
        theirs = max(
            (get_visible_victory_points(state, c) for c in state.colors
             if c != self.agent_color),
            default=0,
        )
        return self.weight * (mine - theirs)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._prev_potential = self._potential()
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        potential = 0.0 if done else self._potential()
        shaping = self.gamma * potential - self._prev_potential
        self._prev_potential = potential
        return obs, reward + shaping, terminated, truncated, info


class SettlementBonusWrapper(Wrapper):
    """A flat bonus each time the agent builds a settlement past the opening.

    **This is not potential-based and it is not policy-invariant.** That is the
    point: :class:`PotentialShapingWrapper` telescopes to a constant and
    therefore *cannot* make the agent prefer settlements, only learn the same
    preference sooner. This wrapper deliberately moves the optimum, which makes
    it a close relative of the deleted ``RewardShapingWrapper`` and subject to
    the same suspicion. It exists to answer one question the VP-differential
    potential could not.

    Why the VP potential could not. Its ``Phi`` is the victory-point
    differential, and at this project's ruleset Longest Road is worth **+2 VP**
    -- so that potential pays *twice as much* for a road spree that takes the
    award as for a settlement. A settlement-specific term has no such hole.
    Live games show the behaviour it targets: 7 roads built and 0 settlements
    across 28 turns, with every road costing a brick a settlement also needed.

    Three deliberate limits, because a flat bonus can be gamed in a way a
    potential cannot:

    - **Settlements only, never cities.** A city bonus would re-import the
      blind spot above -- cities are already the agent's preferred spend.
    - **Capped at ``max_bonuses``.** Buildings cap at 5 settlements and 2 are
      placed in the opening, so 3 is every settlement the agent can ever build
      by its own choice. Without the cap the bonus would also pay for
      settlements rebuilt after an upgrade returns the piece.
    - **The opening is excluded for free.** Under ``PlacementWrapper`` the
      opening is played inside ``reset()``, so those settlements never pass
      through ``step()``. The cap makes that robust rather than incidental.

    A run using this **must be benchmarked with search on**. The critic learns a
    value inflated by expected future bonuses while ``src.agent.mcts`` evaluates
    leaves in the unshaped game; the last reward change to skip that check read
    50.3% bare and 42.2% under search.

    Place it INSIDE :class:`EpisodeStatsWrapper` and OUTSIDE
    :class:`TurnLimitWrapper`, matching :class:`PotentialShapingWrapper`.

    Args:
        env: the base environment.
        bonus: reward added per settlement, in units of the terminal +/-1.
        max_bonuses: how many settlements may be paid for in one episode.
        agent_color: whose settlements count.
    """

    def __init__(self, env, bonus: float, max_bonuses: int = 3,
                 agent_color=Color.BLUE):
        super().__init__(env)
        self.bonus = bonus
        self.max_bonuses = max_bonuses
        self.agent_color = agent_color
        self._paid = 0
        self._prev_settlements_avail = 5

    def _settlements_available(self) -> int:
        state = self.env.unwrapped.game.state
        key = f"P{state.color_to_index[self.agent_color]}"
        return state.player_state[f"{key}_SETTLEMENTS_AVAILABLE"]

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._paid = 0
        self._prev_settlements_avail = self._settlements_available()
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        # The piece pool falling is what a build looks like from here. An
        # upgrade to a city *returns* the piece, so the pool rises again and a
        # rebuilt settlement would otherwise be paid for twice; the cap is what
        # stops that, not this counter.
        avail = self._settlements_available()
        if avail < self._prev_settlements_avail:
            built = self._prev_settlements_avail - avail
            payable = max(0, min(built, self.max_bonuses - self._paid))
            if payable:
                reward += self.bonus * payable
                self._paid += payable
        self._prev_settlements_avail = avail

        if terminated or truncated:
            info["settlement_bonus_paid"] = self._paid * self.bonus
        return obs, reward, terminated, truncated, info


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

    def _dev_bought(self, color):
        """Development cards bought over the game, in hand and played alike.

        Derived from the piece counts rather than by counting
        BUY_DEVELOPMENT_CARD actions: a card is only ever in one of the two
        places, so the sum is exact and costs no action-log scan. The arena
        counts the same quantity off the log, and the two agree.
        """
        state = self.env.unwrapped.game.state
        key = f"P{state.color_to_index[color]}"
        return sum(
            state.player_state[f"{key}_{card}_IN_HAND"]
            + state.player_state[f"{key}_PLAYED_{card}"]
            for card in DEV_CARDS
        )

    def _vp_from_dev(self, color):
        """VP the agent holds in victory-point cards.

        The split that ``final_vp`` alone hides. Buildings cap at 9 VP, so at a
        12- or 15-VP target the rest has to come from Largest Army, Longest Road
        or these; this is the only one of the three that is pure card luck
        bought with ore and sheep, and a run drifting into a dev-card monoculture
        shows up here first.
        """
        state = self.env.unwrapped.game.state
        key = f"P{state.color_to_index[color]}"
        return (state.player_state[f"{key}_VICTORY_POINT_IN_HAND"]
                + state.player_state[f"{key}_PLAYED_VICTORY_POINT"])

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
            info["dev_bought"] = self._dev_bought(self.agent_color)
            info["vp_from_dev"] = self._vp_from_dev(self.agent_color)
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
