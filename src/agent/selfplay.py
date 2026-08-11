"""AlphaZero self-play: turn a network into training data.

Each searched decision yields one training sample ``(obs, mask, pi, z)``:

- ``pi`` is the MCTS visit distribution -- the improved policy the network is
  trained to imitate. This is the whole point of the architecture: search, not a
  gradient step, is the policy improvement operator.
- ``z`` is the value target (see :func:`compute_value_targets`).

Forced decisions (a single legal action) are played without search and produce no
sample.

**Why value targets are bootstrapped.** Vanilla AlphaZero labels every position in
a game with the final result. That works in Go, where play determines the outcome.
Catan is dice-driven: the same position can win or lose on the roll, so the final
result is a very noisy label and the value head fits poorly -- which degrades leaf
evaluation, which degrades search. We therefore blend the outcome with an n-step
bootstrap off the search's own root values, trading a little bias for a large
variance reduction. ``value_mix=1.0`` recovers textbook AlphaZero.
"""

import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field, asdict

import numpy as np

from catanatron import Color

from src.agent.encoding import action_size, obs_size
from src.agent.evaluator import NetEvaluator
from src.agent.mcts import MCTS
from src.agent.net import AlphaZeroNet
from src.env.catan_env import MAX_TURNS, make_1v1_game

P0 = Color.BLUE
P1 = Color.RED


@dataclass
class SelfPlayConfig:
    """Knobs for a self-play batch."""

    simulations: int = 100
    c_puct: float = 1.5
    dirichlet_alpha: float = 0.3
    dirichlet_epsilon: float = 0.25
    fpu_reduction: float = 0.25
    temperature: float = 1.0
    temperature_moves: int = 20
    max_turns: int = MAX_TURNS
    value_nstep: int = 24
    value_mix: float = 0.5
    batch_size: int = 1

    def mcts_kwargs(self) -> dict:
        return {
            "simulations": self.simulations,
            "c_puct": self.c_puct,
            "dirichlet_alpha": self.dirichlet_alpha,
            "dirichlet_epsilon": self.dirichlet_epsilon,
            "fpu_reduction": self.fpu_reduction,
            "max_turns": self.max_turns,
            "batch_size": self.batch_size,
        }


@dataclass
class GameResult:
    """Training samples plus per-game telemetry."""

    obs: np.ndarray
    masks: np.ndarray
    policies: np.ndarray
    values: np.ndarray
    stats: dict = field(default_factory=dict)


def _outcome(winner, color) -> float:
    if winner is None:
        return 0.0
    return 1.0 if winner == color else -1.0


def compute_value_targets(colors, root_values, winner, nstep: int, mix: float):
    """Blend the game outcome with an n-step bootstrap off search root values.

    Args:
        colors: player to move at each recorded decision.
        root_values: search value at each decision, from that player's POV.
        winner: winning Color, or None for a draw/truncation.
        nstep: how many decisions forward to bootstrap from.
        mix: weight on the final outcome; ``1 - mix`` goes to the bootstrap.

    Returns:
        float32 array of value targets, aligned with the decisions.
    """
    total = len(colors)
    targets = np.zeros(total, dtype=np.float32)
    for t in range(total):
        own = _outcome(winner, colors[t])
        future = t + nstep
        if future >= total:
            # Nothing left to bootstrap from; the outcome *is* the true value.
            bootstrap = own
        else:
            sign = 1.0 if colors[future] == colors[t] else -1.0
            bootstrap = sign * root_values[future]
        targets[t] = mix * own + (1.0 - mix) * bootstrap
    return targets


def play_game(evaluator, config: SelfPlayConfig, seed=None, rng=None) -> GameResult:
    """Play one self-play game, both seats driven by the same evaluator."""
    rng = rng or np.random.default_rng(seed)
    game = make_1v1_game(seed=seed)
    mcts = MCTS(evaluator, **config.mcts_kwargs())

    obs_list, mask_list, policy_list, colors, root_values = [], [], [], [], []
    searched = 0

    while game.winning_color() is None and game.state.num_turns < config.max_turns:
        actions = game.state.playable_actions
        if len(actions) == 1:
            game.execute(actions[0], validate_action=False)
            continue

        result = mcts.search(game, rng=rng)
        obs_list.append(result.obs)
        mask_list.append(result.mask)
        policy_list.append(result.policy)
        colors.append(result.color)
        root_values.append(result.value)

        temperature = (
            config.temperature if searched < config.temperature_moves else 0.0
        )
        action = (
            result.sample_action(temperature, rng)
            if temperature > 0
            else result.best_action()
        )
        searched += 1
        game.execute(action, validate_action=False)

    winner = game.winning_color()
    values = compute_value_targets(
        colors, root_values, winner, config.value_nstep, config.value_mix
    )

    return GameResult(
        obs=np.asarray(obs_list, dtype=np.float32).reshape(-1, obs_size()),
        masks=np.asarray(mask_list, dtype=bool).reshape(-1, action_size()),
        policies=np.asarray(policy_list, dtype=np.float32).reshape(-1, action_size()),
        values=values,
        stats=_game_stats(game, winner, searched),
    )


def _game_stats(game, winner, searched: int) -> dict:
    state = game.state
    stats = {
        "turns": state.num_turns,
        "decisions": searched,
        "draw": winner is None,
    }
    for label, color in (("p0", P0), ("p1", P1)):
        key = f"P{state.color_to_index[color]}"
        stats[f"{label}_vp"] = state.player_state[f"{key}_ACTUAL_VICTORY_POINTS"]
        stats[f"{label}_settlements"] = (
            5 - state.player_state[f"{key}_SETTLEMENTS_AVAILABLE"]
        )
        stats[f"{label}_cities"] = 4 - state.player_state[f"{key}_CITIES_AVAILABLE"]
        stats[f"{label}_roads"] = 15 - state.player_state[f"{key}_ROADS_AVAILABLE"]
    return stats


# --------------------------------------------------------------------------
# Parallel generation
# --------------------------------------------------------------------------
def _worker(payload):
    """Play a batch of games in a fresh process. Must be importable top-level."""
    import torch

    net_blob, config_dict, num_games, seed = payload
    # Each worker gets one core; the pool provides the parallelism, and letting
    # torch spawn its own threads per worker oversubscribes the machine badly.
    torch.set_num_threads(1)

    net = AlphaZeroNet(**net_blob["config"])
    net.load_state_dict(net_blob["state_dict"])
    net.eval()

    evaluator = NetEvaluator(net)
    config = SelfPlayConfig(**config_dict)
    rng = np.random.default_rng(seed)
    return [
        play_game(evaluator, config, seed=int(rng.integers(2**31 - 1)), rng=rng)
        for _ in range(num_games)
    ]


def generate_games(net, config: SelfPlayConfig, num_games: int, workers: int = 0,
                   seed=None) -> list:
    """Generate ``num_games`` self-play games, optionally across processes.

    Args:
        net: the current AlphaZeroNet.
        config: self-play settings.
        num_games: total games to play.
        workers: process count; 0 or 1 runs in-process (easier to debug/profile).
        seed: base RNG seed.

    Returns:
        list[GameResult].
    """
    rng = np.random.default_rng(seed)
    config_dict = asdict(config)

    if workers <= 1:
        evaluator = NetEvaluator(net)
        return [
            play_game(evaluator, config, seed=int(rng.integers(2**31 - 1)), rng=rng)
            for _ in range(num_games)
        ]

    net_blob = {
        "config": net.config(),
        "state_dict": {k: v.cpu() for k, v in net.state_dict().items()},
    }
    workers = min(workers, num_games, os.cpu_count() or workers)
    per_worker = [num_games // workers] * workers
    for i in range(num_games % workers):
        per_worker[i] += 1

    payloads = [
        (net_blob, config_dict, count, int(rng.integers(2**31 - 1)))
        for count in per_worker
        if count > 0
    ]

    results = []
    with ProcessPoolExecutor(max_workers=len(payloads)) as pool:
        for batch in pool.map(_worker, payloads):
            results.extend(batch)
    return results
