"""Head-to-head match play and a registry for building any agent by name.

Used by the AlphaZero promotion gate (``src.agent.train_az``) and by the
evaluation entry points (``src.eval.benchmark``, ``src.eval.stage0``), so every
comparison in the project runs the same alternating-seat protocol.

Seats alternate between games. In 1v1 Catan the first player picks first in the
initial placement, which is a real edge; without alternation a benchmark mostly
measures who got P0.

**Matches can run across processes.** Games are independent, so ``workers`` fans
them over a pool. That needs agents to survive pickling, which the factory
closures below do not -- hence :class:`AgentSpec`, a declarative description each
worker rebuilds locally. Serial play still accepts plain factories.
"""

import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

import numpy as np

from catanatron import Color
from catanatron.models.player import RandomPlayer
from catanatron.players.search import VictoryPointPlayer
from catanatron.players.weighted_random import WeightedRandomPlayer

from src.agent.evaluator import NetEvaluator, PPOEvaluator, UniformEvaluator
from src.agent.mcts import MCTSPlayer
from src.agent.net import AlphaZeroNet
from src.env.catan_env import MAX_TURNS, make_1v1_game

BASELINE_BOTS = {
    "random": RandomPlayer,
    "weighted": WeightedRandomPlayer,
    "value": VictoryPointPlayer,
}


class MatchResult:
    """Aggregate outcome of a match, from the challenger's point of view."""

    __slots__ = ("wins", "losses", "draws", "games", "mean_turns",
                 "mean_vp", "mean_opp_vp", "mean_settlements", "mean_cities",
                 "mean_roads", "mean_knights")

    def __init__(self, **kwargs):
        for key in self.__slots__:
            setattr(self, key, kwargs.get(key, 0.0))

    @property
    def win_rate(self) -> float:
        return self.wins / self.games if self.games else 0.0

    @property
    def score(self) -> float:
        """Win rate counting a draw as half a win."""
        return (self.wins + 0.5 * self.draws) / self.games if self.games else 0.0

    def summary(self) -> str:
        return (
            f"{self.wins}W-{self.losses}L-{int(self.draws)}D "
            f"(score {self.score:.1%}) | "
            f"avg turns {self.mean_turns:.0f}, VP {self.mean_vp:.1f} vs "
            f"{self.mean_opp_vp:.1f}, built {self.mean_settlements:.1f} settlements / "
            f"{self.mean_cities:.1f} cities / {self.mean_roads:.1f} roads, "
            f"played {self.mean_knights:.1f} knights"
        )


def _player_stats(state, color) -> dict:
    key = f"P{state.color_to_index[color]}"
    return {
        "vp": state.player_state[f"{key}_ACTUAL_VICTORY_POINTS"],
        "settlements": 5 - state.player_state[f"{key}_SETTLEMENTS_AVAILABLE"],
        "cities": 4 - state.player_state[f"{key}_CITIES_AVAILABLE"],
        "roads": 15 - state.player_state[f"{key}_ROADS_AVAILABLE"],
        # Knights *played*, not bought -- an unplayed knight is worth nothing and
        # counts toward nothing. With Longest Road disabled, Largest Army is the
        # only +2 on the board, so this is the tell for whether the agent has
        # found that route: three knights is the threshold, and a mean well under
        # 3 means it is buying development cards without cashing them in.
        "knights": state.player_state[f"{key}_PLAYED_KNIGHT"],
    }


def _play_one_game(challenger_factory, opponent_factory, index: int, seed: int) -> dict:
    """Play game ``index`` and return its record from the challenger's side.

    Seat is decided by index parity and the seed is drawn per index, so splitting
    games across processes never changes which seat or seed a given game gets.

    That is *not* enough to make a match bit-reproducible, and it is worth knowing
    which knobs actually control that. Measured:

    - same seed, same ``workers``, ``PYTHONHASHSEED`` pinned -- identical.
    - hash seed unpinned -- differs; catanatron's action generation iterates
      hash-ordered containers, so per-process string hashing reorders equal-value
      actions and the argmax tie-break lands elsewhere.
    - different ``workers`` -- differs; games are not fully independent, as engine
      global RNG state survives between games in a process.

    So pin both ``--workers`` and ``PYTHONHASHSEED`` to reproduce a number, and
    otherwise treat match results as samples with real variance.
    """
    challenger_color = Color.BLUE if index % 2 == 0 else Color.RED
    opponent_color = Color.RED if index % 2 == 0 else Color.BLUE
    challenger = challenger_factory(challenger_color)
    opponent = opponent_factory(opponent_color)
    players = (
        [challenger, opponent]
        if challenger_color == Color.BLUE
        else [opponent, challenger]
    )

    game = make_1v1_game(players=players, seed=seed)
    while game.winning_color() is None and game.state.num_turns < MAX_TURNS:
        game.play_tick()

    winner = game.winning_color()
    mine = _player_stats(game.state, challenger_color)
    theirs = _player_stats(game.state, opponent_color)
    return {
        "index": index,
        "outcome": ("win" if winner == challenger_color
                    else "draw" if winner is None else "loss"),
        "turns": game.state.num_turns,
        "vp": mine["vp"],
        "opp_vp": theirs["vp"],
        "settlements": mine["settlements"],
        "cities": mine["cities"],
        "roads": mine["roads"],
        "knights": mine["knights"],
        "opp_knights": theirs["knights"],
    }


def _collect(records, num_games: int) -> MatchResult:
    """Fold per-game records into a :class:`MatchResult`."""
    outcomes = [r["outcome"] for r in records]

    def mean(key):
        return float(np.mean([r[key] for r in records])) if records else 0.0

    return MatchResult(
        wins=outcomes.count("win"), losses=outcomes.count("loss"),
        draws=outcomes.count("draw"), games=num_games,
        mean_turns=mean("turns"), mean_vp=mean("vp"), mean_opp_vp=mean("opp_vp"),
        mean_settlements=mean("settlements"), mean_cities=mean("cities"),
        mean_roads=mean("roads"), mean_knights=mean("knights"),
    )


def _match_worker(payload):
    """Play a slice of a match in a fresh process. Must be importable top-level."""
    import torch

    challenger_spec, opponent_spec, assignments = payload
    # One core per worker; the pool supplies the parallelism, and batch-1 search
    # forwards lose to thread synchronisation anyway.
    torch.set_num_threads(1)

    challenger = build_agent_from_spec(challenger_spec)
    opponent = build_agent_from_spec(opponent_spec)
    return [
        _play_one_game(challenger, opponent, index, seed)
        for index, seed in assignments
    ]


def play_match(challenger, opponent, num_games: int, seed=None,
               progress=False, workers: int = 0) -> MatchResult:
    """Play ``num_games`` alternating-seat games between two agents.

    Args:
        challenger: callable(Color) -> Player, or an :class:`AgentSpec`. The
            agent being measured.
        opponent: callable(Color) -> Player, or an :class:`AgentSpec`.
        num_games: how many games to play.
        seed: base RNG seed.
        progress: print a line per finished game (serial) or chunk (parallel).
        workers: processes to spread games over. 0/1 runs in-process. Requires
            both agents to be :class:`AgentSpec` -- factory closures cannot be
            pickled.

    Returns:
        A :class:`MatchResult` from the challenger's perspective.
    """
    rng = np.random.default_rng(seed)
    seeds = [int(rng.integers(2**31 - 1)) for _ in range(num_games)]

    if workers <= 1:
        challenger_factory = _as_factory(challenger)
        opponent_factory = _as_factory(opponent)
        records = []
        for index in range(num_games):
            record = _play_one_game(
                challenger_factory, opponent_factory, index, seeds[index]
            )
            records.append(record)
            if progress:
                print(f"  game {index + 1}/{num_games}: {record['outcome']}"
                      f" in {record['turns']} turns", flush=True)
        return _collect(records, num_games)

    for agent in (challenger, opponent):
        if not isinstance(agent, AgentSpec):
            raise TypeError(
                "workers > 1 requires AgentSpec for both agents; a factory "
                "closure cannot cross a process boundary."
            )

    workers = min(workers, num_games, os.cpu_count() or workers)
    chunks = [[] for _ in range(workers)]
    for index in range(num_games):
        chunks[index % workers].append((index, seeds[index]))

    payloads = [
        (challenger, opponent, assignments)
        for assignments in chunks if assignments
    ]
    records = []
    with ProcessPoolExecutor(max_workers=len(payloads)) as pool:
        for batch in pool.map(_match_worker, payloads):
            records.extend(batch)
            if progress:
                print(f"  {len(records)}/{num_games} games done", flush=True)
    return _collect(records, num_games)


# --------------------------------------------------------------------------
# Agent factories
# --------------------------------------------------------------------------
def net_factory(net, simulations: int, **mcts_kwargs):
    """MCTS on an AlphaZeroNet. Evaluation play: greedy, no root noise."""
    evaluator = NetEvaluator(net)
    return lambda color: MCTSPlayer(
        color, evaluator, simulations=simulations,
        dirichlet_epsilon=0.0, **mcts_kwargs
    )


@dataclass
class AgentSpec:
    """A picklable description of an agent, for matches that span processes.

    ``net_blob`` carries an in-memory network (config + CPU state dict) so the
    training loop can arena its live challenger without writing a checkpoint
    first; ``model_path`` covers agents loaded from disk.
    """

    kind: str
    model_path: str = None
    simulations: int = 100
    batch_size: int = 1
    net_blob: dict = None

    @classmethod
    def from_net(cls, net, simulations: int, batch_size: int = 1) -> "AgentSpec":
        return cls(
            kind="az", simulations=simulations, batch_size=batch_size,
            net_blob={
                "config": net.config(),
                "state_dict": {k: v.cpu() for k, v in net.state_dict().items()},
            },
        )


def build_agent_from_spec(spec: AgentSpec):
    """Rebuild an agent factory from its spec, inside whatever process needs it."""
    if spec.net_blob is not None:
        net = AlphaZeroNet(**spec.net_blob["config"])
        net.load_state_dict(spec.net_blob["state_dict"])
        net.eval()
        return net_factory(net, spec.simulations, batch_size=spec.batch_size)
    return build_agent(
        spec.kind, spec.model_path, spec.simulations, spec.batch_size
    )


def _as_factory(agent):
    """Accept either a factory or an :class:`AgentSpec`."""
    return build_agent_from_spec(agent) if isinstance(agent, AgentSpec) else agent


def build_agent(spec: str, model_path=None, simulations: int = 100,
                batch_size: int = 1):
    """Build an agent factory from a short name.

    Specs:
        ``random`` / ``weighted`` / ``value`` -- catanatron's built-in bots.
        ``mcts``      -- bare PUCT search with uniform priors and no value net.
                         The control that isolates lookahead from knowledge.
        ``ppo``       -- a MaskablePPO checkpoint played greedily (the old agent).
        ``ppo-mcts``  -- MCTS using that same PPO net for priors and values.
        ``az``        -- MCTS on an AlphaZeroNet checkpoint.

    Args:
        spec: one of the names above.
        model_path: checkpoint path, required for the network-backed specs.
        simulations: playouts per decision for search-backed specs.
        batch_size: leaves per evaluator call; see :class:`~src.agent.mcts.MCTS`.

    Returns:
        callable(Color) -> Player.
    """
    if spec in BASELINE_BOTS:
        bot = BASELINE_BOTS[spec]
        return lambda color: bot(color)

    if spec == "mcts":
        evaluator = UniformEvaluator()
        return lambda color: MCTSPlayer(
            color, evaluator, simulations=simulations, dirichlet_epsilon=0.0,
            batch_size=batch_size,
        )

    if spec in ("ppo", "ppo-mcts"):
        if model_path is None:
            raise ValueError(f"'{spec}' requires --model")
        if spec == "ppo":
            from sb3_contrib import MaskablePPO
            from src.agent.opponent import PolicyPlayer
            model = MaskablePPO.load(
                str(model_path), device="cpu", custom_objects={"n_steps": 1}
            )
            return lambda color: PolicyPlayer(color, model)
        evaluator = PPOEvaluator.from_path(model_path)
        return lambda color: MCTSPlayer(
            color, evaluator, simulations=simulations, dirichlet_epsilon=0.0,
            batch_size=batch_size,
        )

    if spec == "az":
        if model_path is None:
            raise ValueError("'az' requires --model")
        return net_factory(
            AlphaZeroNet.load(model_path), simulations, batch_size=batch_size
        )

    raise ValueError(f"Unknown agent spec: {spec}")
