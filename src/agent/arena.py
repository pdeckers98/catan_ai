"""Head-to-head match play and a registry for building any agent by name.

Used by the AlphaZero promotion gate (``src.agent.train_az``) and by the
evaluation entry points (``src.eval.benchmark``, ``src.eval.stage0``), so every
comparison in the project runs the same alternating-seat protocol.

Seats alternate between games. In 1v1 Catan the first player picks first in the
initial placement, which is a real edge; without alternation a benchmark mostly
measures who got P0.
"""

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
                 "mean_roads")

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
            f"{self.mean_cities:.1f} cities / {self.mean_roads:.1f} roads"
        )


def _player_stats(state, color) -> dict:
    key = f"P{state.color_to_index[color]}"
    return {
        "vp": state.player_state[f"{key}_ACTUAL_VICTORY_POINTS"],
        "settlements": 5 - state.player_state[f"{key}_SETTLEMENTS_AVAILABLE"],
        "cities": 4 - state.player_state[f"{key}_CITIES_AVAILABLE"],
        "roads": 15 - state.player_state[f"{key}_ROADS_AVAILABLE"],
    }


def play_match(challenger_factory, opponent_factory, num_games: int,
               seed=None, progress=False) -> MatchResult:
    """Play ``num_games`` alternating-seat games between two agent factories.

    Args:
        challenger_factory: callable(Color) -> Player, the agent being measured.
        opponent_factory: callable(Color) -> Player.
        num_games: how many games to play.
        seed: base RNG seed.
        progress: print a line per game.

    Returns:
        A :class:`MatchResult` from the challenger's perspective.
    """
    rng = np.random.default_rng(seed)
    wins = losses = draws = 0
    turns, vps, opp_vps, settlements, cities, roads = [], [], [], [], [], []

    for index in range(num_games):
        challenger_color = Color.BLUE if index % 2 == 0 else Color.RED
        opponent_color = Color.RED if index % 2 == 0 else Color.BLUE
        challenger = challenger_factory(challenger_color)
        opponent = opponent_factory(opponent_color)
        players = (
            [challenger, opponent]
            if challenger_color == Color.BLUE
            else [opponent, challenger]
        )

        game = make_1v1_game(players=players, seed=int(rng.integers(2**31 - 1)))
        while game.winning_color() is None and game.state.num_turns < MAX_TURNS:
            game.play_tick()

        winner = game.winning_color()
        if winner == challenger_color:
            wins += 1
        elif winner is None:
            draws += 1
        else:
            losses += 1

        mine = _player_stats(game.state, challenger_color)
        theirs = _player_stats(game.state, opponent_color)
        turns.append(game.state.num_turns)
        vps.append(mine["vp"])
        opp_vps.append(theirs["vp"])
        settlements.append(mine["settlements"])
        cities.append(mine["cities"])
        roads.append(mine["roads"])

        if progress:
            print(f"  game {index + 1}/{num_games}: "
                  f"{'win' if winner == challenger_color else 'loss' if winner else 'draw'}"
                  f" in {game.state.num_turns} turns", flush=True)

    return MatchResult(
        wins=wins, losses=losses, draws=draws, games=num_games,
        mean_turns=float(np.mean(turns)), mean_vp=float(np.mean(vps)),
        mean_opp_vp=float(np.mean(opp_vps)),
        mean_settlements=float(np.mean(settlements)),
        mean_cities=float(np.mean(cities)), mean_roads=float(np.mean(roads)),
    )


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


def build_agent(spec: str, model_path=None, simulations: int = 100):
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

    Returns:
        callable(Color) -> Player.
    """
    if spec in BASELINE_BOTS:
        bot = BASELINE_BOTS[spec]
        return lambda color: bot(color)

    if spec == "mcts":
        evaluator = UniformEvaluator()
        return lambda color: MCTSPlayer(
            color, evaluator, simulations=simulations, dirichlet_epsilon=0.0
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
            color, evaluator, simulations=simulations, dirichlet_epsilon=0.0
        )

    if spec == "az":
        if model_path is None:
            raise ValueError("'az' requires --model")
        return net_factory(AlphaZeroNet.load(model_path), simulations)

    raise ValueError(f"Unknown agent spec: {spec}")
