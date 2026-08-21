"""Head-to-head match play and a registry for building any agent by name.

Used by the PPO self-play ladder (``src.agent.train``) and by
``src.eval.benchmark``, so every comparison in the project runs the same
alternating-seat protocol.

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
from catanatron.models.enums import ActionType
from catanatron.models.player import RandomPlayer
from catanatron.players.search import VictoryPointPlayer
from catanatron.players.weighted_random import WeightedRandomPlayer

from src.agent.evaluator import PPOEvaluator, UniformEvaluator
from src.agent.mcts import MCTSPlayer
from src.env.catan_env import MAX_TURNS, make_1v1_game
# The constant only, so this stays a torch-free import; wrap_factory is still
# imported lazily inside _with_placement.
from src.placement.chooser import PARTNER_RANK

BASELINE_BOTS = {
    "random": RandomPlayer,
    "weighted": WeightedRandomPlayer,
    "value": VictoryPointPlayer,
}


class MatchResult:
    """Aggregate outcome of a match, from the challenger's point of view."""

    __slots__ = ("wins", "losses", "draws", "games", "mean_turns",
                 "mean_vp", "mean_opp_vp", "mean_settlements", "mean_cities",
                 "mean_roads", "mean_knights", "mean_opp_knights",
                 "mean_dev_bought", "mean_dev_unplayed", "mean_vp_from_dev",
                 "mean_end_hand",
                 "mean_final_hand", "mean_trailing_roads",
                 "loss_final_hand", "loss_dev_unplayed", "loss_trailing_roads")

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

    @property
    def knights_diff(self) -> float:
        """Knights played minus the opponent's, averaged over the match.

        Largest Army is a race, not a threshold met in isolation -- four knights
        is dominant against an opponent playing one and irrelevant against one
        playing five. The absolute count cannot tell those apart; this can.
        """
        return self.mean_knights - self.mean_opp_knights

    def summary(self) -> str:
        lines = (
            f"{self.wins}W-{self.losses}L-{int(self.draws)}D "
            f"(score {self.score:.1%}) | "
            f"avg turns {self.mean_turns:.0f}, VP {self.mean_vp:.1f} vs "
            f"{self.mean_opp_vp:.1f}, built {self.mean_settlements:.1f} settlements / "
            f"{self.mean_cities:.1f} cities / {self.mean_roads:.1f} roads, "
            f"played {self.mean_knights:.1f} knights ({self.knights_diff:+.1f}), "
            f"{self.mean_vp_from_dev:.1f} VP from dev cards\n"
            f"waste: {self.mean_dev_bought:.1f} dev bought "
            f"({self.mean_dev_unplayed:.1f} dead), "
            f"{self.mean_trailing_roads:.1f} trailing roads, hand "
            f"{self.mean_end_hand:.1f} at end-turn / "
            f"{self.mean_final_hand:.1f} at game end"
        )
        if self.losses:
            lines += (
                f" (losses: {self.loss_final_hand:.1f} held, "
                f"{self.loss_dev_unplayed:.1f} dead dev, "
                f"{self.loss_trailing_roads:.1f} trailing)"
            )
        return lines


RESOURCES = ("WOOD", "BRICK", "SHEEP", "WHEAT", "ORE")
# VICTORY_POINT is deliberately absent: a VP card in hand scored, it is not dead.
DEAD_DEV_CARDS = ("KNIGHT", "MONOPOLY", "YEAR_OF_PLENTY", "ROAD_BUILDING")


def _hand_size(state, color) -> int:
    key = f"P{state.color_to_index[color]}"
    return sum(state.player_state[f"{key}_{r}_IN_HAND"] for r in RESOURCES)


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
        # Resources still in hand when the game ended: in a loss, cards that
        # were hoarded past the point of usefulness (or never spendable).
        "final_hand": _hand_size(state, color),
        # Development cards bought but never cashed in -- pure waste.
        "dev_unplayed": sum(
            state.player_state[f"{key}_{card}_IN_HAND"] for card in DEAD_DEV_CARDS
        ),
        # VP held in victory-point cards. Buildings cap at 9 VP, so above that
        # target the remainder comes from Largest Army, Longest Road or these;
        # separating them says which route the agent actually took.
        "vp_from_dev": (state.player_state[f"{key}_VICTORY_POINT_IN_HAND"]
                        + state.player_state[f"{key}_PLAYED_VICTORY_POINT"]),
    }


def _action_log_stats(actions, color) -> dict:
    """Waste telemetry mined from a finished game's action log.

    ``trailing_roads`` counts roads built after the player's last settlement or
    city -- roads that never enabled anything. The first two settlements and two
    roads are initial placement and excluded, so a player who never builds
    another building has every later road counted as trailing.
    """
    dev_bought = 0
    placement_settlements = placement_roads = 0
    roads_since_building = 0
    for action in actions:
        if action.color != color:
            continue
        if action.action_type == ActionType.BUY_DEVELOPMENT_CARD:
            dev_bought += 1
        elif action.action_type == ActionType.BUILD_SETTLEMENT:
            if placement_settlements < 2:
                placement_settlements += 1
            else:
                roads_since_building = 0
        elif action.action_type == ActionType.BUILD_CITY:
            roads_since_building = 0
        elif action.action_type == ActionType.BUILD_ROAD:
            if placement_roads < 2:
                placement_roads += 1
            else:
                roads_since_building += 1
    return {"dev_bought": dev_bought, "trailing_roads": roads_since_building}


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
    # Hand size at each of the challenger's END_TURNs has to be sampled live --
    # the action log records the decision but not the hand it was made with.
    end_hands = []
    actions_seen = 0
    while game.winning_color() is None and game.state.num_turns < MAX_TURNS:
        game.play_tick()
        for action in game.state.actions[actions_seen:]:
            if (action.color == challenger_color
                    and action.action_type == ActionType.END_TURN):
                end_hands.append(_hand_size(game.state, challenger_color))
        actions_seen = len(game.state.actions)

    winner = game.winning_color()
    mine = _player_stats(game.state, challenger_color)
    theirs = _player_stats(game.state, opponent_color)
    log_stats = _action_log_stats(game.state.actions, challenger_color)
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
        "final_hand": mine["final_hand"],
        "dev_unplayed": mine["dev_unplayed"],
        "vp_from_dev": mine["vp_from_dev"],
        "dev_bought": log_stats["dev_bought"],
        "trailing_roads": log_stats["trailing_roads"],
        "end_hand": float(np.mean(end_hands)) if end_hands else 0.0,
    }


def _collect(records, num_games: int) -> MatchResult:
    """Fold per-game records into a :class:`MatchResult`."""
    outcomes = [r["outcome"] for r in records]
    lost = [r for r in records if r["outcome"] == "loss"]

    def mean(key, over=records):
        return float(np.mean([r[key] for r in over])) if over else 0.0

    return MatchResult(
        wins=outcomes.count("win"), losses=outcomes.count("loss"),
        draws=outcomes.count("draw"), games=num_games,
        mean_turns=mean("turns"), mean_vp=mean("vp"), mean_opp_vp=mean("opp_vp"),
        mean_settlements=mean("settlements"), mean_cities=mean("cities"),
        mean_roads=mean("roads"), mean_knights=mean("knights"),
        mean_opp_knights=mean("opp_knights"),
        mean_dev_bought=mean("dev_bought"), mean_dev_unplayed=mean("dev_unplayed"),
        mean_vp_from_dev=mean("vp_from_dev"),
        mean_end_hand=mean("end_hand"), mean_final_hand=mean("final_hand"),
        mean_trailing_roads=mean("trailing_roads"),
        loss_final_hand=mean("final_hand", lost),
        loss_dev_unplayed=mean("dev_unplayed", lost),
        loss_trailing_roads=mean("trailing_roads", lost),
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
@dataclass
class AgentSpec:
    """A picklable description of an agent, for matches that span processes."""

    kind: str
    model_path: str = None
    simulations: int = 100
    horizon: int = None
    batch_size: int = 1
    placement_path: str = None
    bundle_path: str = None
    partner_rank: int = PARTNER_RANK


def build_agent_from_spec(spec: AgentSpec):
    """Rebuild an agent factory from its spec, inside whatever process needs it."""
    return build_agent(
        spec.kind, spec.model_path, spec.simulations, spec.batch_size,
        spec.placement_path, spec.bundle_path, spec.partner_rank,
        horizon=spec.horizon,
    )


def _with_placement(factory, placement_path, bundle_path=None,
                    partner_rank=PARTNER_RANK):
    """Optionally hand the opening to a learned placement scorer."""
    if placement_path is None:
        return factory
    from src.placement.player import wrap_factory
    return wrap_factory(factory, placement_path, bundle_path, partner_rank)


def _as_factory(agent):
    """Accept either a factory or an :class:`AgentSpec`."""
    return build_agent_from_spec(agent) if isinstance(agent, AgentSpec) else agent


def build_agent(spec: str, model_path=None, simulations: int = 100,
                batch_size: int = 1, placement_path=None, bundle_path=None,
                partner_rank: int = PARTNER_RANK, horizon: int = None):
    """Build an agent factory from a short name.

    Specs:
        ``random`` / ``weighted`` / ``value`` -- catanatron's built-in bots.
        ``mcts``      -- bare PUCT search with uniform priors and no value net.
                         The control that isolates lookahead from knowledge.
        ``ppo``       -- a MaskablePPO checkpoint played greedily.
        ``ppo-mcts``  -- MCTS using that same PPO net for priors and values.
                         **This is the shipped agent**; it is worth ~10 points
                         over ``ppo`` on the same weights.

    Args:
        spec: one of the names above.
        model_path: checkpoint path, required for the network-backed specs.
        simulations: playouts per decision for search-backed specs.
        batch_size: leaves per evaluator call; see :class:`~src.agent.mcts.MCTS`.
        horizon: cap the search at this many game turns past the root; ``None``
            lets the simulation budget set the depth. Search-backed specs only.
        placement_path: optional PlacementNet checkpoint. Any agent above can be
            given one; it then takes over the opening settlements and nothing
            else, so the same spec with and without it isolates the opening.
        bundle_path: optional BundleNet checkpoint. Requires ``placement_path``,
            which shortlists the corners it searches over. Openings are then
            chosen as a pair rather than greedily one corner at a time.
        partner_rank: how pessimistic the pair search is about the first seat's
            second settlement surviving; see
            :data:`~src.placement.chooser.PARTNER_RANK`. Only meaningful
            alongside ``bundle_path``.

    Returns:
        callable(Color) -> Player.
    """
    return _with_placement(
        _build_core_agent(spec, model_path, simulations, batch_size, horizon),
        placement_path, bundle_path, partner_rank,
    )


def _build_core_agent(spec: str, model_path, simulations: int, batch_size: int,
                      horizon: int = None):
    if spec in BASELINE_BOTS:
        bot = BASELINE_BOTS[spec]
        return lambda color: bot(color)

    if spec == "mcts":
        evaluator = UniformEvaluator()
        return lambda color: MCTSPlayer(
            color, evaluator, simulations=simulations, dirichlet_epsilon=0.0,
            batch_size=batch_size, horizon=horizon,
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
            batch_size=batch_size, horizon=horizon,
        )

    raise ValueError(f"Unknown agent spec: {spec}")
