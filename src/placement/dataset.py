"""Generate outcome-labelled opening placements by playing games.

The labelling is the whole design, so it is worth stating plainly: **no human
ever says which placement is good.** Openings are chosen uniformly at random,
the game is played out, and the label is what happened. That sidesteps the
discounting problem entirely -- the target is the undiscounted result -- and it
fixes the data starvation that broke the main policy, because here every sample
is a placement decision instead of two in three hundred.

**Duplicate-board pairing.** A single game's outcome is mostly dice and board.
Labelling a placement with a raw win/loss buries the placement's contribution
under that noise, so each sample comes from a *pair* of games: the same board is
played twice with the four opening nodes swapped between the seats, and the label
is the difference. If the same seat wins both, the board explained the result and
both openings are labelled 0. If swapping the openings swaps the winner, the
openings explained it and the labels go to +-1.

The honest caveat: this duplicates the **board**, not the dice. Catanatron rolls
off the global RNG, and once the two seats hold different corners they take
different actions and the roll sequences diverge. Board layout is the larger and
more systematic of the two nuisance terms and it is fully controlled; dice are
only correlated early. It is variance reduction, not elimination.

The rollout agent that plays the other ~120 turns is a knob. Weighted-random is
the default because it is fast and symmetric, which keeps labels about the
opening rather than about one side's mid-game skill.
"""

import argparse
import os
import random
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from catanatron import Color
from catanatron.models.enums import ActionType
from catanatron.models.player import Player, RandomPlayer
from catanatron.players.weighted_random import WeightedRandomPlayer

from src.env.catan_env import MAX_TURNS, make_1v1_game
from src.placement.features import (
    encode_candidates,
    feature_size,
    node_features,
)

ROLLOUT_BOTS = {
    "random": RandomPlayer,
    "weighted": WeightedRandomPlayer,
}


class ExplorerPlayer(Player):
    """Plays openings for data collection, then hands over to a rollout bot.

    Settlement choices during the initial build phase come from one of three
    places, in priority order: a forced node list (the swapped replay), the
    current model with probability ``1 - epsilon``, or uniform random. Everything
    else -- initial roads included -- is delegated to ``rollout``.

    Uniform random is the default and is not laziness. Sampling openings from a
    scorer, learned or hand-written, would fill the dataset with that scorer's
    preferences and leave the model no counter-examples to learn from; coverage
    is the point of the exploration phase.

    Args:
        color: seat.
        rollout: the Player that decides everything outside opening settlements.
        rng: ``random.Random`` for exploration, kept separate from the engine's
            global RNG so exploration never perturbs the dice.
        model: optional :class:`~src.placement.model.PlacementNet` for
            epsilon-greedy exploration in later iterations.
        epsilon: probability of a uniform-random pick when ``model`` is set.
        forced: node ids to play in order, for the duplicate replay.
    """

    def __init__(self, color, rollout, rng, model=None, epsilon=1.0, forced=None):
        super().__init__(color)
        self.rollout = rollout
        self.rng = rng
        self.model = model
        self.epsilon = epsilon
        self.forced = list(forced) if forced else []
        # (node_id, features-at-decision-time) for each opening settlement.
        self.picks = []

    def decide(self, game, playable_actions):
        settlements = [
            a for a in playable_actions
            if a.action_type == ActionType.BUILD_SETTLEMENT
        ]
        if not (game.state.is_initial_build_phase and settlements):
            return self.rollout.decide(game, playable_actions)

        nodes = [a.value for a in settlements]
        if self.forced:
            chosen = self.forced.pop(0)
            if chosen not in nodes:
                raise ValueError(
                    f"replay desync: node {chosen} is not legal for {self.color}"
                )
        elif self.model is not None and self.rng.random() >= self.epsilon:
            scores = self.model.score(encode_candidates(game, self.color, nodes))
            chosen = nodes[int(np.argmax(scores))]
        else:
            chosen = self.rng.choice(nodes)

        self.picks.append((chosen, node_features(game, self.color, chosen)))
        return next(a for a in settlements if a.value == chosen)


def _outcome(game, color) -> float:
    """+1 win, -1 loss, 0 draw or turn-limit truncation."""
    winner = game.winning_color()
    if winner is None:
        return 0.0
    return 1.0 if winner == color else -1.0


def _play(seed, rollout_kind, rng_seed, forced_by_color=None, model=None,
          epsilon=1.0):
    """Play one game with explorer openings; returns (game, {color: explorer})."""
    forced_by_color = forced_by_color or {}
    bot = ROLLOUT_BOTS[rollout_kind]
    explorers = {}
    players = []
    for offset, color in enumerate((Color.BLUE, Color.RED)):
        explorer = ExplorerPlayer(
            color,
            rollout=bot(color),
            # Distinct streams per seat, deterministic given rng_seed.
            rng=random.Random(rng_seed * 2 + offset),
            model=model,
            epsilon=epsilon,
            forced=forced_by_color.get(color),
        )
        explorers[color] = explorer
        players.append(explorer)

    game = make_1v1_game(players=players, seed=seed)
    while game.winning_color() is None and game.state.num_turns < MAX_TURNS:
        game.play_tick()
    return game, explorers


def _draft_order(game):
    """The four opening settlements as (color, node_id), in the order played."""
    picks = [
        (a.color, a.value) for a in game.state.actions
        if a.action_type == ActionType.BUILD_SETTLEMENT
    ]
    return picks[:4]


def generate_pair(seed: int, rollout_kind: str = "weighted", model=None,
                  epsilon: float = 1.0):
    """Play a board twice with the openings swapped; return labelled samples.

    Returns:
        (features, labels) -- float32 (4, F) and (4,), or empty arrays if the
        pair could not be completed.
    """
    game_a, explorers = _play(seed, rollout_kind, seed, model=model, epsilon=epsilon)

    order = _draft_order(game_a)
    if len(order) < 4:
        return np.zeros((0, feature_size()), np.float32), np.zeros(0, np.float32)

    # Snake draft: seats pick n1, n2, n3, n4 in the order P0, P1, P1, P0. Swapping
    # ownership means P0 now takes {n2, n3} and P1 takes {n1, n4}, which in draft
    # order is n2, n1, n4, n3. Every node keeps its non-adjacency to the other
    # three, so the swapped assignment is always legal.
    nodes = [node for _, node in order]
    swapped = [nodes[1], nodes[0], nodes[3], nodes[2]]
    first_seat = order[0][0]
    second_seat = order[1][0]
    forced = {
        first_seat: [swapped[0], swapped[3]],
        second_seat: [swapped[1], swapped[2]],
    }

    game_b, _ = _play(seed, rollout_kind, seed, forced_by_color=forced)

    # Paired label: how much better the first seat did holding its own opening
    # than the second seat did holding that same opening on the same board.
    delta = (_outcome(game_a, first_seat) - _outcome(game_b, first_seat)) / 2.0

    features, labels = [], []
    for color, explorer in explorers.items():
        label = delta if color == first_seat else -delta
        for _, vector in explorer.picks:
            features.append(vector)
            labels.append(label)

    return (
        np.stack(features).astype(np.float32),
        np.array(labels, dtype=np.float32),
    )


def _worker(payload):
    import torch

    torch.set_num_threads(1)
    seeds, rollout_kind, model_path, epsilon = payload
    model = None
    if model_path is not None:
        from src.placement.model import PlacementNet
        model = PlacementNet.load(model_path)

    chunks = [generate_pair(s, rollout_kind, model, epsilon) for s in seeds]
    features = [f for f, _ in chunks if len(f)]
    labels = [ln for _, ln in chunks if len(ln)]
    if not features:
        return np.zeros((0, feature_size()), np.float32), np.zeros(0, np.float32)
    return np.concatenate(features), np.concatenate(labels)


def generate(num_pairs: int, seed: int = 0, rollout_kind: str = "weighted",
             workers: int = 0, model_path=None, epsilon: float = 1.0,
             progress: bool = False):
    """Generate ``num_pairs`` duplicate-board pairs (two games each).

    Returns:
        (features, labels) -- float32 (N, F) and (N,), N up to ``4 * num_pairs``.
    """
    rng = np.random.default_rng(seed)
    seeds = [int(rng.integers(2**31 - 1)) for _ in range(num_pairs)]

    if workers <= 1:
        payloads = [(seeds, rollout_kind, model_path, epsilon)]
    else:
        workers = min(workers, num_pairs, os.cpu_count() or workers)
        buckets = [seeds[i::workers] for i in range(workers)]
        payloads = [
            (bucket, rollout_kind, model_path, epsilon)
            for bucket in buckets if bucket
        ]

    if len(payloads) == 1:
        results = [_worker(payloads[0])]
    else:
        results = []
        with ProcessPoolExecutor(max_workers=len(payloads)) as pool:
            for result in pool.map(_worker, payloads):
                results.append(result)
                if progress:
                    done = sum(len(f) for f, _ in results)
                    print(f"  {done} samples collected", flush=True)

    features = [f for f, _ in results if len(f)]
    labels = [ln for _, ln in results if len(ln)]
    if not features:
        return np.zeros((0, feature_size()), np.float32), np.zeros(0, np.float32)
    return np.concatenate(features), np.concatenate(labels)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--pairs", type=int, default=500,
                        help="duplicate-board pairs; each plays two games")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rollout", choices=sorted(ROLLOUT_BOTS), default="weighted",
                        help="bot that plays everything after the opening")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--model", default=None,
                        help="PlacementNet for epsilon-greedy exploration "
                             "(later Expert Iteration rounds)")
    parser.add_argument("--epsilon", type=float, default=1.0,
                        help="uniform-random share of openings when --model is set")
    parser.add_argument("--out", default="data/placement/samples.npz")
    args = parser.parse_args()

    features, labels = generate(
        args.pairs, seed=args.seed, rollout_kind=args.rollout,
        workers=args.workers, model_path=args.model, epsilon=args.epsilon,
        progress=True,
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, features=features, labels=labels)

    informative = int(np.count_nonzero(labels))
    print(f"{len(labels)} samples -> {out}")
    print(f"  {informative} informative ({informative / max(1, len(labels)):.1%}); "
          f"the rest are boards where swapping the openings did not swap the winner")


if __name__ == "__main__":
    main()
