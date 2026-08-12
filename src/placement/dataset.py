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

**Common random numbers.** The pair also duplicates the dice, via
:func:`~src.env.dice.fixed_dice`: both games draw rolls from the same dedicated
stream, so roll *k* is identical in each and only the openings differ. Without
it the two games share a seed but drift apart the moment the seats act
differently, which is immediately. Dev-card draws and robber steals still come
off the global RNG and still diverge; dice are the dominant term.

**The rollout agent decides what the label means.** Weighted-random is the fast
default, but it makes the label answer "which opening beats a bot". ``--rollout
ppo --rollout-model <checkpoint>`` plays the ~120 remaining turns with the
trained agent instead, so the label answers "which opening suits how *we* play"
-- an ore-wheat corner is worth much more to something that actually converts to
cities. Both seats get the same checkpoint, so the comparison stays symmetric,
and the agent's inability to place is irrelevant here because the opening is
forced by the generator either way.

**On-disk format.** ``pairs`` is (P, 4, F): each row is one duplicate-board pair,
holding the first seat's two corners followed by the second seat's two, all
featurised at the moment they were picked in game A. ``deltas`` is (P,), positive
when the first seat's bundle was the better one. ``features``/``labels`` are the
flattened per-node view of exactly the same data, kept for the regression loss.
"""

import argparse
import contextlib
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
from src.env.dice import fixed_dice
from src.placement.features import (
    encode_candidates,
    feature_size,
    node_features,
)

# Corners per pair: the first seat's two, then the second seat's two.
PAIR_NODES = 4

ROLLOUT_BOTS = {
    "random": RandomPlayer,
    "weighted": WeightedRandomPlayer,
    # Needs --rollout-model; built lazily per worker by _make_rollout.
    "ppo": None,
}


def _make_rollout(kind, color, model_path):
    """The player that decides everything after the opening settlements."""
    if kind != "ppo":
        return ROLLOUT_BOTS[kind](color)
    if model_path is None:
        raise ValueError("--rollout ppo needs --rollout-model")
    from src.agent.opponent import PolicyPlayer
    # Stochastic on purpose. A greedy policy plays one fixed line per position,
    # which would label openings by how well that single script converts them
    # rather than by their value across the play the agent actually produces.
    return PolicyPlayer(color, model_path=model_path, deterministic=False)


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

    def __init__(self, color, rollout, rng, model=None, epsilon=1.0, forced=None,
                 opening_rollout=None):
        super().__init__(color)
        self.rollout = rollout
        # Initial roads. Kept off the main rollout because a PPO checkpoint
        # trained through PlacementWrapper never saw the initial phase at all --
        # its head is untrained there, so asking it would be noise dressed up as
        # policy. Weighted-random is what those checkpoints trained against.
        self.opening_rollout = opening_rollout if opening_rollout is not None else rollout
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
        if game.state.is_initial_build_phase and not settlements:
            return self.opening_rollout.decide(game, playable_actions)
        if not settlements or not game.state.is_initial_build_phase:
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
          epsilon=1.0, dice_seed=None, rollout_model=None):
    """Play one game with explorer openings; returns (game, {color: explorer}).

    ``dice_seed`` puts the rolls on their own stream so both games of a pair see
    the same sequence; ``None`` leaves them on the global RNG.
    """
    forced_by_color = forced_by_color or {}
    explorers = {}
    players = []
    for offset, color in enumerate((Color.BLUE, Color.RED)):
        explorer = ExplorerPlayer(
            color,
            rollout=_make_rollout(rollout_kind, color, rollout_model),
            opening_rollout=WeightedRandomPlayer(color),
            # Distinct streams per seat, deterministic given rng_seed.
            rng=random.Random(rng_seed * 2 + offset),
            model=model,
            epsilon=epsilon,
            forced=forced_by_color.get(color),
        )
        explorers[color] = explorer
        players.append(explorer)

    game = make_1v1_game(players=players, seed=seed)
    with contextlib.ExitStack() as stack:
        if dice_seed is not None:
            stack.enter_context(fixed_dice(dice_seed))
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


def _empty_pair():
    return np.zeros((0, PAIR_NODES, feature_size()), np.float32), np.zeros(0, np.float32)


def generate_pair(seed: int, rollout_kind: str = "weighted", model=None,
                  epsilon: float = 1.0, common_dice: bool = True,
                  rollout_model=None):
    """Play a board twice with the openings swapped; return one labelled pair.

    Returns:
        (pair, delta) -- float32 (1, 4, F) and (1,), or empty arrays if the pair
        could not be completed. The four corners are the first seat's two
        followed by the second seat's two, and ``delta`` is positive when the
        first seat's bundle won the comparison.
    """
    dice_seed = seed if common_dice else None
    game_a, explorers = _play(seed, rollout_kind, seed, model=model, epsilon=epsilon,
                              dice_seed=dice_seed, rollout_model=rollout_model)

    order = _draft_order(game_a)
    if len(order) < 4:
        return _empty_pair()

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

    game_b, _ = _play(seed, rollout_kind, seed, forced_by_color=forced,
                      dice_seed=dice_seed, rollout_model=rollout_model)

    # Paired label: how much better the first seat did holding its own opening
    # than the second seat did holding that same opening on the same board.
    delta = (_outcome(game_a, first_seat) - _outcome(game_b, first_seat)) / 2.0

    corners = []
    for color in (first_seat, second_seat):
        corners.extend(vector for _, vector in explorers[color].picks)
    if len(corners) != PAIR_NODES:
        return _empty_pair()

    return (
        np.stack(corners).astype(np.float32)[None, ...],
        np.array([delta], dtype=np.float32),
    )


def flatten_pairs(pairs: np.ndarray, deltas: np.ndarray):
    """Per-node view of paired data, for the regression loss.

    Every corner inherits its bundle's outcome: the first seat's two corners take
    ``+delta`` and the second seat's two take ``-delta``.
    """
    if len(pairs) == 0:
        return np.zeros((0, feature_size()), np.float32), np.zeros(0, np.float32)
    features = pairs.reshape(-1, pairs.shape[-1])
    signs = np.array([1.0, 1.0, -1.0, -1.0], dtype=np.float32)
    labels = (deltas[:, None] * signs[None, :]).reshape(-1)
    return features.astype(np.float32), labels.astype(np.float32)


def _worker(payload):
    import torch

    torch.set_num_threads(1)
    seeds, rollout_kind, model_path, epsilon, common_dice, rollout_model = payload
    model = None
    if model_path is not None:
        from src.placement.model import PlacementNet
        model = PlacementNet.load(model_path)

    chunks = [
        generate_pair(s, rollout_kind, model, epsilon, common_dice, rollout_model)
        for s in seeds
    ]
    pairs = [p for p, _ in chunks if len(p)]
    deltas = [d for _, d in chunks if len(d)]
    if not pairs:
        return _empty_pair()
    return np.concatenate(pairs), np.concatenate(deltas)


def generate(num_pairs: int, seed: int = 0, rollout_kind: str = "weighted",
             workers: int = 0, model_path=None, epsilon: float = 1.0,
             common_dice: bool = True, rollout_model=None,
             progress: bool = False):
    """Generate ``num_pairs`` duplicate-board pairs (two games each).

    Returns:
        (pairs, deltas) -- float32 (P, 4, F) and (P,), P up to ``num_pairs``.
    """
    rng = np.random.default_rng(seed)
    seeds = [int(rng.integers(2**31 - 1)) for _ in range(num_pairs)]

    common = (rollout_kind, model_path, epsilon, common_dice, rollout_model)
    if workers <= 1:
        payloads = [(seeds, *common)]
    else:
        workers = min(workers, num_pairs, os.cpu_count() or workers)
        buckets = [seeds[i::workers] for i in range(workers)]
        payloads = [(bucket, *common) for bucket in buckets if bucket]

    if len(payloads) == 1:
        results = [_worker(payloads[0])]
    else:
        results = []
        with ProcessPoolExecutor(max_workers=len(payloads)) as pool:
            for result in pool.map(_worker, payloads):
                results.append(result)
                if progress:
                    done = sum(len(p) for p, _ in results)
                    print(f"  {done} pairs collected", flush=True)

    pairs = [p for p, _ in results if len(p)]
    deltas = [d for _, d in results if len(d)]
    if not pairs:
        return _empty_pair()
    return np.concatenate(pairs), np.concatenate(deltas)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--pairs", type=int, default=500,
                        help="duplicate-board pairs; each plays two games")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rollout", choices=sorted(ROLLOUT_BOTS), default="weighted",
                        help="bot that plays everything after the opening")
    parser.add_argument("--rollout-model", default=None,
                        help="MaskablePPO checkpoint, required by --rollout ppo. "
                             "Labels then say which openings suit how the agent "
                             "actually plays, not how a bot does.")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--model", default=None,
                        help="PlacementNet for epsilon-greedy exploration "
                             "(later Expert Iteration rounds)")
    parser.add_argument("--epsilon", type=float, default=1.0,
                        help="uniform-random share of openings when --model is set")
    parser.add_argument("--free-dice", action="store_true",
                        help="leave the dice on the global RNG instead of giving "
                             "each pair a shared roll sequence")
    parser.add_argument("--out", default="data/placement/samples.npz")
    args = parser.parse_args()

    pairs, deltas = generate(
        args.pairs, seed=args.seed, rollout_kind=args.rollout,
        workers=args.workers, model_path=args.model, epsilon=args.epsilon,
        common_dice=not args.free_dice, rollout_model=args.rollout_model,
        progress=True,
    )
    features, labels = flatten_pairs(pairs, deltas)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, pairs=pairs, deltas=deltas,
                        features=features, labels=labels)

    informative = int(np.count_nonzero(deltas))
    print(f"{len(pairs)} pairs ({len(labels)} corners) -> {out}")
    print(f"  {informative} informative ({informative / max(1, len(deltas)):.1%}); "
          f"the rest are boards where swapping the openings did not swap the winner")


if __name__ == "__main__":
    main()
