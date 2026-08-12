"""Diagnostics for a placement scorer, on boards it never trained on.

Validation loss says how well the model fits noisy outcome labels; it says
nothing about whether the model learned that an 8 beats a 3. These two numbers
do:

``mean_rank``
    Where the model's chosen corner sits in the hand-written scorer's ordering
    of every legal corner, 1 being the best on the board. This is the exact
    measurement that exposed the original PPO policy: it chose 3-tile corners
    (so it had learned adjacency) but ranked in the bottom half of them (so it
    had not learned numbers).

``rho``
    Rank correlation between the model's scores and the hand-written scorer's,
    across all legal corners. The old policy sat at ~0 on every board.

Both are measured against :mod:`src.placement.heuristic`, which encodes a human
opinion -- so they are a sanity check, not a target. A model that matches the
heuristic exactly has merely rediscovered the beginner rule. The verdict that
counts is the head-to-head win rate from :mod:`src.eval.benchmark`; these
numbers exist to catch a model that has learned nothing at all.
"""

import numpy as np

from catanatron import Color

from src.env.catan_env import make_1v1_game
from src.placement.features import candidate_nodes, encode_candidates
from src.placement.heuristic import pip_score, rank_of, spearman


def diagnose(model, seeds, color=Color.BLUE) -> dict:
    """Score every legal first settlement on each board and compare to the yardstick.

    Args:
        model: a :class:`~src.placement.model.PlacementNet`.
        seeds: board seeds to test. Keep these disjoint from training seeds.
        color: seat to evaluate from.

    Returns:
        dict with ``mean_rank``, ``mean_candidates``, ``mean_rho``, ``boards``.
    """
    ranks, rhos, sizes = [], [], []
    for seed in seeds:
        game = make_1v1_game(seed=seed)
        nodes = candidate_nodes(game.state.playable_actions)
        if len(nodes) < 2:
            continue

        scores = model.score(encode_candidates(game, color, nodes))
        chosen = nodes[int(np.argmax(scores))]
        catan_map = game.state.board.map

        ranks.append(rank_of(catan_map, chosen, nodes))
        sizes.append(len(nodes))
        rhos.append(
            spearman([pip_score(catan_map, n) for n in nodes], list(scores))
        )

    return {
        "boards": len(ranks),
        "mean_rank": float(np.mean(ranks)) if ranks else 0.0,
        "mean_candidates": float(np.mean(sizes)) if sizes else 0.0,
        "mean_rho": float(np.mean(rhos)) if rhos else 0.0,
    }


def format_diagnosis(result: dict) -> str:
    return (
        f"placement: chosen corner ranks {result['mean_rank']:.1f}/"
        f"{result['mean_candidates']:.0f} by the yardstick, "
        f"rho {result['mean_rho']:+.2f} over {result['boards']} boards"
    )
