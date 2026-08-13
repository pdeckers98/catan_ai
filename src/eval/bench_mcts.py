"""Measure the MCTS compute budget.

Self-play throughput sets the whole training schedule, and in this codebase it is
dominated by ``Game.copy()`` -- one deep copy per simulation step, several hundred
thousand per game. This script separates the three costs so you know which one to
attack:

    copy      Game.copy() alone
    step      copy + execute (an engine transition)
    eval      one network forward pass on a single position
    search    a full search at the configured simulation count

Then it plays one full self-play game to report end-to-end seconds/game, which is
the number that actually determines how many games/hour a given core count buys.

Usage:
    python -m src.eval.bench_mcts --simulations 100
"""

import argparse
import time

import numpy as np

from src.agent.encoding import encode_observation, legal_action_mask
from src.agent.evaluator import NetEvaluator
from src.agent.mcts import MCTS
from src.agent.net import AlphaZeroNet
from src.agent.selfplay import SelfPlayConfig, play_game
from src.env.catan_env import make_1v1_game


def _advance(game, plies: int):
    """Play random legal moves to reach a mid-game position with a real choice.

    Keeps going past ``plies`` until the position actually branches -- timing a
    search from a forced move would understate the per-simulation cost.
    """
    rng = np.random.default_rng(0)
    for step in range(plies * 4):
        if game.winning_color() is not None:
            break
        actions = game.state.playable_actions
        if step >= plies and len(actions) > 1:
            break
        game.execute(actions[int(rng.integers(len(actions)))], validate_action=False)
    return game


def _time(label, fn, repeats):
    started = time.perf_counter()
    for _ in range(repeats):
        fn()
    elapsed = time.perf_counter() - started
    print(f"  {label:<8} {elapsed / repeats * 1e6:9.1f} us  "
          f"({repeats / elapsed:,.0f}/s)")
    return elapsed / repeats


def main():
    parser = argparse.ArgumentParser(description="Time the MCTS inner loop.")
    parser.add_argument("--simulations", type=int, default=100)
    parser.add_argument("--plies", type=int, default=60,
                        help="Random plies to reach a representative position.")
    parser.add_argument("--repeats", type=int, default=2000)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--blocks", type=int, default=4)
    parser.add_argument("--skip-game", action="store_true",
                        help="Skip the full self-play game timing.")
    args = parser.parse_args()

    net = AlphaZeroNet(width=args.width, blocks=args.blocks)
    evaluator = NetEvaluator(net)
    game = _advance(make_1v1_game(seed=42), args.plies)
    color = game.state.current_color()
    obs = encode_observation(game, color)
    mask = legal_action_mask(game.state.playable_actions)

    print(f"Position: turn {game.state.num_turns}, "
          f"{len(game.state.playable_actions)} legal actions")
    print(f"Net: width {args.width}, {args.blocks} blocks\n")

    print("Per-operation cost:")
    _time("copy", lambda: game.copy(), args.repeats)

    def step():
        clone = game.copy()
        actions = clone.state.playable_actions
        clone.execute(actions[0], validate_action=False)

    _time("step", step, args.repeats)
    _time("encode", lambda: encode_observation(game, color), args.repeats // 4)
    _time("eval", lambda: evaluator.evaluate(obs, mask), args.repeats // 4)

    mcts = MCTS(evaluator, simulations=args.simulations)
    started = time.perf_counter()
    mcts.search(game)
    search_seconds = time.perf_counter() - started
    print(f"\nsearch   {search_seconds * 1000:.0f} ms for {args.simulations} sims "
          f"({search_seconds / args.simulations * 1e6:.0f} us/sim)")

    if not args.skip_game:
        config = SelfPlayConfig(simulations=args.simulations)
        started = time.perf_counter()
        result = play_game(evaluator, config, seed=1)
        elapsed = time.perf_counter() - started
        print(f"\nFull self-play game: {elapsed:.1f}s, "
              f"{result.stats['turns']} turns, "
              f"{result.stats['decisions']} searched decisions "
              f"({elapsed / max(result.stats['decisions'], 1) * 1000:.0f} ms/decision)")
        print(f"  => {3600 / elapsed:.0f} games/hour/core")


if __name__ == "__main__":
    # Pin PYTHONHASHSEED first, so a seed reproduces the game and not
    # merely the board; this relaunches once when it is unset.
    from src.env.determinism import ensure_hash_seed

    ensure_hash_seed()
    main()
