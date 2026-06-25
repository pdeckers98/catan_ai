"""Phase 1 visual smoke test: watch a 1v1 game play out on a live board plot.

Plays one game choosing uniformly among legal actions (same policy as
``smoke_test.py``) and redraws the board after every N steps so you can watch
settlements/roads/cities appear in a popup window.

Run: ``python -m src.env.visual_test``
"""

import argparse
import random

import matplotlib.pyplot as plt

from src.env.catan_env import make_1v1_env
from src.env.render import render_board


def main():
    parser = argparse.ArgumentParser(description="Catan 1v1 visual smoke test")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--redraw-every", type=int, default=1,
        help="redraw the board every N steps (lower = slower but smoother)",
    )
    parser.add_argument(
        "--pause", type=float, default=0.05,
        help="seconds to pause between redraws",
    )
    args = parser.parse_args()

    rng = random.Random(args.seed)
    env = make_1v1_env()
    _, info = env.reset()

    plt.ion()
    fig, ax = plt.subplots(figsize=(8, 8))
    game = env.unwrapped.game
    render_board(game, ax=ax)
    plt.pause(args.pause)

    terminated = truncated = False
    steps = 0
    while not (terminated or truncated):
        action = rng.choice(info["valid_actions"])
        _, reward, terminated, truncated, info = env.step(action)
        steps += 1
        if steps % args.redraw_every == 0 or terminated or truncated:
            render_board(game, ax=ax)
            plt.pause(args.pause)

    outcome = "WIN" if reward > 0 else "LOSS" if reward < 0 else "DRAW"
    render_board(game, ax=ax, title=f"Game over: P0 {outcome} ({steps} steps)")
    print(f"Game over: P0 {outcome} after {steps} steps.")
    plt.ioff()
    plt.show()


if __name__ == "__main__":
    main()
