"""Phase 1 smoke test: play full 1v1 games with a legal-random agent.

Verifies the env wiring end-to-end -- that we can reset, always pick from
``info["valid_actions"]``, step to termination, and read the outcome -- before
any learning is involved.

Run: ``python -m src.env.smoke_test --games 20``
"""

import argparse
import random

from src.env.catan_env import make_1v1_env


def play_one_game(env, rng):
    """Play a single game choosing uniformly among legal actions.

    Returns:
        (reward, terminated, truncated, steps): final-step reward (+1 win /
        -1 loss / 0 draw for P0) and bookkeeping.
    """
    _, info = env.reset()
    terminated = truncated = False
    reward = 0.0
    steps = 0
    while not (terminated or truncated):
        action = rng.choice(info["valid_actions"])
        _, reward, terminated, truncated, info = env.step(action)
        steps += 1
    return reward, terminated, truncated, steps


def main():
    parser = argparse.ArgumentParser(description="Catan 1v1 env smoke test")
    parser.add_argument("--games", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    env = make_1v1_env()

    wins = losses = draws = truncations = 0
    total_steps = 0
    for _ in range(args.games):
        reward, terminated, truncated, steps = play_one_game(env, rng)
        total_steps += steps
        if truncated and not terminated:
            truncations += 1
        if reward > 0:
            wins += 1
        elif reward < 0:
            losses += 1
        else:
            draws += 1
    env.close()

    print(f"Games:        {args.games}")
    print(f"P0 wins:      {wins}")
    print(f"P0 losses:    {losses}")
    print(f"Draws:        {draws}")
    print(f"Truncated:    {truncations} (hit turn limit)")
    print(f"Avg steps:    {total_steps / args.games:.1f}")
    print("Smoke test OK: full 1v1 games ran using only legal actions.")


if __name__ == "__main__":
    main()
