"""Benchmark any agent against any other, over alternating seats.

Examples:
    # AlphaZero net vs the strongest built-in bot
    python -m src.eval.benchmark --agent az --model checkpoints/run/best.pt \\
        --opponent value --games 200

    # Does search help the old PPO net at all? (see also src.eval.stage0)
    python -m src.eval.benchmark --agent ppo-mcts --model checkpoints/old/agent_final.zip \\
        --opponent ppo --opponent-model checkpoints/old/agent_final.zip --games 100

Agent specs: random, weighted, value, mcts, ppo, ppo-mcts, az.
"""

import argparse

from src.agent.arena import build_agent, play_match


def main():
    parser = argparse.ArgumentParser(description="Benchmark two agents head to head.")
    parser.add_argument("--agent", default="az", help="Challenger spec.")
    parser.add_argument("--model", default=None, help="Challenger checkpoint.")
    parser.add_argument("--opponent", default="weighted", help="Opponent spec.")
    parser.add_argument("--opponent-model", default=None, help="Opponent checkpoint.")
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--simulations", type=int, default=100,
                        help="Playouts per decision for search-backed specs.")
    parser.add_argument("--opponent-simulations", type=int, default=None,
                        help="Defaults to --simulations.")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--progress", action="store_true")
    args = parser.parse_args()

    challenger = build_agent(args.agent, args.model, args.simulations)
    opponent = build_agent(
        args.opponent,
        args.opponent_model,
        args.opponent_simulations or args.simulations,
    )

    print(f"{args.agent} vs {args.opponent} over {args.games} games "
          f"({args.simulations} sims/move)...", flush=True)
    result = play_match(
        challenger, opponent, args.games, seed=args.seed, progress=args.progress
    )
    print(result.summary())


if __name__ == "__main__":
    main()
