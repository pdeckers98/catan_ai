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

import torch

from src.agent.arena import AgentSpec, play_match


def main():
    # Search evaluates one leaf at a time, so every forward pass is batch 1.
    # Torch's intra-op threads cost more in synchronisation than they save on
    # matmuls that small -- left at the default this spawns ~6 threads per
    # process and runs slower than a single core. Same reasoning as the
    # self-play workers in src/agent/selfplay.py.
    torch.set_num_threads(1)

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
    parser.add_argument("--workers", type=int, default=0,
                        help="Processes to spread games over. 0/1 runs "
                             "in-process. Games are independent, so this scales "
                             "close to linearly.")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="MCTS leaves per network call. >1 trades exact "
                             "search reproducibility for throughput.")
    args = parser.parse_args()

    challenger = AgentSpec(
        kind=args.agent, model_path=args.model,
        simulations=args.simulations, batch_size=args.batch_size,
    )
    opponent = AgentSpec(
        kind=args.opponent, model_path=args.opponent_model,
        simulations=args.opponent_simulations or args.simulations,
        batch_size=args.batch_size,
    )

    print(f"{args.agent} vs {args.opponent} over {args.games} games "
          f"({args.simulations} sims/move, {args.workers or 1} workers)...",
          flush=True)
    result = play_match(
        challenger, opponent, args.games, seed=args.seed,
        progress=args.progress, workers=args.workers,
    )
    print(result.summary())


if __name__ == "__main__":
    main()
