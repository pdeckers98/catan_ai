"""Benchmark any agent against any other, over alternating seats.

Examples:
    # The shipped agent vs the strongest built-in bot, under the target ruleset
    python -m src.eval.benchmark --vps-to-win 15 --longest-road --max-turns 1500 \\
        --agent ppo-mcts --model checkpoints/archive/ppo-15vp-lr-step400000.zip \\
        --simulations 50 --opponent value --games 200 \\
        --placement-model checkpoints/placement/scorer_ppo.pt \\
        --bundle-model    checkpoints/placement/bundle_noroads.pt

    # What is search alone worth? Mirror match, same weights, search on one side.
    python -m src.eval.benchmark --agent ppo-mcts --model <ckpt> --simulations 50 \\
        --opponent ppo --opponent-model <ckpt> --games 800

Agent specs: random, weighted, value, mcts, ppo, ppo-mcts.

Budget 800+ games before believing any difference under ~5 points; 200 games
resolves nothing finer than ~7.
"""

import argparse

# Before any engine import: src.env.rules decides at import time whether Longest
# Road pays its VP, so the ruleset has to be selected first. See src/env/ruleset.py.
from src.env.ruleset import apply_cli_overrides

apply_cli_overrides()

import torch

from src.agent.arena import AgentSpec, play_match
from src.env import ruleset
from src.placement.chooser import PARTNER_RANK


def main():
    # Search evaluates one leaf at a time, so every forward pass is batch 1.
    # Torch's intra-op threads cost more in synchronisation than they save on
    # matmuls that small -- left at the default this spawns ~6 threads per
    # process and runs slower than a single core. Same reasoning as the arena
    # match workers in src/agent/arena.py.
    torch.set_num_threads(1)

    parser = argparse.ArgumentParser(description="Benchmark two agents head to head.")
    parser.add_argument("--agent", default="ppo-mcts", help="Challenger spec.")
    parser.add_argument("--model", default=None, help="Challenger checkpoint.")
    parser.add_argument("--opponent", default="weighted", help="Opponent spec.")
    parser.add_argument("--opponent-model", default=None, help="Opponent checkpoint.")
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--simulations", type=int, default=100,
                        help="Playouts per decision for search-backed specs.")
    parser.add_argument("--opponent-simulations", type=int, default=None,
                        help="Defaults to --simulations.")
    parser.add_argument("--horizon", type=int, default=None,
                        help="Cap the search at this many game turns past "
                             "the current position; positions beyond it "
                             "are scored by the value head instead of "
                             "expanded. Default searches as deep as the "
                             "simulation budget reaches.")
    parser.add_argument("--opponent-horizon", type=int, default=None,
                        help="Same, for the opponent. Independent of "
                             "--horizon, so the two can be compared directly.")
    parser.add_argument("--vps-to-win", type=int, default=ruleset.VPS_TO_WIN,
                        help="Victory points to win. Must match the ruleset the "
                             "checkpoints were trained under to mean anything.")
    parser.add_argument("--longest-road", action=argparse.BooleanOptionalAction,
                        default=ruleset.LONGEST_ROAD_VP,
                        help="Award Longest Road its +2 VP.")
    parser.add_argument("--max-turns", type=int, default=ruleset.MAX_TURNS,
                        help="Turn cap before a game is scored as a draw.")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--progress", action="store_true")
    parser.add_argument("--workers", type=int, default=0,
                        help="Processes to spread games over. 0/1 runs "
                             "in-process. Games are independent, so this scales "
                             "close to linearly.")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="MCTS leaves per network call. >1 trades exact "
                             "search reproducibility for throughput.")
    parser.add_argument("--placement-model", default=None,
                        help="PlacementNet checkpoint for the challenger's "
                             "opening. Run the same agent with and without it "
                             "to measure the opening in isolation.")
    parser.add_argument("--opponent-placement-model", default=None,
                        help="Same, for the opponent.")
    parser.add_argument("--bundle-model", default=None,
                        help="BundleNet checkpoint for the challenger. Needs "
                             "--placement-model, which shortlists the corners "
                             "it searches; the opening is then chosen as a pair.")
    parser.add_argument("--opponent-bundle-model", default=None,
                        help="Same, for the opponent.")
    parser.add_argument("--partner-rank", type=int, default=PARTNER_RANK,
                        help="How pessimistic the challenger's pair search is "
                             "about its second settlement surviving the "
                             "opponent's two picks: score each first corner by "
                             "its Nth best partner. 1 is the plain max. Only "
                             "affects the first seat, and needs --bundle-model.")
    parser.add_argument("--opponent-partner-rank", type=int, default=PARTNER_RANK,
                        help="Same, for the opponent.")
    args = parser.parse_args()
    print(f"[Rules] {ruleset.describe()}")

    challenger = AgentSpec(
        kind=args.agent, model_path=args.model,
        simulations=args.simulations, horizon=args.horizon,
        batch_size=args.batch_size,
        placement_path=args.placement_model,
        bundle_path=args.bundle_model,
        partner_rank=args.partner_rank,
    )
    opponent = AgentSpec(
        kind=args.opponent, model_path=args.opponent_model,
        simulations=args.opponent_simulations or args.simulations,
        horizon=args.opponent_horizon,
        batch_size=args.batch_size,
        placement_path=args.opponent_placement_model,
        bundle_path=args.opponent_bundle_model,
        partner_rank=args.opponent_partner_rank,
    )

    # Only the search specs consult --simulations; printing it unconditionally
    # made a bare policy-vs-policy match read as though search had been on.
    searching = [
        f"{args.simulations} sims" if "mcts" in args.agent else None,
        f"{args.opponent_simulations or args.simulations} sims"
        if "mcts" in args.opponent else None,
    ]
    if any(searching):
        sims = " vs ".join(s or "no search" for s in searching)
    else:
        sims = "no search"
    print(f"{args.agent} vs {args.opponent} over {args.games} games "
          f"({sims}, {args.workers or 1} workers)...",
          flush=True)
    result = play_match(
        challenger, opponent, args.games, seed=args.seed,
        progress=args.progress, workers=args.workers,
    )
    print(result.summary())


if __name__ == "__main__":
    # Pin PYTHONHASHSEED first, so a seed reproduces the game and not
    # merely the board; this relaunches once when it is unset.
    from src.env.determinism import ensure_hash_seed

    ensure_hash_seed()
    main()
