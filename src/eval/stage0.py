"""Stage 0: is the existing PPO network worth building on?

Before committing to a full AlphaZero rewrite, wrap MCTS around the *already
trained* MaskablePPO checkpoint and see whether lookahead rescues it. Three
configurations play the same opponent:

    ppo        the checkpoint played greedily -- today's agent
    mcts       PUCT search with uniform priors and no value net -- lookahead alone
    ppo-mcts   PUCT search using the checkpoint for priors and leaf values

Reading the result:

- **ppo-mcts clearly beats both** -- the network has usable structure; its priors
  and values are steering the tree. Start AlphaZero from these weights.
- **ppo-mcts ~= mcts** -- the network adds nothing over uniform priors; the value
  head is the weak link. Train from scratch and prioritise the value target.
- **ppo-mcts < mcts** -- the network's priors are actively misleading the search.
  Definitely start from scratch.

The `mcts` control is the important one: without it, any improvement from
`ppo-mcts` is unattributable between "the net is fine" and "search fixes anything".

Note the caveat baked into ``PPOEvaluator``: PPO's critic predicts a shaped,
discounted return rather than a win probability, so its values are squashed with
tanh and are ordered-but-uncalibrated. Treat this as a directional read, not a
measurement.

Usage:
    python -m src.eval.stage0 --model checkpoints/<run>/agent_final.zip --games 60
"""

import argparse

from src.agent.arena import build_agent, play_match


def main():
    parser = argparse.ArgumentParser(description="Stage 0 MCTS-on-PPO diagnostic.")
    parser.add_argument("--model", required=True,
                        help="MaskablePPO checkpoint (.zip) to diagnose.")
    parser.add_argument("--opponent", default="weighted",
                        help="Shared yardstick: random, weighted or value.")
    parser.add_argument("--games", type=int, default=60)
    parser.add_argument("--simulations", type=int, default=100)
    parser.add_argument("--seed", type=int, default=12345,
                        help="Shared across configurations so they face the same "
                             "boards; vary it to check the result is stable.")
    args = parser.parse_args()

    opponent = build_agent(args.opponent)
    configurations = [
        ("ppo      ", "ppo", args.model),
        ("mcts     ", "mcts", None),
        ("ppo-mcts ", "ppo-mcts", args.model),
    ]

    print(f"Stage 0: {args.games} games vs '{args.opponent}', "
          f"{args.simulations} sims/move\n")
    scores = {}
    for label, spec, model_path in configurations:
        agent = build_agent(spec, model_path, args.simulations)
        result = play_match(agent, opponent, args.games, seed=args.seed)
        scores[spec] = result.score
        print(f"  {label} {result.summary()}", flush=True)

    print("\nVerdict:")
    delta = scores["ppo-mcts"] - scores["mcts"]
    if delta > 0.05:
        print("  The PPO net is adding real signal on top of search. Initialise "
              "the AlphaZero trunk from it.")
    elif delta < -0.05:
        print("  The PPO net is steering search worse than uniform priors. "
              "Train AlphaZero from scratch.")
    else:
        print("  The PPO net is roughly interchangeable with uniform priors -- "
              "search is doing the work. Train from scratch and prioritise the "
              "value target (--value-mix / --value-nstep).")


if __name__ == "__main__":
    main()
