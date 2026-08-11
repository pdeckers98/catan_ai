"""AlphaZero training loop: self-play -> replay buffer -> supervised update.

One iteration is:

1. Play ``--games-per-iter`` self-play games with the current network, each move
   chosen by MCTS with Dirichlet noise at the root.
2. Push the resulting ``(obs, mask, pi, z)`` samples into a bounded replay buffer.
3. Take ``--train-steps`` gradient steps on minibatches from the buffer, fitting
   the policy head to the visit counts and the value head to the value targets.
4. Every ``--eval-every`` iterations, play an arena match against the current best
   network and promote the challenger if it clears ``--promote-threshold``.

There is no reward shaping here, deliberately. The dense signal the milestone
bonuses in ``RewardShapingWrapper`` were standing in for now comes from search: a
position that leads to a settlement in eight plies gets a better backed-up value
than one that does not, without anyone hand-tuning a bonus for it.

Usage:
    python -m src.agent.train_az --iterations 200 --games-per-iter 64 --workers 16
"""

import argparse
import random
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import wandb

from catanatron.players.weighted_random import WeightedRandomPlayer

from src.agent.arena import net_factory, play_match
from src.agent.net import AlphaZeroNet, masked_policy_loss
from src.agent.selfplay import SelfPlayConfig, generate_games

BEST_MODEL = "best.pt"
LATEST_MODEL = "latest.pt"


class ReplayBuffer:
    """Bounded FIFO of self-play samples, sampled uniformly.

    Holding several iterations of games smooths the target distribution; too large
    a window and the policy head chases visit counts produced by networks it has
    already outgrown.
    """

    def __init__(self, capacity: int):
        self.obs = deque(maxlen=capacity)
        self.masks = deque(maxlen=capacity)
        self.policies = deque(maxlen=capacity)
        self.values = deque(maxlen=capacity)

    def __len__(self):
        return len(self.obs)

    def extend(self, results):
        for result in results:
            self.obs.extend(result.obs)
            self.masks.extend(result.masks)
            self.policies.extend(result.policies)
            self.values.extend(result.values)

    def sample(self, batch_size: int, rng):
        indices = rng.integers(0, len(self.obs), size=batch_size)
        obs = np.stack([self.obs[i] for i in indices])
        masks = np.stack([self.masks[i] for i in indices])
        policies = np.stack([self.policies[i] for i in indices])
        values = np.asarray([self.values[i] for i in indices], dtype=np.float32)
        return (
            torch.from_numpy(obs),
            torch.from_numpy(masks),
            torch.from_numpy(policies),
            torch.from_numpy(values),
        )


def train_epoch(net, optimizer, buffer, steps: int, batch_size: int,
                value_coef: float, rng) -> dict:
    """Run ``steps`` gradient steps; returns mean losses."""
    net.train()
    policy_losses, value_losses = [], []
    for _ in range(steps):
        obs, masks, target_pi, target_v = buffer.sample(batch_size, rng)
        logits, value = net(obs)

        policy_loss = masked_policy_loss(logits, target_pi, masks)
        value_loss = F.mse_loss(value, target_v)
        loss = policy_loss + value_coef * value_loss

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        optimizer.step()

        policy_losses.append(policy_loss.item())
        value_losses.append(value_loss.item())
    net.eval()
    return {
        "loss/policy": float(np.mean(policy_losses)),
        "loss/value": float(np.mean(value_losses)),
    }


def aggregate_stats(results) -> dict:
    """Mean of the per-game telemetry, prefixed for W&B."""
    if not results:
        return {}
    keys = results[0].stats.keys()
    stats = {
        f"selfplay/{key}": float(np.mean([r.stats[key] for r in results]))
        for key in keys
    }
    stats["selfplay/samples"] = int(sum(len(r.values) for r in results))
    return stats


def main():
    parser = argparse.ArgumentParser(description="AlphaZero self-play training.")
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--games-per-iter", type=int, default=64)
    parser.add_argument("--simulations", type=int, default=100,
                        help="MCTS playouts per decision during self-play.")
    parser.add_argument("--workers", type=int, default=0,
                        help="Self-play processes. 0/1 runs in-process.")
    parser.add_argument("--buffer-size", type=int, default=200_000)
    parser.add_argument("--train-steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--value-coef", type=float, default=1.0)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--blocks", type=int, default=4)
    parser.add_argument("--value-nstep", type=int, default=24,
                        help="Decisions to look ahead for the bootstrapped value "
                             "target. Lower = less variance, more bias.")
    parser.add_argument("--value-mix", type=float, default=0.5,
                        help="Weight on the final game outcome vs. the n-step "
                             "bootstrap. 1.0 is textbook AlphaZero.")
    parser.add_argument("--temperature-moves", type=int, default=20)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--eval-games", type=int, default=40)
    parser.add_argument("--eval-simulations", type=int, default=100)
    parser.add_argument("--promote-threshold", type=float, default=0.55)
    parser.add_argument("--min-buffer", type=int, default=5_000,
                        help="Samples required before the first gradient step.")
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--w-b-project", type=str, default="catan-ai")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to a .pt checkpoint to continue from.")
    args = parser.parse_args()

    seed = args.seed if args.seed is not None else random.randint(0, 2**31 - 1)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    run = wandb.init(
        project=args.w_b_project,
        config={**vars(args), "seed": seed, "algorithm": "alphazero"},
    )
    checkpoint_dir = Path("checkpoints") / (args.run_name or run.name)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    print(f"[Run] Checkpoints -> {checkpoint_dir}")

    if args.resume:
        net = AlphaZeroNet.load(args.resume)
        print(f"[Resume] Loaded {args.resume}")
    else:
        net = AlphaZeroNet(width=args.width, blocks=args.blocks)
    net.eval()

    best_net = AlphaZeroNet(**net.config())
    best_net.load_state_dict(net.state_dict())
    best_net.eval()
    net.save(checkpoint_dir / BEST_MODEL)

    optimizer = torch.optim.AdamW(
        net.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    buffer = ReplayBuffer(args.buffer_size)
    selfplay_config = SelfPlayConfig(
        simulations=args.simulations,
        temperature_moves=args.temperature_moves,
        value_nstep=args.value_nstep,
        value_mix=args.value_mix,
    )

    for iteration in range(1, args.iterations + 1):
        started = time.time()
        results = generate_games(
            net, selfplay_config, args.games_per_iter,
            workers=args.workers, seed=int(rng.integers(2**31 - 1)),
        )
        buffer.extend(results)
        selfplay_seconds = time.time() - started

        log = {
            "iteration": iteration,
            "selfplay/seconds": selfplay_seconds,
            "buffer/size": len(buffer),
            **aggregate_stats(results),
        }

        if len(buffer) >= args.min_buffer:
            started = time.time()
            log.update(
                train_epoch(
                    net, optimizer, buffer, args.train_steps,
                    args.batch_size, args.value_coef, rng,
                )
            )
            log["train/seconds"] = time.time() - started
        else:
            print(f"[Iter {iteration}] Buffer {len(buffer)}/{args.min_buffer}; "
                  "skipping update")

        net.save(checkpoint_dir / LATEST_MODEL)

        if iteration % args.eval_every == 0:
            arena = play_match(
                net_factory(net, args.eval_simulations),
                net_factory(best_net, args.eval_simulations),
                args.eval_games,
                seed=int(rng.integers(2**31 - 1)),
            )
            log["eval/score_vs_best"] = arena.score
            log["eval/turns_vs_best"] = arena.mean_turns
            promoted = arena.score >= args.promote_threshold
            log["eval/promoted"] = float(promoted)
            if promoted:
                best_net.load_state_dict(net.state_dict())
                net.save(checkpoint_dir / BEST_MODEL)

            baseline = play_match(
                net_factory(net, args.eval_simulations),
                lambda color: WeightedRandomPlayer(color),
                max(args.eval_games // 2, 2),
                seed=int(rng.integers(2**31 - 1)),
            )
            log["eval/score_vs_weighted_random"] = baseline.score
            log["eval/settlements_vs_weighted_random"] = baseline.mean_settlements
            log["eval/cities_vs_weighted_random"] = baseline.mean_cities
            log["eval/roads_vs_weighted_random"] = baseline.mean_roads
            print(f"[Iter {iteration}] vs best {arena.score:.1%} "
                  f"({'promoted' if promoted else 'kept'}), "
                  f"vs weighted-random {baseline.summary()}")

        wandb.log(log)
        print(f"[Iter {iteration}/{args.iterations}] "
              f"selfplay {selfplay_seconds:.0f}s, buffer {len(buffer)}, "
              f"policy_loss {log.get('loss/policy', float('nan')):.3f}, "
              f"value_loss {log.get('loss/value', float('nan')):.3f}")

    net.save(checkpoint_dir / LATEST_MODEL)
    wandb.finish()
    print(f"Training complete. Best model at {checkpoint_dir / BEST_MODEL}")


if __name__ == "__main__":
    main()
