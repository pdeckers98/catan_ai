"""Phase 2 training: MaskablePPO with self-play and W&B logging.

Usage:
    python -m src.agent.train --total-steps 1000000 --eval-interval 50000 --w-b-project "catan-ai"
"""

import argparse
import random
from pathlib import Path

import numpy as np
import wandb
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker

from catanatron import Color
from catanatron.players.weighted_random import WeightedRandomPlayer
from src.agent.checkpoint_manager import (
    list_checkpoints, prune_checkpoints, save_checkpoint,
)
from src.env.catan_env import make_1v1_env, valid_action_mask


def sample_opponent():
    """Sample an opponent: either WeightedRandom or a random checkpoint from pool."""
    checkpoints = list_checkpoints()
    if not checkpoints or random.random() < 0.3:
        return WeightedRandomPlayer(Color.RED)

    checkpoint = random.choice(checkpoints)
    model = MaskablePPO.load(str(checkpoint), device="cpu")
    from src.agent.opponent import PolicyPlayer
    return PolicyPlayer(Color.RED, model)


def evaluate(model, opponent, num_games: int = 50) -> float:
    """Play num_games against opponent, return win rate.

    Args:
        model: trained MaskablePPO.
        opponent: catanatron Player instance.
        num_games: how many games to play.

    Returns:
        Win rate [0, 1].
    """
    wins = 0
    for _ in range(num_games):
        env = ActionMasker(make_1v1_env(enemy=opponent), valid_action_mask)
        obs, info = env.reset()
        done = False
        while not done:
            mask = valid_action_mask(env)
            action, _ = model.predict(obs, action_masks=mask, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        if reward > 0:
            wins += 1
        env.close()
    return wins / num_games


def main():
    parser = argparse.ArgumentParser(description="Train MaskablePPO agent with self-play.")
    parser.add_argument("--total-steps", type=int, default=1_000_000)
    parser.add_argument("--eval-interval", type=int, default=50_000)
    parser.add_argument("--w-b-project", type=str, default="catan-ai")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    wandb.init(
        project=args.w_b_project,
        config={
            "total_steps": args.total_steps,
            "eval_interval": args.eval_interval,
            "model": "MaskablePPO",
            "policy": "MlpPolicy",
            "net_arch": [64, 64],
        },
    )

    env = ActionMasker(make_1v1_env(), valid_action_mask)
    model = MaskablePPO(
        "MlpPolicy",
        env,
        policy_kwargs={"net_arch": [64, 64]},
        verbose=1,
        device="cpu",
    )

    steps_done = 0
    eval_step = 0
    while steps_done < args.total_steps:
        interval = min(args.eval_interval, args.total_steps - steps_done)
        model.learn(total_timesteps=interval, progress_bar=True)
        steps_done += interval
        eval_step += 1

        opponent = sample_opponent()
        win_rate = evaluate(model, opponent, num_games=50)

        wandb.log({
            "step": steps_done,
            "eval/win_rate": win_rate,
            "eval/opponent": opponent.__class__.__name__,
        })
        print(f"Step {steps_done}: win_rate={win_rate:.2%} vs {opponent.__class__.__name__}")

        if eval_step % 2 == 0:
            save_checkpoint(model, steps_done)
            prune_checkpoints(keep_n=3)
            wandb.log({"checkpoint/step": steps_done})

        opponent = sample_opponent()
        env = ActionMasker(make_1v1_env(enemy=opponent), valid_action_mask)
        model.set_env(env)

    model.save(str(Path("checkpoints") / "agent_final"))
    wandb.finish()
    print("Training complete. Model saved to checkpoints/agent_final")


if __name__ == "__main__":
    main()
