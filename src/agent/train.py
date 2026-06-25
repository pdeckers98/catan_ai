"""Phase 2 training: MaskablePPO with self-play and W&B logging.

Usage:
    python -m src.agent.train --total-steps 500000 --eval-interval 50000 --w-b-project "catan-ai"
"""

import argparse
import random
from pathlib import Path

import numpy as np
import wandb
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import SubprocVecEnv

from catanatron import Color
from catanatron.players.weighted_random import WeightedRandomPlayer
from src.agent.checkpoint_manager import (
    list_checkpoints, prune_checkpoints, save_checkpoint,
)
from src.env.catan_env import (
    make_1v1_env, valid_action_mask, TurnLimitWrapper, RewardShapingWrapper,
)


class GameTurnCallback(BaseCallback):
    """Track how many turns games last and log the rolling mean.

    Reads the ``game_turns`` field that ``TurnLimitWrapper`` injects into ``info``
    when an episode ends. If the agent is learning to win efficiently, this mean
    should trend *down* over training. Logged to both the SB3 table (visible in the
    console) and W&B.
    """

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self._turns: list[int] = []
        self._vps: list[int] = []
        self._opp_vps: list[int] = []

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "game_turns" in info:
                self._turns.append(info["game_turns"])
            if "final_vp" in info:
                self._vps.append(info["final_vp"])
            if "opp_vp" in info:
                self._opp_vps.append(info["opp_vp"])
        return True

    def _on_rollout_end(self) -> None:
        if not self._turns:
            return
        mean_turns = float(np.mean(self._turns))
        self.logger.record("rollout/mean_game_turns", mean_turns)
        log = {
            "train/mean_game_turns": mean_turns,
            "train/games_finished": len(self._turns),
            "step": self.num_timesteps,
        }
        if self._vps:
            mean_vp = float(np.mean(self._vps))
            self.logger.record("rollout/mean_agent_vp", mean_vp)
            log["train/mean_agent_vp"] = mean_vp
        if self._opp_vps:
            mean_opp_vp = float(np.mean(self._opp_vps))
            self.logger.record("rollout/mean_opp_vp", mean_opp_vp)
            log["train/mean_opp_vp"] = mean_opp_vp
        if wandb.run is not None:
            wandb.log(log)
        self._turns, self._vps, self._opp_vps = [], [], []


def make_vec_env(num_envs: int, enemy=None):
    """Create a vectorized environment with num_envs parallel games.

    Args:
        num_envs: number of parallel environments.
        enemy: opponent Player instance. Defaults to WeightedRandomPlayer.

    Returns:
        SubprocVecEnv with num_envs workers.
    """
    if enemy is None:
        enemy = WeightedRandomPlayer(Color.RED)

    def make_env():
        def _init():
            env = make_1v1_env(enemy=enemy)
            env = TurnLimitWrapper(ActionMasker(env, valid_action_mask), max_turns=300)
            env = RewardShapingWrapper(env)
            return env
        return _init

    return SubprocVecEnv([make_env() for _ in range(num_envs)])


def sample_opponent():
    """Sample an opponent: either WeightedRandom or a random checkpoint from pool.

    30% of the time (or whenever the pool is empty) use the built-in
    WeightedRandomPlayer; otherwise wrap a saved checkpoint in PolicyPlayer for
    self-play. PolicyPlayer builds its own observation from the live game, so it
    works inside SubprocVecEnv workers.
    """
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
    parser.add_argument("--total-steps", type=int, default=500_000)
    parser.add_argument("--eval-interval", type=int, default=50_000)
    parser.add_argument("--w-b-project", type=str, default="catan-ai")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-steps", type=int, default=2048,
                        help="PPO rollout length per env before each update.")
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
            "num_envs": 8,
            "n_steps": args.n_steps,
        },
    )

    num_envs = 8
    env = make_vec_env(num_envs=num_envs)
    model = MaskablePPO(
        "MlpPolicy",
        env,
        n_steps=args.n_steps,
        policy_kwargs={"net_arch": [64, 64]},
        verbose=1,
        device="cpu",
    )

    turn_callback = GameTurnCallback()

    steps_done = 0
    eval_step = 0
    while steps_done < args.total_steps:
        interval = min(args.eval_interval, args.total_steps - steps_done)
        print(f"\n[Training] Steps {steps_done}->{steps_done + interval} / {args.total_steps}")
        model.learn(
            total_timesteps=interval,
            progress_bar=True,
            callback=turn_callback,
            reset_num_timesteps=False,
        )
        steps_done += interval
        eval_step += 1

        opponent = sample_opponent()
        print(f"[Eval] Testing against {opponent.__class__.__name__}...", end="", flush=True)
        win_rate = evaluate(model, opponent, num_games=50)

        wandb.log({
            "step": steps_done,
            "eval/win_rate": win_rate,
            "eval/opponent": opponent.__class__.__name__,
        })
        print(f" OK win_rate={win_rate:.2%}")

        if eval_step % 2 == 0:
            print(f"[Checkpoint] Saving model at step {steps_done}")
            save_checkpoint(model, steps_done)
            prune_checkpoints(keep_n=3)
            wandb.log({"checkpoint/step": steps_done})

        opponent = sample_opponent()
        print(f"[Self-play] Swapping to {opponent.__class__.__name__}")
        env = make_vec_env(num_envs=num_envs, enemy=opponent)
        model.set_env(env)

    model.save(str(Path("checkpoints") / "agent_final"))
    wandb.finish()
    print("Training complete. Model saved to checkpoints/agent_final")


if __name__ == "__main__":
    main()
