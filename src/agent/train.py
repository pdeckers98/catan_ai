"""Phase 2 training: MaskablePPO with self-play and W&B logging.

Usage:
    python -m src.agent.train --total-steps 500000 --eval-interval 50000 --w-b-project "catan-ai"
"""

import argparse
import random
from pathlib import Path

import numpy as np
import wandb
from wandb.integration.sb3 import WandbCallback
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
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
        self._settlements: list[int] = []

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "game_turns" in info:
                self._turns.append(info["game_turns"])
            if "final_vp" in info:
                self._vps.append(info["final_vp"])
            if "opp_vp" in info:
                self._opp_vps.append(info["opp_vp"])
            if "settlements_built" in info:
                self._settlements.append(info["settlements_built"])
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
        if self._settlements:
            mean_settlements = float(np.mean(self._settlements))
            self.logger.record("rollout/mean_settlements_built", mean_settlements)
            log["train/mean_settlements_built"] = mean_settlements
        if self.model is not None:
            lr = self.model.lr_schedule(self.model._current_progress_remaining)
            self.logger.record("train/learning_rate", lr)
            log["train/learning_rate"] = lr
        if wandb.run is not None:
            wandb.log(log)
        self._turns, self._vps, self._opp_vps, self._settlements = [], [], [], []


def make_vec_env(num_envs: int, enemy=None, building_bonus: float = 0.05):
    """Create a vectorized environment with num_envs parallel games.

    Args:
        num_envs: number of parallel environments.
        enemy: opponent Player instance. Defaults to WeightedRandomPlayer.
        building_bonus: reward added each time the agent places a settlement or city.

    Returns:
        SubprocVecEnv with num_envs workers.
    """
    if enemy is None:
        enemy = WeightedRandomPlayer(Color.RED)

    def make_env():
        def _init():
            env = make_1v1_env(enemy=enemy)
            env = TurnLimitWrapper(ActionMasker(env, valid_action_mask), max_turns=300)
            env = RewardShapingWrapper(env, building_bonus=building_bonus)
            return env
        return _init

    return SubprocVecEnv([make_env() for _ in range(num_envs)])


def sample_opponent(checkpoint_dir):
    """Sample an opponent from the checkpoint pool, or WeightedRandom if empty.

    Uses WeightedRandom only as a bootstrap before any checkpoint exists.
    Once the pool has at least one entry, always sample from it so the agent
    trains against its own past policies rather than a fixed bot.
    """
    checkpoints = list_checkpoints(checkpoint_dir)
    if not checkpoints:
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
    parser.add_argument("--eval-interval", type=int, default=100_000)
    parser.add_argument("--w-b-project", type=str, default="catan-ai")
    parser.add_argument("--run-name", type=str, default=None,
                        help="Subdirectory under checkpoints/ for this run. "
                             "Defaults to the W&B run name.")
    parser.add_argument("--seed", type=int, default=None,
                        help="RNG seed. Omit to pick one randomly.")
    parser.add_argument("--n-steps", type=int, default=4096,
                        help="PPO rollout length per env before each update.")
    parser.add_argument("--batch-size", type=int, default=256,
                        help="PPO minibatch size. Must divide n_steps * num_envs.")
    parser.add_argument("--ent-coef", type=float, default=0.05,
                        help="Entropy bonus coefficient (raised from 0.01 to discourage "
                             "collapsing to road-heavy policies).")
    parser.add_argument("--net-arch", type=int, nargs="+", default=[32, 32, 32],
                        help="Hidden layer sizes, e.g. --net-arch 256 256.")
    parser.add_argument("--building-bonus", type=float, default=0.05,
                        help="Reward added each time the agent places a settlement or city. "
                             "Counteracts the incentive to road-rush for longest road.")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to a checkpoint zip to resume from. "
                             "Step count is parsed from the filename (agent_step_XXXXXXXX).")
    args = parser.parse_args()

    seed = args.seed if args.seed is not None else random.randint(0, 2**31 - 1)
    random.seed(seed)
    np.random.seed(seed)

    run = wandb.init(
        project=args.w_b_project,
        config={
            "total_steps": args.total_steps,
            "eval_interval": args.eval_interval,
            "model": "MaskablePPO",
            "policy": "MlpPolicy",
            "net_arch": args.net_arch,
            "num_envs": 8,
            "n_steps": args.n_steps,
            "batch_size": args.batch_size,
            "ent_coef": args.ent_coef,
            "building_bonus": args.building_bonus,
            "seed": seed,
        },
    )

    run_name = args.run_name or run.name
    checkpoint_dir = Path("checkpoints") / run_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    print(f"[Run] Checkpoints -> {checkpoint_dir}")

    num_envs = 8
    env = make_vec_env(num_envs=num_envs, building_bonus=args.building_bonus)

    if args.resume:
        resume_path = Path(args.resume)
        model = MaskablePPO.load(str(resume_path), env=env, device="cpu")
        # Parse step count from filename, e.g. agent_step_01200000[.zip]
        stem = resume_path.stem  # strips .zip if present
        try:
            steps_done = int(stem.split("_")[-1])
        except ValueError:
            steps_done = 0
        print(f"[Resume] Loaded {resume_path}, continuing from step {steps_done}")
    else:
        model = MaskablePPO(
            "MlpPolicy",
            env,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            ent_coef=args.ent_coef,
            policy_kwargs={"net_arch": args.net_arch},
            verbose=1,
            device="cpu",
        )
        steps_done = 0

    turn_callback = GameTurnCallback()
    wandb_callback = WandbCallback(verbose=0)
    callbacks = CallbackList([turn_callback, wandb_callback])

    eval_step = 0
    while steps_done < args.total_steps:
        interval = min(args.eval_interval, args.total_steps - steps_done)
        print(f"\n[Training] Steps {steps_done}->{steps_done + interval} / {args.total_steps}")
        model.learn(
            total_timesteps=interval,
            progress_bar=True,
            callback=callbacks,
            reset_num_timesteps=False,
        )
        steps_done += interval
        eval_step += 1

        opponent = sample_opponent(checkpoint_dir)
        print(f"[Eval] Testing against {opponent.__class__.__name__}...", end="", flush=True)
        win_rate = evaluate(model, opponent, num_games=25)

        wandb.log({
            "step": steps_done,
            "eval/win_rate": win_rate,
            "eval/opponent": opponent.__class__.__name__,
        })
        print(f" OK win_rate={win_rate:.2%}")

        if eval_step % 2 == 0:
            print(f"[Checkpoint] Saving model at step {steps_done}")
            save_checkpoint(model, steps_done, checkpoint_dir)
            prune_checkpoints(checkpoint_dir, keep_n=3)
            wandb.log({"checkpoint/step": steps_done})

        opponent = sample_opponent(checkpoint_dir)
        print(f"[Self-play] Swapping to {opponent.__class__.__name__}")
        env = make_vec_env(num_envs=num_envs, enemy=opponent, building_bonus=args.building_bonus)
        model.set_env(env)

    model.save(str(checkpoint_dir / "agent_final"))
    wandb.finish()
    print(f"Training complete. Model saved to {checkpoint_dir / 'agent_final'}")


if __name__ == "__main__":
    main()
