"""Phase 2 training: MaskablePPO with self-play and W&B logging.

Long treated as the superseded track -- ``src.agent.train_az`` replaces the
policy-gradient update with AlphaZero-style search + supervised distillation --
but the evidence for abandoning it does not survive inspection. Every PPO run
happened on 2026-06-26; the rules that disable the Longest Road VP bonus landed
on 2026-08-11, in the same sitting as AlphaZero. So the failure that condemned
PPO ("the agent only builds roads") was observed when Longest Road was worth +2
VP out of 10, i.e. when road-spam genuinely *was* the highest-EV line and the
policy was right to find it. The rule change and the algorithm change are
completely confounded, and PPO has never run under the current rules.

This module is set up to settle that, cheaply: PPO spends one network forward
per decision against AlphaZero's ~200, so if it works at all it is worth roughly
two orders of magnitude of compute.

The re-test is two arms, differing only in ``--shaping``:

- ``--no-shaping`` -- sparse win/loss only. The arm that answers whether PPO can
  actually solve this. Run this one first.
- ``--shaping`` -- milestone VP bonuses. A crutch tuned against the old rules;
  useful only as a fallback if the sparse arm flatlines.

Evaluation deliberately goes through ``src.agent.arena``, the same harness the
AlphaZero track uses, so the numbers are directly comparable to it (the feas02
checkpoint scored 70.8% vs weighted-random over 200 games).

Usage:
    python -m src.agent.train --total-steps 2000000 --no-shaping --run-name ppo-sparse
"""

import argparse
import random
import shutil
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
from src.agent.arena import AgentSpec, play_match
from src.agent.checkpoint_manager import (
    list_checkpoints, prune_checkpoints, save_checkpoint,
)
from src.env.catan_env import (
    MAX_TURNS, make_1v1_env, valid_action_mask, TurnLimitWrapper,
    RewardShapingWrapper,
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
        self._roads: list[int] = []
        self._cities: list[int] = []

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
            if "roads_built" in info:
                self._roads.append(info["roads_built"])
            if "cities_built" in info:
                self._cities.append(info["cities_built"])
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
        if self._roads:
            mean_roads = float(np.mean(self._roads))
            self.logger.record("rollout/mean_roads_built", mean_roads)
            log["train/mean_roads_built"] = mean_roads
        if self._cities:
            mean_cities = float(np.mean(self._cities))
            self.logger.record("rollout/mean_cities_built", mean_cities)
            log["train/mean_cities_built"] = mean_cities
        if self.model is not None:
            lr = self.model.lr_schedule(self.model._current_progress_remaining)
            self.logger.record("train/learning_rate", lr)
            log["train/learning_rate"] = lr
        if wandb.run is not None:
            wandb.log(log)
        self._turns, self._vps, self._opp_vps, self._settlements, self._roads, \
            self._cities = [], [], [], [], [], []


def make_vec_env(num_envs: int, enemy=None, shaping: bool = True):
    """Create a vectorized environment with num_envs parallel games.

    Args:
        num_envs: number of parallel environments.
        enemy: opponent Player instance. Defaults to WeightedRandomPlayer.
        shaping: wrap in :class:`RewardShapingWrapper`. This is the re-test's
            independent variable, so it must be switchable -- the wrapper's
            milestone bonuses were tuned against the pre-08-11 rules and leaving
            them permanently on would carry that confound into the answer.

    Returns:
        SubprocVecEnv with num_envs workers.

    Note:
        Without shaping the per-episode ``info`` fields the wrapper injects
        (``final_vp``, ``settlements_built``, ...) are absent, so
        :class:`GameTurnCallback` logs turns only. Build telemetry still arrives
        from the fixed evaluation, which does not depend on the wrapper.
    """
    if enemy is None:
        enemy = WeightedRandomPlayer(Color.RED)

    def make_env():
        def _init():
            env = make_1v1_env(enemy=enemy)
            env = TurnLimitWrapper(
                ActionMasker(env, valid_action_mask), max_turns=MAX_TURNS
            )
            if shaping:
                env = RewardShapingWrapper(env)
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


def evaluate(model_path, num_games: int, workers: int = 0, seed=None,
             gauntlet_games: int = 0, gauntlet_simulations: int = 400) -> dict:
    """Score a checkpoint against fixed yardsticks, via the shared arena.

    The previous version of this played against ``sample_opponent()`` -- a
    *randomly drawn past checkpoint*. That measures the agent against a moving
    target, so the resulting curve could not be compared across time, across
    runs, or against the AlphaZero track. These opponents never change, which is
    the whole point.

    Args:
        model_path: path to a saved MaskablePPO zip. Passed by path rather than
            by object so the match can be fanned across processes.
        num_games: games against WeightedRandomPlayer.
        workers: processes for match play. 0/1 runs in-process.
        seed: base RNG seed.
        gauntlet_games: games against bare PUCT search; 0 skips it. Unlike the
            scripted bots this yardstick does not saturate.
        gauntlet_simulations: playouts for that opponent.

    Returns:
        Metrics dict ready for ``wandb.log``.
    """
    challenger = AgentSpec(kind="ppo", model_path=str(model_path))
    baseline = play_match(
        challenger, AgentSpec(kind="weighted"), num_games,
        seed=seed, workers=workers,
    )
    metrics = {
        "eval/score_vs_weighted_random": baseline.score,
        "eval/win_rate": baseline.win_rate,
        "eval/turns": baseline.mean_turns,
        "eval/vp": baseline.mean_vp,
        "eval/opp_vp": baseline.mean_opp_vp,
        "eval/settlements": baseline.mean_settlements,
        "eval/cities": baseline.mean_cities,
        "eval/roads": baseline.mean_roads,
    }
    # The failure mode this whole re-test is about. Roads per victory point is
    # the sharpest single tell: feas02 improved 2.41 -> 1.54 as it learned.
    if baseline.mean_vp > 0:
        metrics["eval/roads_per_vp"] = baseline.mean_roads / baseline.mean_vp

    if gauntlet_games:
        gauntlet = play_match(
            challenger,
            AgentSpec(kind="mcts", simulations=gauntlet_simulations),
            gauntlet_games, seed=seed, workers=workers,
        )
        metrics["eval/score_vs_mcts"] = gauntlet.score
    return metrics, baseline


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
    parser.add_argument("--ent-coef", type=float, default=0.01,
                        help="Entropy bonus coefficient. Was 0.05, raised at the "
                             "time to fight 'collapsing to road-heavy policies' "
                             "-- a symptom of the Longest Road VP bonus that no "
                             "longer exists. Back to the SB3 default so the "
                             "re-test is not pre-compensating for a dead cause.")
    parser.add_argument("--net-arch", type=int, nargs="+", default=[256, 256],
                        help="Hidden layer sizes. Was [32, 32, 32], which is very "
                             "small for a 614-dim observation and 294 actions and "
                             "is a plausible independent cause of the original "
                             "failure; the AlphaZero net that does learn here is "
                             "256 wide.")
    parser.add_argument("--num-envs", type=int, default=8,
                        help="Parallel envs for rollout collection.")
    parser.add_argument("--shaping", action=argparse.BooleanOptionalAction,
                        default=False,
                        help="Milestone VP reward bonuses. Defaults off: the "
                             "sparse arm is the one that answers whether PPO can "
                             "solve this without a hand-tuned crutch.")
    parser.add_argument("--opponent", choices=["weighted", "selfplay"],
                        default="weighted",
                        help="'weighted' trains against a fixed bot throughout -- "
                             "one moving part, interpretable curve. 'selfplay' "
                             "swaps in sampled past checkpoints (the old "
                             "behaviour), which is stronger but confounds the "
                             "learning curve with a drifting opponent.")
    parser.add_argument("--eval-games", type=int, default=100)
    parser.add_argument("--eval-workers", type=int, default=0,
                        help="Processes for evaluation matches.")
    parser.add_argument("--gauntlet-games", type=int, default=0,
                        help="Games vs bare PUCT at each eval; 0 disables.")
    parser.add_argument("--gauntlet-simulations", type=int, default=400)
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
            "num_envs": args.num_envs,
            "n_steps": args.n_steps,
            "batch_size": args.batch_size,
            "ent_coef": args.ent_coef,
            "shaping": args.shaping,
            "opponent": args.opponent,
            "seed": seed,
        },
    )

    run_name = args.run_name or run.name
    checkpoint_dir = Path("checkpoints") / run_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    print(f"[Run] Checkpoints -> {checkpoint_dir}")

    num_envs = args.num_envs
    env = make_vec_env(num_envs=num_envs, shaping=args.shaping)

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

    # ``prune_checkpoints`` keeps the most *recent* three, which is what self-play
    # sampling wants and is not at all what a human wants to play against. Track
    # the best-scoring one separately; the name is outside the ``agent_step_*``
    # glob, so pruning leaves it alone.
    best_path = checkpoint_dir / "best.zip"
    best_score = -1.0
    best_step = None

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

        # Evaluation loads from disk so it can run across processes, so the
        # checkpoint has to exist before the match, not after it.
        print(f"[Checkpoint] Saving model at step {steps_done}")
        latest = save_checkpoint(model, steps_done, checkpoint_dir)
        prune_checkpoints(checkpoint_dir, keep_n=3)

        print("[Eval] vs weighted-random...", end="", flush=True)
        metrics, baseline = evaluate(
            latest, args.eval_games, workers=args.eval_workers,
            seed=int(np.random.randint(2**31 - 1)),
            gauntlet_games=args.gauntlet_games,
            gauntlet_simulations=args.gauntlet_simulations,
        )
        score = metrics["eval/score_vs_weighted_random"]
        if score > best_score:
            best_score, best_step = score, steps_done
            shutil.copyfile(Path(f"{latest}.zip"), best_path)
            print(f"  new best ({score:.1%}) -> {best_path}")
        metrics["eval/best_score"] = best_score
        wandb.log({"step": steps_done, "checkpoint/step": steps_done, **metrics})
        print(f" {baseline.summary()}")

        if args.opponent == "selfplay":
            opponent = sample_opponent(checkpoint_dir)
            print(f"[Self-play] Swapping to {opponent.__class__.__name__}")
            env = make_vec_env(
                num_envs=num_envs, enemy=opponent, shaping=args.shaping
            )
            model.set_env(env)

    model.save(str(checkpoint_dir / "agent_final"))
    wandb.finish()
    print(f"Training complete. Final model at {checkpoint_dir / 'agent_final'}")
    if best_step is not None:
        print(f"Best checkpoint: step {best_step}, {best_score:.1%} vs "
              f"weighted-random -> {best_path}")
        print(f"Play it: python -m src.eval.play --agent ppo --model {best_path}")


if __name__ == "__main__":
    main()
