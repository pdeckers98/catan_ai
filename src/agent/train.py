"""MaskablePPO training with self-play and W&B logging -- the agent's only trainer.

The reward is the sparse win/loss outcome and nothing else; there is no shaping
and no machinery for it. ``EpisodeStatsWrapper`` keeps the end-of-episode
telemetry (VPs, settlements, cities, roads) without touching the reward.

``--opponent pool`` is the strongest setting: frozen past checkpoints mixed with
a slice of scripted games, rated by Elo against the run's own ladder rather than
by a win rate that cannot exceed 100%. Every run so far plateaus once it beats
its references, which is why **search at inference, or a harder opponent, is a
better next move than more steps** -- 50-sim search is worth ~+10 points on the
same weights, where 800k further steps of fine-tuning measured as nothing.

Evaluation goes through ``src.agent.arena``, the same harness ``src.eval.benchmark``
uses, so in-run numbers and reported numbers are directly comparable.

The ruleset (VP target, Longest Road, turn cap) is per-run and set through
``src.env.ruleset``; see the note on ``apply_cli_overrides`` below for why it is
read before the imports rather than from ``args``. Raise ``--gamma`` with the VP
target: games run ~2.4x longer at 15 VP than at 8, and the terminal result is the
only reward there is.

Usage:
    python -m src.agent.train --total-steps 2000000 --run-name ppo-sparse
    python -m src.agent.train --vps-to-win 15 --longest-road --max-turns 1500 \
        --gamma 0.999 --lookahead \
        --placement-model checkpoints/placement/scorer_ppo.pt \
        --bundle-model checkpoints/placement/bundle_noroads.pt
    python -m src.agent.train --opponent pool --resume checkpoints/<run>/best.zip
"""

import argparse
import random
import shutil
from pathlib import Path

# MUST run before any import that reaches the engine: ``src.env.rules`` decides
# at *import* time whether to suppress the Longest Road VP award, and the env
# constants are read at import too. By the time argparse runs inside main() the
# ruleset is already baked into this process -- and into every worker it spawns.
from src.env.ruleset import apply_cli_overrides

apply_cli_overrides()

import numpy as np
import torch.nn as nn
import wandb
from wandb.integration.sb3 import WandbCallback
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
try:  # SB3 >= 2.7 renamed the constant-schedule helper.
    from stable_baselines3.common.utils import FloatSchedule as constant_lr
except ImportError:  # pragma: no cover - older SB3
    from stable_baselines3.common.utils import get_schedule_fn as constant_lr
from stable_baselines3.common.vec_env import SubprocVecEnv

from catanatron import Color
from catanatron.players.weighted_random import WeightedRandomPlayer
from src.agent.arena import AgentSpec, play_match
from src.agent.checkpoint_manager import (
    list_checkpoints, prune_checkpoints, save_checkpoint,
)
from src.agent.elo import Ladder, elo_delta
from src.agent import pool as opponent_pool
from src.env.catan_env import (
    MAX_TURNS, make_1v1_env, valid_action_mask, TurnLimitWrapper,
    EpisodeStatsWrapper,
)
from src.env import ruleset


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
        self._dev_bought: list[int] = []
        self._vp_from_dev: list[int] = []

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
            if "dev_bought" in info:
                self._dev_bought.append(info["dev_bought"])
            if "vp_from_dev" in info:
                self._vp_from_dev.append(info["vp_from_dev"])
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
        if self._dev_bought:
            mean_dev = float(np.mean(self._dev_bought))
            self.logger.record("rollout/mean_dev_bought", mean_dev)
            log["train/mean_dev_bought"] = mean_dev
        if self._vp_from_dev:
            mean_dev_vp = float(np.mean(self._vp_from_dev))
            self.logger.record("rollout/mean_vp_from_dev", mean_dev_vp)
            log["train/mean_vp_from_dev"] = mean_dev_vp
        if self.model is not None:
            lr = self.model.lr_schedule(self.model._current_progress_remaining)
            self.logger.record("train/learning_rate", lr)
            log["train/learning_rate"] = lr
        if wandb.run is not None:
            wandb.log(log)
        self._turns, self._vps, self._opp_vps, self._settlements, self._roads, \
            self._cities, self._dev_bought, self._vp_from_dev = (
                [], [], [], [], [], [], [], [])


def _policy_kwargs(args) -> dict:
    """Assemble ``policy_kwargs``, with or without the shared trunk.

    Without --trunk this is exactly what the run always used, so the flag is
    additive and an unflagged command is bit-for-bit the old configuration.

    With it, ``net_arch`` stops meaning "the towers" and starts meaning "the
    heads on top of the trunk" -- SB3 applies ``net_arch`` *after* the features
    extractor. Passing the default [256, 256] alongside a trunk therefore builds
    something deeper than intended rather than something wrong, which is easy to
    do by accident, so it is worth being deliberate about the head sizes.
    """
    if not args.trunk:
        return {"net_arch": args.net_arch}
    from src.agent.trunk import SharedTrunk
    return {
        "net_arch": args.net_arch,
        "features_extractor_class": SharedTrunk,
        "features_extractor_kwargs": {"hidden_sizes": tuple(args.trunk)},
        # The trunk normalises and uses GELU; leaving the heads on Tanh would
        # reintroduce the saturation the trunk exists to avoid.
        "activation_fn": nn.GELU,
    }


def _opponent_opens_with_scorer(enemy) -> bool:
    """Whether this opponent should be handed the placement scorer.

    Trained opponents: yes, and it is not optional. A checkpoint trained through
    ``PlacementWrapper`` never received a gradient on placement, so without the
    scorer it opens with an untrained head -- measured at 100.0% -> 88.2% vs
    weighted-random. The learner would spend those envs beating an opponent that
    threw the game away before move one.

    Scripted bots: no. They are in the mixture as *fixed difficulty references*,
    and a bot that plays randomly for 300 turns off a grandmaster opening is
    neither the reference it was chosen to be nor an opponent that exists.
    The cost is real and worth naming: on those envs the learner starts with an
    opening edge it will not have against anything competent, so its win rate
    there overstates it. That is why they are a small slice.
    """
    from src.agent.opponent import PolicyPlayer
    return isinstance(enemy, PolicyPlayer)


def make_vec_env(num_envs: int, enemy=None, enemies=None,
                 placement_model=None, lookahead: bool = False,
                 bundle_model=None):
    """Create a vectorized environment with num_envs parallel games.

    Args:
        num_envs: number of parallel environments.
        enemy: opponent Player instance. Defaults to WeightedRandomPlayer.
        enemies: one opponent per env, overriding ``enemy``. Self-play uses this
            to face a mixture within a single batch -- see
            :func:`src.agent.pool.sample_enemies`.
        placement_model: PlacementNet checkpoint. The learner then opens with the
            scorer, inside ``reset()``, so its episode starts at a mid-game
            position and contains no placement decisions at all. Trained
            *opponents* get the scorer too; the scripted bots deliberately do not
            -- see :func:`_opponent_opens_with_scorer`.
        bundle_model: BundleNet checkpoint. Openings are then chosen as a *pair*
            of corners rather than greedily one at a time, which measured
            53.5% over 1200 games. Without it training opens greedily while
            ``src.eval.benchmark`` can be pointed at a bundle, so the agent
            would be trained on one distribution of openings and graded on
            another.
        lookahead: append the two-roll dice projection to the observation
            (614 -> 642). Must match the checkpoint at evaluation time, which
            :class:`~src.agent.opponent.PolicyPlayer` infers from the model's
            observation space rather than being told.

    Returns:
        SubprocVecEnv with num_envs workers.

    Note:
        The ruleset is not an argument here. It travels to the workers through
        the environment (:mod:`src.env.ruleset`), because ``SubprocVecEnv``
        spawns rather than forks on Windows and re-imports every module in the
        child -- an argument captured in this closure would not survive the
        engine's import-time patching.
    """
    if enemies is None:
        if enemy is None:
            enemy = WeightedRandomPlayer(Color.RED)
        enemies = [enemy] * num_envs
    if len(enemies) != num_envs:
        raise ValueError(f"need {num_envs} enemies, got {len(enemies)}")

    def make_env(env_enemy):
        def _init():
            # Runs inside the SubprocVecEnv worker. Torch defaults to one
            # intra-op thread per core, so 8 workers each running the opponent's
            # net (and the placement scorer) oversubscribe the CPU and stall
            # each other; measured 576 -> 819+ steps/s total when pinned. The
            # learner process keeps its own default thread count.
            import torch
            torch.set_num_threads(1)
            if placement_model is not None:
                from src.placement.env_wrapper import make_placement_env
                env = make_placement_env(
                    placement_model, enemy=env_enemy, bundle_path=bundle_model,
                    opponent_scorer=_opponent_opens_with_scorer(env_enemy),
                )
            else:
                env = make_1v1_env(enemy=env_enemy)
            if lookahead:
                from src.env.lookahead import LookaheadWrapper
                env = LookaheadWrapper(env)
            env = TurnLimitWrapper(
                ActionMasker(env, valid_action_mask), max_turns=MAX_TURNS
            )
            return EpisodeStatsWrapper(env)
        return _init

    return SubprocVecEnv([make_env(e) for e in enemies])


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
             gauntlet_games: int = 0, gauntlet_simulations: int = 400,
             placement_model=None, bundle_model=None,
             greedy_games: int = 0) -> dict:
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
        placement_model: PlacementNet checkpoint. Must be passed whenever
            training used one -- otherwise the run trains with good openings and
            is graded with the policy's own, which measures neither cleanly.
        bundle_model: BundleNet checkpoint. Same argument: match training, or
            the openings differ between the run and its own scoreboard.
        greedy_games: games against VictoryPointPlayer; 0 skips it. Worth paying
            for whenever greedy is in the training mixture, because the two
            scripted bots saturate at different points -- weighted-random tops
            out near 92% and stops discriminating long before greedy does.

    Returns:
        Metrics dict ready for ``wandb.log``.
    """
    challenger = AgentSpec(kind="ppo", model_path=str(model_path),
                           placement_path=placement_model,
                           bundle_path=bundle_model)
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
        "eval/knights_played": baseline.mean_knights,
        "eval/opp_knights_played": baseline.mean_opp_knights,
        "eval/knights_diff": baseline.knights_diff,
        "eval/dev_bought": baseline.mean_dev_bought,
        "eval/vp_from_dev": baseline.mean_vp_from_dev,
    }
    # The failure mode this whole re-test is about. Roads per victory point is
    # the sharpest single tell: feas02 improved 2.41 -> 1.54 as it learned.
    if baseline.mean_vp > 0:
        metrics["eval/roads_per_vp"] = baseline.mean_roads / baseline.mean_vp

    if greedy_games:
        greedy = play_match(
            challenger, AgentSpec(kind="value"), greedy_games,
            seed=seed, workers=workers,
        )
        metrics["eval/score_vs_greedy"] = greedy.score
        metrics["eval/vp_vs_greedy"] = greedy.mean_vp
        metrics["eval/opp_vp_vs_greedy"] = greedy.mean_opp_vp

    if gauntlet_games:
        gauntlet = play_match(
            challenger,
            AgentSpec(kind="mcts", simulations=gauntlet_simulations),
            gauntlet_games, seed=seed, workers=workers,
        )
        metrics["eval/score_vs_mcts"] = gauntlet.score
    return metrics, baseline


def evaluate_ladder(model_path, ladder: Ladder, num_games: int, workers: int = 0,
                    seed=None, promote_score: float = 0.70,
                    promote_path=None, step: int = 0,
                    placement_model=None, bundle_model=None):
    """Rate a checkpoint against the top of its own Elo ladder.

    Only the top anchor is played, not every anchor. A full round robin would
    give a better-conditioned rating, but it costs a match per anchor per
    evaluation and the extra precision buys nothing here -- what the run needs
    to answer is "is this stronger than the best thing we have made so far",
    which is one match.

    Args:
        model_path: challenger checkpoint.
        ladder: the run's ladder; must already be seeded.
        num_games: games against the top anchor.
        workers: processes for match play.
        seed: base RNG seed.
        promote_score: score at which the challenger joins the ladder. Above a
            coin flip by a healthy margin, so noise cannot ratchet the reference.
        promote_path: durable path to register if promoted (a pool entry --
            ``agent_step_*`` files get pruned, and an anchor must outlive that).
        step: training step, recorded with the anchor.
        placement_model: PlacementNet checkpoint, given to both sides.
        bundle_model: BundleNet checkpoint, given to both sides.

    Returns:
        ``(rating, result, promoted)``.
    """
    anchor = ladder.top()
    # Both sides get the opening scorer. Leaving it off is symmetric and so does
    # not bias the score, but a checkpoint trained through PlacementWrapper has
    # an untrained placement head -- both agents would open near-randomly and the
    # ladder would be rating them on positions neither will ever play from.
    result = play_match(
        AgentSpec(kind="ppo", model_path=str(model_path),
                  placement_path=placement_model, bundle_path=bundle_model),
        AgentSpec(kind="ppo", model_path=anchor.path,
                  placement_path=placement_model, bundle_path=bundle_model),
        num_games, seed=seed, workers=workers,
    )
    rating = anchor.elo + elo_delta(result.score, num_games)

    promoted = False
    if result.score >= promote_score and promote_path is not None:
        ladder.add(promote_path, rating, step)
        promoted = True
    return rating, result, promoted


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
    parser.add_argument("--learning-rate", type=float, default=3e-4,
                        help="Constant Adam step size; 3e-4 is the SB3 default and "
                             "suits training from a random init. Lower it to ~1e-4 "
                             "when *resuming* an already-converged policy, where "
                             "the job is refinement: the ppo-10vp run resumed at "
                             "5e-4 and drifted to 45.5%% over 500 games against its "
                             "own starting checkpoint, never once scoring above it.")
    parser.add_argument("--gamma", type=float, default=0.999,
                        help="Discount factor. Must be read against episode "
                             "length: ~150 agent steps at 8 VP, and games run "
                             "~2.4x longer at 15 VP (median 417 turns vs 172). "
                             "The terminal win/loss is the only reward this "
                             "project gives, so the horizon decides how much of "
                             "it survives back to the early game. SB3's 0.99 is "
                             "a 100-step horizon -- ~22%% reaching the opening "
                             "at 8 VP. 0.995 was the 8-VP setting at ~50%%; it "
                             "would deliver ~16%% at 15 VP, which is why the "
                             "default is now 0.999 (a ~1000-step horizon). Drop "
                             "it back to 0.995 for short-game runs.")
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
                             "256 wide. With --trunk these are the *head* sizes "
                             "on top of the shared trunk, so pass something "
                             "thinner (e.g. 256).")
    parser.add_argument("--trunk", type=int, nargs="+", default=None,
                        help="Share a trunk of these widths between the policy "
                             "and value heads (e.g. --trunk 512 512). Off by "
                             "default, which is SB3's two independent towers. "
                             "Sparse reward makes the critic the only dense "
                             "signal in the run, and without a shared trunk none "
                             "of what it learns reaches the policy. Also swaps "
                             "Tanh for LayerNorm+GELU. Costs ~24%% of rollout "
                             "throughput at 512x512.")
    parser.add_argument("--num-envs", type=int, default=8,
                        help="Parallel envs for rollout collection.")
    # Ruleset flags. Declared here so --help and W&B see them, but they are read
    # from sys.argv by apply_cli_overrides() at import time -- argparse runs far
    # too late to affect the engine's import-time patching. Keep the names in
    # sync with src/env/ruleset.py.
    parser.add_argument("--vps-to-win", type=int, default=ruleset.VPS_TO_WIN,
                        help="Victory points to win. Buildings cap at 9 VP (5 "
                             "settlements, 4 of them upgraded), so at 12 the "
                             "agent still needs Largest Army or Longest Road, "
                             "and at 15 it needs VP cards as well. Raise "
                             "--gamma with this: episodes run ~172 turns at 8 "
                             "VP and ~417 at 15.")
    parser.add_argument("--longest-road", action=argparse.BooleanOptionalAction,
                        default=ruleset.LONGEST_ROAD_VP,
                        help="Award Longest Road its +2 VP. Off historically "
                             "because at 8 VP it was a fifth of the win "
                             "condition and drove road-spam; on for the "
                             "colonist.io 1v1 target ruleset.")
    parser.add_argument("--max-turns", type=int, default=ruleset.MAX_TURNS,
                        help="Turn cap before a game is truncated as a draw. A "
                             "truncated episode pays 0 -- neither win nor loss "
                             "-- so this has to clear honest games by a wide "
                             "margin or the agent trains on unlearnable "
                             "episodes.")
    parser.add_argument("--placement-model", default=None,
                        help="PlacementNet checkpoint. Both seats then open "
                             "with the scorer during reset(), so the episode the "
                             "learner sees contains no placement decisions and "
                             "starts from a strong opening.")
    parser.add_argument("--bundle-model", default=None,
                        help="BundleNet checkpoint, used with --placement-model. "
                             "Openings are chosen as a pair of corners rather "
                             "than greedily one at a time (53.5%% over 1200 "
                             "games). Pass it whenever evaluation will, or the "
                             "run trains and is graded on different openings.")
    parser.add_argument("--lookahead", action=argparse.BooleanOptionalAction,
                        default=False,
                        help="Append the two-roll dice projection to the "
                             "observation (614 -> 642): expected production, "
                             "P(afford each build) after one and two rolls, and "
                             "discard exposure on both sides. Public "
                             "information only.")
    parser.add_argument("--opponent", choices=["weighted", "pool", "selfplay"],
                        default="weighted",
                        help="'weighted' trains against a fixed bot throughout -- "
                             "one moving part, interpretable curve, but the agent "
                             "only ever learns to beat that bot. 'pool' mixes "
                             "frozen past checkpoints with a slice of scripted "
                             "games (see --pool-weighted-frac). 'selfplay' is the "
                             "old behaviour: one sampled checkpoint at a time.")
    parser.add_argument("--pool-weighted-frac", type=float, default=0.1,
                        help="Share of envs facing WeightedRandomPlayer under "
                             "--opponent pool. Rounded up to at least one env, so "
                             "the fixed reference never disappears entirely.")
    parser.add_argument("--pool-greedy-frac", type=float, default=0.1,
                        help="Share of envs facing VictoryPointPlayer (greedy) "
                             "under --opponent pool. Kept as a slice of its own "
                             "rather than folded into --pool-weighted-frac "
                             "because the two fail differently: weighted-random "
                             "is broad and weak, greedy walks straight at the "
                             "win condition. Also rounded up to at least one env.")
    parser.add_argument("--pool-max", type=int, default=25,
                        help="Opponent pool size (~6 MB per entry).")
    parser.add_argument("--pool-deterministic", action="store_true",
                        help="Play pool opponents greedily. Off by default: a "
                             "greedy opponent shows the learner one line per "
                             "position, which is easy to overfit to.")
    parser.add_argument("--elo-games", type=int, default=100,
                        help="Games vs the top ladder anchor at each eval; 0 "
                             "disables the ladder.")
    parser.add_argument("--elo-promote", type=float, default=0.70,
                        help="Score needed to become the new top anchor.")
    parser.add_argument("--eval-games", type=int, default=100)
    parser.add_argument("--eval-greedy-games", type=int, default=0,
                        help="Games vs VictoryPointPlayer at each eval; 0 "
                             "disables. Pass it whenever greedy is in the "
                             "training mixture -- weighted-random saturates "
                             "near 92%% and stops telling you anything.")
    parser.add_argument("--eval-workers", type=int, default=0,
                        help="Processes for evaluation matches.")
    parser.add_argument("--gauntlet-games", type=int, default=0,
                        help="Games vs bare PUCT at each eval; 0 disables.")
    parser.add_argument("--gauntlet-simulations", type=int, default=400)
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to a checkpoint zip to resume from. "
                             "Step count is parsed from the filename (agent_step_XXXXXXXX).")
    args = parser.parse_args()

    # The ruleset was fixed at import time by apply_cli_overrides(). If argparse
    # disagrees with it, the pre-pass did not see these flags -- main() called in
    # process, say -- and the run would silently train under the wrong rules
    # while reporting the requested ones. Fail loudly instead.
    requested = (args.vps_to_win, args.longest_road, args.max_turns)
    active = (ruleset.VPS_TO_WIN, ruleset.LONGEST_ROAD_VP, ruleset.MAX_TURNS)
    if requested != active:
        raise SystemExit(
            f"ruleset mismatch: requested {requested}, engine imported with "
            f"{active}. src.env.ruleset.apply_cli_overrides() must run before "
            f"any engine import; see the note at the top of this module."
        )
    print(f"[Rules] {ruleset.describe()}")

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
            "trunk": args.trunk,
            "num_envs": args.num_envs,
            "n_steps": args.n_steps,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "gamma": args.gamma,
            "ent_coef": args.ent_coef,
            **ruleset.as_config(),
            "placement_model": args.placement_model,
            "bundle_model": args.bundle_model,
            "opponent": args.opponent,
            "pool_weighted_frac": args.pool_weighted_frac,
            "pool_greedy_frac": args.pool_greedy_frac,
            "pool_max": args.pool_max,
            "seed": seed,
        },
    )

    run_name = args.run_name or run.name
    checkpoint_dir = Path("checkpoints") / run_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    print(f"[Run] Checkpoints -> {checkpoint_dir}")

    num_envs = args.num_envs
    ladder = Ladder.load(checkpoint_dir / "ladder.json")
    rng = random.Random(seed)

    steps_done = 0
    resume_path = None
    if args.resume:
        resume_path = Path(args.resume)
        # Parse step count from filename, e.g. agent_step_01200000[.zip]
        try:
            steps_done = int(resume_path.stem.split("_")[-1])
        except ValueError:
            steps_done = 0

        # Seed the pool *before* the first env is built, or the opening interval
        # would train against weighted-random alone despite --opponent pool.
        if args.opponent == "pool" and not opponent_pool.list_pool(checkpoint_dir):
            origin = opponent_pool.add_to_pool(
                resume_path, steps_done, checkpoint_dir, args.pool_max
            )
            ladder.seed(origin, steps_done)
            print(f"[Pool] Seeded from {resume_path} -> {origin}")

    if args.opponent == "pool":
        enemies = opponent_pool.sample_enemies(
            num_envs, checkpoint_dir, args.pool_weighted_frac, rng,
            deterministic=args.pool_deterministic,
            greedy_frac=args.pool_greedy_frac,
        )
        print(f"[Pool] Opponents: {opponent_pool.describe(enemies)}")
        env = make_vec_env(
            num_envs=num_envs, enemies=enemies,
            placement_model=args.placement_model, lookahead=args.lookahead,
            bundle_model=args.bundle_model,
        )
    else:
        env = make_vec_env(
            num_envs=num_envs,
            placement_model=args.placement_model, lookahead=args.lookahead,
            bundle_model=args.bundle_model,
        )

    if args.resume:
        model = MaskablePPO.load(str(resume_path), env=env, device="cpu")
        # Architecture is baked into the weights, so --trunk cannot be applied to
        # a checkpoint that was not trained with it (or dropped from one that
        # was). ``load`` would silently keep the saved architecture and the run
        # would report a configuration it is not using -- the same failure the
        # ruleset check above exists to prevent.
        from src.agent.trunk import SharedTrunk
        loaded_trunk = isinstance(model.policy.features_extractor, SharedTrunk)
        if loaded_trunk != bool(args.trunk):
            raise SystemExit(
                f"--trunk mismatch: {resume_path.name} was trained "
                f"{'with' if loaded_trunk else 'without'} a shared trunk, but "
                f"--trunk was {'given' if args.trunk else 'omitted'}. "
                f"Architecture cannot change on a resume; start a new run."
            )
        # ``load`` restores the *saved* hyperparameters, so --learning-rate and
        # --gamma would be silently ignored on every resumed run. Rebind the lr
        # in both places: SB3 reads ``lr_schedule`` each update, but
        # ``learning_rate`` is what gets serialised into the next checkpoint.
        model.learning_rate = args.learning_rate
        model.lr_schedule = constant_lr(args.learning_rate)
        # Changing gamma invalidates the loaded critic -- every value it learned
        # is on the old discount -- so the first evaluations after a gamma change
        # can dip while the value head re-fits. That is the price of the change,
        # not a sign the run is broken.
        changed_gamma = model.gamma != args.gamma
        model.gamma = args.gamma
        if model.rollout_buffer is not None:
            model.rollout_buffer.gamma = args.gamma
        print(f"[Resume] Loaded {resume_path}, continuing from step {steps_done} "
              f"at lr {args.learning_rate:g}, gamma {args.gamma:g}"
              f"{' (changed -- critic will re-fit)' if changed_gamma else ''}")
    else:
        model = MaskablePPO(
            "MlpPolicy",
            env,
            learning_rate=args.learning_rate,
            gamma=args.gamma,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            ent_coef=args.ent_coef,
            policy_kwargs=_policy_kwargs(args),
            verbose=1,
            device="cpu",
        )

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
    # Under --opponent pool the selection criterion is the ladder rating, not the
    # win rate vs weighted-random: that bot is saturated at ~92%, so picking the
    # best on it is mostly picking whichever eval drew a lucky hundred games.
    best_metric = "elo" if args.opponent == "pool" and args.elo_games else "weighted"
    rating = 0.0

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
            placement_model=args.placement_model,
            bundle_model=args.bundle_model,
            greedy_games=args.eval_greedy_games,
        )
        score = metrics["eval/score_vs_weighted_random"]
        print(f" {baseline.summary()}")
        if "eval/score_vs_greedy" in metrics:
            print(f"[Eval] vs greedy: {metrics['eval/score_vs_greedy']:.1%} "
                  f"(VP {metrics['eval/vp_vs_greedy']:.1f} vs "
                  f"{metrics['eval/opp_vp_vs_greedy']:.1f})")

        # Grow the pool before rating, so a promoted checkpoint has a durable
        # file to point at (``agent_step_*`` entries get pruned three intervals
        # later, which would leave the ladder referencing a deleted anchor).
        pool_entry = None
        if args.opponent == "pool":
            pool_entry = opponent_pool.add_to_pool(
                latest, steps_done, checkpoint_dir, args.pool_max,
                protected=ladder.paths(),
            )
            if not len(ladder):
                ladder.seed(pool_entry, steps_done)

        if args.opponent == "pool" and args.elo_games and len(ladder):
            anchor = ladder.top()
            print(f"[Elo] vs anchor step {anchor.step} ({anchor.elo:+.0f})...",
                  end="", flush=True)
            rating, ladder_result, promoted = evaluate_ladder(
                latest, ladder, args.elo_games, workers=args.eval_workers,
                seed=int(np.random.randint(2**31 - 1)),
                promote_score=args.elo_promote, promote_path=pool_entry,
                step=steps_done, placement_model=args.placement_model,
                bundle_model=args.bundle_model,
            )
            metrics["eval/elo"] = rating
            metrics["eval/score_vs_anchor"] = ladder_result.score
            metrics["eval/ladder_size"] = len(ladder)
            # The knight race against a trained opponent, which is the one that
            # actually contests Largest Army -- weighted-random barely plays any.
            metrics["eval/knights_vs_anchor"] = ladder_result.mean_knights
            metrics["eval/knights_diff_vs_anchor"] = ladder_result.knights_diff
            print(f" {ladder_result.score:.1%} -> Elo {rating:+.0f}"
                  f"{'  [new anchor]' if promoted else ''}")

        selection = rating if best_metric == "elo" else score
        if selection > best_score:
            best_score, best_step = selection, steps_done
            shutil.copyfile(Path(f"{latest}.zip"), best_path)
            label = (f"Elo {selection:+.0f}" if best_metric == "elo"
                     else f"{selection:.1%}")
            print(f"  new best ({label}) -> {best_path}")
        metrics["eval/best_score"] = best_score
        wandb.log({"step": steps_done, "checkpoint/step": steps_done, **metrics})

        if args.opponent in ("pool", "selfplay"):
            if args.opponent == "pool":
                enemies = opponent_pool.sample_enemies(
                    num_envs, checkpoint_dir, args.pool_weighted_frac, rng,
                    deterministic=args.pool_deterministic,
                    greedy_frac=args.pool_greedy_frac,
                )
                print(f"[Pool] Opponents: {opponent_pool.describe(enemies)}")
                new_env = make_vec_env(
                    num_envs=num_envs, enemies=enemies,
                    placement_model=args.placement_model,
                    lookahead=args.lookahead, bundle_model=args.bundle_model,
                )
            else:
                opponent = sample_opponent(checkpoint_dir)
                print(f"[Self-play] Swapping to {opponent.__class__.__name__}")
                new_env = make_vec_env(
                    num_envs=num_envs, enemy=opponent,
                    placement_model=args.placement_model,
                    lookahead=args.lookahead, bundle_model=args.bundle_model,
                )
            model.set_env(new_env)
            # The old vector env owns ``num_envs`` live subprocesses. Rebinding
            # the name without closing it leaks them all, every interval.
            env.close()
            env = new_env

    model.save(str(checkpoint_dir / "agent_final"))
    env.close()
    wandb.finish()
    print(f"Training complete. Final model at {checkpoint_dir / 'agent_final'}")
    if best_step is not None:
        label = (f"Elo {best_score:+.0f}" if best_metric == "elo"
                 else f"{best_score:.1%} vs weighted-random")
        print(f"Best checkpoint: step {best_step}, {label} -> {best_path}")
        print(f"Play it: python -m src.eval.play --agent ppo --model {best_path}")


if __name__ == "__main__":
    # Pin PYTHONHASHSEED first, so a seed reproduces the game and not
    # merely the board; this relaunches once when it is unset.
    from src.env.determinism import ensure_hash_seed

    ensure_hash_seed()
    main()
