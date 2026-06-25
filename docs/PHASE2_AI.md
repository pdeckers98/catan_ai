# Phase 2 — Build the AI (deep RL)

**Goal:** train a strong 1v1 agent that beats the built-in bots, then push further with self-play.
**Status:** ⏳ in progress — self-play infrastructure + MaskablePPO baseline built.

## Why action masking is non-negotiable

The action space is `Discrete(290)` but only a handful of actions are legal each turn (54 at the
opening). An unmasked policy spends ~all its probability mass on illegal actions and never learns.
Always feed the mask from `src/env/catan_env.py:valid_action_mask` into the policy.

## Setup

- Install: `pip install -r requirements.txt` (includes `wandb`).
- W&B: requires a free account at https://wandb.ai. Login once with `wandb login` (saved to `~/.netrc`).
- Each training run is logged to your W&B project in real-time (metrics, checkpoints, hyperparams).

## What was built

- `src/agent/train.py` — main training loop (MaskablePPO + self-play + W&B logging).
  - Runs **8 games in parallel** via `SubprocVecEnv` (one worker per subprocess).
  - Trains for `--total-steps` (default 1M), evals every `--eval-interval` (default 50k).
  - Uses rotating checkpoint pool (keep 3 most recent, prune older) for self-play.
  - Opponent sampled from pool or defaults to WeightedRandomPlayer; swapped every 2 evals.
  - Logs to W&B: step, win rate vs. current opponent, checkpoint markers, num_envs config.
  - Run: `python -m src.agent.train --total-steps 1000000 --w-b-project catan-ai`

- `src/agent/checkpoint_manager.py` — checkpoint I/O and pruning.
  - `save_checkpoint(model, step)` — save SB3 model.
  - `list_checkpoints()` — list all saved checkpoints.
  - `prune_checkpoints(keep_n=3)` — delete all but N most recent.

- `src/agent/opponent.py` — `PolicyPlayer` wrapper to use trained models as opponents.
  - Frozen policy plugs into `config["enemies"]` for env init.

## Track A (start here): MaskablePPO baseline

`sb3_contrib.MaskablePPO` is PyTorch under the hood, handles masking out of the box. Policy is
small MLP `[64, 64]` to train quickly on CPU. Start training with:

```bash
python -m src.agent.train --total-steps 1000000 --eval-interval 50000 --w-b-project "catan-ai"
```

**First milestone:** beat `WeightedRandomPlayer` clearly (>70% win rate). This validates env,
reward, and masking end-to-end before anything fancier.

## Track B (later): custom raw-PyTorch agent

Once the baseline learns, optionally migrate to a hand-written policy/value network + PPO loop for
full control. Apply the mask as `-inf` on illegal logits before softmax. Keep `src/agent/` engine-
agnostic behind a single `act(obs, mask) -> action` interface so the *same* code serves training,
local inference (GTX 1660 Super), and the Phase 3 colonist.io bridge.

## Reward shaping

Default reward is sparse (`+1/-1` at game end). Pass a custom `reward_function(game, p0_color)`
via `make_1v1_env(reward_function=...)` to add dense signals (VP gained, longest road / largest
army, production potential). Keep it swappable and A/B it against the sparse baseline.

## Self-play (the strength ceiling)

The packaged bots top out at `VictoryPointPlayer` (greedy). To go beyond, periodically snapshot the
current policy and use it as the (frozen) `enemies` opponent, refreshing every N updates. Wrap the
snapshot policy in a `catanatron` `Player` subclass so it plugs into `config["enemies"]`.

## Parallel game collection

Training runs **8 games in parallel** via `SubprocVecEnv`, one worker per subprocess. Each worker
runs independent games with its own environment and random seed. This is critical since the workload
is **CPU-bound** — throughput is dominated by stepping game simulations, not policy computation.

To adjust parallelism, edit `num_envs` in `src/agent/train.py:main()`. The value should match your
CPU core count (or slightly higher for some oversubscription if I/O permits).

## Cloud training

Self-play generates data on the fly, so there is little to pre-upload. **Decide the exact sync
mechanism (block volume vs. plain checkpoint `scp`) during deployment**, once we see checkpoint
sizes and iteration speed. Pull the final `.pt` / `.zip` down for local inference on the GTX 1660
Super.

## Evaluation harness (`src/eval/`)

Catanatron runs thousands of games/sec, so benchmark over many games. Report win rate vs each rung
of the ladder: `RandomPlayer → WeightedRandomPlayer → VictoryPointPlayer` (the available bots —
see PHASE1 limitation note). Gate every training change on this harness.

```bash
python -m src.eval.benchmark --games 500
```

## Exit criteria for Phase 2

- Beats `WeightedRandomPlayer` > 80% and `VictoryPointPlayer` > 60% in 1v1.
- A reproducible trained checkpoint in `checkpoints/` plus a local-inference entry point.
