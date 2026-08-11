# Phase 2 — Build the AI (AlphaZero)

**Goal:** train a strong 1v1 agent that beats the built-in bots, then push further with self-play.
**Status:** ⏳ in progress — AlphaZero stack built (search, self-play, training loop, eval);
the MaskablePPO baseline is retained as a legacy track.

---

## Why we moved off PPO

The PPO agent had a persistent failure mode: **it only built roads.** That is a long-horizon
credit-assignment failure, and it is structural rather than a hyperparameter problem.

- Roads are almost always legal and are the cheapest build (brick + wood are the most common
  resources). Settlements and cities are rarely legal.
- The payoff chain is `road now → settlement in ~8-20 plies → VP → win`. PPO has to *stumble into*
  that chain by exploration and then propagate credit back through a value function that is itself
  badly fit over a multi-hundred-ply game.
- Raising `ent_coef` (0.01 → 0.05) and hand-tuning milestone reward bonuses treated symptoms. The
  git history is three consecutive commits of shaping tweaks; that is a treadmill.

**PPO and AlphaZero differ in their policy improvement operator.** PPO improves by a gradient step
on advantage estimates. AlphaZero improves by *search*: the MCTS visit distribution is provably a
better policy than the raw network prior, and the network is then trained to imitate it. Search
doesn't have to discover the road→settlement chain by luck — it simulates forward until the
settlement appears.

That also removes the shaping treadmill: **`train_az.py` trains on the sparse win/loss outcome
only.** The density the milestone bonuses were faking now comes from the tree.

**What survived the switch:** `src/env/rules.py` in full, the 614-dim feature encoder, the
checkpoint manager, W&B logging, the catanatron `Player` interface, and the eval harness.
**What was dropped:** the PPO update itself, its `ent_coef`/`n_steps` tuning, and
`RewardShapingWrapper` (still applied on the legacy track only).

### Stage 0 — check before you commit

Before trusting the rewrite, wrap search around the *existing* PPO checkpoint:

```bash
python -m src.eval.stage0 --model checkpoints/<run>/agent_step_01600000.zip --games 60
```

It runs three configurations against a shared opponent — `ppo` (greedy, today's agent),
`mcts` (uniform priors, no value net — lookahead alone), and `ppo-mcts` (search using the PPO net).
The `mcts` control is the point: without it, any gain from `ppo-mcts` is unattributable between
"the net is fine" and "search fixes anything." The script prints a verdict on whether to initialise
the AlphaZero trunk from the PPO weights or start from scratch.

Caveat baked into `PPOEvaluator`: PPO's critic predicts a shaped, discounted return, not a win
probability. Values are squashed with `tanh`, so ordering is preserved but calibration is not.
Treat Stage 0 as directional.

---

## Rule changes that shape the learning problem

Both set in `src/env/` and applied to every code path:

- **7 VP to win** (`VPS_TO_WIN` in `src/env/catan_env.py`). Games resolve in ~80-150 turns instead
  of stretching past a turn cap, which shortens the credit-assignment horizon and lets search see
  terminal states from realistic positions.
- **Longest Road awards no victory points** (`_patch_no_longest_road`). At 7 VP a 2-point swing is
  nearly a third of the win condition, and it pays out for exactly the road-spam behaviour we are
  training away from. `LONGEST_ROAD_LENGTH` is still tracked and still appears in the observation
  vector; only the VP award and the `HAS_ROAD` flag are suppressed.

---

## The stack

### `src/agent/net.py` — AlphaZeroNet

Residual MLP trunk (default width 256, 4 blocks) with two heads:

- **policy** → logits over the 294-slot action space; illegal slots are masked to `-inf` before the
  softmax, so MCTS receives a distribution over legal moves only.
- **value** → scalar in `[-1, 1]`, the expected result **from the perspective of the player to
  move**. Observations are encoded from the mover's POV, so one network serves both seats and the
  sign flips on backup.

The feature vector is a flat mix of one-hot board features and raw counts spanning very different
scales, so the input projection is followed immediately by a `LayerNorm`.

### `src/agent/mcts.py` — PUCT search

Three Catan-specific decisions:

- **Stochastic transitions** (dice, dev-card draws, robber steals) are handled with **sampled chance
  nodes**. Each edge holds a dict of children keyed by the realized outcome and lets the engine's
  own RNG sample it, so repeated visits land on children in proportion to the true probabilities.
  That gives an unbiased expectation without an 11-way fan-out per roll. Dice children are keyed by
  the *sum*, since (2,5) and (3,4) are the same event.
- **Non-alternating perspective.** A Catan turn is many consecutive decisions by one player. Every
  node records who is to move; values are negated on backup whenever the perspective flips, which
  handles within-turn runs and hand-offs uniformly.
- **Forced moves are free.** A large fraction of Catan plies have exactly one legal action. Those
  are played without search and generate no training sample.

`fpu_reduction` pessimizes unvisited children relative to the parent's own value, which stops the
search fanning out uniformly across Catan's very wide action lists (54 legal moves at the opening).

### `src/agent/selfplay.py` — data generation and value targets

Each searched decision yields `(obs, mask, pi, z)`, where `pi` is the visit distribution.

**Value targets are bootstrapped, and this matters.** Textbook AlphaZero labels every position with
the final result. That works in Go, where play determines the outcome. Catan is dice-driven: the
same position wins or loses on the roll, so `z ∈ {-1, +1}` is a very noisy label. A poorly fit value
head degrades leaf evaluation, which degrades search — a feedback loop in the wrong direction.

So the target blends the outcome with an n-step bootstrap off the search's own root values:

```
z_t = value_mix * outcome + (1 - value_mix) * (± root_value[t + value_nstep])
```

The sign flips if the player to move differs. Defaults: `--value-nstep 24`, `--value-mix 0.5`.
Setting `--value-mix 1.0` recovers textbook AlphaZero. **If the value loss plateaus high, lower
`value-mix` first** — that is the knob this whole design exists for.

### `src/agent/train_az.py` — the loop

Each iteration:

1. Play `--games-per-iter` self-play games (Dirichlet noise at the root, temperature sampling for
   the first `--temperature-moves` decisions).
2. Push samples into a bounded replay buffer (`--buffer-size`, default 200k).
3. Take `--train-steps` gradient steps: cross-entropy to `pi` + `--value-coef` × MSE to `z`.
4. Every `--eval-every` iterations, play an arena match against the current best net and promote
   the challenger at `--promote-threshold` (default 55%). Also reports score vs `WeightedRandomPlayer`.

```bash
python -m src.agent.train_az --iterations 200 --games-per-iter 64 --workers 16
```

Checkpoints land in `checkpoints/<run-name>/` as `best.pt` and `latest.pt`.

---

## Compute budget

Self-play throughput sets the whole schedule. Measure it before picking a simulation count:

```bash
python -m src.eval.bench_mcts --simulations 100
```

Measured on the dev machine (width 256, 4 blocks, 100 sims/move):

| operation | cost |
| --- | --- |
| `Game.copy()` | ~32 µs |
| copy + `execute()` | ~59 µs |
| `encode_observation` | ~131 µs |
| **network forward (batch 1)** | **~604 µs** |
| full search | ~147 ms/move |
| full self-play game | ~16 s → **~230 games/hour/core** |

**The network forward pass dominates, not the state copy.** Two consequences:

- Shrinking the net (`--width` / `--blocks`) buys self-play throughput almost linearly. Start small.
- The highest-leverage optimization is **batched leaf evaluation** (parallel descent with virtual
  loss), which would amortize that 604 µs across many leaves. `Evaluator` is already batched
  (`evaluate_batch`) so the search can adopt it without touching the evaluators.

A tree-reuse optimization (carrying the subtree across moves) is not implemented; chance outcomes
make it fiddly and it was not the bottleneck.

## Parallel self-play

`generate_games(..., workers=N)` fans games out over a `ProcessPoolExecutor`; each worker rebuilds
the net from a broadcast `state_dict` and sets `torch.set_num_threads(1)` so the pool provides all
the parallelism. Set `--workers` to the core count. `--workers 0` runs in-process, which is what you
want when profiling or debugging.

## Setup

- Install: `pip install -r requirements.txt`.
- W&B: free account at https://wandb.ai, then `wandb login` once. Set `WANDB_MODE=offline` for
  smoke runs.

## Evaluation harness (`src/eval/`)

`src/agent/arena.py` builds any agent from a short name — `random`, `weighted`, `value`, `mcts`,
`ppo`, `ppo-mcts`, `az` — and plays **alternating-seat** matches. Alternation matters: the first
player picks first in the initial placement, which is a real edge, and without it a benchmark
mostly measures who got P0.

```bash
python -m src.eval.benchmark --agent az --model checkpoints/<run>/best.pt \
    --opponent value --games 200
```

Report win rate against each rung: `RandomPlayer → WeightedRandomPlayer → VictoryPointPlayer`.
Gate every training change on this harness.

## Cloud training

Self-play generates data on the fly, so there is little to pre-upload. Checkpoints are small
(`.pt`, a few MB). **Decide the exact sync mechanism (block volume vs. plain `scp`) during
deployment.** Pull `best.pt` down for local inference on the GTX 1660 Super.

## Exit criteria for Phase 2

- Beats `WeightedRandomPlayer` > 80% and `VictoryPointPlayer` > 60% in 1v1.
- A reproducible trained checkpoint in `checkpoints/` plus a local-inference entry point.

## Legacy track: MaskablePPO

`src/agent/train.py` still runs the original loop (`SubprocVecEnv` × 8, rotating self-play
checkpoint pool, `RewardShapingWrapper` with milestone VP bonuses rescaled to the 7-VP game). It is
kept so old checkpoints stay loadable and so `src/eval/stage0.py` has a baseline to diagnose. New
work goes on the AlphaZero track.
