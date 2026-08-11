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
python -m src.agent.train_az --iterations 200 --games-per-iter 64 --workers 16 \
    --simulations 200 --search-batch 16 --eval-workers 8
```

Checkpoints land in `checkpoints/<run-name>/` as `best.pt` and `latest.pt`. Note `best.pt` only
moves on promotion, so if nothing ever clears the threshold it stays at random init — benchmark
`latest.pt` in that case.

`--min-buffer` defaults to `games_per_iter × 100`, roughly one iteration of self-play. It used to be
a hardcoded 5000, which silently skipped the first gradient step on any run with fewer than ~50
games per iteration.

---

## Simulation count is the first thing to get right

**`--simulations` must comfortably exceed the branching factor, or the loop cannot learn.** This is
the single most expensive lesson from the first feasibility runs, and it is cheap to get wrong
because nothing crashes — training simply flatlines.

Catan's opening has ~54 legal actions. At 50 simulations, search cannot visit every child even
once, so visit counts are dominated by the prior plus noise. The policy target `pi` is then a
slightly-perturbed copy of the network's own prior, the network trains to imitate itself, the priors
stay uniform, and the search stays undiscriminating. A self-reinforcing null that more iterations
never escape.

The signature is a **policy loss that plateaus at `ln(k)`** for `k` the mean legal-action count —
that is the cross-entropy floor when the prediction is uniform, so a loss stuck there means the
target carries no information. The first run parked at 1.585 ≈ ln(4.9) from iteration 3 onward.

Diagnose it by comparing the entropy of the visit distribution against the entropy of the raw prior
on the same position. If `H(pi) ≈ H(prior)`, search is adding nothing:

| position | sims | H(π) | H(prior) | verdict |
| --- | --- | --- | --- | --- |
| 8 legal actions | 50 | 1.750 | 1.741 | search adds nothing |
| 5 legal actions | 800 | 0.351 | 1.275 | search is discriminating |

Note this comparison must be made *within* a single position — see the seeding caveat below.

## Compute budget

Self-play throughput sets the whole schedule. Measure it before picking a simulation count:

```bash
python -m src.eval.bench_mcts --simulations 100
```

Measured on the dev machine (12 logical / 6 physical cores, CPU-only torch):

| operation | width 256, 4 blocks | width 128, 2 blocks |
| --- | --- | --- |
| `Game.copy()` | ~32 µs | ~34 µs |
| copy + `execute()` | ~59 µs | ~45 µs |
| `encode_observation` | ~131 µs | ~135 µs |
| **network forward (batch 1)** | **~604 µs** | **~345 µs** |
| full search | ~147 ms/move @100 sims | ~49 ms/move @50 sims |
| full self-play game | ~16 s | ~9.6 s → ~376 games/hour/core |

**The network forward pass dominates, not the state copy.** Consequences:

- Shrinking the net buys throughput, but **sublinearly** — 4× fewer parameters bought only 1.75×,
  because `encode_observation` and tensor setup are fixed overhead. Below width 128 you pay
  accuracy for almost nothing.
- Batched leaf evaluation is the real lever (below).
- Never let torch spawn intra-op threads for search. Batch-1 forwards lose more to synchronisation
  than they gain: measured 711 µs/eval at 1 thread vs 858 µs at 6, i.e. one sixth the cores and
  21% *faster*. Every search entry point calls `torch.set_num_threads(1)`.

### Batched leaf evaluation

`MCTS(batch_size=N)` collects N leaves before calling `evaluate_batch` once, using **virtual loss**
to stop every descent in a batch converging on the same leaf. Measured at 200 sims, width 128:

| batch | ms/search | speedup |
| --- | --- | --- |
| 1 | 222 | 1.0× |
| 8 | 138 | 1.6× |
| 16 | 123 | 1.8× |
| 32 | 111 | 2.0× |

The ceiling is ~2× because the forward pass is only about half the cost; `Game.copy()` and encoding
make up the rest. **16 is the sweet spot** — beyond that a large fraction of the budget is in flight
against stale statistics and search quality degrades.

`batch_size=1` (the default) reproduces the serial search exactly, so it is opt-in: turning it on
changes results for a given seed and would invalidate comparisons against earlier runs.

A tree-reuse optimization (carrying the subtree across moves) is not implemented; chance outcomes
make it fiddly and it was not the bottleneck.

## Parallel self-play

`generate_games(..., workers=N)` fans games out over a `ProcessPoolExecutor`; each worker rebuilds
the net from a broadcast `state_dict` and sets `torch.set_num_threads(1)` so the pool provides all
the parallelism. Set `--workers` to the core count. `--workers 0` runs in-process, which is what you
want when profiling or debugging.

On a 12-logical / 6-physical-core box, expect ~40% more wall clock than a naive
cores × games estimate — hyperthreads do not deliver a full core each.

## Parallel evaluation

`play_match(..., workers=N)` spreads a match over processes too. This matters more than it looks:
eval was serial originally, which forced arena matches down to a size that could not resolve the
55% promotion threshold — a 12-game gate has a ±14% standard error, so promotion decisions were
close to coin flips.

Parallelism requires agents that survive pickling, and the factory closures do not. Hence
`AgentSpec`, a declarative description each worker rebuilds locally; `AgentSpec.from_net` carries an
in-memory network so the training loop can arena its live challenger without writing a checkpoint
first. Serial play still accepts plain factories.

Measured: 20 games at 50 sims, 102 s serial vs 38 s at `--workers 10` (2.7×). The gap to linear is
process startup plus load imbalance — game lengths vary a lot, and 20 games over 10 workers is only
2 each. It amortizes better over a 200-game benchmark.

### Reproducibility

Match results are reproducible only under specific conditions, all measured:

| condition | reproducible |
| --- | --- |
| same seed, same `--workers`, `PYTHONHASHSEED` pinned | yes, exactly |
| same seed, hash seed unpinned | no |
| same seed, different `--workers` | no |

Two independent causes. Catanatron's action generation iterates hash-ordered containers, so
per-process string hashing reorders equal-value actions and the argmax tie-break lands elsewhere.
Separately, engine global RNG state survives between games within a process, so how games are
partitioned across workers leaks into results.

Pin both `--workers` and `PYTHONHASHSEED` to reproduce a number; otherwise treat match results as
samples with real variance and size them accordingly.

### Seeding caveat (fixed)

`make_1v1_game(seed=...)` used to **not control the board layout**. `Game.__init__` reseeds the
global `random` module, but `build_map()` shuffles the board off that same module — and as a plain
argument it ran *first*, leaving the layout at the mercy of whatever had consumed the RNG
beforehand. Two games with the same seed got different maps. `make_1v1_game` now seeds before
building the map. Any measurement that predates this fix had board variance as an uncontrolled
factor.

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

**That ladder is not actually ordered under these house rules.** Measured over 200 games each, the
same checkpoint scored 70.8% vs `WeightedRandomPlayer` and 71.8% vs `VictoryPointPlayer` —
statistically identical (±3.2%). `VictoryPointPlayer` is supposed to be the harder rung. Two likely
reasons: with Longest Road awarding no VP it loses a chunk of what it normally chases, and at 7 VP
the game is short enough that greedy myopia costs less than usual. Do not assume a result against
one transfers to the other, and re-derive any threshold that was set assuming the ordering held.

Size matches properly. At 200 games the standard error is ±3.2%; at 12 games it is ±14%. During the
first feasibility run the in-training 6-game evals reported 83.3% and 66.7% for a checkpoint whose
true strength was 70.8% — both were pure noise.

## Cloud training

Self-play generates data on the fly, so there is little to pre-upload. Checkpoints are small
(`.pt`, a few MB). **Decide the exact sync mechanism (block volume vs. plain `scp`) during
deployment.** Pull `best.pt` down for local inference on the GTX 1660 Super.

## Feasibility runs (2026-08-11)

Two short runs on the dev laptop, width 128 / 2 blocks, 10 workers. They exist to answer "does
AlphaZero fix the road-spam failure that killed PPO", not to produce a strong agent.

| | feas01 | feas02 |
| --- | --- | --- |
| simulations | 50 | 200 |
| games/iter × iterations | 48 × 9 | 32 × 12 |
| self-play turns | 192 → 195 (**flat**) | 220 → **136** |
| draws | 13% → 17% (flat) | 34% → **3%** |
| policy loss | 1.668 → 1.585, flat from iter 3 | 1.884 → **1.621**, still falling at the end |
| roads per VP | ~1.8 (flat) | 2.41 → **1.54** |

**feas01 learned nothing** — every signal flat, for the simulation-count reason above.
**feas02 worked.** Games went from grinding to the 300-turn cap to resolving decisively, and the
road-spam pathology loosened measurably. Nothing had converged when the run ended.

Final benchmark of `feas02/best.pt`, 200 games each at 100 sims:

| opponent | record | score | target |
| --- | --- | --- | --- |
| `WeightedRandomPlayer` | 133W-50L-17D | 70.8% | >80% ✗ |
| `VictoryPointPlayer` | 135W-48L-17D | 71.8% | >60% ✓ |

Read as a feasibility result, not a strength result: ~71% after 12 iterations and ~70 minutes on a
laptop, with the loss still descending. Caveats worth carrying forward: total structures stayed flat
at ~2.6 the whole run (the VP gain came from **upgrading settlements to cities**, not from building
more), and roads only fell 9.8 → 8.7 in absolute terms.

One trap when reading self-play stats: `p0_settlements` *falls* as the agent improves, because
building a city returns the settlement piece. Track `settlements + cities`, and expect `p0_vp` to
fall as games get shorter, since the losing seat ends with fewer VP.

## Exit criteria for Phase 2

- Beats `WeightedRandomPlayer` > 80% and `VictoryPointPlayer` > 60% in 1v1, measured over **at least
  200 games** (see the note on the ladder not being ordered — both numbers are needed).
- A reproducible trained checkpoint in `checkpoints/` plus a local-inference entry point.

## Legacy track: MaskablePPO

`src/agent/train.py` still runs the original loop (`SubprocVecEnv` × 8, rotating self-play
checkpoint pool, `RewardShapingWrapper` with milestone VP bonuses rescaled to the 7-VP game). It is
kept so old checkpoints stay loadable and so `src/eval/stage0.py` has a baseline to diagnose. New
work goes on the AlphaZero track.
