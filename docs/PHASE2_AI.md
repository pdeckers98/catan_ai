# Phase 2 — Build the AI

**Goal:** a strong 1v1 agent, trained by self-play. **Status:** ⏳ in progress.

Two components, trained separately because they are different problems:

| | trains on | entry point | best result |
| --- | --- | --- | --- |
| **PPO** (MaskablePPO) | sparse win/loss, gym env | `src.agent.train` | **~92%** vs `weighted` at 3M steps |
| **AlphaZero** (PUCT + `AlphaZeroNet`) | self-play, sparse win/loss | `src.agent.train_az` | feasibility only (70.8%) |
| **Opening placement** | random openings labelled by game outcome | `src.placement.train` | +85.5% head-to-head over the same agent without it |

**PPO is currently the stronger *and* cheaper track** — one network forward per decision against
AlphaZero's ~200 — and the case against it does not survive inspection. Every PPO run predates
2026-08-11, the day the Longest-Road-awards-no-VP rule landed. The failure that condemned it
("the agent only builds roads") was observed while Longest Road was worth +2 VP out of 10, i.e.
when road-spam genuinely *was* the highest-EV line. The rule change and the algorithm change are
completely confounded.

The strongest playable agent is `checkpoints/ppo-8vp-scratch/best.zip`: 89.2% vs `weighted` alone,
**97.0%** with the placement scorer attached.

---

## Rules that shape the learning problem

Set in `src/env/` and applied on every code path:

- **8 VP to win** (`VPS_TO_WIN`), resolving in ~100-125 turns / ~150 agent decisions.
  `MAX_TURNS = 300` caps runaway games. A `ppo-10vp` run pushed to the standard 10 and games grew
  to ~176 turns; the policy never beat its own starting checkpoint, so the target went back to 8.
- **Longest Road awards no VP** (`_patch_no_longest_road`). At 8 VP a 2-point swing is nearly a
  third of the win condition, and it pays out for exactly the road-spam behaviour we train away
  from. Length is still tracked and still appears in the observation; only the VP award and the
  `HAS_ROAD` flag are suppressed.
- Discard above 9 cards, per-resource discard (action space 290 → **294**), Colonist.io robber
  restrictions, no dev card the turn it was bought. Full list in `src/env/rules.py`.

There is **no reward shaping and no discount** on the AlphaZero track — the tree supplies the
dense signal that milestone bonuses used to fake.

---

## The stack

### `src/agent/net.py` — AlphaZeroNet

Residual MLP trunk (width 256, 4 blocks), two heads: **policy** → 294 logits, illegal slots masked
to `-inf` before the softmax; **value** → scalar in `[-1, 1]` **from the perspective of the player
to move**, so one network serves both seats and the sign flips on backup. The 614-dim input mixes
one-hots with raw counts up to ~95, so the input projection is followed immediately by `LayerNorm`.

### `src/agent/mcts.py` — PUCT search

- **Stochastic transitions are sampled, not enumerated.** Each edge holds a dict of children keyed
  by the *realized* outcome (`_outcome_key`: dice sum, dev card drawn, resource stolen) and lets
  the engine's RNG sample it. Repeated visits land on children in proportion to true probabilities,
  giving an unbiased expectation without an 11-way fan-out per roll. **See the END_TURN limitation
  below — this is where the design currently bites.**
- **Non-alternating perspective.** A Catan turn is many consecutive decisions by one player. Each
  node records who is to move; values negate on backup whenever perspective flips.
- **Forced moves are free** — single-legal-action plies are played without search and produce no
  training sample.
- `fpu_reduction` pessimizes unvisited children relative to the parent's value, stopping the search
  fanning out uniformly across Catan's very wide action lists (54 legal moves at the opening).

### `src/agent/selfplay.py` — data generation

Each searched decision yields `(obs, mask, pi, z)` with `pi` the visit distribution.

**Value targets are bootstrapped, and this matters.** Catan is dice-driven, so the final result is
a very noisy label; a poorly fit value head degrades leaf evaluation, which degrades search. The
target blends outcome with an n-step bootstrap off the search's own root values:

```
z_t = value_mix * outcome + (1 - value_mix) * (± root_value[t + value_nstep])
```

Defaults `--value-nstep 24 --value-mix 0.5`; `--value-mix 1.0` recovers textbook AlphaZero.
**If value loss plateaus high, lower `value-mix` first** — that is the knob this design exists for.

### `src/agent/train_az.py` — the loop

Per iteration: self-play → replay buffer (`--buffer-size`, default 200k) → `--train-steps`
gradient steps (cross-entropy to `pi` + `--value-coef` × MSE to `z`) → every `--eval-every`
iterations an arena match against the current best, promoting at `--promote-threshold` (55%).

```bash
python -m src.agent.train_az --iterations 200 --games-per-iter 64 --workers 16 \
    --simulations 200 --search-batch 16 --eval-workers 8
```

- `best.pt` **only moves on promotion** — if nothing clears the threshold it stays at random init.
  Benchmark `latest.pt` in that case.
- `--min-buffer` defaults to `games_per_iter × 100`. It was once a hardcoded 5000, which silently
  skipped the first gradient step on any run with under ~50 games/iteration.

### `src/placement/` — the opening specialist

Placement is structurally unlike the rest of the game: no dice at decision time, ~2 decisions per
game, fully observable, hugely decisive — and it receives only **~2 of ~300 gradient samples per
episode**. That data starvation, not the discount, is why the main policy never learned it.

**Design constraint: no hardcoded placement knowledge.** The line is facts vs judgements. A tile's
roll probability is a property of dice and belongs in the features; whether 6s beat 5s or ore+wheat
beats brick+wood is a judgement and must be learned from outcomes.

| module | role |
| --- | --- |
| `features.py` | 45 mechanical facts per node: production rate per resource, tile counts, dice-number histogram, desert count, port one-hot, first/second-settlement flag, both players' holdings, buildable production 2 and 3 edges out. No weighting between blocks. |
| `dataset.py` | Openings explored **uniformly at random** (sampling from a scorer would bake in its preferences), played out, labelled with the actual outcome. |
| `model.py` | 64-wide 2-layer MLP → scalar. Normalisation stats fitted on the train split, stored as buffers so inference cannot disagree with training. |
| `heuristic.py` | Hand-written pips × diversity scorer. **Quarantined**: evaluation yardstick only, never in a decision path or in data generation. |
| `player.py` | Intercepts initial-phase settlements only; everything else, initial roads included, goes to the inner agent. |
| `env_wrapper.py` | The gym-side equivalent, for learners that step an env rather than acting as a `Player`. |

**Duplicate-board pairing** is the variance reduction. Each board is played twice with the four
opening nodes swapped between seats; the label is the difference. Same seat wins both → the board
explained it, both openings labelled 0. Swapping flips the winner → the openings explained it,
labels ±1. ~73% of pairs come back informative. The snake draft (P0, P1, P1, P0) makes the swap
always legal: replay in the order n2, n1, n4, n3 and every node keeps its non-adjacency.

*Caveat: this duplicates the **board**, not the dice.* Rolls come off the global RNG, so once the
seats hold different corners their actions and their roll sequences diverge. Variance reduction,
not elimination.

```bash
python -m src.placement.dataset --pairs 8000 --workers 8 --seed 1   # ~1 min on 8 workers
python -m src.placement.train --data data/placement/samples.npz
```

**Training overfits fast** — validation loss bottoms out around epoch 7 of 200 and climbs steadily.
Early stopping is load-bearing, not tidiness.

Attach to any agent with `--placement-model`, or `AgentSpec(placement_path=...)`.

**For a gym-based learner**, `make_placement_env(model_path)` returns an env that plays the whole
initial build phase **inside `reset()`** -- both seats -- so the learner's first observation is a
mid-game position:

```python
from src.placement.env_wrapper import make_placement_env
env = make_placement_env("checkpoints/placement/scorer.pt")
```

Doing it in `reset()` rather than intercepting mid-episode is what keeps the rollout buffer honest:
an overridden action would otherwise sit in the buffer as if the learner had chosen it. It also
makes placement 0 of ~300 samples rather than 2, removing the starvation problem by construction.
The opponent gets the scorer too by default -- training against bad openings teaches the agent to
exploit an edge it will not have.

**A checkpoint trained this way depends on the scorer permanently.** Placement decisions never
reach the rollout buffer, so the policy head gets no gradient on them and cannot place without
help. Measured on the 2M run: 100.0% -> 88.2% vs weighted-random when the scorer is removed, and
42.2% vs a bare `ppo-8vp-scratch` (against 53.0% when both have it) -- bare, it is beaten by the
older agent whose placement was merely bad rather than absent. Ship the two together.

Two caveats. **Initial roads are random here**, where `PlacementPlayer` delegates them to the inner
agent -- a small train/play mismatch on a 2-3 option decision. And **`env.reset(seed=n)` does not
control the board layout**, and is not even repeatable across two resets: the gym env builds its
map off the global `random` module at reset time. Same bug `make_1v1_game` was fixed for, still
live on the gym path. Rank placements against `env.unwrapped.game.state.board.map`, never against
a board built separately from the same seed.

---

## Hard-won constraints

### `--simulations` must comfortably exceed the branching factor

The single most expensive lesson from the feasibility runs, and cheap to get wrong because nothing
crashes — training simply flatlines. At 50 sims against Catan's ~54-action opening, search cannot
visit every child once, so visit counts are dominated by prior plus noise: `pi` becomes a
perturbed copy of the network's own prior, the network trains to imitate itself, and the loop
never escapes.

**Signature: policy loss plateaus at `ln(k)`** for `k` the mean legal-action count. The first run
parked at 1.585 ≈ ln(4.9) from iteration 3 onward. Diagnose by comparing `H(pi)` against
`H(prior)` **within a single position**; if they match, search is adding nothing.

| position | sims | H(π) | H(prior) | verdict |
| --- | --- | --- | --- | --- |
| 8 legal actions | 50 | 1.750 | 1.741 | search adds nothing |
| 5 legal actions | 800 | 0.351 | 1.275 | search is discriminating |

### Compute budget

Measured on the dev box (12 logical / 6 physical cores, CPU-only torch), width 256 / 4 blocks:

| operation | cost |
| --- | --- |
| `Game.copy()` | ~32 µs |
| `encode_observation` | ~131 µs |
| **network forward (batch 1)** | **~604 µs** |
| full search | ~147 ms/move @ 100 sims |
| full self-play game | ~16 s |

**The forward pass dominates, not the state copy.** Consequences:

- Shrinking the net buys throughput **sublinearly** — 4× fewer parameters bought only 1.75×,
  because encoding and tensor setup are fixed overhead. Below width 128 you pay accuracy for
  almost nothing.
- **Never let torch spawn intra-op threads for search.** Measured 711 µs/eval at 1 thread vs
  858 µs at 6 — one sixth the cores and 21% *faster*. Every search entry point calls
  `torch.set_num_threads(1)`.
- **Batched leaf evaluation** (`MCTS(batch_size=N)`, virtual loss to stop descents converging on
  one leaf) is the real lever: 1.6× at 8, 1.8× at 16, 2.0× at 32. Ceiling is ~2× because the
  forward pass is only half the cost. **16 is the sweet spot** — beyond that too much of the
  budget is in flight against stale statistics. `batch_size=1` reproduces serial search exactly,
  so it is opt-in.
- Parallel self-play and parallel eval both fan over `ProcessPoolExecutor`. On 12 logical / 6
  physical cores expect ~40% more wall clock than a naive cores × games estimate.

### Reproducibility

| condition | reproducible |
| --- | --- |
| same seed, same `--workers`, `PYTHONHASHSEED` pinned | yes, exactly |
| same seed, hash seed unpinned | no |
| same seed, different `--workers` | no |

Two causes: catanatron iterates hash-ordered containers, so per-process string hashing reorders
equal-value actions and the argmax tie-break moves; and engine global RNG state survives between
games in a process, so how games partition across workers leaks into results.

`make_1v1_game` seeds *before* `build_map`, which is what makes a seed reproduce a board.
Measurements predating that fix had board layout as an uncontrolled variable.

### Benchmarking

`src/agent/arena.py` builds any agent by name — `random`, `weighted`, `value`, `mcts`, `ppo`,
`ppo-mcts`, `az` — and plays **alternating-seat** matches (the first player picks first in
placement, a real edge).

- **The bot ladder is not ordered under these house rules.** Over 200 games each, one checkpoint
  scored 70.8% vs `WeightedRandomPlayer` and 71.8% vs `VictoryPointPlayer` — statistically
  identical. Do not assume a result against one transfers to the other.
- **Size matches properly.** 200 games is ±3.2% standard error; 12 games is ±14%. In-training
  6-game evals once reported 83.3% and 66.7% for a checkpoint whose true strength was 70.8%.
- **Don't use `feas01`/`feas02` for calibration probes** — neither converged, so a null result
  there is uninformative rather than evidence.
- Self-play stat trap: `p0_settlements` *falls* as the agent improves, because building a city
  returns the settlement piece. Track `settlements + cities`.

### Waste telemetry

Every match summary carries a second line, from the challenger's side:

```
waste: 9.3 dev bought (0.5 dead), 0.5 trailing roads, hand 4.4 at end-turn / 4.0 at game end
       (losses: 5.0 held, 0.7 dead dev, 1.9 trailing)
```

**trailing roads** = roads built after the player's last settlement or city, i.e. roads that never
enabled anything (initial placement excluded). **hand at end-turn** is sampled live during play —
the action log records the decision but not the hand it was made with. The `losses:` clause is the
same numbers over lost games only; divergence from the overall means shows what the agent does
when its plan stalls.

---

## Measured results

| measurement | result |
| --- | --- |
| `ppo-8vp-scratch` vs `weighted`, 200 games, seed 42 | 89.2% (177W-20L-3D), avg 128 turns |
| …with the placement scorer attached | **97.0%** (194W-6L), avg 85 turns |
| …vs the identical agent *without* the scorer | **85.5%** (171W-29L) |
| `ppo` / `mcts` / `ppo-mcts`, 40 games @ 50 sims | 80.0% / 22.5% / **95.0%** |
| `feas02/best.pt` vs `weighted` / `value`, 200 games each | 70.8% / 71.8% |

**Placement diagnosis (2026-08-12).** Across seeds 7/11/23/42/99 the PPO policy picked a 3-tile
node 5/5 times but ranked **7-17th of the ~18-22 available 3-tile nodes** — bottom half every
time, numbers like [4,5,9] and [4,12,5], almost no 6s or 8s. `rho(production, policy prior) ≈ 0`
on every board. So the policy learned adjacent-tile count but is **blind to dice numbers**. The
critic is no better: rho swung from -0.833 (seed 7) to **+0.730** (seed 42), i.e. board-dependent
noise rather than reliable inversion, over a narrow value spread. Ruled out: port confound, and
placement being bypassed in training (it runs the same policy/env loop as every other decision).

After training the specialist: chosen-corner rank **~11/54 → 4.2/54**, rho **~0 → +0.88**.

**Dev-card monoculture (2026-08-12).** `ppo-8vp-scratch` buys 9.3 dev cards/game (~28 resources)
but leaves only 0.5 unplayed — it cashes them. Yet it finishes with ~2.8 buildings (~0.8 net
beyond placement); its 7.7 VP ≈ 3.9 buildings + 2 Largest Army + ~1.8 VP cards. The policy
collapsed onto place-two → one city → pump dev cards, and never learned to expand. Losses hold 5.0
cards (vs 4.0) with 1.9 trailing roads (vs 0.5): when the dev line stalls it hoards and builds
roads to nowhere. **Search barely moved this** (dev 8.5→8.4, cities 1.1→1.4), so search's gain
reads as tactical rather than a cure for the monoculture.

---

## The two-roll lookahead (`src/env/lookahead.py`)

The hypothesis behind the waste telemetry is that the policy undervalues `END_TURN` because a kept
hand only pays off on the far side of a dice roll.

**In MCTS this remains unfixed**, and the general search does not cover it:

- PUCT spreads ~100 simulations across the whole action list, so `END_TURN` collects perhaps 10-15
  visits — used to sample an 11-outcome distribution (121 for two rolls deep). That estimate is
  noise.
- It is **biased against ending turn**, not merely noisy: `_select` applies FPU reduction to
  under-visited children, so a branch whose few sampled rolls came up bad gets a poor Q and then
  stops being visited. The search cannot tell an unreliable estimate from a genuinely low one.

**For PPO the fix took a different shape, because a value override does not apply.** PPO picks
actions from policy logits and its critic scores *states*, not actions — there is no per-action
value to replace. So the lookahead is delivered as **observation features** instead
(`--lookahead`, 614 → 642):

| block | count |
| --- | --- |
| E[gain] and Var[gain] per resource, next roll | 10 |
| **P(afford road / settlement / city / dev) after 1 roll, and after 2** | 8 |
| P(you must discard), E[cards lost], E[hand size] after 1 and 2 rolls | 4 |
| Opponent E[gain] per resource, and P(they must discard) | 6 |

**Probabilities, not expectations, are the payload.** "Expected 0.5 wheat" could be a near-certain
half wheat or 3 wheat one time in six, and the END_TURN question is a threshold — *will waiting let
me afford a settlement?* — which a mean cannot answer.

**Public information only.** The observation exposes 14 `P1_*` features and none break out the
opponent's hand: only `P1_NUM_RESOURCES_IN_HAND` (a count, public in a real game) and
`P1_NUM_DEVS_IN_HAND`. So no `P(opponent affords X)` feature exists — that would need their hand
composition, i.e. fabricated hidden state. What is used from their side is their *production*
(derivable from their buildings on the board) and their exact card count, which says whether a 7
would force them to discard. **Standing rule: anything derived for the observation must be
computable from what a human player can see** — Phase 3 puts this agent on colonist.io with only
the public view.

Cost: **114 µs per call**. The first implementation looped the 121 roll pairs and cost ~1 ms, the
same order as an entire PPO step; vectorising the grid to `(11, 11, 5)` gave 9×.

Approximations, all deliberate: the opponent acts between the two rolls and none of it is modelled
(so the projection is optimistic about their turn); a discard is assumed to remove cards
proportionally, where the real choice is the policy's; and the robber is read where it stands.

## Other open threads

- **Expert Iteration / warm start.** `AlphaZeroNet` and the PPO net share the 614-dim observation
  and 294-action space, so "start from PPO" means distilling PPO's masked policy and critic into a
  fresh `AlphaZeroNet` (`net.py:masked_policy_loss` + MSE), then running `train_az.py` from that
  checkpoint rather than random init.
- **Placement iteration 2.** `dataset.py --model <scorer> --epsilon 0.3` explores around the
  current scorer instead of uniformly. Labels also currently come from weighted-random rollouts;
  rolling out with the real agent makes them "good openings *for how we actually play*", which is
  why the scorer needs periodic retraining as the agent improves.
- **Cheap control never run.** A handcrafted cap on dev-card buys (~5/game). If a dumb constraint
  moves the win rate, it bounds what any search fix can recover.

## Setup and cloud

- `pip install -r requirements.txt`. W&B: `wandb login` once; `WANDB_MODE=offline` for smoke runs.
- Self-play generates data on the fly, so there is little to pre-upload and checkpoints are a few
  MB. Pull `best.pt` down for local inference on the GTX 1660 Super.

## Exit criteria

- Beats `WeightedRandomPlayer` > 80% **and** `VictoryPointPlayer` > 60%, over at least 200 games
  each (both are needed — the ladder is not ordered).
- A reproducible trained checkpoint in `checkpoints/` plus a local-inference entry point.
