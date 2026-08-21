# Phase 2 — The AI

**Goal:** a strong 1v1 agent, trained by self-play. **Status:** ✅ a shipped agent exists.

The agent is three pieces that are trained separately because they are different problems, and
**all three must be deployed together** — none of them is a standalone player:

| | trains on | entry point | role at play time |
| --- | --- | --- | --- |
| **PPO** (MaskablePPO) | sparse win/loss, gym env | `src.agent.train` | the policy and the critic |
| **PUCT search** | nothing — it is inference-only | `src.agent.mcts` | ~+10 points on the same weights |
| **Opening placement** | random openings labelled by game outcome | `src.placement.train` | the two opening settlements |

The shipped configuration, under the target ruleset (15 VP, Longest Road on, 1500-turn cap):

```bash
python -m src.eval.benchmark --vps-to-win 15 --longest-road --max-turns 1500 \
    --agent ppo-mcts --model checkpoints/archive/ppo-15vp-lr-step400000.zip \
    --simulations 50 --opponent value --games 200 \
    --placement-model checkpoints/placement/scorer_ppo.pt \
    --bundle-model    checkpoints/placement/bundle_noroads.pt
```

---

## Rules that shape the learning problem

The VP target, the Longest Road award and the turn cap are **per-run**, selected through
`src/env/ruleset.py` (`--vps-to-win` / `--longest-road` / `--max-turns`, or the matching `CATAN_*`
environment variables). Defaults reproduce the historical setup: 8 VP, no Longest Road, 1000 turns.

**Why 15 VP is a different game.** Buildings cap at **9 VP** — 5 settlements, 4 of them upgraded to
cities. So 12 VP is unreachable without Largest Army or Longest Road, and 15 is unreachable without
VP cards on top. Road-building stops being the pathology the old 8-VP rules made it and becomes
mandatory. Raise `--max-turns` to 1500 at 15 VP: the default 1000 is already binding there, and a
truncated episode pays 0, so capped games teach nothing. Raise `--gamma` too — games run ~2.4x
longer at 15 VP than at 8 (median 417 turns vs 172, over 30 WeightedRandom mirror games).

Beyond the ruleset, `src/env/rules.py` monkeypatches the engine at import time (seven patches; full
list in that module): discard above **9** cards rather than 7, per-resource one-card-at-a-time
discard the policy actually chooses (action space 290 → **294**), correct multi-discarder
sequencing, Colonist.io 1v1 robber restrictions, Longest Road awarding no VP by default, no dev card
the turn it was bought, and `_discard_remaining` surviving `State.copy()` so MCTS works.

**How the ruleset travels.** `src/env/rules.py` decides at *import* time whether to suppress the
Longest Road award, and `SubprocVecEnv` / arena workers are **spawned** on Windows, so they
re-import everything from a fresh interpreter. A function argument cannot carry the ruleset; it
lives in environment variables, which children inherit for free. Entry points call
`ruleset.apply_cli_overrides()` as their **first statement, above the engine imports**, and
`train.py` hard-fails if argparse disagrees with the ruleset the engine imported under. This is why
`.flake8` grants E402 to `train.py` and `benchmark.py`.

There is **no reward shaping**, and the machinery is gone rather than switched off.
`EpisodeStatsWrapper` keeps the end-of-episode telemetry the old `RewardShapingWrapper` also carried
(VPs, settlements, cities, roads) without touching the reward.

---

## What actually helps

Measured 2026-08-16, all at 12/15 VP with Longest Road:

| change | effect | games |
| --- | --- | --- |
| **50-sim search at inference** | **+9.8 points** (59.8%) | 200, mirror match |
| 100-sim search | 60.8% — search saturates by 50 | 200, mirror match |
| shared policy/value trunk | +2.9, not significant | 800 |
| 800k steps of 15 VP fine-tuning | −1.5, not significant | — |

**Search at inference dominates further training.** In a mirror match — identical weights, the only
difference being search — 50 sims/move is worth ~10 points, while a full day of architecture and
training work bought nothing measurable.

Search saturating by 50 sims says the ceiling is **critic quality, not search budget**: PPO's value
head estimates a discounted return rather than a win probability, and it never trained on the
positions search explores. Every PPO run so far plateaus once it beats its references, so **reach
for search, or harder opponents, before another training run.**

### Sample sizes

A 300-game head-to-head reported a 3-point edge that a 500-game run at a different seed did not
reproduce (56.0% then 51.0%; pooled 52.9% ± 1.8% over 800). Budget **800+ games** before believing
any difference under ~5 points, and treat 200 games as resolving nothing finer than ~7 points.

---

## The stack

### `src/agent/train.py` — MaskablePPO

The only trainer. Reward is the sparse win/loss outcome and nothing else. `--opponent pool` is the
strongest setting: frozen past checkpoints mixed with a slice of scripted games
(`src/agent/pool.py`), rated by Elo against the run's own ladder (`src/agent/elo.py`) rather than by
a win rate that cannot exceed 100%.

`src/agent/trunk.py` (`--trunk`, off by default) gives the policy and value heads a shared
LayerNorm+GELU trunk instead of SB3's two independent towers. It measured a statistical tie, but
**it cannot be deleted**: `ppo-15vp-lr-step400000.zip` stores `features_extractor_class` as the
literal string `src.agent.trunk.SharedTrunk`, so the shipped model will not unpickle without that
module at that exact path.

```bash
python -m src.agent.train --vps-to-win 15 --longest-road --max-turns 1500 --gamma 0.999 \
    --total-steps 3000000 --lookahead \
    --placement-model checkpoints/placement/scorer_ppo.pt \
    --bundle-model    checkpoints/placement/bundle_noroads.pt --run-name <name>
```

### `src/agent/mcts.py` — PUCT search

- **Stochastic transitions are sampled, not enumerated.** Each edge holds a dict of children keyed
  by the *realized* outcome (`_outcome_key`: dice sum, dev card drawn, resource stolen). Repeated
  visits land on children in proportion to true probabilities, giving an unbiased expectation
  without a full fan-out per roll.
- **Dead rolls are pooled into one chance outcome.** A dice sum is *live* if either player has a
  building on a tile carrying that number which the robber is not on; 7 is always live. Everything
  else pays nobody, and catanatron's non-7 roll branch does nothing but pay production and set the
  prompt — so two dead sums leave byte-identical successor states, which
  `test_pooled_dead_rolls_leave_identical_positions` pins down. The search therefore samples the roll itself from a
  compressed table — one entry per live sum at its true 1/36-table probability, plus a single
  pooled entry carrying all the dead mass — and forces the dice via `Action(color, ROLL, (d1, d2))`,
  which the engine honours (`state.py`: `dices = action.value or roll_dice()`). **The distribution
  over successor states is unchanged, so the estimator stays exactly unbiased**; what changes is
  that identical positions stop scattering across up to ten children that each cost a network call
  and each get one visit.
  Measured over 1063 mid-game roll positions, **7.0 of 11 sums are live**, so a roll edge branches
  ~8 ways instead of 11. The effect is modest and shows up as depth, not node count (an MCTS
  simulation always creates exactly one node): at 200 sims mean tree depth goes 3.85 → 4.16 and max
  depth 4.83 → 5.83; at 50 sims, 2.79 → 2.97. Biggest early, when few numbers are built on, and
  whenever the robber blanks a number.
- **`horizon` caps the search in game turns.** Off by default — depth is normally a budget, not a
  horizon, and every measurement on record was taken without it. Set it and a position that many
  turns past the root is scored by the value head instead of expanded (`_beyond_horizon`, applied in
  `_expand` where the leaf value is already in hand), trading depth for breadth near the root. Turns
  count the way `max_turns` counts them, so in 1v1 `horizon=1` reaches the end of the opponent's
  reply and `horizon=2` the end of your next turn. Exposed as `--horizon` / `--opponent-horizon` on
  `src.eval.benchmark` and `--horizon` on `src.eval.play`.
- **Non-alternating perspective.** A Catan turn is many consecutive decisions by one player. Each
  node records who is to move; values negate on backup whenever perspective flips.
- **Forced moves are free** — single-legal-action plies are played without search.
- `fpu_reduction` pessimizes unvisited children relative to the parent's value, stopping the search
  fanning out uniformly across Catan's very wide action lists (54 legal moves at the opening).
- **Batched leaf evaluation** (`MCTS(batch_size=N)`, virtual loss to stop descents converging on one
  leaf) is the throughput lever: 1.6× at 8, 1.8× at 16, 2.0× at 32. Ceiling is ~2× because the
  forward pass is only half the cost. **16 is the sweet spot.** `batch_size=1` reproduces serial
  search exactly, so it is opt-in.

The evaluator declares `wants_lookahead` (inferred from its own checkpoint's observation width) and
the search reads it once at construction. **Any search measurement recorded before `c54ea46` is
void** — MCTS built 614-value observations and handed them to nets expecting 642, so `ppo-mcts`
crashed against every checkpoint trained with `--lookahead`.

### Compute budget

Measured on the dev box (12 logical / 6 physical cores, CPU-only torch):

| operation | cost |
| --- | --- |
| `Game.copy()` | ~32 µs |
| `encode_observation` | ~131 µs |
| **network forward (batch 1)** | **~604 µs** |

**The forward pass dominates, not the state copy.** Consequences: shrinking the net buys throughput
only sublinearly, because encoding and tensor setup are fixed overhead; and **never let torch spawn
intra-op threads for search** — measured 711 µs/eval at 1 thread vs 858 µs at 6, one sixth the cores
and 21% *faster*. Every search entry point calls `torch.set_num_threads(1)`.

### `src/placement/` — the opening specialist

Placement is structurally unlike the rest of the game: no dice at decision time, ~2 decisions per
game, fully observable, hugely decisive — and it receives only **~2 of ~300 gradient samples per
episode**. That data starvation is why the main policy never learned it: across seeds 7/11/23/42/99
the PPO policy picked a 3-tile node 5/5 times but ranked **7-17th of the ~18-22 available 3-tile
nodes**, with almost no 6s or 8s. It learned adjacent-tile count and stayed blind to dice numbers.

**Design constraint: no hardcoded placement knowledge.** The line is facts vs judgements. A tile's
roll probability is a property of dice and belongs in the features; whether 6s beat 5s or ore+wheat
beats brick+wood is a judgement and must be learned from outcomes.

| module | role |
| --- | --- |
| `features.py` | 45 mechanical facts per node: production rate per resource, tile counts, dice-number histogram, desert count, port one-hot, first/second-settlement flag, both players' holdings, buildable production 2 and 3 edges out. No weighting between blocks. |
| `dataset.py` | Openings explored **uniformly at random** (sampling from a scorer would bake in its preferences), played out, labelled with the actual outcome. Emits duplicate-board *pairs*. |
| `../env/dice.py` | `fixed_dice(seed)` — a dedicated roll stream, so both games of a pair see the same dice. |
| `model.py` | `PlacementNet`: 64-wide 2-layer MLP → scalar, scoring one corner. `BundleNet`: same body over two concatenated corners, scoring a whole opening. Normalisation stats fitted on the train split, stored as buffers so inference cannot disagree with training. |
| `chooser.py` | `OpeningChooser` — the selection rule. The corner scorer shortlists 12 first picks and 30 partners; the bundle scorer ranks every pairing. |
| `player.py` | Intercepts initial-phase settlements only; everything else, initial roads included, goes to the inner agent. |
| `env_wrapper.py` | The gym-side equivalent, for learners that step an env rather than acting as a `Player`. |

**Duplicate-board pairing** is the variance reduction. Each board is played twice with the four
opening nodes swapped between seats; the label is the difference. Same seat wins both → the board
explained it, both openings labelled 0. Swapping flips the winner → the openings explained it,
labels ±1. ~73% of pairs come back informative. The snake draft (P0, P1, P1, P0) makes the swap
always legal: replay in the order n2, n1, n4, n3 and every node keeps its non-adjacency. The two
opening **roads travel with their settlements** on the replay, which is what makes the pair a
controlled comparison rather than a comparison with a free variable in it.

**Common random numbers** duplicate the dice too. `src/env/dice.py:fixed_dice` swaps
`catanatron.state.roll_dice` for a dedicated `random.Random`, so roll *k* is identical in both games
of a pair no matter how the bots' actions diverge. *It measured neutral* — 49.0% head-to-head, and
the informative-pair rate barely moved (73.1% → 73.9%). Left on because it is free and makes the
pair an actually controlled comparison.

**The selection rule was the lever, not the labels.** `BundleNet` scores both opening corners at
once, the second encoded with `assume_owned=(first,)` so complementarity is visible; `OpeningChooser`
searches corner *pairs* instead of taking corners greedily. Measured **642W-558L = 53.5%** over 1200
games, CI [50.7, 56.3], against the same agent with the same corner scorer and the same training
data — only the selection rule differs. Six seeds were needed; the first two read 48.5% and 60.0%.

**The best partner may not survive.** Between the first seat's two picks the opponent takes two
corners, so taking the max over partners is optimistic. The first seat therefore scores a first
corner by its `PARTNER_RANK`-th best partner (default 3). Measured: rank 3 beats rank 1 **52.2%**
over 2400 games, CI [50.2, 54.2]. Some pessimism is what pays, not its precise amount — rank 2
scored 50.7% and rank 5 scored 52.3%.

```bash
# ~11 min on 8 workers with weighted rollouts; ~72 min with --rollout ppo
python -m src.placement.dataset --pairs 8000 --workers 8 --seed 1 \
    --rollout ppo --rollout-model checkpoints/archive/ppo-15vp-lr-step400000.zip
python -m src.placement.train --data data/placement/samples.npz \
    --target corner --out checkpoints/placement/scorer_ppo.pt
python -m src.placement.train --data data/placement/samples.npz \
    --target bundle --out checkpoints/placement/bundle_noroads.pt
```

Both targets read the same `.npz`: `--target corner` trains on the flattened per-settlement view,
`--target bundle` on the `bundles` array. **Training overfits fast** — early stopping is load-bearing,
not tidiness. Validate a refit with `src.eval.benchmark` over 800+ games; the loss is not the number
to steer by, and the rank/rho diagnostic that used to print here was removed with
`src/placement/evaluate.py`.

**Attach both models, always:**

```bash
--placement-model checkpoints/placement/scorer_ppo.pt \
--bundle-model    checkpoints/placement/bundle_noroads.pt
```

The corner scorer is not optional and not superseded — in pair-search it is the shortlister that
makes the search affordable, and the fallback when nothing pairs. `--placement-model` alone drops to
greedy corner-at-a-time selection, which is what every measurement before `51a3097` used.
`src/agent/train.py` accepts both flags, so PPO runs can train against pair-searched openings.

**For a gym-based learner**, `make_placement_env(model_path, bundle_path=...)` returns an env that
plays the whole initial build phase **inside `reset()`** — both seats — so the learner's first
observation is a mid-game position. Doing it in `reset()` rather than intercepting mid-episode is
what keeps the rollout buffer honest: an overridden action would otherwise sit in the buffer as if
the learner had chosen it. The opponent gets the scorer too by default — training against bad
openings teaches the agent to exploit an edge it will not have.

Two caveats. **Initial roads are random here**, where `PlacementPlayer` delegates them to the inner
agent — a small train/play mismatch on a 2-3 option decision. And **`env.reset(seed=n)` does not
control the board layout**, and is not even repeatable across two resets: the gym env builds its map
off the global `random` module at reset time. Rank placements against
`env.unwrapped.game.state.board.map`, never against a board built separately from the same seed.

### The two-roll lookahead (`src/env/lookahead.py`)

The hypothesis behind the waste telemetry is that the policy undervalues `END_TURN` because a kept
hand only pays off on the far side of a dice roll. PPO picks actions from policy logits and its
critic scores *states*, not actions, so there is no per-action value to override — the lookahead is
delivered as **observation features** instead (`--lookahead`, 614 → 642):

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
composition, i.e. fabricated hidden state. **Standing rule: anything derived for the observation
must be computable from what a human player can see** — Phase 3 puts this agent on colonist.io with
only the public view.

Cost: **114 µs per call**. The first implementation looped the 121 roll pairs and cost ~1 ms;
vectorising the grid to `(11, 11, 5)` gave 9×. Approximations, all deliberate: the opponent acts
between the two rolls and none of it is modelled; a discard is assumed to remove cards
proportionally; and the robber is read where it stands.

---

## Benchmarking

`src/agent/arena.py` builds any agent by name — `random`, `weighted`, `value`, `mcts`, `ppo`,
`ppo-mcts` — and plays **alternating-seat** matches (the first player picks first in placement, a
real edge).

**Waste telemetry.** Every match summary carries a second line, from the challenger's side:

```
waste: 9.3 dev bought (0.5 dead), 0.5 trailing roads, hand 4.4 at end-turn / 4.0 at game end
       (losses: 5.0 held, 0.7 dead dev, 1.9 trailing)
```

**trailing roads** = roads built after the player's last settlement or city, i.e. roads that never
enabled anything (initial placement excluded). **hand at end-turn** is sampled live during play. The
`losses:` clause is the same numbers over lost games only; divergence from the overall means shows
what the agent does when its plan stalls.

Self-play stat trap: `p0_settlements` *falls* as the agent improves, because building a city returns
the settlement piece. Track `settlements + cities`. (Per player: 15 roads, 5 settlements, 4 cities.)

Note that `value` here is **`VictoryPointPlayer`** — this build of catanatron ships only
`RandomPlayer`, `WeightedRandomPlayer` and `VictoryPointPlayer`, and the pool already runs that
same bot as its `greedy` slice. There is no stronger scripted opponent to reach for.

### `src/eval/waste.py` — what a turn actually buys

A decision-level audit, and the answer to a question the arena's waste line cannot reach: the
arena counts what was bought, this counts what was *thrown away*. Measured 2026-08-19,
`ppo-15vp-lr-step400000` over 30 self-play games at 15 VP:

| | our agent | `VictoryPointPlayer` | weighted-random |
| --- | --- | --- | --- |
| turns containing a maritime trade | **36%** | 20% | 19% |
| trades made while something was already affordable | **68%** | 40% | 30% |
| trades giving away a resource acquired the same turn | **42%** | 24% | 21% |
| cards paid to the bank per game | **136** | 126 | 116 |
| dev cards bought while one card short of a city | **32%** | 17% | — |

It trades constantly, mostly while it can already build, and undoes itself in a single turn
nearly half the time. It never buys a development card while `BUILD_CITY` is *legal* — it buys
one when it is a single card away, spending the wheat and ore the city was waiting on.

**Why no benchmark caught it.** Both sides of a mirror burn cards, so burning them costs nothing
relative to the opponent and the match scores 50%. Exactly the blindness that hid the refusal to
expand — the pathology has to be *counted*, not scored.

**Why training allows it.** The terminal win/loss is one bit spread over ~600 of the agent's own
decisions at 15 VP with `--gamma 0.999`. Three cards handed to the bank move the return by far
less than the noise in the advantage estimate, so the wasteful trade is **free in the loss**;
nothing pushes it down and the entropy bonus keeps probability mass on it. Self-play compounds it,
because the opponent is wasting too.

Run it on any agent spec `benchmark` accepts:

```bash
python -m src.eval.waste --vps-to-win 15 --longest-road --max-turns 1500     --agent ppo --model checkpoints/archive/ppo-15vp-lr-step400000.zip     --placement-model checkpoints/placement/scorer_ppo.pt     --bundle-model    checkpoints/placement/bundle_noroads.pt --games 30
```

### Potential-based shaping (`--shaping-weight`) — run on 2026-08-21, and it lost

`PotentialShapingWrapper` adds `F(s, s') = γΦ(s') − Φ(s)` with
`Φ(s) = w·(my actual VP − their visible VP)`. This is the Ng/Harada/Russell form: over an episode
it telescopes to `γ^T Φ(s_T) − Φ(s_0)`, and with `Φ` forced to zero in the absorbing state the
total added return is the constant `−Φ(s_0)`. **It cannot invent a new optimal policy.** Verified
on real games — four different policies on one board seed each added exactly `+0.05`.

This is *not* the deleted `RewardShapingWrapper` returning. That one paid one-time bonuses for
crossing VP milestones, which do not telescope and genuinely could move the optimum; that is why
it was a crutch and why it is still not coming back.

Two deliberate choices, both tested in `tests/test_shaping.py`:

- **The opponent contributes their *visible* VP**, ours our actual. Scoring their actual would
  leak a face-down victory-point card into the reward and teach the critic to expect a signal it
  cannot observe.
- **Truncation is absorbing too.** A turn-limited game pays 0 and teaches nothing; leaving `Φ`
  standing there would let a policy bank shaping for a lead it never converted, which is the one
  way this wrapper could stop being policy-invariant.

**The run.** From a random init at `--shaping-weight 0.05`, 15 VP + Longest Road, 1500-turn cap,
`--gamma 0.999 --lookahead --opponent pool`, evaluated every 300k steps. Stopped by hand at
**1.8M of 3M** once the picture stopped changing (`checkpoints/ppo-15vp-shaped/`, W&B run
`c9hanngt`). It did exactly what it was designed to do, and it did not help.

**It fixed the trade waste.** `src/eval/waste.py`, mirror games, against the
`ppo-15vp-lr-step400000` baseline in the section above:

| | archive | 600k | 900k | 1.2M | 1.5M | 1.8M (40 games) |
| --- | --- | --- | --- | --- | --- | --- |
| trades/game | 56 | 35 | 35 | 71 | 28 | **31** |
| % of turns trading | 36% | 29% | 31% | 40% | 27% | **30%** |
| bought nothing that turn | 77% | 66% | 63% | 81% | 60% | **60%** |
| made while already affordable | 68% | 53% | 55% | 75% | 52% | **55%** |
| same-turn giveaway | 42% | 22% | 19% | 38% | 20% | **20%** |
| cards to the bank/game | 136 | 91 | 85 | 157 | 74 | **80** |
| dev bought one short of a city | 32% | 29% | 29% | 32% | 39% | **32%** |

Cards to the bank fell **136 → 80**, a 41% cut, stable across four evals. The mechanism is the
stated one: a trade moves no victory point so it earns what it always earned, while a city now
pays immediately, and building wins the local comparison that 600 decisions of credit assignment
erase. What did *not* change is the character of the decision — 55% of trades are still made with
something already affordable, 60% still buy nothing, and `dev bought one card short of a city`
returned to the archive's exact 32%. Shaping cut the volume of waste, not the judgement.

**The 1.2M column is a 20-game sampling artefact, not a regression.** It was read as one at the
time and it was wrong to do so. Audits at 20 games swing by 2x on these rates; the per-eval audit
went to 40 games from 1.8M on. **Budget 40+ games for a waste audit** — the same discipline the
head-to-heads already have.

**It did not expand.** Settlements per game fell monotonically, 2.9 → 2.7 → 2.5 → 2.4, then held
at ~2.5 for the rest of the run while cities sat at 2.8 and dev buys at ~15. Games got 28% shorter
over the same span (230 → 166 turns). **Speed and expansion moved in opposite directions**: the
fast line is the dev-card/city line that needs no brick and no roads. Anything proposed on the
theory that quicker wins imply more building has to answer this table first.

**And the score went nowhere.** Against `ppo-15vp-lr-step400000`, both sides with the placement
models:

| comparison | score | note |
| --- | --- | --- |
| bare policy, 800 games | **50.3%** (397W-392L-11D) | ±1.8%; rules out any edge above ~4 points |
| **50-sim search both sides, 200 games** | **42.2%** (83W-114L-3D) | ±3.5% |

Note the reference is *handicapped*: `ppo-15vp-lr-step400000` trained under the pre-2026-08-19
friendly-robber rule and before the Road Building fix, so it plays these games under rules it
never saw. A tie against it is uninformative; **losing to it is the signal.**

**Why search makes it worse — the part worth remembering.** Potential shaping leaves the optimal
policy alone but it does *not* leave the value function alone: the critic of the shaped MDP learns
`V(s) − Φ(s)`. That is harmless to PPO, which only ever forms advantages inside the shaped MDP.
It is not harmless to MCTS, which takes that same value head as a leaf evaluator and runs it over
the **unshaped** game — `mcts.py` searches raw `Game` objects and never adds `Φ` back along the
path. Every leaf is therefore scored with a systematic `−w·(VP lead)` bias, marking down exactly
the positions the agent should be steering toward, and more simulations apply the bias more
thoroughly. This predicts the observed pattern — neutral bare, harmful under search — and matches
the 1.5 VP deficit in the final margin (11.2 vs 12.7). It is a hypothesis with one confirming
experiment, not a proven cause; the cheap test is to add `Φ` at the leaf and re-run the 200 games.

So the finding is stronger than "shaping bought nothing": **shaping is incompatible with the
deployed configuration**, which is always search plus the policy, never the policy alone.

**Two lessons that outlive this experiment:**

1. **Measure reward changes with search on.** The bare number said "no effect, harmless" and that
   was wrong by 8 points. Any reward change alters the critic, and the critic is what search runs on.
2. **A behavioural metric improving is not evidence of strength.** The waste audit measured
   something real, moved it 41% in the intended direction, and converted to nothing. It stays
   useful as a diagnosis of *what* a policy does; it is not a substitute for a head-to-head.

`--pool-search-frac` is the other half of the same idea from the opponent side: a slice of the
pool played with search on top, since search is worth ~+9.8 points and there is no scripted bot
above `VictoryPointPlayer` to reach for instead. It is off by default because every simulation is
a forward pass on the rollout workers' own cores. **It has still not been run.**

### Reproducibility

Every entry point calls `src/env/determinism.py:ensure_hash_seed()`, which **relaunches the process
once under `PYTHONHASHSEED=0`** when the variable is unset. Without it a seed fixes the board but not
the game: Catanatron builds `playable_actions` off sets of enum members, and `enum.Enum.__hash__`
hashes the member *name string*, so randomised string hashing reshuffles action order every process.
The same seed produced five different games across five runs before this landed.

Two consequences: results are now identical regardless of `--workers`, and **any measurement
recorded before `c34ab25` is not reproducible** — its statistics are still valid, but the exact games
cannot be recovered.

---

## Checkpoints worth keeping

`checkpoints/` is git-ignored, so `checkpoints/archive/` is the convention for models that should
outlive their run directory (a rerun with the same `--run-name` overwrites everything else).

| file | rules | notes |
| --- | --- | --- |
| `ppo-15vp-lr-step400000.zip` | 15 VP, LR | **current agent.** Fine-tuned from the 12 VP trunk model; run it with `--agent ppo-mcts --simulations 50` |
| `ppo-12vp-trunk-step2000000.zip` | 12 VP, LR | shared trunk, 932k params; its parent |
| `ppo-12vp-lr-step1800000.zip` | 12 VP, LR | two-tower `[256,256]`, 537k params; a statistical tie with the trunk model |

The 12 VP models play 15 VP without retraining (80–0 vs weighted-random out of the box, 15.0 VP in
145 turns) — the win condition changes, the mechanics do not.

---

## Caveats

**No trained checkpoint here is a standalone agent — every one must ship with a placement scorer.**
Any run trained through `PlacementWrapper` plays the opening inside `reset()`, so those decisions
never enter the rollout buffer and the policy head receives **zero gradient on placement**. Deprived
of the scorer it places with an untrained head. Measured over 200 games each: 100.0% → 88.2% vs
weighted-random, and head-to-head against `ppo-8vp-scratch` it goes from 53.0% (both scored) to
42.2% (neither) — bare, it is *worse* than the older agent that at least learned placement badly.
**This applies to the Phase 3 colonist.io bridge too: the scorer is part of the deployed agent, not
an eval-time extra.**

**The placement specialist was fitted under the old rules**, on rollouts that never ran to 12 or 15
VP. Whether corner values shift when the game runs longer is untested, and it is the most likely
place a stale assumption is still costing points. Refitting it is the clearest open piece of work,
which is why the whole dataset/train pipeline is kept.

---

## Dead ends, so they are not re-run

- **AlphaZero.** A full track existed (`train_az.py`, `selfplay.py`, `net.py`, PUCT supplying an
  improved policy target). It never got past feasibility — 70.8% vs weighted-random — and the code
  was removed once PPO + search shipped. Recoverable from git history if the critic-quality argument
  above is ever worth acting on. The lesson worth keeping: **`--simulations` must comfortably exceed
  the branching factor.** At 50 sims against Catan's ~54-action opening, search cannot visit every
  child once, `pi` becomes a perturbed copy of the network's own prior, and the network trains to
  imitate itself. Signature: policy loss plateaus at `ln(k)` for `k` the mean legal-action count.
- **Roads in the opening bundle.** A bundle model over (settlement, road) × 2 scored **44.1%**
  against the settlement-only bundle over 1200 games, wiping out the entire pair-scoring gain. On 32
  of 40 boards the two models picked different first settlements, and the roads model sat *closer*
  to the plain corner scorer (mean rank 1.70 vs 3.02) — the 24 road dimensions swamped the
  complementarity signal. The road *features* have been removed; the road *replay* in data
  generation was neutral and is kept, because controlling a variable is right even when modelling it
  is not. Roads remain unsolved and worth solving, with a separate model conditioned on the chosen
  pair.
- **Refitting the placement scorer under the 15 VP ruleset.** The most-suspected stale assumption
  in the project, measured on 2026-08-19 and **refuted**. Regenerating the labels under 15 VP +
  Longest Road (8000 pairs, rolled out with `ppo-15vp-lr-step400000`, 85.4% informative) does change
  what the model reaches for, exactly as predicted: brick goes from 1.63 to 4.15 mean pips and
  brickless openings from 57% to 22% of boards, paid for out of sheep. And the openings are worse.

  | comparison, 1600 games each, seats alternating | score | 95% CI |
  | --- | --- | --- |
  | same pipeline refit on the **old** data vs the shipped models — the control | 49.9% | [47.5, 52.4] |
  | **15 VP** models vs shipped, weighted-random both sides | 45.7% | [43.3, 48.2] |
  | **15 VP** models vs shipped, `ppo-15vp-lr-step400000` both sides | 47.4% | [44.9, 49.9] |

  The control is what makes this readable: retraining on the old file reproduces the shipped models
  to within noise, so the pipeline is sound and the deficit belongs to the new labels. And it shows
  up under weighted-random, which is fitted to neither opening, so it is not "our policy cannot use
  brick" — the brick-rich openings are simply worse.

  **What the same runs show instead:** in every configuration above, both agents build about **2.2
  settlements** — the opening two, plus a fraction. Handing the policy a brick-rich opening did not
  make it expand. The refusal to expand is a property of the policy, not of the board it is handed,
  which is why placement was never the lever. `data/placement/samples_15vp.npz` is kept as evidence,
  and `dataset.py` now stamps its ruleset into every file it writes.
- **A Bradley-Terry ranking loss** on placement pairs: 45.3% over 400 games. Probable cause is the
  additive assumption `s(X) = s(n1) + s(n4)` — the first corner is featurised before the second
  exists. See `git show 5a3e2eb`.
- **Label-quality work generally.** CRN dice, the ranking loss, and agent-strength rollouts all left
  play strength where it was, despite two of them measurably improving label quality. 13.75× more
  data improved the model but not the play (50.7% over 9600 games). The label pipeline is not what
  limits this scorer.
- **Bigger models.** Plateaus here are opponent difficulty and exploration, not capacity; width
  costs ~40% throughput for nothing.
- **`feas01` / `feas02` checkpoints** never converged. A null result measured against them is
  uninformative, not evidence.
- **Potential-based reward shaping (`--shaping-weight 0.05`).** Run from scratch on 2026-08-21 and
  stopped at 1.8M steps. It cut cards-paid-to-the-bank 136 → 80 exactly as designed, and scored
  **50.3%** over 800 bare games and **42.2%** over 200 games with 50-sim search against
  `ppo-15vp-lr-step400000` — a reference handicapped by the old friendly-robber rule. The waste it
  removed was real and worth nothing; the search result is worse than nothing, most likely because
  the shaped critic learns `V − Φ` while `mcts.py` evaluates leaves in the unshaped game. Full
  write-up in the shaping section above. `checkpoints/ppo-15vp-shaped/best.zip` is kept as evidence.

## Setup and cloud

- `pip install -r requirements.txt`. W&B: `wandb login` once; `WANDB_MODE=offline` for smoke runs.
- Self-play generates data on the fly, so there is little to pre-upload and checkpoints are a few
  MB. Pull `best.zip` down for local inference on the GTX 1660 Super.
