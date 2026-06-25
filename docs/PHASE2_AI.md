# Phase 2 — Build the AI (deep RL)

**Goal:** train a strong 1v1 agent that beats the built-in bots, then push further with self-play.
**Status:** ⏳ not started (Phase 1 complete).

## Why action masking is non-negotiable

The action space is `Discrete(290)` but only a handful of actions are legal each turn (54 at the
opening). An unmasked policy spends ~all its probability mass on illegal actions and never learns.
Always feed the mask from `src/env/catan_env.py:valid_action_mask` into the policy.

## Track A (start here): MaskablePPO baseline

`sb3_contrib.MaskablePPO` is PyTorch under the hood and handles masking out of the box.

```python
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from src.env.catan_env import make_1v1_env, valid_action_mask

env = ActionMasker(make_1v1_env(), valid_action_mask)
model = MaskablePPO("MlpPolicy", env, verbose=1, device="cpu")  # small MLP; CPU is fine
model.learn(total_timesteps=2_000_000)
model.save("checkpoints/maskppo_v0")
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

## Cloud training (CPU-bound)

This workload is **CPU-bound** — throughput is dominated by stepping game simulations, and the net
is a small MLP. Provision a **high-core-count CPU instance** and run many envs in parallel with
`stable_baselines3.common.vec_env.SubprocVecEnv`. Self-play generates data on the fly, so there is
little to pre-upload. **Decide the exact sync mechanism (block volume vs. plain checkpoint `scp`)
during this phase**, once we see iteration speed and checkpoint sizes. Pull the final `.pt` /
`.zip` down for local inference on the GTX 1660 Super.

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
