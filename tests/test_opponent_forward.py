"""PolicyPlayer._forward must reproduce MaskablePPO.predict exactly.

The fast path skips SB3's per-call plumbing (eval-mode walk, double Categorical
construction), so the thing to prove is that it changes nothing about the
chosen actions: identical argmax under deterministic=True, and identical
samples *and RNG consumption* under deterministic=False.
"""

import numpy as np
import pytest
import torch

from catanatron import Color

from src.env.catan_env import make_1v1_env  # noqa: F401 -- applies rule patches
from src.agent.opponent import PolicyPlayer

N_ACTIONS = 294
OBS_DIM = 614
N_TRIALS = 200


@pytest.fixture(scope="module")
def policy():
    from sb3_contrib import MaskablePPO
    env = make_1v1_env()
    # Untrained random-init weights: equivalence is a property of the forward
    # computation, not of any particular checkpoint.
    return MaskablePPO("MlpPolicy", env, device="cpu", n_steps=64, seed=3)


def random_inputs(rng):
    obs = rng.standard_normal(OBS_DIM).astype(np.float64)
    n_legal = rng.integers(2, 40)
    mask = np.zeros(N_ACTIONS, dtype=bool)
    mask[rng.choice(N_ACTIONS, size=n_legal, replace=False)] = True
    return obs, mask


def test_deterministic_matches_predict(policy):
    player = PolicyPlayer(Color.RED, policy, deterministic=True)
    player._ensure_policy()
    rng = np.random.default_rng(11)
    for _ in range(N_TRIALS):
        obs, mask = random_inputs(rng)
        fast = player._forward(policy, obs, mask)
        ref, _ = policy.predict(obs, action_masks=mask, deterministic=True)
        assert fast == int(ref)
        assert mask[fast]


def test_sampling_matches_predict_and_rng(policy):
    player = PolicyPlayer(Color.RED, policy, deterministic=False)
    player._ensure_policy()
    rng = np.random.default_rng(12)
    for i in range(N_TRIALS):
        obs, mask = random_inputs(rng)
        torch.manual_seed(1000 + i)
        fast = player._forward(policy, obs, mask)
        state_after_fast = torch.get_rng_state()
        torch.manual_seed(1000 + i)
        ref, _ = policy.predict(obs, action_masks=mask, deterministic=False)
        assert fast == int(ref)
        assert mask[fast]
        # Same number of RNG draws, so a shared torch seed replays the same
        # game whether the opponent uses the fast path or SB3's.
        assert torch.equal(state_after_fast, torch.get_rng_state())
