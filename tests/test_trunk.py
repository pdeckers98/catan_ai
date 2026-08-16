"""Tests for the shared policy/value trunk and its wiring into train.py."""

import argparse

import gymnasium as gym
import numpy as np
import pytest
import torch
import torch.nn as nn

from src.agent.trunk import SharedTrunk

OBS_DIM = 642


def _obs_space(dim: int = OBS_DIM) -> gym.Space:
    return gym.spaces.Box(low=-np.inf, high=np.inf, shape=(dim,),
                          dtype=np.float32)


def _args(trunk=None, net_arch=None) -> argparse.Namespace:
    return argparse.Namespace(trunk=trunk, net_arch=net_arch or [256, 256])


# --------------------------------------------------------------------------
# The extractor itself
# --------------------------------------------------------------------------
def test_features_dim_is_the_last_trunk_layer():
    """SB3 sizes the heads off features_dim; a wrong value fails far downstream."""
    trunk = SharedTrunk(_obs_space(), hidden_sizes=(512, 384))
    assert trunk.features_dim == 384
    assert trunk(torch.zeros(4, OBS_DIM)).shape == (4, 384)


def test_a_single_layer_trunk_is_allowed():
    trunk = SharedTrunk(_obs_space(), hidden_sizes=(256,))
    assert trunk(torch.zeros(2, OBS_DIM)).shape == (2, 256)


def test_an_empty_trunk_is_rejected():
    """Silently degrading to an identity would look like a working run."""
    with pytest.raises(ValueError):
        SharedTrunk(_obs_space(), hidden_sizes=())


def test_the_trunk_normalises_and_does_not_saturate():
    """The point of LayerNorm+GELU over Tanh: gradient still flows at scale.

    Tanh maps a large input to +-1 and passes almost nothing back; a normalised
    trunk keeps its output in a sane range *and* stays differentiable.
    """
    trunk = SharedTrunk(_obs_space(), hidden_sizes=(128,))
    big = (torch.randn(8, OBS_DIM) * 50.0).requires_grad_(True)
    out = trunk(big)
    out.sum().backward()

    assert out.abs().max() < 50.0, "trunk output tracked the input scale"
    first = trunk.mlp[0].weight
    assert first.grad is not None and first.grad.abs().sum() > 0, \
        "no gradient reached the first layer"


# --------------------------------------------------------------------------
# Wiring into train.py
# --------------------------------------------------------------------------
def test_no_trunk_flag_leaves_the_old_configuration_untouched():
    """The flag must be additive: an unflagged run is the historical setup."""
    from src.agent.train import _policy_kwargs

    assert _policy_kwargs(_args()) == {"net_arch": [256, 256]}


def test_trunk_flag_installs_the_extractor_and_drops_tanh():
    from src.agent.train import _policy_kwargs

    kwargs = _policy_kwargs(_args(trunk=[512, 512], net_arch=[256]))
    assert kwargs["features_extractor_class"] is SharedTrunk
    assert kwargs["features_extractor_kwargs"] == {"hidden_sizes": (512, 512)}
    assert kwargs["activation_fn"] is nn.GELU
    assert kwargs["net_arch"] == [256]


def test_ppo_builds_a_policy_whose_heads_share_the_trunk():
    """The whole point: one set of trunk weights feeding both heads.

    Built through MaskablePPO rather than by hand, because what matters is that
    SB3 accepts the kwargs and wires them the way the flag claims.
    """
    from sb3_contrib import MaskablePPO
    from src.agent.train import _policy_kwargs

    env = gym.make("CartPole-v1")  # any Box/Discrete env exercises the wiring
    model = MaskablePPO(
        "MlpPolicy", env, n_steps=64, batch_size=32, device="cpu",
        policy_kwargs=_policy_kwargs(_args(trunk=[64, 48], net_arch=[32])),
    )
    policy = model.policy

    assert isinstance(policy.features_extractor, SharedTrunk)
    assert policy.features_extractor.features_dim == 48
    # SB3 exposes per-head extractors; shared means they are the *same object*,
    # which is what lets the value loss shape the policy's features.
    assert policy.pi_features_extractor is policy.vf_features_extractor
    env.close()
