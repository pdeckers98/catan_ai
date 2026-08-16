"""A shared policy/value trunk for MaskablePPO.

SB3's ``MlpPolicy`` builds *two independent towers* from ``net_arch``: the policy
and the value function each get their own copy of the hidden layers, and they
share no weights at all. That is a poor fit for this game. Reward is sparse --
one win/loss at the end of a 130-to-400 turn episode -- so the critic is the only
dense learning signal in the run, and it does learn: ``ppo-12vp-lr`` held an
``explained_variance`` of 0.86-0.90 while its policy stopped improving after 1.2M
steps. None of what the critic understood about a Catan position was reaching the
policy's representation, because there was no path for it to travel.

This extractor gives them one. Both heads read the same trunk, so gradients from
the value loss shape the features the policy acts on.

Two smaller departures from the SB3 default, both aimed at optimisation rather
than capacity:

- **GELU instead of Tanh.** Tanh saturates, and a saturated unit passes almost no
  gradient; it is a conservative default for unnormalised networks.
- **LayerNorm on every hidden layer.** The observation mixes wildly different
  scales -- one-hot board features next to resource counts next to the lookahead
  probabilities -- and normalising keeps early training from being dominated by
  whichever block happens to be largest.

Sizing is deliberately modest. The diagnostics above say this run was never
capacity-limited, so the trunk exists to *share* representation, not to add
parameters; 512 wide with 256-wide heads costs ~24% of rollout throughput against
the two-tower baseline (610 -> 466 steps/s measured on the 12 VP mixture), which
a 3-hour budget absorbs easily -- it still collects ~5M steps, and the run this
replaces plateaued at 1.2M.
"""

import gymnasium as gym
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class SharedTrunk(BaseFeaturesExtractor):
    """LayerNorm+GELU MLP whose output feeds both the policy and value heads.

    Args:
        observation_space: the env's (flat, 1-D) observation space.
        hidden_sizes: width of each trunk layer. The last one is the feature
            dimension SB3 hands to ``net_arch``.
    """

    def __init__(self, observation_space: gym.Space,
                 hidden_sizes: tuple[int, ...] = (512, 512)):
        if not hidden_sizes:
            raise ValueError("SharedTrunk needs at least one hidden layer")
        super().__init__(observation_space, features_dim=hidden_sizes[-1])

        layers: list[nn.Module] = []
        in_dim = int(observation_space.shape[0])
        for width in hidden_sizes:
            layers += [nn.Linear(in_dim, width), nn.LayerNorm(width), nn.GELU()]
            in_dim = width
        self.mlp = nn.Sequential(*layers)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.mlp(observations)
