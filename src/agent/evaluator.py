"""Leaf evaluators for MCTS.

An evaluator maps a position to ``(priors, value)``:

- ``priors``: a distribution over the full action space, zero on illegal slots.
- ``value``: the expected result in [-1, 1] **from the perspective of the player
  to move at that position**.

Keeping this behind one interface is what makes Stage 0 possible: the same tree
search runs on a random prior, on the existing MaskablePPO checkpoint, or on the
AlphaZero net, so their strengths are directly comparable.

The interface is batched (``evaluate_batch``) because a batched forward pass is
several times cheaper per leaf than 294-wide single-row inference, and the search
loop is the compute bottleneck.
"""

import numpy as np
import torch

from src.agent.encoding import action_size
from src.agent.net import AlphaZeroNet


class Evaluator:
    """Base class: implement ``evaluate_batch``; ``evaluate`` is derived."""

    def evaluate_batch(self, obs_batch, mask_batch):
        """Args: (B, obs_dim) float32, (B, num_actions) bool.

        Returns: ((B, num_actions) float32 priors, (B,) float32 values).
        """
        raise NotImplementedError

    def evaluate(self, obs, mask):
        """Single-position convenience wrapper."""
        priors, values = self.evaluate_batch(obs[None, :], mask[None, :])
        return priors[0], float(values[0])


class UniformEvaluator(Evaluator):
    """Uniform priors, value 0. The no-network control for Stage 0.

    Search on top of this is pure PUCT with no knowledge, which isolates how much
    of any improvement comes from lookahead versus from the network.
    """

    def evaluate_batch(self, obs_batch, mask_batch):
        counts = mask_batch.sum(axis=-1, keepdims=True)
        priors = mask_batch.astype(np.float32) / np.maximum(counts, 1)
        values = np.zeros(len(mask_batch), dtype=np.float32)
        return priors, values


class NetEvaluator(Evaluator):
    """Wraps an :class:`AlphaZeroNet`."""

    def __init__(self, net: AlphaZeroNet, device="cpu"):
        self.net = net.to(device).eval()
        self.device = device

    @classmethod
    def from_path(cls, path, device="cpu") -> "NetEvaluator":
        return cls(AlphaZeroNet.load(path, device=device), device=device)

    def evaluate_batch(self, obs_batch, mask_batch):
        return self.net.infer(obs_batch, mask_batch)


class PPOEvaluator(Evaluator):
    """Wraps a trained MaskablePPO checkpoint as an MCTS evaluator.

    This is the Stage 0 adapter. Two caveats worth remembering when reading its
    numbers:

    - PPO's critic predicts a *shaped, discounted return*, not a win probability,
      so its scale is arbitrary. We squash it with ``tanh`` to land in [-1, 1].
      Ordering is preserved, which is what PUCT actually needs; calibration is not.
    - That critic was trained only on states the PPO policy itself visited, so it
      is extrapolating on the positions search drags it into.
    """

    def __init__(self, model, value_scale: float = 1.0):
        self.model = model
        self.value_scale = value_scale
        self.num_actions = action_size()

    @classmethod
    def from_path(cls, path, value_scale: float = 1.0) -> "PPOEvaluator":
        from sb3_contrib import MaskablePPO
        model = MaskablePPO.load(str(path), device="cpu", custom_objects={"n_steps": 1})
        return cls(model, value_scale=value_scale)

    @torch.no_grad()
    def evaluate_batch(self, obs_batch, mask_batch):
        policy = self.model.policy
        obs_t, _ = policy.obs_to_tensor(np.asarray(obs_batch, dtype=np.float32))
        distribution = policy.get_distribution(obs_t, action_masks=mask_batch)
        priors = distribution.distribution.probs.cpu().numpy().astype(np.float32)
        values = policy.predict_values(obs_t).squeeze(-1)
        values = torch.tanh(values * self.value_scale).cpu().numpy().astype(np.float32)
        return priors, values
