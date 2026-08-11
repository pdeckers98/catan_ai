"""AlphaZero-style policy+value network.

One trunk, two heads:

- **policy head** -> logits over the 294-slot action space. Illegal actions are
  masked to -inf before the softmax, so the priors MCTS receives are a
  distribution over legal moves only.
- **value head** -> a scalar in [-1, 1] estimating the game result *from the
  perspective of the player to move*. Because observations are encoded from the
  mover's point of view (see ``src.agent.encoding``), one network serves both
  seats and the sign flips on backup in the tree.

The trunk is a residual MLP. Catan's 614-dim feature vector is a flat mix of
one-hot board features and raw counts, so there is no spatial structure for a
conv net to exploit; width and depth are cheap on CPU and that is where self-play
runs.
"""

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.agent.encoding import action_size, obs_size

# Value fed to masked logits before the softmax. Large enough to zero the slot,
# small enough not to produce NaNs in float32.
NEG_INF = -1e9


class ResidualBlock(nn.Module):
    """Pre-activation style residual MLP block with LayerNorm."""

    def __init__(self, width: int):
        super().__init__()
        self.fc1 = nn.Linear(width, width)
        self.norm1 = nn.LayerNorm(width)
        self.fc2 = nn.Linear(width, width)
        self.norm2 = nn.LayerNorm(width)

    def forward(self, x):
        h = F.relu(self.norm1(self.fc1(x)))
        h = self.norm2(self.fc2(h))
        return F.relu(x + h)


class AlphaZeroNet(nn.Module):
    """Policy + value network over the flat Catan feature vector.

    Args:
        obs_dim: observation length. Defaults to the live env's (614).
        num_actions: action-space size. Defaults to the live env's (294).
        width: trunk width.
        blocks: number of residual blocks.
    """

    def __init__(self, obs_dim=None, num_actions=None, width=256, blocks=4):
        super().__init__()
        self.obs_dim = obs_dim if obs_dim is not None else obs_size()
        self.num_actions = num_actions if num_actions is not None else action_size()
        self.width = width
        self.blocks = blocks

        # Raw features span very different scales (booleans next to resource counts
        # up to ~95), so normalize immediately after the input projection rather
        # than relying on the optimizer to rescale the first layer.
        self.stem = nn.Sequential(
            nn.Linear(self.obs_dim, width),
            nn.LayerNorm(width),
            nn.ReLU(),
        )
        self.trunk = nn.Sequential(*[ResidualBlock(width) for _ in range(blocks)])

        self.policy_head = nn.Sequential(
            nn.Linear(width, width // 2),
            nn.ReLU(),
            nn.Linear(width // 2, self.num_actions),
        )
        self.value_head = nn.Sequential(
            nn.Linear(width, width // 2),
            nn.ReLU(),
            nn.Linear(width // 2, 1),
            nn.Tanh(),
        )

    def forward(self, obs):
        """Returns (policy_logits, value) -- logits are *unmasked*."""
        h = self.trunk(self.stem(obs))
        return self.policy_head(h), self.value_head(h).squeeze(-1)

    @torch.no_grad()
    def infer(self, obs, mask):
        """Batched masked inference for search.

        Args:
            obs: float32 array (B, obs_dim) or (obs_dim,).
            mask: bool array of the same leading shape, (B, num_actions).

        Returns:
            (priors, values): priors (B, num_actions) sum to 1 over legal slots;
            values (B,) in [-1, 1]. Both numpy float32.
        """
        obs_t = torch.as_tensor(np.atleast_2d(obs), dtype=torch.float32)
        mask_t = torch.as_tensor(np.atleast_2d(mask), dtype=torch.bool)

        logits, value = self(obs_t)
        logits = logits.masked_fill(~mask_t, NEG_INF)
        priors = torch.softmax(logits, dim=-1)
        return (
            priors.cpu().numpy().astype(np.float32),
            value.cpu().numpy().astype(np.float32),
        )

    # ---- persistence ----------------------------------------------------
    def config(self) -> dict:
        return {
            "obs_dim": self.obs_dim,
            "num_actions": self.num_actions,
            "width": self.width,
            "blocks": self.blocks,
        }

    def save(self, path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"config": self.config(), "state_dict": self.state_dict()}, path)
        return path

    @classmethod
    def load(cls, path, device="cpu") -> "AlphaZeroNet":
        blob = torch.load(Path(path), map_location=device, weights_only=False)
        net = cls(**blob["config"])
        net.load_state_dict(blob["state_dict"])
        net.to(device)
        net.eval()
        return net


def masked_policy_loss(logits, target_pi, mask):
    """Cross-entropy between the MCTS visit distribution and the masked policy.

    ``target_pi`` is already zero on illegal slots, so masking the logits keeps the
    log-partition honest without the zeros contributing gradient.
    """
    logits = logits.masked_fill(~mask, NEG_INF)
    log_probs = torch.log_softmax(logits, dim=-1)
    return -(target_pi * log_probs).sum(dim=-1).mean()
