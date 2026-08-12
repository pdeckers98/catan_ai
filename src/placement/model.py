"""The learned placement scorer.

A small MLP mapping one candidate node's feature vector to a scalar in [-1, 1]:
the expected game result for the player who settles there, +1 for a win and -1
for a loss. At play time every legal node is scored and the best one taken, so
the network never needs to produce a distribution -- it only has to order
corners correctly.

Deliberately tiny (two hidden layers, 64 wide). The input is 45 mechanical
features rather than the 614-dim game vector, the training set is a few thousand
labelled placements, and the target is one number; anything larger would memorise
boards instead of learning that 6s and 8s pay.

Feature scales are wildly different -- roll probabilities near 0.14 sit next to
tile counts near 9 -- so the standardisation statistics are fitted on the
training set and stored as buffers. They travel with the checkpoint, which means
inference cannot silently disagree with training about what the inputs mean.
"""

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from src.placement.features import feature_size


class PlacementNet(nn.Module):
    """Scores a single candidate settlement node.

    Args:
        feature_dim: input width. Defaults to :func:`~src.placement.features.feature_size`.
        width: hidden width.
    """

    def __init__(self, feature_dim=None, width=64):
        super().__init__()
        self.feature_dim = feature_dim if feature_dim is not None else feature_size()
        self.width = width

        # Buffers, not parameters: fitted once from the data, never by gradient.
        self.register_buffer("feature_mean", torch.zeros(self.feature_dim))
        self.register_buffer("feature_std", torch.ones(self.feature_dim))

        self.body = nn.Sequential(
            nn.Linear(self.feature_dim, width),
            nn.ReLU(),
            nn.Linear(width, width),
            nn.ReLU(),
            nn.Linear(width, 1),
            nn.Tanh(),
        )

    def fit_normalizer(self, features: np.ndarray) -> None:
        """Set the standardisation buffers from a training feature matrix."""
        mean = features.mean(axis=0)
        std = features.std(axis=0)
        # Constant columns (a port channel absent from every sampled board, say)
        # would divide by zero; leaving their scale at 1 passes them through as
        # the constant they are.
        std[std < 1e-6] = 1.0
        self.feature_mean.copy_(torch.as_tensor(mean, dtype=torch.float32))
        self.feature_std.copy_(torch.as_tensor(std, dtype=torch.float32))

    def forward(self, features):
        normed = (features - self.feature_mean) / self.feature_std
        return self.body(normed).squeeze(-1)

    @torch.no_grad()
    def score(self, features: np.ndarray) -> np.ndarray:
        """Score an (N, F) batch of candidates. Returns float32 (N,)."""
        features = np.asarray(features)
        if len(features) == 0:
            return np.zeros(0, dtype=np.float32)
        if features.ndim == 1:
            features = features[None, :]
        tensor = torch.as_tensor(features, dtype=torch.float32)
        return self(tensor).cpu().numpy().astype(np.float32)

    # ---- persistence ----------------------------------------------------
    def config(self) -> dict:
        return {"feature_dim": self.feature_dim, "width": self.width}

    def save(self, path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"config": self.config(), "state_dict": self.state_dict()}, path)
        return path

    @classmethod
    def load(cls, path, device="cpu") -> "PlacementNet":
        blob = torch.load(Path(path), map_location=device, weights_only=False)
        net = cls(**blob["config"])
        net.load_state_dict(blob["state_dict"])
        net.to(device)
        net.eval()
        return net


class BundleNet(PlacementNet):
    """Scores an opening as a *pair* of corners rather than one at a time.

    The per-corner scorer has a structural blind spot: it commits to the first
    settlement before knowing what the second will be, so it cannot prefer a
    slightly weaker corner that opens a much better pairing. Feeding it both
    corners at once -- the second encoded with ``assume_owned=(first,)`` -- puts
    the whole opening in front of the network and lets complementarity be part
    of the decision instead of an afterthought.

    Input is the two corners concatenated, in pick order, so the head keeps a
    consistent meaning for "the one I take first". Everything else, including
    the normalisation buffers and the persistence format, is inherited.
    """

    def __init__(self, feature_dim=None, width=64, corners=2):
        self.corners = corners
        per_corner = feature_dim if feature_dim is not None else feature_size()
        super().__init__(feature_dim=per_corner * corners, width=width)
        self.corner_dim = per_corner

    def config(self) -> dict:
        return {"feature_dim": self.corner_dim, "width": self.width,
                "corners": self.corners}

    def forward(self, features):
        # Accept either (N, corners, F) or the already-flat (N, corners * F).
        if features.dim() == 3:
            features = features.reshape(len(features), -1)
        return super().forward(features)

    def fit_normalizer(self, features: np.ndarray) -> None:
        super().fit_normalizer(np.asarray(features).reshape(len(features), -1))
