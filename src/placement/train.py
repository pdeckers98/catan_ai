"""Fit the placement scorer to outcome-labelled openings.

Plain supervised regression: minimise squared error between the network's score
for a corner and the duplicate-board outcome label that corner earned. There is
no discounting and no bootstrapping, because the label already *is* the result.

**Split by pair, not by corner.** A pair's four corners share one outcome -- the
first seat's two take ``+delta``, the second's take ``-delta`` -- so splitting
corners independently puts halves of the same comparison on opposite sides of
the train/val boundary and makes validation look better than it is. Early
stopping is the thing that keeps this model honest, so a leak that flatters
validation makes it stop late, which is exactly the failure mode to avoid.

*A Bradley-Terry ranking loss on those pairs was tried and lost* -- 45.3% over
400 head-to-head games against this regression. Probable cause is the additive
bundle assumption ``s(X) = s(n1) + s(n4)``: the first corner is featurised
before the second exists, so it cannot know what it will be paired with. See
`git show 5a3e2eb`.

Two things about the loss are worth expecting in advance. It will look terrible
-- labels are +-1 and 0 with enormous variance, so a validation MSE near 0.8 is
normal and improving it to 0.7 is a large gain. And it is not the number to
steer by: :mod:`src.placement.evaluate` reports whether the model learned dice
numbers, and :mod:`src.eval.benchmark` reports whether that wins games.

Usage:
    python -m src.placement.dataset --pairs 2000 --workers 8
    python -m src.placement.train --data data/placement/samples.npz
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from src.placement.dataset import flatten_pairs
from src.placement.evaluate import diagnose, format_diagnosis
from src.placement.features import feature_size
from src.placement.model import BundleNet, PlacementNet

FEATURE_SIZE = feature_size()

# Held-out board seeds for the diagnostic. Fixed so the number means the same
# thing across runs, and far away from the default data-generation seeds.
DIAGNOSTIC_SEEDS = tuple(range(900_000, 900_040))


def load_pairs(path):
    """Duplicate-board pairs: (pairs, deltas), float32 (P, 4, F) and (P,).

    Raises:
        KeyError: if the file predates the paired format; regenerate it with
            :mod:`src.placement.dataset`.
    """
    blob = np.load(Path(path))
    if "pairs" not in blob:
        raise KeyError(
            f"{path} has no 'pairs' array -- it predates the paired format. "
            f"Regenerate it with src.placement.dataset."
        )
    return blob["pairs"].astype(np.float32), blob["deltas"].astype(np.float32)


def load_bundles(path):
    """Openings as corner *pairs*: (bundles, deltas), float32 (P, 2, 2, F), (P,).

    Axis 1 is the seat (first, second); axis 2 is the corner in pick order.
    """
    blob = np.load(Path(path))
    if "bundles" not in blob:
        raise KeyError(
            f"{path} has no 'bundles' array. Regenerate it with "
            f"src.placement.dataset."
        )
    return blob["bundles"].astype(np.float32), blob["deltas"].astype(np.float32)


def drop_road_features(bundles):
    """Keep only the settlement half of each corner in a bundle array.

    Data generation records the opening road alongside the settlement, which is
    right -- the road is a real decision and leaving it unrecorded was a bug.
    But a bundle model *trained* on those 12 extra dimensions per corner scored
    44.1% against the settlement-only one over 1200 games: the road features
    swamped the complementarity signal that pair-scoring exists to capture. So
    the shipped bundle model is fitted on this slice. Road features are appended
    after the node features, so the leading columns are exactly the old format.
    """
    return np.ascontiguousarray(bundles[..., :FEATURE_SIZE])


def flatten_bundles(bundles, deltas):
    """(2P, 2, F) openings and (2P,) labels -- one sample per seat per pair."""
    if len(bundles) == 0:
        return bundles.reshape(0, 2, 0), deltas
    samples = bundles.reshape(-1, *bundles.shape[2:])
    labels = np.stack([deltas, -deltas], axis=1).reshape(-1)
    return samples.astype(np.float32), labels.astype(np.float32)


def train_bundle(bundles, deltas, epochs=200, batch_size=256, lr=1e-3,
                 val_fraction=0.15, width=64, seed=0, patience=25, progress=True):
    """Fit a :class:`BundleNet` on whole openings. Returns (net, history).

    Same regression against the same duplicate-board label as the per-corner
    scorer -- the only change is that the network sees both corners at once, so
    complementarity is available to it rather than having to be inferred one
    corner at a time. Split is by pair, as above.
    """
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    order = rng.permutation(len(bundles))
    cut = max(1, int(len(bundles) * (1.0 - val_fraction)))
    train_x, train_y = flatten_bundles(bundles[order[:cut]], deltas[order[:cut]])
    val_x, val_y = flatten_bundles(bundles[order[cut:]], deltas[order[cut:]])

    net = BundleNet(feature_dim=bundles.shape[-1], width=width)
    net.fit_normalizer(train_x)
    return _fit(net, train_x, train_y, val_x, val_y, epochs, batch_size, lr,
                patience, progress)


def train(pairs, deltas, epochs=200, batch_size=256, lr=1e-3, val_fraction=0.15,
          width=64, seed=0, patience=25, progress=True):
    """Fit a :class:`PlacementNet`. Returns (net, history).

    Labels carry a weak signal under enormous noise, so this overfits early and
    hard -- validation loss bottoms out within a few dozen epochs and then climbs
    steadily while training loss keeps falling. Early stopping on validation loss
    is doing real work here, not tidying up: the epoch-200 network ranks corners
    noticeably worse than the epoch-20 one.
    """
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    order = rng.permutation(len(pairs))
    cut = max(1, int(len(pairs) * (1.0 - val_fraction)))
    train_x, train_y = flatten_pairs(pairs[order[:cut]], deltas[order[:cut]])
    val_x, val_y = flatten_pairs(pairs[order[cut:]], deltas[order[cut:]])

    net = PlacementNet(feature_dim=pairs.shape[-1], width=width)
    # Fitted on the training split only -- the validation set must not inform
    # the input scaling any more than it informs the weights.
    net.fit_normalizer(train_x)
    return _fit(net, train_x, train_y, val_x, val_y, epochs, batch_size, lr,
                patience, progress)


def _fit(net, train_x, train_y, val_x, val_y, epochs, batch_size, lr, patience,
         progress):
    """Adam + MSE with early stopping on validation loss. Returns (net, history)."""
    train_x_t = torch.as_tensor(train_x)
    train_y_t = torch.as_tensor(train_y)
    val_x_t = torch.as_tensor(val_x)
    val_y_t = torch.as_tensor(val_y)

    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.MSELoss()

    history = []
    best_val, best_state, best_epoch = float("inf"), None, 0
    for epoch in range(1, epochs + 1):
        net.train()
        perm = torch.randperm(len(train_x_t))
        total = 0.0
        for start in range(0, len(perm), batch_size):
            batch = perm[start:start + batch_size]
            optimizer.zero_grad()
            loss = loss_fn(net(train_x_t[batch]), train_y_t[batch])
            loss.backward()
            optimizer.step()
            total += loss.detach().item() * len(batch)

        net.eval()
        with torch.no_grad():
            val_loss = float(loss_fn(net(val_x_t), val_y_t)) if len(val_x_t) else 0.0
        train_loss = total / max(1, len(perm))
        history.append((epoch, train_loss, val_loss))

        if val_loss < best_val:
            best_val, best_epoch = val_loss, epoch
            best_state = {k: v.clone() for k, v in net.state_dict().items()}

        if progress and (epoch % 20 == 0 or epoch == 1):
            print(f"  epoch {epoch:>4}  train {train_loss:.4f}  val {val_loss:.4f}",
                  flush=True)

        if patience and epoch - best_epoch >= patience:
            if progress:
                print(f"  early stop at epoch {epoch}; best was {best_epoch}",
                      flush=True)
            break

    if best_state is not None:
        net.load_state_dict(best_state)
    net.eval()
    if progress:
        print(f"  keeping epoch {best_epoch} (val {best_val:.4f})", flush=True)
    return net, history


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data", default="data/placement/samples.npz")
    parser.add_argument("--out", default="checkpoints/placement/scorer.pt")
    parser.add_argument("--target", choices=("corner", "bundle"), default="corner",
                        help="corner: score one settlement at a time (default). "
                             "bundle: score both corners of an opening jointly.")
    parser.add_argument("--keep-road-features", action="store_true",
                        help="Fit the bundle on the road dimensions too. Off by "
                             "default: the roads model measured 44.1%% against "
                             "the settlement-only one. See drop_road_features.")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--patience", type=int, default=25,
                        help="epochs without validation improvement before "
                             "stopping; 0 disables early stopping")
    args = parser.parse_args()

    kwargs = dict(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
                  width=args.width, seed=args.seed, patience=args.patience)

    if args.target == "bundle":
        bundles, deltas = load_bundles(args.data)
        if not args.keep_road_features and bundles.shape[-1] > FEATURE_SIZE:
            print(f"dropping road features: {bundles.shape[-1]} -> {FEATURE_SIZE} "
                  f"per corner")
            bundles = drop_road_features(bundles)
        fit = lambda: train_bundle(bundles, deltas, **kwargs)  # noqa: E731
        unit = f"{len(bundles)} pairs -> {2 * len(bundles)} openings"
    else:
        pairs, deltas = load_pairs(args.data)
        fit = lambda: train(pairs, deltas, **kwargs)  # noqa: E731
        unit = f"{len(pairs)} pairs -> {4 * len(pairs)} corners"

    decided = int(np.count_nonzero(deltas))
    print(f"{unit}, {decided} decided ({decided / max(1, len(deltas)):.1%})")

    net, _ = fit()
    path = net.save(args.out)
    print(f"saved -> {path}")
    # The rank/rho diagnostic ranks single corners, so it has nothing to say
    # about a bundle scorer. Games are the metric there.
    if args.target == "corner":
        print(format_diagnosis(diagnose(net, DIAGNOSTIC_SEEDS)))


if __name__ == "__main__":
    main()
