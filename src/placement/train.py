"""Fit the placement scorer to outcome-labelled openings.

Plain supervised regression: minimise squared error between the network's score
for a corner and the duplicate-board outcome label that corner earned. There is
no discounting and no bootstrapping, because the label already *is* the result.

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

from src.placement.evaluate import diagnose, format_diagnosis
from src.placement.model import PlacementNet

# Held-out board seeds for the diagnostic. Fixed so the number means the same
# thing across runs, and far away from the default data-generation seeds.
DIAGNOSTIC_SEEDS = tuple(range(900_000, 900_040))


def load_dataset(path):
    blob = np.load(Path(path))
    return blob["features"].astype(np.float32), blob["labels"].astype(np.float32)


def train(features, labels, epochs=200, batch_size=256, lr=1e-3,
          val_fraction=0.15, width=64, seed=0, patience=25, progress=True):
    """Fit a :class:`PlacementNet`. Returns (net, history).

    Labels carry a weak signal under enormous noise, so this overfits early and
    hard -- validation loss bottoms out within a few dozen epochs and then climbs
    steadily while training loss keeps falling. Early stopping on validation loss
    is doing real work here, not tidying up: the epoch-200 network ranks corners
    noticeably worse than the epoch-20 one.
    """
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    order = rng.permutation(len(features))
    features, labels = features[order], labels[order]
    split = max(1, int(len(features) * (1.0 - val_fraction)))
    train_x, val_x = features[:split], features[split:]
    train_y, val_y = labels[:split], labels[split:]

    net = PlacementNet(feature_dim=features.shape[1], width=width)
    # Fitted on the training split only -- the validation set must not inform
    # the input scaling any more than it informs the weights.
    net.fit_normalizer(train_x)

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
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--patience", type=int, default=25,
                        help="epochs without validation improvement before "
                             "stopping; 0 disables early stopping")
    args = parser.parse_args()

    features, labels = load_dataset(args.data)
    informative = int(np.count_nonzero(labels))
    print(f"{len(labels)} samples, {features.shape[1]} features, "
          f"{informative} informative ({informative / max(1, len(labels)):.1%})")

    net, _ = train(
        features, labels, epochs=args.epochs, batch_size=args.batch_size,
        lr=args.lr, width=args.width, seed=args.seed, patience=args.patience,
    )

    path = net.save(args.out)
    print(f"saved -> {path}")
    print(format_diagnosis(diagnose(net, DIAGNOSTIC_SEEDS)))


if __name__ == "__main__":
    main()
