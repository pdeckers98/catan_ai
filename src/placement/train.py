"""Fit the placement scorer to outcome-labelled openings.

Two losses, both fed by the same duplicate-board data.

**Regression (``--loss mse``, the default).** Squared error between a corner's
score and the duplicate-board outcome it earned. No discounting, no
bootstrapping: the label already *is* the result. Plain, and -- measured -- the
stronger of the two.

**Ranking (``--loss rank``).** A pair is a comparison: on this board, the first seat's
two corners beat the second seat's two, or lost, or tied. Scoring a bundle as the
sum of its corners' scores turns that into a Bradley-Terry fit -- push
``sigmoid(s(X) - s(Y))`` toward ``(delta + 1) / 2``. This matches what the model
is actually for. At play time it argmaxes over corners, so only the *ordering*
has to be right; a calibrated value is wasted effort, and demanding one from a
+-1 outcome label mostly fits noise. A tie (``delta == 0``) becomes a target of
0.5, which is a real statement -- these two openings were equal here -- rather
than the regression's much stronger claim that both are worth zero.

The learnable ``log_scale`` exists because the network's tanh output caps a
bundle difference at +-4. Without a scale the only way to express confidence is
to saturate the tanh, which flattens the ordering resolution the model is
supposed to provide; letting the temperature move absorbs that pressure. It is a
training-time parameter and is deliberately not saved -- inference only ranks.

It lost. Over 400 head-to-head games, an agent given the ranking scorer scored
45.3% against the same agent given the regression scorer. The likely culprit is
the additive bundle assumption: ``s(X) = s(n1) + s(n4)`` treats corners as
independent, but the first pick is featurised before the second exists, so it
cannot know what it will be paired with, and the sigmoid compounds that error
where the regression merely absorbs it. Kept because it is the right shape for
data whose labels come from strong rollouts rather than weighted-random -- an
untested hypothesis, not a recommendation.

Expect ugly numbers either way. Labels are +-1 and 0 under enormous variance, so
neither loss is the thing to steer by: :mod:`src.placement.evaluate` reports
whether the model learned dice numbers, and :mod:`src.eval.benchmark` reports
whether that wins games. Validation *pair accuracy* is the one training-time
number that means something directly -- the share of decided pairs whose winner
the model calls correctly.

Usage:
    python -m src.placement.dataset --pairs 2000 --workers 8
    python -m src.placement.train --data data/placement/samples.npz
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.placement.evaluate import diagnose, format_diagnosis
from src.placement.model import PlacementNet

# Held-out board seeds for the diagnostic. Fixed so the number means the same
# thing across runs, and far away from the default data-generation seeds.
DIAGNOSTIC_SEEDS = tuple(range(900_000, 900_040))

# Corner ordering within a pair: first seat's two, then second seat's two.
_BUNDLE_SIGN = torch.tensor([1.0, 1.0, -1.0, -1.0])


def load_dataset(path):
    """Per-corner view: (features, labels), float32 (N, F) and (N,)."""
    blob = np.load(Path(path))
    return blob["features"].astype(np.float32), blob["labels"].astype(np.float32)


def load_pairs(path):
    """Paired view: (pairs, deltas), float32 (P, 4, F) and (P,).

    Raises:
        KeyError: if the file predates the paired format. Regenerate it with
            :mod:`src.placement.dataset`, or train with ``--loss mse``.
    """
    blob = np.load(Path(path))
    if "pairs" not in blob:
        raise KeyError(
            f"{path} has no 'pairs' array -- it was written before the ranking "
            f"loss existed. Regenerate it, or pass --loss mse."
        )
    return blob["pairs"].astype(np.float32), blob["deltas"].astype(np.float32)


def _split(count, val_fraction, rng):
    order = rng.permutation(count)
    cut = max(1, int(count * (1.0 - val_fraction)))
    return order[:cut], order[cut:]


def _run(net, history, evaluate_fn, step_fn, batches_fn, epochs, patience,
         progress, headline):
    """Shared epoch loop: train, validate, early-stop on validation loss."""
    best_val, best_state, best_epoch, best_extra = float("inf"), None, 0, None
    for epoch in range(1, epochs + 1):
        net.train()
        total, seen = 0.0, 0
        for batch in batches_fn():
            loss = step_fn(batch)
            total += loss * len(batch)
            seen += len(batch)

        net.eval()
        with torch.no_grad():
            val_loss, extra = evaluate_fn()
        train_loss = total / max(1, seen)
        history.append((epoch, train_loss, val_loss, extra))

        if val_loss < best_val:
            best_val, best_epoch, best_extra = val_loss, epoch, extra
            best_state = {k: v.clone() for k, v in net.state_dict().items()}

        if progress and (epoch % 20 == 0 or epoch == 1):
            print(f"  epoch {epoch:>4}  train {train_loss:.4f}  val {val_loss:.4f}"
                  f"{headline(extra)}", flush=True)

        if patience and epoch - best_epoch >= patience:
            if progress:
                print(f"  early stop at epoch {epoch}; best was {best_epoch}",
                      flush=True)
            break

    if best_state is not None:
        net.load_state_dict(best_state)
    net.eval()
    if progress:
        print(f"  keeping epoch {best_epoch} (val {best_val:.4f}"
              f"{headline(best_extra)})", flush=True)
    return net, history


def train_ranking(pairs, deltas, epochs=200, batch_size=256, lr=1e-3,
                  val_fraction=0.15, width=64, seed=0, patience=25, progress=True):
    """Fit a :class:`PlacementNet` by Bradley-Terry ranking. Returns (net, history).

    The split is by *pair*, not by corner. That matters: a bundle's two corners
    share a label and are compared against each other, so splitting corners would
    put half of a comparison in train and half in validation and leak.
    """
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    train_idx, val_idx = _split(len(pairs), val_fraction, rng)
    train_x = torch.as_tensor(pairs[train_idx])
    train_y = torch.as_tensor((deltas[train_idx] + 1.0) / 2.0)
    val_x = torch.as_tensor(pairs[val_idx])
    val_d = torch.as_tensor(deltas[val_idx])
    val_y = (val_d + 1.0) / 2.0

    net = PlacementNet(feature_dim=pairs.shape[-1], width=width)
    net.fit_normalizer(pairs[train_idx].reshape(-1, pairs.shape[-1]))

    # Temperature on the bundle difference; see the module docstring.
    log_scale = torch.zeros((), requires_grad=True)
    optimizer = torch.optim.Adam(
        [{"params": net.parameters(), "weight_decay": 1e-4},
         {"params": [log_scale], "weight_decay": 0.0}],
        lr=lr,
    )

    def logits(x):
        scores = net(x.reshape(-1, x.shape[-1])).reshape(len(x), -1)
        return torch.exp(log_scale) * (scores * _BUNDLE_SIGN).sum(dim=1)

    def step(batch):
        optimizer.zero_grad()
        loss = F.binary_cross_entropy_with_logits(logits(train_x[batch]),
                                                  train_y[batch])
        loss.backward()
        optimizer.step()
        return loss.detach().item()

    def batches():
        perm = torch.randperm(len(train_x))
        return (perm[i:i + batch_size] for i in range(0, len(perm), batch_size))

    def evaluate():
        if not len(val_x):
            return 0.0, 0.0
        val_logits = logits(val_x)
        loss = float(F.binary_cross_entropy_with_logits(val_logits, val_y))
        # Accuracy is only defined where the pair actually decided something.
        decided = val_d != 0
        if not bool(decided.any()):
            return loss, 0.0
        correct = torch.sign(val_logits[decided]) == torch.sign(val_d[decided])
        return loss, float(correct.float().mean())

    history = []
    return _run(net, history, evaluate, step, batches, epochs, patience, progress,
                headline=lambda acc: f"  pair-acc {acc:.3f}" if acc else "")


def train(features, labels, epochs=200, batch_size=256, lr=1e-3,
          val_fraction=0.15, width=64, seed=0, patience=25, progress=True):
    """Fit a :class:`PlacementNet` by regression. Returns (net, history).

    Labels carry a weak signal under enormous noise, so this overfits early and
    hard -- validation loss bottoms out within a few dozen epochs and then climbs
    steadily while training loss keeps falling. Early stopping on validation loss
    is doing real work here, not tidying up: the epoch-200 network ranks corners
    noticeably worse than the epoch-20 one.
    """
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    train_idx, val_idx = _split(len(features), val_fraction, rng)
    train_x = torch.as_tensor(features[train_idx])
    train_y = torch.as_tensor(labels[train_idx])
    val_x = torch.as_tensor(features[val_idx])
    val_y = torch.as_tensor(labels[val_idx])

    net = PlacementNet(feature_dim=features.shape[1], width=width)
    # Fitted on the training split only -- the validation set must not inform
    # the input scaling any more than it informs the weights.
    net.fit_normalizer(features[train_idx])

    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.MSELoss()

    def step(batch):
        optimizer.zero_grad()
        loss = loss_fn(net(train_x[batch]), train_y[batch])
        loss.backward()
        optimizer.step()
        return loss.detach().item()

    def batches():
        perm = torch.randperm(len(train_x))
        return (perm[i:i + batch_size] for i in range(0, len(perm), batch_size))

    def evaluate():
        return (float(loss_fn(net(val_x), val_y)) if len(val_x) else 0.0), None

    history = []
    return _run(net, history, evaluate, step, batches, epochs, patience, progress,
                headline=lambda _: "")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data", default="data/placement/samples.npz")
    parser.add_argument("--out", default="checkpoints/placement/scorer.pt")
    parser.add_argument("--loss", choices=("rank", "mse"), default="mse",
                        help="mse: per-corner regression (default; measured "
                             "stronger). rank: Bradley-Terry on pairs.")
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

    if args.loss == "rank":
        pairs, deltas = load_pairs(args.data)
        decided = int(np.count_nonzero(deltas))
        print(f"{len(pairs)} pairs, {pairs.shape[-1]} features, "
              f"{decided} decided ({decided / max(1, len(deltas)):.1%})")
        net, _ = train_ranking(pairs, deltas, **kwargs)
    else:
        features, labels = load_dataset(args.data)
        informative = int(np.count_nonzero(labels))
        print(f"{len(labels)} samples, {features.shape[1]} features, "
              f"{informative} informative ({informative / max(1, len(labels)):.1%})")
        net, _ = train(features, labels, **kwargs)

    path = net.save(args.out)
    print(f"saved -> {path}")
    print(format_diagnosis(diagnose(net, DIAGNOSTIC_SEEDS)))


if __name__ == "__main__":
    main()
