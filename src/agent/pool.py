"""Durable opponent pool for PPO self-play.

Distinct from ``checkpoint_manager``, which keeps only the three most recent
checkpoints so the run directory does not fill up. That rotation is fine for
crash recovery and useless as an opponent pool -- three adjacent snapshots are
nearly the same policy, and the history the agent should be learning against has
already been deleted. This module keeps a separate, thinned, long-horizon set.

Sampling is uniform over the pool. The tempting alternative -- always train
against the newest checkpoint -- is what makes self-play cycle: the agent chases
the current opponent, forgets the counter to an older one, and rediscovers it
later, going round in circles while every metric looks flat. Sampling uniformly
over history is the fictitious-play fix, and it is why the pool has to keep old
entries rather than a sliding window.
"""

import random
import shutil
from pathlib import Path

from catanatron import Color
from catanatron.players.weighted_random import WeightedRandomPlayer

POOL_DIRNAME = "pool"
POOL_GLOB = "pool_step_*.zip"


def pool_dir(checkpoint_dir) -> Path:
    return Path(checkpoint_dir) / POOL_DIRNAME


def list_pool(checkpoint_dir) -> list[Path]:
    """Pool entries, oldest first."""
    directory = pool_dir(checkpoint_dir)
    if not directory.exists():
        return []
    return sorted(directory.glob(POOL_GLOB))


def add_to_pool(source, step: int, checkpoint_dir, max_size: int = 25,
                protected=None) -> Path:
    """Copy a saved checkpoint into the pool, thinning if it has grown too big.

    Args:
        source: path to the checkpoint zip (with or without the extension).
        step: training step it was taken at; becomes the filename.
        checkpoint_dir: the run directory.
        max_size: entries to keep. Each is ~6 MB.
        protected: paths that must never be deleted (Elo anchors).

    Returns:
        Path to the new pool entry.
    """
    source = Path(source)
    if source.suffix != ".zip":
        source = source.with_suffix(".zip")

    directory = pool_dir(checkpoint_dir)
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / f"pool_step_{step:08d}.zip"
    shutil.copyfile(source, destination)

    thin_pool(checkpoint_dir, max_size, protected)
    return destination


def thin_pool(checkpoint_dir, max_size: int, protected=None) -> list[Path]:
    """Drop entries until the pool fits, keeping its coverage of history.

    Deletes the *second* oldest each time rather than the oldest. Dropping the
    oldest would turn the pool back into a sliding window over recent policies,
    which is the failure this module exists to avoid; this keeps the origin
    checkpoint and progressively coarsens the middle, so the pool stays spread
    across the whole run while getting denser toward the present.
    """
    protected = {str(Path(p)) for p in (protected or ())}
    entries = list_pool(checkpoint_dir)
    removed = []
    while len(entries) > max_size:
        victim = next(
            (e for e in entries[1:-1] if str(e) not in protected), None
        )
        if victim is None:
            break  # everything left is protected or an endpoint
        victim.unlink(missing_ok=True)
        removed.append(victim)
        entries.remove(victim)
    return removed


def make_enemy(path=None, color=Color.RED, deterministic: bool = False):
    """Build one opponent: a pool checkpoint, or the scripted bot if ``path`` is None.

    Pool opponents act *stochastically* by default. A greedy opponent plays one
    fixed line per position, so the learner sees a vanishingly narrow slice of
    the game and can overfit to beating that exact line; sampling from the frozen
    policy keeps the opponent distribution wide.
    """
    if path is None:
        return WeightedRandomPlayer(color)
    from src.agent.opponent import PolicyPlayer
    return PolicyPlayer(color, model_path=path, deterministic=deterministic)


def sample_enemies(num_envs: int, checkpoint_dir, weighted_frac: float = 0.1,
                   rng=None, deterministic: bool = False) -> list:
    """Draw one opponent per environment: a mixture of pool and scripted bot.

    The mixture is spread across environments rather than across time, so every
    PPO update is computed from a batch containing both opponent types. Swapping
    the whole vector env between opponents instead would make each update see
    one opponent only, which is a noisier gradient and lets the policy drift
    toward whatever it faced most recently.

    Keeping a slice of scripted games is deliberate: it is the only opponent
    whose difficulty never moves, so it stops the run from wandering off into a
    private equilibrium where the agent and its ghosts co-adapt to something
    that no longer resembles Catan.

    Args:
        num_envs: how many opponents to draw.
        checkpoint_dir: run directory holding the pool.
        weighted_frac: target share of envs facing WeightedRandomPlayer.
        rng: ``random.Random``, or None for the module-level RNG.
        deterministic: play pool opponents greedily.

    Returns:
        A list of ``num_envs`` Player instances.
    """
    rng = rng or random
    entries = list_pool(checkpoint_dir)
    if not entries:
        return [make_enemy(None) for _ in range(num_envs)]

    # At least one scripted env whenever the fraction is non-zero: with 8 envs a
    # 10% target rounds to 0.8, and silently dropping it would quietly remove the
    # only fixed reference from training.
    num_weighted = round(weighted_frac * num_envs)
    if weighted_frac > 0:
        num_weighted = max(1, num_weighted)
    num_weighted = min(num_weighted, num_envs)

    enemies = [make_enemy(None) for _ in range(num_weighted)]
    enemies += [
        make_enemy(rng.choice(entries), deterministic=deterministic)
        for _ in range(num_envs - num_weighted)
    ]
    rng.shuffle(enemies)
    return enemies


def describe(enemies) -> str:
    """One-line summary of a drawn opponent set, for the training log."""
    from src.agent.opponent import PolicyPlayer
    steps = [
        Path(e.model_path).stem.split("_")[-1]
        for e in enemies if isinstance(e, PolicyPlayer)
    ]
    scripted = len(enemies) - len(steps)
    return f"{scripted} weighted-random + pool steps {sorted(set(steps))}"
