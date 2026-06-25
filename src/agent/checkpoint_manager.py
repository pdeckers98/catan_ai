"""Rotating checkpoint pool for self-play: keep N recent models, delete older ones."""

from pathlib import Path

CHECKPOINT_DIR = Path("checkpoints")


def checkpoint_path(step: int) -> Path:
    """Return the checkpoint path for a given step count."""
    return CHECKPOINT_DIR / f"agent_step_{step:08d}"


def save_checkpoint(model, step: int) -> Path:
    """Save model to checkpoint; return path.

    Args:
        model: SB3 model with a .save(path) method.
        step: current step count.

    Returns:
        Path to the saved checkpoint.
    """
    CHECKPOINT_DIR.mkdir(exist_ok=True)
    path = checkpoint_path(step)
    model.save(str(path))
    return path


def list_checkpoints() -> list[Path]:
    """Return list of saved checkpoints, sorted by step (oldest first)."""
    if not CHECKPOINT_DIR.exists():
        return []
    checkpoints = sorted(CHECKPOINT_DIR.glob("agent_step_*"))
    return checkpoints


def prune_checkpoints(keep_n: int = 3) -> None:
    """Delete all but the N most recent checkpoints.

    Args:
        keep_n: number of most recent checkpoints to preserve.
    """
    checkpoints = list_checkpoints()
    if len(checkpoints) <= keep_n:
        return
    to_delete = checkpoints[:-keep_n]
    for path in to_delete:
        import shutil
        if path.is_dir():
            shutil.rmtree(path)
