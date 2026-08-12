"""Rotating checkpoint pool for self-play: keep N recent models, delete older ones."""

from pathlib import Path


def checkpoint_path(step: int, checkpoint_dir: Path) -> Path:
    return checkpoint_dir / f"agent_step_{step:08d}"


def save_checkpoint(model, step: int, checkpoint_dir: Path) -> Path:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    path = checkpoint_path(step, checkpoint_dir)
    model.save(str(path))
    return path


def list_checkpoints(checkpoint_dir: Path) -> list[Path]:
    if not checkpoint_dir.exists():
        return []
    return sorted(checkpoint_dir.glob("agent_step_*"))


def prune_checkpoints(checkpoint_dir: Path, keep_n: int = 3) -> None:
    checkpoints = list_checkpoints(checkpoint_dir)
    if len(checkpoints) <= keep_n:
        return
    import shutil
    for path in checkpoints[:-keep_n]:
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink(missing_ok=True)
