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
from catanatron.players.search import VictoryPointPlayer
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


SCRIPTED_BOTS = {
    "weighted": WeightedRandomPlayer,
    "greedy": VictoryPointPlayer,
}


def make_enemy(path=None, color=Color.RED, deterministic: bool = False,
               kind: str = "weighted", simulations: int = 0):
    """Build one opponent: a pool checkpoint, or a scripted bot if ``path`` is None.

    Pool opponents act *stochastically* by default. A greedy opponent plays one
    fixed line per position, so the learner sees a vanishingly narrow slice of
    the game and can overfit to beating that exact line; sampling from the frozen
    policy keeps the opponent distribution wide.

    Args:
        path: pool checkpoint to load, or None for a scripted bot.
        color: seat colour.
        deterministic: play a pool opponent greedily.
        kind: which scripted bot, when ``path`` is None. ``weighted`` is
            catanatron's WeightedRandomPlayer; ``greedy`` is VictoryPointPlayer,
            which plays the move that most immediately raises its own VP. Those
            are the only two this build of catanatron ships -- there is no
            stronger scripted bot to reach for, which is why the hard opponent
            here has to be one of our own checkpoints.
        simulations: play a pool checkpoint with tree search at this budget
            instead of straight off the policy head. Ignored when ``path`` is
            None. See :class:`~src.agent.opponent.SearchPlayer`; it costs this
            many forward passes per opponent move.
    """
    if path is None:
        return SCRIPTED_BOTS[kind](color)
    if simulations:
        from src.agent.opponent import SearchPlayer
        return SearchPlayer(color, model_path=path, simulations=simulations)
    from src.agent.opponent import PolicyPlayer
    return PolicyPlayer(color, model_path=path, deterministic=deterministic)


def sample_enemies(num_envs: int, checkpoint_dir, weighted_frac: float = 0.1,
                   rng=None, deterministic: bool = False,
                   greedy_frac: float = 0.1, search_frac: float = 0.0,
                   search_simulations: int = 10) -> list:
    """Draw one opponent per environment: a mixture of pool and scripted bots.

    The mixture is spread across environments rather than across time, so every
    PPO update is computed from a batch containing every opponent type. Swapping
    the whole vector env between opponents instead would make each update see
    one opponent only, which is a noisier gradient and lets the policy drift
    toward whatever it faced most recently.

    Keeping a slice of scripted games is deliberate: they are the only opponents
    whose difficulty never moves, so they stop the run from wandering off into a
    private equilibrium where the agent and its ghosts co-adapt to something
    that no longer resembles Catan. The two are kept as *separate* slices
    because they fail differently -- weighted-random is broad and weak, greedy is
    narrow and pointed straight at the win condition -- and a run that beats one
    while losing to the other has told you something a merged slice would hide.

    Pool opponents are drawn *without replacement* while the pool is at least as
    big as the slots available, so a batch covers as much of the history as it
    can rather than spending envs on duplicates. Across intervals the draw is
    still uniform over the whole pool -- old checkpoints are as likely as recent
    ones, which is the fictitious-play property this module exists for.

    Args:
        num_envs: how many opponents to draw.
        checkpoint_dir: run directory holding the pool.
        weighted_frac: target share of envs facing WeightedRandomPlayer.
        rng: ``random.Random``, or None for the module-level RNG.
        deterministic: play pool opponents greedily.
        greedy_frac: target share of envs facing VictoryPointPlayer.
        search_frac: share of the *pool* slice played with tree search rather
            than off the policy head. Zero by default: it is the only opponent
            stronger than the learner's own past selves, and also the only one
            that costs forward passes on the rollout workers' own cores.
        search_simulations: search budget for that slice.

    Returns:
        A list of ``num_envs`` Player instances.
    """
    rng = rng or random
    entries = list_pool(checkpoint_dir)

    # At least one env per scripted bot whenever its fraction is non-zero: with 8
    # envs a 10% target rounds to 0.8, and silently dropping it would quietly
    # remove a fixed reference from training.
    def slice_size(frac):
        count = round(frac * num_envs)
        return max(1, count) if frac > 0 else 0

    num_weighted = slice_size(weighted_frac)
    num_greedy = slice_size(greedy_frac)
    # The pool is what gives way when the fractions over-subscribe the envs, but
    # it cannot go below zero -- trim greedy first, then weighted, so a
    # degenerate config still returns exactly ``num_envs`` players.
    num_greedy = min(num_greedy, num_envs - num_weighted) if num_envs else 0
    num_weighted = min(num_weighted, num_envs)

    enemies = [make_enemy(None, kind="weighted") for _ in range(num_weighted)]
    enemies += [make_enemy(None, kind="greedy") for _ in range(num_greedy)]

    remaining = num_envs - len(enemies)
    if entries:
        # Without replacement while the pool can cover the slots. Drawing
        # independently per env is uniform over history but wastes slots on
        # duplicates: 6 draws from an 11-entry pool average 4.8 distinct
        # opponents, and a measured interval of ``ppo-12vp-lr`` ran 6 envs
        # against only 4 different checkpoints. Every duplicate is a batch that
        # sees less of the history the pool exists to preserve. Past that point
        # the pool is smaller than the batch and duplicates are unavoidable.
        picked = []
        while len(picked) < remaining:
            take = min(remaining - len(picked), len(entries))
            picked += rng.sample(entries, take)
        # At least one searching env whenever the fraction is non-zero, for the
        # same reason the scripted slices round up: a 10% target over 6 pool
        # envs rounds to 0.6, and silently dropping it would remove the only
        # opponent in the run that is stronger than the learner's own head.
        num_search = min(max(1, round(search_frac * len(picked)))
                         if search_frac > 0 else 0, len(picked))
        enemies += [
            make_enemy(path, deterministic=deterministic,
                       simulations=search_simulations if i < num_search else 0)
            for i, path in enumerate(picked)
        ]
    else:
        # No pool yet -- the first interval of a from-scratch run, before any
        # checkpoint exists. Fill with the scripted bots rather than leaving the
        # envs unassigned, split in proportion to the two fractions so the phase
        # is not silently all weighted-random.
        total = weighted_frac + greedy_frac
        share = greedy_frac / total if total else 0.0
        extra_greedy = round(share * remaining)
        enemies += [make_enemy(None, kind="greedy") for _ in range(extra_greedy)]
        enemies += [make_enemy(None, kind="weighted")
                    for _ in range(remaining - extra_greedy)]

    rng.shuffle(enemies)
    return enemies


def describe(enemies) -> str:
    """One-line summary of a drawn opponent set, for the training log."""
    from src.agent.opponent import PolicyPlayer, SearchPlayer
    steps = [
        Path(e.model_path).stem.split("_")[-1]
        for e in enemies if isinstance(e, PolicyPlayer)
    ]
    weighted = sum(isinstance(e, WeightedRandomPlayer) for e in enemies)
    greedy = sum(isinstance(e, VictoryPointPlayer) for e in enemies)
    searching = sum(isinstance(e, SearchPlayer) for e in enemies)
    steps += [Path(e.model_path).stem.split("_")[-1]
              for e in enemies if isinstance(e, SearchPlayer)]
    line = (f"{weighted} weighted-random + {greedy} greedy + "
            f"pool steps {sorted(set(steps))}")
    if searching:
        sims = next(e.simulations for e in enemies
                    if isinstance(e, SearchPlayer))
        line += f" ({searching} of them searching at {sims} sims)"
    return line
