"""Pin ``PYTHONHASHSEED`` so a seed reproduces a game, not just a board.

``make_1v1_game(seed=n)`` reseeds the global ``random`` module, which fixes the
board. It does not fix the *game*: Catanatron builds ``playable_actions`` by
iterating sets whose members are enum values, and ``enum.Enum.__hash__`` hashes
the member's *name string*. String hashing is randomised per process, so the
action order shifts between runs and any bot choosing off that list walks a
different game. Measured before this module existed -- the same seed, five
processes, 400 ticks::

    turn 143 / 157 / 157 / 153 / 158     <- five different games
    PYTHONHASHSEED=0:  160 / 160 / 160   <- identical

**The interpreter reads this variable before it starts**, so assigning
``os.environ["PYTHONHASHSEED"]`` from running code does nothing at all -- the
hash table is already seeded. The only fix is to hand the value to a fresh
interpreter, which is why :func:`ensure_hash_seed` relaunches.

Relaunching uses :mod:`subprocess` rather than ``os.execv``. On Windows
``execv`` does not replace the process as it does on POSIX; it spawns a
replacement and kills the caller, so the shell sees the command finish while
work continues in the background. That breaks any script that sequences runs,
which is most of how this project is driven.

Cost is one extra interpreter startup. Set ``PYTHONHASHSEED`` yourself -- to
``random`` to opt out, or to any other value to pin a different one -- and this
does nothing.
"""

import os
import subprocess
import sys

DEFAULT_HASH_SEED = "0"


def ensure_hash_seed(value: str = DEFAULT_HASH_SEED) -> None:
    """Relaunch under a pinned ``PYTHONHASHSEED`` unless one is already set.

    Call first thing in an entry point's ``main()``. Returns normally when the
    seed is already pinned -- including in the relaunched child, which is what
    terminates the recursion. Otherwise it does not return: the child runs to
    completion and its exit status is propagated.

    Any pre-existing value is honoured, so ``PYTHONHASHSEED=random`` opts out.
    """
    if os.environ.get("PYTHONHASHSEED"):
        return

    env = dict(os.environ, PYTHONHASHSEED=value)
    # orig_argv, not argv: it keeps the interpreter's own flags and the "-m
    # package.module" form, which plain argv has already stripped.
    completed = subprocess.run([sys.executable, *sys.orig_argv[1:]], env=env)
    sys.exit(completed.returncode)
