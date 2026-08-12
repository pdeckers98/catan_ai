"""A dedicated dice stream, for common random numbers across paired games.

Catanatron rolls with ``random.randint`` off the global RNG, which means two
games that share a seed only stay in sync while they consume randomness
identically. As soon as the players act differently -- which is the entire point
of a duplicate-board pair -- the streams drift and the dice stop being a
controlled variable.

:func:`fixed_dice` gives the dice their own generator. Roll *k* is then the same
in both games of a pair regardless of what the bots did in between, which is the
common-random-numbers trick from simulation: hold the noise fixed and the
difference between two runs is attributable to the thing you changed.

Turns alternate, so turn *k* belongs to the same seat in both games and the
alignment is meaningful. Other engine randomness -- dev-card draws, robber steals
-- still comes off the global RNG and still diverges; dice are the dominant term
and the only one worth the patch.
"""

import contextlib
import random

import catanatron.state as _state


@contextlib.contextmanager
def fixed_dice(seed: int):
    """Draw every roll inside the block from ``random.Random(seed)``.

    Restores the original roller on exit, including on exception. Not
    thread-safe (it rebinds a module global), but each data-generation worker is
    its own process, so that is not a constraint here.
    """
    rng = random.Random(seed)
    original = _state.roll_dice

    def roll():
        return (rng.randint(1, 6), rng.randint(1, 6))

    _state.roll_dice = roll
    try:
        yield
    finally:
        _state.roll_dice = original
