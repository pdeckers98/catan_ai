"""Per-run rule settings that must survive a process boundary.

Three settings vary per run and are read by code that executes in *other
processes*: ``SubprocVecEnv`` rollout workers, and the worker pool
``src.agent.arena`` fans matches across. On Windows those are spawned, not
forked, so every module is re-imported from scratch in the child and any value
passed as a function argument at the top of ``main()`` is simply not there.

That rules out the obvious design. ``src.env.rules`` compounds it: the Longest
Road patch is installed at *import* time, before any argument could reach it.

So the ruleset lives in the environment, which children inherit for free -- the
same mechanism ``src.env.determinism`` already uses to pin ``PYTHONHASHSEED``:

- ``CATAN_VPS_TO_WIN``   -- victory points to win (default 8)
- ``CATAN_MAX_TURNS``    -- turn cap before a game is called a draw (default 1000)
- ``CATAN_LONGEST_ROAD`` -- 1 to award Longest Road its +2 VP (default 0, off)

The cost of that convenience is a global: a variable left over from a previous
command silently applies to the next one. Two guards. :func:`describe` is logged
at the top of every run and recorded in the W&B config, so the ruleset a number
was measured under is always on the record. And :func:`apply_cli_overrides` must
run before ``src.env.catan_env`` or ``src.env.rules`` is imported, which is why
entry points call it as their first statement -- import it late and you get the
default ruleset with no warning.
"""

import os

_DEFAULTS = {
    "CATAN_VPS_TO_WIN": "8",
    "CATAN_MAX_TURNS": "1000",
    "CATAN_LONGEST_ROAD": "0",
}

_TRUE = {"1", "true", "yes", "on"}


def _flag(name: str) -> bool:
    return os.environ.get(name, _DEFAULTS[name]).strip().lower() in _TRUE


def _int(name: str) -> int:
    return int(os.environ.get(name, _DEFAULTS[name]))


def _read_env() -> None:
    """(Re)read the ruleset from the environment into this module's globals."""
    global VPS_TO_WIN, MAX_TURNS, LONGEST_ROAD_VP
    VPS_TO_WIN = _int("CATAN_VPS_TO_WIN")
    MAX_TURNS = _int("CATAN_MAX_TURNS")
    LONGEST_ROAD_VP = _flag("CATAN_LONGEST_ROAD")


# Read at import. Everything downstream imports these names by value, so a change
# after *those* imports has no effect -- which is the point: the ruleset is fixed
# for the lifetime of the process, and every process in a run agrees on it.
VPS_TO_WIN = MAX_TURNS = 0
LONGEST_ROAD_VP = False
_read_env()


def apply_cli_overrides(argv=None) -> None:
    """Translate ruleset flags on the command line into environment variables.

    Call this *before* importing anything that touches the engine. It is a
    deliberate pre-pass over ``sys.argv`` rather than part of the entry point's
    ``argparse`` setup, because argparse necessarily runs inside ``main()`` --
    long after ``src.env.rules`` has already installed (or not installed) the
    Longest Road patch at import time.

    Unrecognised arguments are ignored; the entry point's own parser still
    declares these flags, so ``--help`` and validation behave normally and the
    values reach the W&B config through the usual path.

    Args:
        argv: argument list to scan. Defaults to ``sys.argv[1:]``.
    """
    if argv is None:
        import sys
        argv = sys.argv[1:]

    values = {"--vps-to-win": "CATAN_VPS_TO_WIN", "--max-turns": "CATAN_MAX_TURNS"}
    for index, arg in enumerate(argv):
        name, _, inline = arg.partition("=")
        if name in values:
            value = inline if inline else (argv[index + 1] if index + 1 < len(argv) else None)
            if value is not None:
                os.environ[values[name]] = value
        elif name == "--longest-road":
            os.environ["CATAN_LONGEST_ROAD"] = "1"
        elif name == "--no-longest-road":
            os.environ["CATAN_LONGEST_ROAD"] = "0"

    # Setting the variables is not enough. This module's constants were read at
    # *its* import -- which is the line that imported this function, one
    # statement earlier -- so without re-reading them, the overrides would reach
    # spawned children (which re-import from a fresh interpreter) but never the
    # parent, and the parent is what installs the Longest Road patch.
    _read_env()


def describe() -> str:
    """One-line summary of the active ruleset, for logs."""
    road = "on" if LONGEST_ROAD_VP else "off"
    return (f"{VPS_TO_WIN} VP, longest-road {road}, "
            f"turn cap {MAX_TURNS}")


def as_config() -> dict:
    """The active ruleset as a dict, for W&B config / run metadata."""
    return {
        "vps_to_win": VPS_TO_WIN,
        "max_turns": MAX_TURNS,
        "longest_road_vp": LONGEST_ROAD_VP,
    }
