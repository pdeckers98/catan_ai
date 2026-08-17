"""Assemble the agent the bridge deploys.

The deployed agent is **three artifacts, not one**: the PPO checkpoint, 50-sim
PUCT search on top of it, and both placement models. Dropping the search gives
up ~10 points; dropping the placement models leaves the opening to a policy head
that received zero gradient on placement and is measurably *worse* than the older
agent that at least learned it badly (100.0% -> 88.2% vs weighted-random). See
the Caveats in ``docs/PHASE2_AI.md``.

Every other entry point in the project makes those three optional flags, which is
right for benchmarking -- the whole point there is to measure each piece. Live
play has no such excuse, so this builder requires all three and refuses to
produce a half-equipped agent.
"""

from src.agent.arena import build_agent
from src.placement.chooser import PARTNER_RANK

#: Search saturates here: 50 sims is worth +9.8 points in a mirror match and 100
#: only reaches 60.8%, because the ceiling is critic quality, not budget.
DEFAULT_SIMULATIONS = 50


def build_bridge_player(color, model_path, placement_path, bundle_path,
                        simulations: int = DEFAULT_SIMULATIONS,
                        partner_rank: int = PARTNER_RANK):
    """Build the full deployed agent for one seat.

    Args:
        color: the seat the agent occupies in the live game.
        model_path: PPO checkpoint. Must have been trained under the ruleset the
            live lobby uses.
        placement_path: PlacementNet checkpoint (shortlists opening corners).
        bundle_path: BundleNet checkpoint (ranks whole corner pairs).
        simulations: playouts per decision.
        partner_rank: pair-search pessimism; see
            :data:`~src.placement.chooser.PARTNER_RANK`.

    Returns:
        A catanatron ``Player``. Ask it for a move with
        ``player.decide(replay.game, replay.playable_actions)``.

    Raises:
        ValueError: if any of the three artifacts is missing.
    """
    missing = [
        name for name, path in (
            ("model_path", model_path),
            ("placement_path", placement_path),
            ("bundle_path", bundle_path),
        ) if not path
    ]
    if missing:
        raise ValueError(
            f"the deployed agent is three artifacts; missing {', '.join(missing)}. "
            "A checkpoint without search and both placement models is not the "
            "agent that was measured."
        )

    factory = build_agent(
        "ppo-mcts",
        model_path=str(model_path),
        simulations=simulations,
        placement_path=str(placement_path),
        bundle_path=str(bundle_path),
        partner_rank=partner_rank,
    )
    return factory(color)
