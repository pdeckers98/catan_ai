"""Bolt a learned placement scorer onto any existing agent.

The wrapper answers exactly one kind of decision -- which node to settle during
the initial build phase -- and forwards everything else untouched. That narrow
seam is what makes the comparison clean: the same agent with and without the
scorer differs only in its opening, so a benchmark between them measures the
opening and nothing else.

Initial *roads* are left to the inner agent on purpose. The scorer ranks corners,
and inventing a road policy here would smuggle in an extra untested change.
"""

from catanatron.models.enums import ActionType
from catanatron.models.player import Player

from src.placement.chooser import PARTNER_RANK, OpeningChooser
from src.placement.model import BundleNet, PlacementNet


class PlacementPlayer(Player):
    """``inner`` plays the game; the scorer picks the opening settlements.

    Args:
        color: seat.
        inner: the Player handling every non-opening decision.
        model: a :class:`~src.placement.model.PlacementNet`.
        bundle_model: optional :class:`~src.placement.model.BundleNet`; when
            given, the opening is chosen as a pair of corners rather than one
            corner at a time.
        partner_rank: forwarded to the chooser; see
            :data:`~src.placement.chooser.PARTNER_RANK`.
    """

    def __init__(self, color, inner, model, bundle_model=None,
                 partner_rank=PARTNER_RANK):
        super().__init__(color)
        self.inner = inner
        self.model = model
        # Per-seat: the chooser remembers this player's first pick.
        self.chooser = OpeningChooser(model, bundle_model,
                                      partner_rank=partner_rank)

    def decide(self, game, playable_actions):
        if game.state.is_initial_build_phase:
            settlements = [
                a for a in playable_actions
                if a.action_type == ActionType.BUILD_SETTLEMENT
            ]
            if settlements:
                nodes = [a.value for a in settlements]
                chosen = self.chooser.choose(game, self.color, nodes)
                return next(a for a in settlements if a.value == chosen)

            roads = [
                a for a in playable_actions
                if a.action_type == ActionType.BUILD_ROAD
            ]
            # Only when the chooser is actually planning openings. Without a
            # bundle model it has no opinion about roads, and the inner agent
            # keeps them -- which is what every earlier measurement did.
            if roads and self.chooser.plans_roads:
                edge = self.chooser.choose_road(
                    game, self.color, [a.value for a in roads]
                )
                return next(
                    a for a in roads if tuple(sorted(a.value)) == tuple(sorted(edge))
                )
        return self.inner.decide(game, playable_actions)

    def reset_state(self):
        self.chooser.reset()
        self.inner.reset_state()


def wrap_factory(inner_factory, model_path, bundle_path=None,
                 partner_rank=PARTNER_RANK):
    """Wrap a ``callable(Color) -> Player`` so its openings come from the scorer.

    Loads the checkpoints once and shares them across seats; the models are
    stateless at inference. The per-seat search state lives in the chooser, so
    each :class:`PlacementPlayer` builds its own.
    """
    model = PlacementNet.load(model_path)
    bundle = BundleNet.load(bundle_path) if bundle_path else None
    return lambda color: PlacementPlayer(
        color, inner_factory(color), model, bundle, partner_rank=partner_rank
    )
