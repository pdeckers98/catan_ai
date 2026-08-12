"""Bolt a learned placement scorer onto any existing agent.

The wrapper answers exactly one kind of decision -- which node to settle during
the initial build phase -- and forwards everything else untouched. That narrow
seam is what makes the comparison clean: the same agent with and without the
scorer differs only in its opening, so a benchmark between them measures the
opening and nothing else.

Initial *roads* are left to the inner agent on purpose. The scorer ranks corners,
and inventing a road policy here would smuggle in an extra untested change.
"""

import numpy as np

from catanatron.models.enums import ActionType
from catanatron.models.player import Player

from src.placement.features import encode_candidates
from src.placement.model import PlacementNet


class PlacementPlayer(Player):
    """``inner`` plays the game; ``model`` picks the opening settlements.

    Args:
        color: seat.
        inner: the Player handling every non-opening decision.
        model: a :class:`~src.placement.model.PlacementNet`.
    """

    def __init__(self, color, inner, model):
        super().__init__(color)
        self.inner = inner
        self.model = model

    def decide(self, game, playable_actions):
        if game.state.is_initial_build_phase:
            settlements = [
                a for a in playable_actions
                if a.action_type == ActionType.BUILD_SETTLEMENT
            ]
            if settlements:
                nodes = [a.value for a in settlements]
                scores = self.model.score(
                    encode_candidates(game, self.color, nodes)
                )
                return settlements[int(np.argmax(scores))]
        return self.inner.decide(game, playable_actions)

    def reset_state(self):
        self.inner.reset_state()


def wrap_factory(inner_factory, model_path):
    """Wrap a ``callable(Color) -> Player`` so its openings come from the scorer.

    Loads the checkpoint once and shares it across seats; the model is stateless
    at inference, so there is nothing to keep separate.
    """
    model = PlacementNet.load(model_path)
    return lambda color: PlacementPlayer(color, inner_factory(color), model)
