"""Self-relative Elo: a yardstick that does not saturate.

Every evaluation in this project used to be a win rate against a fixed scripted
bot, which stops carrying information once the agent beats it. ``ppo-8vp-long``
finished at ~92% vs weighted-random with ~1.4% standard error, so the entire
remaining range is four standard errors wide -- at that point the metric mostly
measures which hundred games got drawn, and "did another million steps help?"
becomes unanswerable.

Elo against the agent's *own* past checkpoints has no ceiling: as the agent
improves, so does the reference, and the number keeps climbing. The cost is that
it is purely relative -- a rising ladder rating says the agent beats its former
self, which is not the same as playing Catan well (a policy can cycle: A beats
B beats C beats A). Keep scoring against weighted-random alongside it as an
absolute sanity check; the pair is informative in a way neither is alone.

Ratings are anchored at 0 for the first checkpoint on the ladder, so a rating is
read as "Elo above where this run started".
"""

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

# A match is a finite sample, so a clean sweep does not mean infinite skill.
# ``elo_delta`` clamps the score away from 0 and 1 by half a game, and this
# caps the result regardless -- 800 is already far past anything meaningful.
MAX_DELTA = 800.0


def elo_delta(score: float, games: int) -> float:
    """Rating difference implied by scoring ``score`` over ``games``.

    Inverts the logistic expectation ``E = 1 / (1 + 10 ** (-d / 400))``.

    Args:
        score: challenger score in [0, 1], draws counting half.
        games: games played, used to bound a 0% or 100% result.

    Returns:
        Elo points to add to the opponent's rating. Positive means stronger.
    """
    if games <= 0:
        raise ValueError("games must be positive")
    margin = 0.5 / games
    clamped = min(max(score, margin), 1.0 - margin)
    delta = -400.0 * math.log10(1.0 / clamped - 1.0)
    return max(-MAX_DELTA, min(MAX_DELTA, delta))


@dataclass
class Anchor:
    """A frozen checkpoint with a rating, used as an Elo reference point."""

    path: str
    elo: float
    step: int


class Ladder:
    """An ordered set of rated anchors, persisted as JSON.

    New anchors are added only when the challenger clearly beats the current
    top one. Promoting on a coin-flip result would let measurement noise ratchet
    the reference upward, inflating every later rating -- the ladder would climb
    even for a policy that never improved.
    """

    def __init__(self, path, anchors=None):
        self.path = Path(path)
        self.anchors: list[Anchor] = list(anchors or [])

    @classmethod
    def load(cls, path) -> "Ladder":
        path = Path(path)
        if not path.exists():
            return cls(path)
        data = json.loads(path.read_text())
        return cls(path, [Anchor(**entry) for entry in data["anchors"]])

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps({"anchors": [asdict(a) for a in self.anchors]}, indent=2)
        )

    def __len__(self) -> int:
        return len(self.anchors)

    def top(self) -> Anchor:
        """The strongest anchor -- the one a challenger is measured against."""
        if not self.anchors:
            raise ValueError("ladder is empty; seed it first")
        return max(self.anchors, key=lambda a: a.elo)

    def seed(self, path, step: int) -> Anchor:
        """Install the origin anchor at rating 0, if the ladder is empty."""
        if not self.anchors:
            self.anchors.append(Anchor(path=str(path), elo=0.0, step=step))
            self.save()
        return self.top()

    def add(self, path, elo: float, step: int) -> Anchor:
        anchor = Anchor(path=str(path), elo=float(elo), step=step)
        self.anchors.append(anchor)
        self.save()
        return anchor

    def paths(self) -> set[str]:
        """Files the ladder depends on; pool thinning must not delete these."""
        return {a.path for a in self.anchors}
