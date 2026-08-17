"""Pick an opening: the two settlements, chosen as a pair.

The per-corner scorer has a structural blind spot. It commits to the first
settlement before the second exists, so it cannot trade a slightly weaker corner
for a much better pairing -- and complementarity between your two openings is a
real part of the decision. Three label-pipeline changes failed to move play
strength, which is what pointed at the selection rule rather than the labels.

Pair-scoring works: **642W-558L = 53.5%** over 1200 games against greedy
selection, same agent and same corner scorer on both sides.

**Roads are deliberately not modelled here.** An opening road is a real decision
-- Longest Road is worth no VP by default, so a road buys exactly one thing,
access to the corner you settle next -- but folding it into this search was
measured and lost badly: a bundle model over (settlement, road) x 2 scored
**44.1%** against the settlement-only bundle over 1200 games, wiping out the
entire pair-scoring gain. Diagnosis: on 32 of 40 boards the two models picked
different first settlements, and the roads model sat *closer* to the plain corner
scorer (mean rank 1.70 vs 3.02) -- the 24 road dimensions swamped the
complementarity signal that was doing the work. That path has been removed;
roads go to the inner agent. Solving them properly needs a separate model
conditioned on the chosen pair, not a wider version of this search.

Approximations, stated rather than hidden:

- **The corner scorer proposes, the bundle scorer disposes.** Searching all
  openings exactly would be slow, so the per-corner model shortlists first
  corners and partners. The partner list is kept much wider than the first list,
  because an individually mediocre corner that complements well is exactly the
  case this is meant to catch and a tight shortlist would discard it.
- **The best partner may not survive.** Between the first seat's two picks the
  opponent takes two corners, so taking the max over partners is optimistic --
  a first corner whose case rests on one specific partner is being overrated.
  So the first seat scores a first corner by its ``partner_rank``-th best
  partner rather than its best, which is a cheap stand-in for what is likely to
  still be there. The second seat picks consecutively, nothing can be taken in
  between, and the max is exact for it. Either way the chooser still *plans* the
  best partner and re-searches when its intended corner is gone.

  Measured: rank 3 beats rank 1 **1252W-1148L = 52.2%** over 2400 games, CI
  [50.2, 54.2], p ~ 0.034. It changes the first settlement on 40.5% of boards.
  Because only the first seat is affected, the first-seat-conditional rate is
  about 54.4%. The choice of rank barely matters past 2: rank 2 scored 50.7%
  and rank 5 scored 52.3%, both over 1200 games -- so some pessimism is what
  pays, not its precise amount, and 3 is kept because two corners is what the
  opponent actually takes.
"""

import numpy as np

from catanatron.models.board import STATIC_GRAPH

from src.placement.features import (
    encode_candidates,
    node_features,
    open_nodes,
)

# How many first corners to search, and how many partners to consider for each.
FIRST_K = 12
PARTNER_K = 30

# Which partner stands in for "the second settlement I will actually get" when
# scoring a first corner, for the first seat only. The opponent takes two
# corners in between, so the top two partners are the ones most at risk and the
# third is the first that has a fair chance of surviving.
PARTNER_RANK = 3


class OpeningChooser:
    """Chooses one seat's opening settlements, across one game.

    Args:
        corner_model: :class:`~src.placement.model.PlacementNet`, always used --
            alone in corner mode, as the shortlister in bundle mode.
        bundle_model: optional :class:`~src.placement.model.BundleNet`. When
            given, the two settlements are chosen as a pair.
        first_k / partner_k: search width; see the module docstring.
        partner_rank: how pessimistic to be about the second settlement
            surviving the opponent's two picks. 1 restores the plain max.
    """

    def __init__(self, corner_model, bundle_model=None, first_k=FIRST_K,
                 partner_k=PARTNER_K, partner_rank=PARTNER_RANK):
        self.corner_model = corner_model
        self.bundle_model = bundle_model
        self.first_k = first_k
        self.partner_k = partner_k
        self.partner_rank = max(1, int(partner_rank))
        self.reset()

    def reset(self):
        """Forget the plan. Call between games."""
        self._plan = None
        self._first_corner = None      # settlement features of the first pick
        self._partner_nodes = {}       # node -> settlement features

    # ---- settlements ----------------------------------------------------
    def choose(self, game, color, nodes):
        """Return the node id to settle, from the legal ``nodes``."""
        if not nodes:
            raise ValueError("no candidate nodes")
        if self.bundle_model is None:
            scores = self.corner_model.score(encode_candidates(game, color, nodes))
            return nodes[int(np.argmax(scores))]
        return (self._choose_first(game, color, nodes) if self._plan is None
                else self._choose_second(game, color, nodes))

    # ---- bundle search --------------------------------------------------
    def _shortlist(self, game, color, nodes, count):
        scores = self.corner_model.score(encode_candidates(game, color, nodes))
        order = np.argsort(scores)[::-1][:count]
        return [nodes[i] for i in order]

    @staticmethod
    def _is_first_seat(game):
        """True when this is the very first settlement of the game.

        Only the first seat has its two picks separated by the opponent's, so
        only it needs the pessimistic partner. ``buildings`` holds settlements
        and cities, not roads, so it is empty exactly at that moment.
        """
        return not game.state.board.buildings

    @staticmethod
    def _partner_value(scores, rank):
        """Score a first corner by its ``rank``-th best partner.

        Falls back to the worst available when fewer than ``rank`` partners
        exist.
        """
        ordered = sorted(scores, reverse=True)
        return float(ordered[min(rank, len(ordered)) - 1])

    def _choose_first(self, game, color, nodes):
        firsts = self._shortlist(game, color, nodes, self.first_k)
        pool = self._shortlist(game, color, open_nodes(game), self.partner_k)
        rank = self.partner_rank if self._is_first_seat(game) else 1

        best = (-np.inf, None)
        for first in firsts:
            excluded = {first} | set(STATIC_GRAPH.neighbors(first))
            partners = [m for m in pool if m not in excluded]
            if not partners:
                continue

            own = node_features(game, color, first)
            partner_features = encode_candidates(
                game, color, partners, assume_owned=(first,)
            )
            # Every (first corner, second corner) pairing.
            openings = np.concatenate(
                [np.repeat(own[None, :], len(partners), axis=0), partner_features],
                axis=1,
            )
            scores = self.bundle_model.score(openings)
            # Judge the first corner pessimistically, but still plan the best
            # pairing -- _choose_second re-searches if it has been taken, so the
            # plan is a preference and costs nothing if it does not survive.
            top = int(np.argmax(scores))
            value = (float(scores[top]) if rank == 1
                     else self._partner_value(scores, rank))
            if value > best[0]:
                best = (value, {
                    "first": first,
                    "first_corner": own,
                    "second": partners[top],
                    "partner_nodes": dict(zip(partners, partner_features)),
                })

        if best[1] is None:  # nothing pairable; fall back to the corner scorer
            scores = self.corner_model.score(encode_candidates(game, color, nodes))
            return nodes[int(np.argmax(scores))]

        plan = best[1]
        self._plan = plan
        self._first_corner = plan["first_corner"]
        self._partner_nodes = plan["partner_nodes"]
        return plan["first"]

    def _choose_second(self, game, color, nodes):
        """Play the corner the plan committed to, or re-search if it is gone."""
        planned = self._plan["second"]
        if planned in nodes:
            return planned

        right = np.stack([
            self._partner_nodes.get(node)
            if self._partner_nodes.get(node) is not None
            else node_features(game, color, node)
            for node in nodes
        ])
        openings = np.concatenate(
            [np.repeat(self._first_corner[None, :], len(right), axis=0), right],
            axis=1,
        )
        return nodes[int(np.argmax(self.bundle_model.score(openings)))]
