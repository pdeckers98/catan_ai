"""Pick an opening: two settlements and the two roads that come with them.

The per-corner scorer has a structural blind spot. It commits to the first
settlement before the second exists, so it cannot trade a slightly weaker corner
for a much better pairing -- and complementarity between your two openings is a
real part of the decision. Three label-pipeline changes failed to move play
strength, which is what pointed at the selection rule rather than the labels.

Pair-scoring works: **642W-558L = 53.5%** over 1200 games against greedy
selection, same agent and same corner scorer on both sides.

**Folding roads into that search does not, and the measurement is emphatic.**
An opening road is a real decision -- Longest Road is worth no VP here, so a
road buys exactly one thing, access to the corner you settle next -- and leaving
it random is plainly wrong. But a bundle model over (settlement, road) x 2
scored **44.1%** against the settlement-only bundle over 1200 games, wiping out
the entire pair-scoring gain. Diagnosis: on 32 of 40 boards the two models pick
different first settlements, and the roads model sits *closer* to the plain
corner scorer (mean rank 1.70 vs 3.02). The 24 road dimensions swamped the
complementarity signal that was doing the work, and the settlement choice
collapsed back toward greedy. Roads are still worth solving -- but with a
separate model conditioned on the chosen pair, not by widening this search.

So :class:`OpeningChooser` searches settlement pairs. It reads the corner width
off the bundle checkpoint and only plans roads if given a model trained on them
(``plans_roads``); with the settlement-only model that is measured stronger, the
road falls back to the inner agent.

Approximations, stated rather than hidden (the last two apply only in the
measured-worse road mode):

- **The corner scorer proposes, the bundle scorer disposes.** Searching all
  openings exactly would be slow, so the per-corner model shortlists first
  corners and partners. The partner list is kept much wider than the first list,
  because an individually mediocre corner that complements well is exactly the
  case this is meant to catch and a tight shortlist would discard it.
- **Road reach ignores your other settlement.** A partner's road features are
  computed once at the base board rather than per candidate first corner. The
  two settlements are non-adjacent and usually far apart, so the blocking they
  do to each other's expansion rings rarely overlaps. Data generation uses the
  same convention, so train and play agree.
- **The best partner may not survive.** Between the first seat's two picks the
  opponent takes two corners, so taking the max over partners is optimistic --
  a first corner whose case rests on one specific partner is being overrated.
  So the first seat scores a first corner by its ``partner_rank``-th best
  partner rather than its best, which is a cheap stand-in for what is likely to
  still be there. The second seat picks consecutively, nothing can be taken in
  between, and the max is exact for it. Either way the chooser still *plans* the
  best partner and re-searches when its intended corner or road is gone.

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
    corner_feature_size,
    encode_candidates,
    legal_road_edges,
    node_features,
    open_nodes,
    road_features,
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
    """Chooses one seat's opening, across one game.

    Args:
        corner_model: :class:`~src.placement.model.PlacementNet`, always used --
            alone in corner mode, as the shortlister in bundle mode.
        bundle_model: optional :class:`~src.placement.model.BundleNet`. When
            given, the opening is chosen whole: both settlements and both roads.
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
        # A bundle model trained before roads joined the opening takes bare
        # settlement vectors. Read which it is off the checkpoint rather than
        # asking the caller, so an older scorer keeps working and simply leaves
        # the roads to the inner agent as it always did.
        self.plans_roads = (
            bundle_model is not None
            and bundle_model.corner_dim == corner_feature_size()
        )
        self.reset()

    def reset(self):
        """Forget the plan. Call between games."""
        self._plan = None
        self._first_corner = None      # settlement + road features, first pick
        self._partner_nodes = {}       # node -> settlement features
        self._pending_road = None
        self._placed = 0

    # ---- settlements ----------------------------------------------------
    def choose(self, game, color, nodes):
        """Return the node id to settle, from the legal ``nodes``."""
        if not nodes:
            raise ValueError("no candidate nodes")
        if self.bundle_model is None:
            scores = self.corner_model.score(encode_candidates(game, color, nodes))
            return nodes[int(np.argmax(scores))]
        node = (self._choose_first(game, color, nodes) if self._plan is None
                else self._choose_second(game, color, nodes))
        self._placed += 1
        return node

    def choose_road(self, game, color, edges):
        """Return the edge to build, from the legal ``edges``.

        Normally this is the road the opening search already committed to. If it
        has been taken, the offered edges are re-scored against the counterpart
        corner, which keeps the decision inside the same model.
        """
        if not edges:
            raise ValueError("no candidate edges")
        edges = [tuple(sorted(e)) for e in edges]
        if not self.plans_roads or self._plan is None:
            return self._pending_road if self._pending_road in edges else edges[0]
        if self._pending_road in edges:
            return self._pending_road
        return self._rescore_road(game, color, edges)

    # ---- bundle search --------------------------------------------------
    def _shortlist(self, game, color, nodes, count):
        scores = self.corner_model.score(encode_candidates(game, color, nodes))
        order = np.argsort(scores)[::-1][:count]
        return [nodes[i] for i in order]

    def _corners(self, game, color, node, features, assume_owned=()):
        """Every (edge, corner vector) for one settlement node.

        With a settlement-only bundle model there is one "corner" and no road to
        choose, so the edge is None and the search collapses to the corner pair.
        """
        if not self.plans_roads:
            return [(None, features)]
        return [
            (tuple(edge), np.concatenate([
                features,
                road_features(game, color, node, edge, assume_owned=assume_owned),
            ]))
            for edge in legal_road_edges(node)
        ]

    @staticmethod
    def _is_first_seat(game):
        """True when this is the very first settlement of the game.

        Only the first seat has its two picks separated by the opponent's, so
        only it needs the pessimistic partner. ``buildings`` holds settlements
        and cities, not roads, so it is empty exactly at that moment.
        """
        return not game.state.board.buildings

    def _partner_value(self, scores, second, rank):
        """Score a first corner by its ``rank``-th best *distinct* partner node.

        Ranking is by node rather than by row because in road mode one partner
        contributes several rows -- one per legal road -- and three roads off the
        same corner are not three surviving partners. Falls back to the worst
        available when fewer than ``rank`` partners exist.
        """
        best_per_node = {}
        width = len(second)
        for index, score in enumerate(scores):
            node = second[index % width][0]
            if score > best_per_node.get(node, -np.inf):
                best_per_node[node] = score
        ordered = sorted(best_per_node.values(), reverse=True)
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

            own = self._corners(game, color, first,
                                node_features(game, color, first))
            partner_features = encode_candidates(
                game, color, partners, assume_owned=(first,)
            )
            second = []
            for node, features in zip(partners, partner_features):
                second.extend(
                    (node, edge, vector) for edge, vector
                    in self._corners(game, color, node, features)
                )

            left = np.stack([vector for _, vector in own])
            right = np.stack([vector for _, _, vector in second])
            # Every (first road) x (second corner, second road) combination.
            openings = np.concatenate(
                [np.repeat(left, len(right), axis=0),
                 np.tile(right, (len(left), 1))],
                axis=1,
            )
            scores = self.bundle_model.score(openings)
            # Judge the first corner pessimistically, but still plan the best
            # pairing -- _choose_second re-searches if it has been taken, so the
            # plan is a preference and costs nothing if it does not survive.
            top = int(np.argmax(scores))
            value = (float(scores[top]) if rank == 1
                     else self._partner_value(scores, second, rank))
            if value > best[0]:
                i, j = divmod(top, len(right))
                best = (value, {
                    "first": (first, own[i][0]),
                    "first_corner": own[i][1],
                    "second": (second[j][0], second[j][1]),
                    "partner_nodes": dict(zip(partners, partner_features)),
                })

        if best[1] is None:  # nothing pairable; fall back to the corner scorer
            scores = self.corner_model.score(encode_candidates(game, color, nodes))
            return nodes[int(np.argmax(scores))]

        plan = best[1]
        self._plan = plan
        self._first_corner = plan["first_corner"]
        self._partner_nodes = plan["partner_nodes"]
        self._pending_road = plan["first"][1]
        return plan["first"][0]

    def _choose_second(self, game, color, nodes):
        """Play the corner the plan committed to, or re-search if it is gone."""
        planned, planned_road = self._plan["second"]
        if planned in nodes:
            self._pending_road = planned_road
            return planned

        candidates = []
        for node in nodes:
            features = self._partner_nodes.get(node)
            if features is None:
                features = node_features(game, color, node)
            candidates.extend(
                (node, edge, vector)
                for edge, vector in self._corners(game, color, node, features)
            )
        right = np.stack([vector for _, _, vector in candidates])
        openings = np.concatenate(
            [np.repeat(self._first_corner[None, :], len(right), axis=0), right],
            axis=1,
        )
        top = int(np.argmax(self.bundle_model.score(openings)))
        node, edge, _ = candidates[top]
        self._pending_road = edge
        return node

    def _rescore_road(self, game, color, edges):
        """Choose among the roads actually on offer, holding the rest fixed."""
        node, _ = self._plan["first" if self._placed == 1 else "second"]
        features = (self._partner_nodes.get(node) if self._placed == 2
                    else node_features(game, color, node))
        if features is None:
            features = node_features(game, color, node)

        corners = np.stack([
            np.concatenate([features, road_features(game, color, node, edge)])
            for edge in edges
        ])
        counterpart = (self._first_corner if self._placed == 2
                       else self._plan_second_corner(game, color))
        pairs = np.concatenate(
            [corners, np.repeat(counterpart[None, :], len(edges), axis=0)]
            if self._placed == 1 else
            [np.repeat(counterpart[None, :], len(edges), axis=0), corners],
            axis=1,
        )
        return edges[int(np.argmax(self.bundle_model.score(pairs)))]

    def _plan_second_corner(self, game, color):
        node, edge = self._plan["second"]
        features = self._partner_nodes.get(node)
        if features is None:
            features = node_features(game, color, node)
        return np.concatenate([
            features, road_features(game, color, node, edge),
        ])
