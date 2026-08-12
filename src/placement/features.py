"""Mechanical per-node board encoding for opening placement.

Every number in this vector is a *fact about the board*, never a judgement about
it. The distinction matters, because the whole point of this package is that
placement skill has to be learned rather than typed in:

- "this node touches a tile numbered 8, whose roll probability is 5/36" is a
  fact -- 5/36 is a property of two dice, not an opinion about Catan.
- "pips x 1.15 for a third resource" is a judgement, and lives in
  :mod:`src.placement.heuristic`, which is only ever used as a yardstick.

So the encoder hands the model raw production rates, a number histogram, port
type, what each player already owns, and how much room is left to expand -- with
no weighting between them. Resource *value* (ore+wheat vs brick+wood), the worth
of diversity, whether a 6 is better than a 5: all of that is recoverable from
these features, and the model is expected to learn it from outcomes.

Why not the 614-dim game vector? Because it is the representation that already
failed. In it a node's numbers are scattered across the board encoding, so
"a corner with an 8 and a 6" looks different on every map and the model would
have to relearn it per layout. Here it is the same handful of dimensions
everywhere, which is what makes learning from a few thousand games realistic.
"""

import functools

import numpy as np

from catanatron.models.board import STATIC_GRAPH, get_node_distances
from catanatron.models.enums import SETTLEMENT

# Fixed resource order. Used for every per-resource block below, so the model
# sees the same channel meaning in each of them.
RESOURCES = ("WOOD", "BRICK", "SHEEP", "WHEAT", "ORE")

# Dice numbers that can appear on a tile. 7 is the robber and never printed.
NUMBERS = (2, 3, 4, 5, 6, 8, 9, 10, 11, 12)

# Port channels: the five resource ports plus the generic 3:1. A node with no
# port is the all-zero vector, so "no port" needs no channel of its own.
PORT_KINDS = RESOURCES + ("ANY",)

# Expansion room is measured at these road distances from the candidate. A node
# two edges away is the closest spot the distance rule allows a second
# settlement, and three is the next ring out; together they describe whether a
# corner is a dead end or an opening onto the board.
EXPANSION_DISTANCES = (2, 3)


@functools.lru_cache(maxsize=1)
def _distances():
    """All-pairs node distances on the static board graph (cached upstream)."""
    return get_node_distances()


@functools.lru_cache(maxsize=None)
def _nodes_at_distance(node_id: int, distance: int):
    dist = _distances()[node_id]
    return tuple(other for other, d in dist.items() if d == distance)


@functools.lru_cache(maxsize=4)
def _port_lookup(catan_map):
    """node id -> port channel index, built from the map's port node sets."""
    lookup = {}
    for resource, nodes in catan_map.port_nodes.items():
        kind = "ANY" if resource is None else str(resource)
        index = PORT_KINDS.index(kind)
        for node_id in nodes:
            lookup[node_id] = index
    return lookup


def _production_vector(catan_map, node_id) -> np.ndarray:
    """Per-resource expected production rate for one node, in dice probability."""
    counter = catan_map.node_production[node_id]
    return np.array(
        [counter.get(resource, 0.0) for resource in RESOURCES], dtype=np.float32
    )


def _owned_production(game, color) -> np.ndarray:
    """Summed production of every settlement ``color`` already holds.

    Cities cannot exist during the opening, so a plain sum over settlements is
    the whole picture. For the second placement this is what lets the model work
    out complementarity on its own -- it can see the first settlement's resource
    profile alongside the candidate's, and nothing tells it which pairings are
    good.
    """
    catan_map = game.state.board.map
    total = np.zeros(len(RESOURCES), dtype=np.float32)
    for node_id, (owner, building) in game.state.board.buildings.items():
        if owner == color and building == SETTLEMENT:
            total += _production_vector(catan_map, node_id)
    return total


def _occupied_or_blocked(game) -> set:
    """Nodes that can never take a settlement again: built on, or adjacent to one."""
    blocked = set()
    for node_id in game.state.board.buildings:
        blocked.add(node_id)
        blocked.update(STATIC_GRAPH.neighbors(node_id))
    return blocked


def _expansion_features(game, node_id, blocked) -> np.ndarray:
    """How much productive room is left within reach of this corner.

    Three numbers per distance ring: how many spots are still legal, the best
    total production among them, and the mean. All three are counts and rates --
    the model decides whether a wide-but-poor corner beats a narrow-but-rich one.
    """
    catan_map = game.state.board.map
    # The candidate itself would block its own neighbours once built.
    blocked = blocked | {node_id} | set(STATIC_GRAPH.neighbors(node_id))

    # The static graph spans all 96 nodes; only the land nodes can ever be built
    # on, and only they have production entries.
    land = catan_map.land_nodes

    features = []
    for distance in EXPANSION_DISTANCES:
        totals = [
            float(sum(catan_map.node_production[other].values()))
            for other in _nodes_at_distance(node_id, distance)
            if other in land and other not in blocked
        ]
        features.extend(
            [
                float(len(totals)),
                max(totals) if totals else 0.0,
                float(np.mean(totals)) if totals else 0.0,
            ]
        )
    return np.array(features, dtype=np.float32)


def node_features(game, color, node_id: int) -> np.ndarray:
    """Encode one candidate settlement node from ``color``'s point of view.

    Args:
        game: the live ``Game``, read-only, mid initial-build phase.
        color: the player who would settle here.
        node_id: candidate node.

    Returns:
        float32 vector of length :func:`feature_size`.
    """
    catan_map = game.state.board.map
    adjacent = catan_map.adjacent_tiles[node_id]

    # --- what the corner produces -------------------------------------
    production = _production_vector(catan_map, node_id)

    tile_counts = np.zeros(len(RESOURCES), dtype=np.float32)
    number_hist = np.zeros(len(NUMBERS), dtype=np.float32)
    deserts = 0.0
    for tile in adjacent:
        if tile.resource is None:
            deserts += 1.0
            continue
        tile_counts[RESOURCES.index(str(tile.resource))] += 1.0
        number_hist[NUMBERS.index(tile.number)] += 1.0

    # Raw counts and probabilities are both present on purpose: the count says
    # how many ways a resource can arrive, the probability says how often. A
    # single 6 and a pair of 3s are close on one and far apart on the other.
    scarcity = np.array([deserts, float(len(adjacent))], dtype=np.float32)

    ports = np.zeros(len(PORT_KINDS), dtype=np.float32)
    port_index = _port_lookup(catan_map).get(node_id)
    if port_index is not None:
        ports[port_index] = 1.0

    # --- context: whose turn in the snake draft, and what is already down ---
    opponent = next(c for c in game.state.colors if c != color)
    own_owned = _owned_production(game, color)
    opp_owned = _owned_production(game, opponent)
    # 0.0 on the first settlement, 1.0 on the second. The two decisions have
    # genuinely different logic (the second answers the first), and this is what
    # lets one model serve both instead of needing two.
    placement_index = np.array(
        [1.0 if own_owned.any() else 0.0], dtype=np.float32
    )

    expansion = _expansion_features(game, node_id, _occupied_or_blocked(game))

    return np.concatenate(
        [
            production,        # 5  expected rate per resource
            tile_counts,       # 5  adjacent tiles per resource
            number_hist,       # 10 adjacent tiles per dice number
            scarcity,          # 2  desert count, total adjacent land tiles
            ports,             # 6  port type one-hot (all-zero = no port)
            placement_index,   # 1  first or second settlement
            own_owned,         # 5  production already secured
            opp_owned,         # 5  production the opponent already secured
            expansion,         # 6  room left at distance 2 and 3
        ]
    ).astype(np.float32)


# Where each block starts in the vector, in the order node_features concatenates
# them. Kept as data rather than arithmetic scattered through the code so tests
# and diagnostics can name a block instead of counting offsets by hand.
_BLOCK_WIDTHS = (
    ("production", len(RESOURCES)),
    ("tile_counts", len(RESOURCES)),
    ("number_hist", len(NUMBERS)),
    ("scarcity", 2),
    ("ports", len(PORT_KINDS)),
    ("placement_index", 1),
    ("own_production", len(RESOURCES)),
    ("opponent_production", len(RESOURCES)),
    ("expansion", 3 * len(EXPANSION_DISTANCES)),
)


def _build_offsets():
    offsets, cursor = {}, 0
    for name, width in _BLOCK_WIDTHS:
        offsets[name] = (cursor, cursor + width)
        cursor += width
    return offsets, cursor


FEATURE_SLICES, _TOTAL_WIDTH = _build_offsets()


def feature_size() -> int:
    """Length of a :func:`node_features` vector."""
    return _TOTAL_WIDTH


def candidate_nodes(playable_actions):
    """Settlement node ids among ``playable_actions``, in a stable order."""
    from catanatron.models.enums import ActionType

    return [
        action.value
        for action in playable_actions
        if action.action_type == ActionType.BUILD_SETTLEMENT
    ]


def encode_candidates(game, color, node_ids) -> np.ndarray:
    """Stack :func:`node_features` for several candidates into one (N, F) batch."""
    if not node_ids:
        return np.zeros((0, feature_size()), dtype=np.float32)
    return np.stack([node_features(game, color, n) for n in node_ids])
