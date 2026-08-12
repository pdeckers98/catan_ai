"""The hand-written placement scorer -- a yardstick, never a policy.

This is the textbook beginner rule: total dice-roll production across adjacent
tiles, nudged up for resource variety. It encodes a human opinion about Catan,
so **nothing in the training pipeline may use it**. It is not consulted when
generating data (openings are explored uniformly at random) and it is not
consulted at play time.

It exists for exactly one job: measuring. Ranking the learned model's choices
against this scorer on held-out boards is how we know whether the model has
picked up dice numbers at all -- the diagnostic that found the original policy
settling on 3-tile corners made of 3s, 4s and 12s. A learned scorer that only
matches this one has learned the beginner rule; one that beats it head-to-head
has learned something we did not know to type in.
"""

from collections import Counter

from catanatron.models.map import number_probability

# Weight added per extra distinct resource. Deliberately mild -- this scorer is
# meant to be a competent baseline, not a tuned opponent.
DIVERSITY_BONUS = 0.15


def pip_score(catan_map, node_id: int) -> float:
    """Production probability across adjacent tiles, scaled by resource variety."""
    production = Counter()
    for tile in catan_map.adjacent_tiles[node_id]:
        if tile.resource is not None:
            production[tile.resource] += number_probability(tile.number)
    variety = 1.0 + DIVERSITY_BONUS * max(0, len(production) - 1)
    return sum(production.values()) * variety


def rank_of(catan_map, node_id: int, candidates) -> int:
    """1-based rank of ``node_id`` among ``candidates`` by :func:`pip_score`.

    Rank 1 is the best corner on the board. This is the number to quote when
    asking "how good is the placement this agent actually chose?".
    """
    ordered = sorted(candidates, key=lambda n: -pip_score(catan_map, n))
    return ordered.index(node_id) + 1


def spearman(xs, ys) -> float:
    """Rank correlation, ties averaged. Returns 0.0 when either side is constant.

    Hand-rolled because scipy is not a dependency of this project and one
    correlation does not justify adding it.
    """
    def ranks(values):
        order = sorted(range(len(values)), key=lambda i: values[i])
        out = [0.0] * len(values)
        start = 0
        while start < len(order):
            stop = start
            while (stop + 1 < len(order)
                   and values[order[stop + 1]] == values[order[start]]):
                stop += 1
            average = (start + stop) / 2.0
            for k in range(start, stop + 1):
                out[order[k]] = average
            start = stop + 1
        return out

    rx, ry = ranks(xs), ranks(ys)
    n = len(xs)
    if n < 2:
        return 0.0
    mx, my = sum(rx) / n, sum(ry) / n
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    vx = sum((a - mx) ** 2 for a in rx)
    vy = sum((b - my) ** 2 for b in ry)
    return cov / ((vx * vy) ** 0.5) if vx and vy else 0.0
