"""Rebuild an exact Catanatron board from an observed layout.

The bridge cannot use ``build_map("BASE")``: that shuffles. To play a live game
we need a ``CatanMap`` whose tiles carry the resources and numbers colonist.io
actually dealt, while keeping catanatron's own node and edge numbering -- the
numbering every downstream module (features, action space, placement scorer) is
written against.

``initialize_tiles`` already takes the three shuffled lists as parameters, which
is the seam we want: hand it explicit lists and the topology walk assigns node,
edge, tile and port ids exactly as it would for a random board. The lists are
consumed with ``pop()`` (from the end), so they are reversed on the way in --
:func:`spec_from_map` is the inverse and the round-trip is what the tests pin.

A :class:`BoardSpec` is deliberately *not* a translation of colonist's board
message; it is the target that translation has to hit. Keeping it a plain,
validated value object means a mistranslated board fails here, loudly, instead
of surfacing later as an agent that inexplicably plays badly.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

from catanatron.models.map import (
    BASE_MAP_TEMPLATE, CatanMap, initialize_tiles,
)

#: (resource, number) for one land tile. Resource ``None`` is the desert, which
#: is the only tile allowed a ``None`` number.
TileSpec = Tuple[Optional[str], Optional[int]]


def _template_counts():
    """Expected multisets for a BASE board: tiles, numbers, ports."""
    return (
        sorted(BASE_MAP_TEMPLATE.tile_resources, key=str),
        sorted(BASE_MAP_TEMPLATE.numbers),
        sorted(BASE_MAP_TEMPLATE.port_resources, key=str),
    )


@dataclass(frozen=True)
class BoardSpec:
    """A fully-determined BASE board.

    Args:
        tiles: ``(resource, number)`` per land tile, in **tile-id order** --
            which is the order ``BASE_MAP_TEMPLATE.topology`` walks its
            coordinates, not any ordering colonist.io uses.
        ports: port resource (``None`` for a 3:1 port) per port, in port-id
            order, same walk.
    """

    tiles: Tuple[TileSpec, ...]
    ports: Tuple[Optional[str], ...]

    def __post_init__(self):
        self.validate()

    def validate(self) -> None:
        """Fail on anything that is not a legal BASE board.

        The multiset checks are the point: a translator that drops a tile, or
        reads numbers in the wrong order, produces a board that is still
        *playable* and therefore silently wrong. Nineteen tiles of the right
        composition in the wrong places would still pass -- that failure is
        caught by the observation round-trip in the verification ladder, not
        here.
        """
        want_tiles, want_numbers, want_ports = _template_counts()

        # Before the multiset checks: a numbered desert would otherwise be
        # reported as the wrong count of numbers, which points at the wrong bug.
        for resource, number in self.tiles:
            if (resource is None) != (number is None):
                raise ValueError(
                    "exactly the desert carries no number; got "
                    f"resource={resource!r} number={number!r}"
                )

        resources = [r for r, _ in self.tiles]
        numbers = [n for _, n in self.tiles if n is not None]
        if sorted(resources, key=str) != want_tiles:
            raise ValueError(
                f"tile resources are not a BASE board: got {sorted(resources, key=str)}"
            )
        if sorted(numbers) != want_numbers:
            raise ValueError(f"tile numbers are not a BASE board: got {sorted(numbers)}")
        if sorted(self.ports, key=str) != want_ports:
            raise ValueError(
                f"ports are not a BASE board: got {sorted(self.ports, key=str)}"
            )


def build_map_from_spec(spec: BoardSpec) -> CatanMap:
    """Build the ``CatanMap`` this spec describes."""
    resources = [resource for resource, _ in spec.tiles]
    numbers = [number for _, number in spec.tiles if number is not None]

    # Reversed because initialize_tiles pops from the end; copies because it
    # pops from the lists we hand it.
    tiles = initialize_tiles(
        BASE_MAP_TEMPLATE,
        shuffled_numbers_param=list(reversed(numbers)),
        shuffled_port_resources_param=list(reversed(spec.ports)),
        shuffled_tile_resources_param=list(reversed(resources)),
    )
    return CatanMap.from_tiles(tiles)


def spec_from_map(catan_map: CatanMap) -> BoardSpec:
    """Read a spec back off a built map. Inverse of :func:`build_map_from_spec`."""
    tiles = tuple(
        (catan_map.tiles_by_id[i].resource, catan_map.tiles_by_id[i].number)
        for i in sorted(catan_map.tiles_by_id)
    )
    ports = tuple(
        catan_map.ports_by_id[i].resource for i in sorted(catan_map.ports_by_id)
    )
    return BoardSpec(tiles=tiles, ports=ports)
