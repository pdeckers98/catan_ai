"""Matplotlib renderer for a Catanatron board/game state.

Catanatron's pip package ships no visualizer (the official web UI is a
separate Flask+React app run via Docker, not on PyPI). This draws the hex
board, settlements/cities, roads, robber, dice, resource hands, and VPs
directly from ``catanatron.game.Game``, so we can watch a game live without
any extra infra.

Node/edge IDs have no inherent pixel position in catanatron -- only tile cube
coordinates do. We derive node/tile pixel positions ourselves via standard
pointy-top hex math; shared nodes between neighboring tiles resolve to the
same pixel position by construction, since cube coordinates are consistent.
"""

import math

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, RegularPolygon, Rectangle

from catanatron.models.enums import RESOURCES, ActionType
from catanatron.models.map import NodeRef
from catanatron.models.player import Color

HEX_SIZE = 1.0

RESOURCE_COLORS = {
    "WOOD": "#2d6a4f",
    "BRICK": "#bc6c25",
    "SHEEP": "#a7c957",
    "WHEAT": "#e9c46a",
    "ORE": "#6c757d",
    None: "#f1e6c8",  # desert
}

PLAYER_COLORS = {
    Color.RED: "#d62828",
    Color.BLUE: "#1d3557",
    Color.ORANGE: "#f4a261",
    Color.WHITE: "#dddddd",
}

# Corner angle (degrees, 0 = +x axis, CCW) for a pointy-top hex, matching
# catanatron's NodeRef naming.
NODE_ANGLES = {
    NodeRef.NORTH: 90,
    NodeRef.NORTHEAST: 30,
    NodeRef.SOUTHEAST: -30,
    NodeRef.SOUTH: -90,
    NodeRef.SOUTHWEST: -150,
    NodeRef.NORTHWEST: 150,
}


def _tile_center(coordinate):
    x, _y, z = coordinate
    px = HEX_SIZE * math.sqrt(3) * (x + z / 2)
    # catanatron's +z (cube coord) points toward NORTH-ish tiles, which in
    # screen space (y grows upward) means -z must map to +py.
    py = -HEX_SIZE * 1.5 * z
    return px, py


def _node_positions(board):
    """node_id -> (x, y), derived from every tile that references it."""
    positions = {}
    for coordinate, tile in board.map.land_tiles.items():
        cx, cy = _tile_center(coordinate)
        for node_ref, node_id in tile.nodes.items():
            angle = math.radians(NODE_ANGLES[node_ref])
            positions[node_id] = (
                cx + HEX_SIZE * math.cos(angle),
                cy + HEX_SIZE * math.sin(angle),
            )
    return positions


def _last_dice_roll(state):
    for action in reversed(state.actions):
        if action.action_type == ActionType.ROLL and action.value is not None:
            return action.value
    return None


def _player_hand_lines(state, color):
    key = f"P{state.color_to_index[color]}"
    parts = [
        f"{r[:4]}:{state.player_state[f'{key}_{r}_IN_HAND']}" for r in RESOURCES
    ]
    dev_cards = sum(
        v for k, v in state.player_state.items()
        if k.startswith(f"{key}_") and k.endswith("_IN_HAND") and k.split("_")[1] not in RESOURCES
    )
    vps = state.player_state[f"{key}_ACTUAL_VICTORY_POINTS"]
    return f"{color.value}: VP={vps}  DevCards={dev_cards}  " + " ".join(parts)


def render_board(game, ax=None, title=None, label_nodes=False, show_info=True):
    """Draw the current board state of a ``catanatron.game.Game``.

    Args:
        game: a ``catanatron.game.Game`` instance.
        ax: optional matplotlib Axes to draw into (cleared first). Creates a
            new figure/axes if omitted.
        title: optional title override (defaults to turn/player info).
        label_nodes: if True, draw each node's id on the board (so a textual
            action like ``BUILD_ROAD edge (12,13)`` can be located visually).
        show_info: if True, draw the built-in dice/hands info panel. Set False
            when the caller renders its own (e.g. a god-mode panel).

    Returns:
        The matplotlib Axes drawn into.
    """
    state = game.state
    board = state.board

    if ax is None:
        _, ax = plt.subplots(figsize=(9, 9))
    ax.clear()
    ax.set_aspect("equal")
    ax.axis("off")

    node_pos = _node_positions(board)

    centers = [_tile_center(c) for c in board.map.land_tiles]
    xs = [c[0] for c in centers]
    ys = [c[1] for c in centers]
    margin = HEX_SIZE * 1.5
    ax.set_xlim(min(xs) - margin, max(xs) + margin)
    ax.set_ylim(min(ys) - margin, max(ys) + margin * 2.5)

    for coordinate, tile in board.map.land_tiles.items():
        cx, cy = _tile_center(coordinate)
        hexagon = RegularPolygon(
            (cx, cy),
            numVertices=6,
            radius=HEX_SIZE,
            orientation=0,
            facecolor=RESOURCE_COLORS[tile.resource],
            edgecolor="black",
            linewidth=1,
        )
        ax.add_patch(hexagon)
        if tile.number is not None:
            ax.text(
                cx, cy, str(tile.number),
                ha="center", va="center",
                fontsize=12, fontweight="bold",
                bbox=dict(boxstyle="circle", facecolor="white", alpha=0.8),
            )
        if coordinate == board.robber_coordinate:
            ax.add_patch(Circle((cx, cy), HEX_SIZE * 0.3, facecolor="black", zorder=5))

    drawn_edges = set()
    for edge, color in board.roads.items():
        key = tuple(sorted(edge))
        if key in drawn_edges:
            continue
        drawn_edges.add(key)
        (x1, y1), (x2, y2) = node_pos[edge[0]], node_pos[edge[1]]
        ax.plot([x1, x2], [y1, y2], color=PLAYER_COLORS[color], linewidth=5, zorder=4)

    for node_id, (color, building_type) in board.buildings.items():
        x, y = node_pos[node_id]
        if building_type == "CITY":
            half = HEX_SIZE * 0.22
            ax.add_patch(
                Rectangle(
                    (x - half, y - half), 2 * half, 2 * half,
                    facecolor=PLAYER_COLORS[color],
                    edgecolor="black",
                    linewidth=2,
                    zorder=6,
                )
            )
        else:  # SETTLEMENT
            ax.add_patch(
                Circle(
                    (x, y), HEX_SIZE * 0.16,
                    facecolor=PLAYER_COLORS[color],
                    edgecolor="black",
                    linewidth=1.5,
                    zorder=6,
                )
            )

    if label_nodes:
        for node_id, (x, y) in node_pos.items():
            ax.text(
                x, y, str(node_id),
                ha="center", va="center", fontsize=6, color="black",
                zorder=7,
                bbox=dict(boxstyle="round,pad=0.05", facecolor="white",
                          edgecolor="none", alpha=0.6),
            )

    if show_info:
        dice = _last_dice_roll(state)
        dice_text = f"Dice: {dice[0]} + {dice[1]} = {sum(dice)}" if dice else "Dice: --"
        hand_lines = [_player_hand_lines(state, c) for c in state.colors]
        info_text = dice_text + "\n" + "\n".join(hand_lines)
        ax.text(
            0.0, 1.0, info_text,
            transform=ax.transAxes, ha="left", va="top", fontsize=9,
            family="monospace",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
        )

    if title is None:
        current_color = state.colors[state.current_turn_index]
        title = f"Turn {state.num_turns} -- current: {current_color.value}"
    ax.set_title(title)

    return ax
