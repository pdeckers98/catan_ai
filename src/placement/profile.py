"""What resources does a placement model's opening actually produce?

The diagnostic that found the brick blind spot. Run it against any pair of
placement checkpoints; it says nothing about strength, only about what the model
reaches for, which is the thing a win rate cannot show you.

    python -m src.placement.profile <corner.pt> <bundle.pt> [boards]
"""
import sys
from collections import Counter

from catanatron.models.enums import ActionType

from src.env.catan_env import make_1v1_game
from src.placement.chooser import OpeningChooser
from src.placement.model import BundleNet, PlacementNet

PIPS = {2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 8: 5, 9: 4, 10: 3, 11: 2, 12: 1}
RES = ("WOOD", "BRICK", "SHEEP", "WHEAT", "ORE")


def production(board, color):
    prod = Counter()
    for node, (owner, _) in board.buildings.items():
        if owner != color:
            continue
        for tile in board.map.land_tiles.values():
            if node in tile.nodes.values() and tile.resource and tile.number:
                prod[tile.resource] += PIPS[tile.number]
    return prod


def main():
    corner_path, bundle_path = sys.argv[1], sys.argv[2]
    boards = int(sys.argv[3]) if len(sys.argv) > 3 else 40
    corner = PlacementNet.load(corner_path)
    bundle = BundleNet.load(bundle_path)

    missing, totals = Counter(), Counter()
    for seed in range(boards):
        game = make_1v1_game(seed=seed)
        state = game.state
        seat = state.colors[0]
        chooser = OpeningChooser(corner, bundle)
        while state.is_initial_build_phase:
            actions = state.playable_actions
            if (state.current_color() == seat
                    and actions[0].action_type == ActionType.BUILD_SETTLEMENT):
                node = chooser.choose(game, seat, [a.value for a in actions])
                action = next(a for a in actions if a.value == node)
            else:
                action = actions[0]
            game.execute(action, validate_action=False)
        prod = production(state.board, seat)
        for resource in RES:
            totals[resource] += prod.get(resource, 0)
            missing[resource] += prod.get(resource, 0) == 0

    print(f"{corner_path} + {bundle_path}, {boards} boards, first seat")
    print("  mean pips:  " + "  ".join(
        f"{r} {totals[r]/boards:5.2f}" for r in RES))
    print("  zero of it: " + "  ".join(
        f"{r} {100*missing[r]/boards:3.0f}%" for r in RES))


if __name__ == "__main__":
    main()
