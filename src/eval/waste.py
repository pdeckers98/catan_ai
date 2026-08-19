"""What does a turn actually buy? A decision-level audit of maritime trading.

A win rate cannot see this. Both sides of a mirror match burn cards at the same
rate, so burning them costs nothing *relative to the opponent* and the score
comes out 50%. The same blindness that hid the refusal to expand hides this:
you have to count the cards.

The numbers that matter are ``bought nothing`` -- trades made on a turn that
built nothing and bought nothing -- and ``cards burnt``, the cards handed to the
bank above what came back. Run it against any agent spec ``benchmark`` accepts::

    python -m src.eval.waste --vps-to-win 15 --longest-road --max-turns 1500 \
        --agent ppo --model checkpoints/archive/ppo-15vp-lr-step400000.zip \
        --games 30
"""

import argparse

# Before any engine import; see src/env/ruleset.py.
from src.env.ruleset import apply_cli_overrides

apply_cli_overrides()

from collections import Counter  # noqa: E402

from catanatron.models.enums import ActionType  # noqa: E402
from catanatron.models.player import Color  # noqa: E402

from src.agent.arena import build_agent  # noqa: E402
from src.env import ruleset  # noqa: E402
from src.env.catan_env import make_1v1_game  # noqa: E402

# Anything that turns resources into board presence or a card. A trade made on a
# turn containing none of these bought nothing.
PURCHASES = (ActionType.BUILD_ROAD, ActionType.BUILD_SETTLEMENT,
             ActionType.BUILD_CITY, ActionType.BUY_DEVELOPMENT_CARD)

RESOURCES = ("WOOD", "BRICK", "SHEEP", "WHEAT", "ORE")
COSTS = (
    {"WOOD": 1, "BRICK": 1},                                # road
    {"WOOD": 1, "BRICK": 1, "SHEEP": 1, "WHEAT": 1},        # settlement
    {"WHEAT": 2, "ORE": 3},                                 # city
    {"SHEEP": 1, "WHEAT": 1, "ORE": 1},                     # development card
)


def _hand(state, color) -> dict:
    key = f"P{state.color_to_index[color]}"
    return {r: state.player_state[f"{key}_{r}_IN_HAND"] for r in RESOURCES}


def distance_to_a_purchase(hand: dict) -> int:
    """Cards still missing for the cheapest thing this hand could aim at.

    Zero means something is already affordable. This is the only fair test of a
    maritime trade: a trade that leaves the number unchanged or larger bought
    nothing now and set nothing up for later, whatever the agent does next turn.
    """
    return min(sum(max(0, need - hand.get(r, 0)) for r, need in cost)
               for cost in (c.items() for c in COSTS))


class TradeSpy:
    """Wraps a player and scores each maritime trade it chooses.

    The hand has to be read at the moment of the decision -- the action log
    records what was traded but not what was held, and the whole question is
    whether the trade closed a gap.
    """

    def __init__(self, inner):
        self.inner = inner
        self.color = inner.color
        self.stats = Counter()

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def decide(self, game, playable_actions):
        action = self.inner.decide(game, playable_actions)
        if action.action_type == ActionType.BUY_DEVELOPMENT_CARD:
            self.stats["dev_bought"] += 1
            hand = _hand(game.state, self.color)
            # A development card costs sheep + wheat + ore; a city costs two
            # wheat and three ore. Buying one while a single card short of a
            # city spends the very cards the city was waiting on.
            short = max(0, 2 - hand["WHEAT"]) + max(0, 3 - hand["ORE"])
            if short == 1:
                self.stats["dev_one_card_short_of_a_city"] += 1
        if action.action_type == ActionType.MARITIME_TRADE:
            hand = _hand(game.state, self.color)
            before = distance_to_a_purchase(hand)
            for give in action.value[:4]:
                if give is not None:
                    hand[give] -= 1
            hand[action.value[-1]] += 1
            if before == 0:
                # Something was already affordable, so no trade can reduce the
                # distance; scoring these as waste would only measure how often
                # the agent trades while solvent. Counted, not judged.
                self.stats["while_solvent"] += 1
            else:
                self.stats["scored"] += 1
                if distance_to_a_purchase(hand) >= before:
                    self.stats["no_closer"] += 1
        return action


def audit_game(game, color) -> Counter:
    """Per-turn trade telemetry for ``color`` from a finished game's log."""
    stats = Counter()
    turn = []
    for action in game.state.actions:
        if action.color != color:
            continue
        turn.append(action)
        if action.action_type != ActionType.END_TURN:
            continue
        trades = [a for a in turn if a.action_type == ActionType.MARITIME_TRADE]
        bought = [a for a in turn if a.action_type in PURCHASES]
        stats["turns"] += 1
        stats["trades"] += len(trades)
        # value is (give, give, give, give, get) with unused gives None, so the
        # cards burnt by one trade is the give count minus the single card back.
        stats["cards_burnt"] += sum(
            sum(g is not None for g in a.value[:4]) - 1 for a in trades)
        if trades:
            stats["turns_with_trade"] += 1
            if not bought:
                stats["bought_nothing"] += len(trades)
            got = Counter(a.value[-1] for a in trades)
            gave = Counter(g for a in trades for g in a.value[:4]
                           if g is not None)
            # A resource both acquired and given away inside one turn: the trade
            # undid itself and paid the bank for the privilege.
            stats["churned"] += sum((got & gave).values())
        turn = []
    return stats


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--agent", default="ppo")
    parser.add_argument("--model", default=None)
    parser.add_argument("--games", type=int, default=30)
    parser.add_argument("--simulations", type=int, default=50)
    parser.add_argument("--seed", type=int, default=7000)
    parser.add_argument("--placement-model", default=None)
    parser.add_argument("--bundle-model", default=None)
    parser.add_argument("--vps-to-win", type=int, default=ruleset.VPS_TO_WIN)
    parser.add_argument("--longest-road", action=argparse.BooleanOptionalAction,
                        default=ruleset.LONGEST_ROAD_VP)
    parser.add_argument("--max-turns", type=int, default=ruleset.MAX_TURNS)
    args = parser.parse_args()

    factory = build_agent(args.agent, args.model, args.simulations,
                          placement_path=args.placement_model,
                          bundle_path=args.bundle_model)
    stats = Counter()
    for i in range(args.games):
        spy = TradeSpy(factory(Color.BLUE))
        game = make_1v1_game(players=[spy, factory(Color.RED)],
                             seed=args.seed + i)
        game.play()
        stats.update(audit_game(game, Color.BLUE))
        stats.update(spy.stats)

    games, turns = args.games, max(stats["turns"], 1)
    trades = max(stats["trades"], 1)
    print(f"[Rules] {ruleset.describe()}")
    print(f"{args.agent} over {games} games, {stats['turns']} own turns")
    print(f"  {stats['trades'] / games:.0f} maritime trades per game "
          f"({100 * stats['turns_with_trade'] / turns:.0f}% of turns trade)")
    print(f"  {100 * stats['bought_nothing'] / trades:.0f}% of them on a turn "
          f"that built nothing and bought nothing")
    print(f"  {100 * stats['while_solvent'] / trades:.0f}% made while something "
          f"was already affordable")
    print(f"  of the rest, {100 * stats['no_closer'] / max(stats['scored'], 1):.0f}% "
          f"left the hand no closer to any purchase than before")
    print(f"  {100 * stats['churned'] / trades:.0f}% gave away a resource the "
          f"same turn acquired")
    print(f"  {stats['cards_burnt'] / games:.0f} cards paid to the bank per game")
    dev = max(stats["dev_bought"], 1)
    print(f"  {stats['dev_bought'] / games:.1f} development cards bought per "
          f"game, {100 * stats['dev_one_card_short_of_a_city'] / dev:.0f}% of "
          f"them while one card short of a city")


if __name__ == "__main__":
    main()
