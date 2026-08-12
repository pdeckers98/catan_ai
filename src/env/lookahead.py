"""Two-roll dice lookahead, as observation features.

The agent's persistent failure is preferring *doing something* over ending the
turn: 9.3 development cards a game against ~0.8 net new buildings. The payoff of
a kept hand only exists on the far side of a dice roll, and nothing in the 614-dim
observation states it -- the hand and the board are both in there, but the policy
would have to learn two-dice arithmetic from win/loss alone, and the placement
diagnosis showed it never learns dice numbers even when the stakes are highest.

So compute it and hand it over. For the next two rolls -- the opponent's, then
yours, both of which produce for both players -- this reports what you expect to
gain and, more importantly, **what you become able to afford**.

**Probabilities, not expectations, are the payload.** "Expected 0.5 wheat" could
mean a near-certain half wheat or 3 wheat one time in six; those are different
decisions. And the question at END_TURN is a threshold -- "will waiting let me
afford a settlement?" -- which a mean cannot answer. Hence the eight
``P(afford ...)`` features.

**Nothing here uses hidden information.** The opponent's hand composition is not
in the observation (only `P1_NUM_RESOURCES_IN_HAND`, a count, exactly as in a real
game) and is not used here. What is used from their side is public: their
production, derived from their buildings on the board, and their exact card count,
which tells you whether a 7 would force them to discard. This constraint is not
cosmetic -- Phase 3 puts this agent on colonist.io with only the public view, and
a policy trained on hidden state degrades the moment it gets there.

**Approximations, all deliberate:**

- The opponent acts between the two rolls (builds, trades, moves the robber on a
  7). None of that is modelled, so the projection is optimistic about what they
  do with their turn. Modelling it means simulating a policy, which is the cost
  this design exists to avoid.
- A discard is assumed to remove cards *proportionally* across resources. The
  real choice is the policy's, and it will not be proportional.
- Robber placement is read as it stands now; a 7 in the window would move it.
"""

import numpy as np
from gymnasium import Wrapper, spaces

from catanatron.models.decks import (
    CITY_COST_FREQDECK,
    DEVELOPMENT_CARD_COST_FREQDECK,
    ROAD_COST_FREQDECK,
    SETTLEMENT_COST_FREQDECK,
)
from catanatron.models.enums import CITY

from src.env.rules import DISCARD_LIMIT

# Freqdeck order, which is what the cost vectors above are written in.
RESOURCES = ("WOOD", "BRICK", "SHEEP", "WHEAT", "ORE")

# Every dice sum, 7 included. No tile carries a 7, so its production row stays
# zero on its own -- the 7 matters here for discards, not for resources.
DICE_SUMS = (2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12)
SEVEN = DICE_SUMS.index(7)
DICE_PROBS = np.array(
    [(6 - abs(7 - n)) / 36.0 for n in DICE_SUMS], dtype=np.float64
)

# Order fixed here because it is also the feature order.
COSTS = np.array(
    [
        ROAD_COST_FREQDECK,
        SETTLEMENT_COST_FREQDECK,
        CITY_COST_FREQDECK,
        DEVELOPMENT_CARD_COST_FREQDECK,
    ],
    dtype=np.float64,
)

LOOKAHEAD_SIZE = 28


def _production_by_sum(game, color) -> np.ndarray:
    """(11, 5) resource gain for ``color`` at each dice sum, robber applied."""
    board = game.state.board
    gains = np.zeros((len(DICE_SUMS), len(RESOURCES)), dtype=np.float64)

    # Tiles carry no coordinate, so the robber has to be resolved to a tile id
    # through the map's coordinate index before it can be compared.
    robbed = board.map.land_tiles.get(board.robber_coordinate)
    robbed_id = robbed.id if robbed is not None else None

    for node_id, (owner, building) in board.buildings.items():
        if owner != color:
            continue
        amount = 2.0 if building == CITY else 1.0
        for tile in board.map.adjacent_tiles[node_id]:
            if tile.resource is None or tile.id == robbed_id:
                continue
            gains[DICE_SUMS.index(tile.number),
                  RESOURCES.index(str(tile.resource))] += amount
    return gains


def _hand(state, color) -> np.ndarray:
    key = f"P{state.color_to_index[color]}"
    return np.array(
        [float(state.player_state[f"{key}_{r}_IN_HAND"]) for r in RESOURCES]
    )


def _after_discard(hands: np.ndarray) -> np.ndarray:
    """House discard rule over a stack of hands, removing cards proportionally.

    Vectorized over every leading axis: this runs on the full (11, 11) roll grid,
    not one hand at a time.
    """
    totals = hands.sum(axis=-1, keepdims=True)
    kept = totals - totals // 2
    # Under the limit nothing is discarded; the guard also covers an empty hand,
    # where the ratio would be 0/0.
    scale = np.where(totals > DISCARD_LIMIT, kept / np.maximum(totals, 1.0), 1.0)
    return hands * scale


def _affordable(hands: np.ndarray) -> np.ndarray:
    """Can each hand pay each cost? (..., 5) -> (..., 4) as float."""
    return (hands[..., None, :] >= COSTS).all(axis=-1).astype(np.float64)


def lookahead_features(game, color) -> np.ndarray:
    """Two-roll projection for ``color``. Returns float32 of ``LOOKAHEAD_SIZE``.

    Layout::

        0-4    E[gain] per resource, next roll
        5-9    Var[gain] per resource, next roll
        10-13  P(afford road / settlement / city / dev) after 1 roll
        14-17  P(afford ...) after 2 rolls
        18     P(you must discard next roll)
        19     E[cards you lose to discard next roll]
        20-21  E[hand size] after 1 roll, after 2 rolls
        22-26  opponent E[gain] per resource, next roll
        27     P(opponent must discard next roll)
    """
    state = game.state
    opponent = next(c for c in state.colors if c != color)

    gains = _production_by_sum(game, color)
    opp_gains = _production_by_sum(game, opponent)
    hand = _hand(state, color)

    # --- distribution of a single roll's production -----------------------
    mean_gain = DICE_PROBS @ gains
    var_gain = DICE_PROBS @ (gains ** 2) - mean_gain ** 2
    opp_mean_gain = DICE_PROBS @ opp_gains

    # --- one roll ahead ---------------------------------------------------
    # (11, 5): the hand after each possible sum. The 7 row produces nothing and
    # instead takes the discard.
    hands_1 = hand + gains
    hands_1[SEVEN] = _after_discard(hand[None, :])[0]

    afford_1 = DICE_PROBS @ _affordable(hands_1)
    hand_size_1 = DICE_PROBS @ hands_1.sum(axis=-1)

    # --- two rolls ahead (opponent rolls, then you) -----------------------
    # (11, 11, 5) over both sums at once. Done as an explicit 121-iteration loop
    # this cost ~1 ms per call -- the same order as an entire PPO step, which
    # would have roughly doubled training time for a feature block.
    hands_2 = hands_1[:, None, :] + gains[None, :, :]
    hands_2[:, SEVEN, :] = _after_discard(hands_1)

    joint = DICE_PROBS[:, None] * DICE_PROBS[None, :]
    afford_2 = np.tensordot(joint, _affordable(hands_2), axes=([0, 1], [0, 1]))
    hand_size_2 = float((joint * hands_2.sum(axis=-1)).sum())

    # --- discard exposure, both sides -------------------------------------
    # P(7) is a constant and would be a dead feature; what varies is whether the
    # hand is over the limit when it lands, which is why these are products.
    p_seven = DICE_PROBS[SEVEN]
    total = hand.sum()
    at_risk = float(total > DISCARD_LIMIT)
    my_discard_p = p_seven * at_risk
    my_discard_loss = my_discard_p * (total // 2)

    opp_key = f"P{state.color_to_index[opponent]}"
    opp_cards = sum(
        state.player_state[f"{opp_key}_{r}_IN_HAND"] for r in RESOURCES
    )
    opp_discard_p = p_seven * float(opp_cards > DISCARD_LIMIT)

    return np.concatenate([
        mean_gain,
        np.maximum(var_gain, 0.0),   # guard float error on zero-variance rows
        afford_1,
        afford_2,
        [my_discard_p, my_discard_loss, hand_size_1, hand_size_2],
        opp_mean_gain,
        [opp_discard_p],
    ]).astype(np.float32)


class LookaheadWrapper(Wrapper):
    """Append :func:`lookahead_features` to the env's observation vector.

    Only for the gym-based (PPO) path. Search and self-play run off a raw
    ``Game`` and would call :func:`lookahead_features` directly; the AlphaZero
    encoder in ``src/agent/encoding.py`` is deliberately left at 614 so the two
    tracks' checkpoints stay distinguishable by observation size.
    """

    def __init__(self, env):
        super().__init__(env)
        base = env.observation_space
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(base.shape[0] + LOOKAHEAD_SIZE,),
            dtype=np.float32,
        )

    def _extend(self, obs):
        game = self.env.unwrapped.game
        color = self.env.unwrapped.p0.color
        extra = lookahead_features(game, color)
        return np.concatenate([np.asarray(obs, dtype=np.float32), extra])

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self._extend(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._extend(obs), reward, terminated, truncated, info
