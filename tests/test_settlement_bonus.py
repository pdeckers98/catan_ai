"""Tests for the flat settlement bonus.

Unlike :mod:`tests.test_shaping`, the property under test is *not* that the
added return is policy-independent -- this wrapper exists precisely to move the
optimum. What has to hold instead is that it pays for the thing it names and
nothing else: settlements, at most three, never a city, never twice for a
settlement rebuilt on a piece a city handed back.
"""

from catanatron import Color

from src.env.catan_env import SettlementBonusWrapper

BONUS = 0.03


class FakeState:
    def __init__(self, settlements_available):
        self.colors = (Color.BLUE, Color.RED)
        self.color_to_index = {Color.BLUE: 0, Color.RED: 1}
        self.player_state = {"P0_SETTLEMENTS_AVAILABLE": settlements_available}


class FakeGame:
    def __init__(self, state):
        self.state = state


class FakeEnv:
    """A scripted env: each step yields the next (available, done) pair."""

    def __init__(self, script, start=3):
        self.script = list(script)
        self.start = start
        self.unwrapped = self
        self.game = FakeGame(FakeState(start))

    def reset(self, **kwargs):
        self.game = FakeGame(FakeState(self.start))
        return None, {}

    def step(self, action):
        available, terminated = self.script.pop(0)
        self.game = FakeGame(FakeState(available))
        return None, 0.0, terminated, False, {}


def _run(script, start=3, bonus=BONUS, max_bonuses=3):
    env = SettlementBonusWrapper(FakeEnv(script, start), bonus, max_bonuses)
    env.reset()
    rewards = []
    while True:
        _, reward, terminated, truncated, _ = env.step(0)
        rewards.append(reward)
        if terminated or truncated:
            return rewards


def test_a_settlement_pays_the_bonus_on_the_step_it_is_built():
    # Starts at 3 available (the opening two are already placed), builds one.
    assert _run([(3, False), (2, False), (2, True)]) == [0.0, BONUS, 0.0]


def test_the_bonus_is_capped_at_three():
    script = [(2, False), (1, False), (0, False), (0, True)]
    assert _run(script, max_bonuses=2) == [BONUS, BONUS, 0.0, 0.0]


def test_an_upgrade_returning_the_piece_pays_nothing():
    # A city hands the settlement piece back: available RISES. That is not a
    # build and must not be paid for.
    assert _run([(2, False), (3, False), (3, True)]) == [BONUS, 0.0, 0.0]


def test_a_settlement_rebuilt_after_an_upgrade_cannot_exceed_the_cap():
    # build, upgrade (piece back), rebuild, upgrade, rebuild: five pool moves
    # but only three payable settlements.
    script = [(2, False), (3, False), (2, False), (3, False), (2, False),
              (3, False), (2, True)]
    paid = sum(_run(script)) / BONUS
    assert paid == 3


def test_two_settlements_in_one_step_pay_twice():
    assert _run([(1, False), (1, True)]) == [2 * BONUS, 0.0]


def test_zero_bonus_leaves_the_reward_untouched():
    assert _run([(2, False), (1, False), (1, True)], bonus=0.0) == [0.0] * 3


def test_the_total_is_reported_at_the_end():
    env = SettlementBonusWrapper(FakeEnv([(2, False), (2, True)]), BONUS)
    env.reset()
    env.step(0)
    _, _, _, _, info = env.step(0)
    assert info["settlement_bonus_paid"] == BONUS
