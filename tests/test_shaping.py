"""Tests for potential-based reward shaping.

The whole case for putting shaping back into this project is that this form
telescopes: over an episode the added return is a constant that does not depend
on the policy, so unlike the deleted VP-milestone bonuses it cannot move the
optimum. That is a property, not a comment, so it is tested here.
"""

import pytest

from catanatron import Color

from src.env.catan_env import PotentialShapingWrapper


class FakeState:
    def __init__(self, vps):
        self.colors = tuple(vps)
        self.color_to_index = {c: i for i, c in enumerate(vps)}
        self.player_state = {}
        for i, (color, vp) in enumerate(vps.items()):
            self.player_state[f"P{i}_ACTUAL_VICTORY_POINTS"] = vp


class FakeGame:
    def __init__(self, state):
        self.state = state


class FakeEnv:
    """A scripted env: each step yields the next (vp, reward, done) triple."""

    def __init__(self, script, start=(2, 2)):
        self.script = list(script)
        self.start = start
        self.unwrapped = self
        self.game = FakeGame(self._state(*start))

    def _state(self, mine, theirs):
        return FakeState({Color.BLUE: mine, Color.RED: theirs})

    def reset(self, **kwargs):
        self.game = FakeGame(self._state(*self.start))
        return None, {}

    def step(self, action):
        mine, theirs, reward, terminated, truncated = self.script.pop(0)
        self.game = FakeGame(self._state(mine, theirs))
        return None, reward, terminated, truncated, {}


@pytest.fixture(autouse=True)
def visible_vp_is_actual_vp(monkeypatch):
    """The wrapper reads the opponent's *visible* VP through catanatron.

    The fake state has no buildings, so the real function would return 0 for
    everyone and the test would be measuring nothing. Point it at the field the
    fake actually carries.
    """
    import catanatron.state_functions as sf

    def visible(state, color):
        return state.player_state[
            f"P{state.color_to_index[color]}_ACTUAL_VICTORY_POINTS"]

    monkeypatch.setattr(sf, "get_visible_victory_points", visible)


def _run(script, weight=0.05, gamma=1.0, start=(2, 2)):
    env = PotentialShapingWrapper(FakeEnv(script, start), weight, gamma)
    env.reset()
    rewards = []
    while True:
        _, reward, terminated, truncated, _ = env.step(0)
        rewards.append(reward)
        if terminated or truncated:
            return rewards


def test_a_city_pays_immediately():
    """The point of the whole exercise: building has a local advantage that
    survives the ~600 decisions between it and the terminal reward."""
    rewards = _run([(3, 2, 0.0, False, False), (3, 2, 1.0, True, False)])
    assert rewards[0] == pytest.approx(0.05)


def test_a_wasted_turn_pays_nothing():
    """A maritime trade that burns three cards moves no victory point, so it
    earns exactly what it earned before. Shaping does not punish the trade; it
    pays for the alternative."""
    rewards = _run([(2, 2, 0.0, False, False), (2, 2, 1.0, True, False)])
    assert rewards[0] == pytest.approx(0.0)


def test_the_opponent_scoring_costs_us():
    rewards = _run([(2, 4, 0.0, False, False), (2, 4, -1.0, True, False)])
    assert rewards[0] == pytest.approx(-0.10)


@pytest.mark.parametrize("script", [
    # A win reached by building.
    [(3, 2, 0.0, False, False), (5, 2, 0.0, False, False),
     (8, 3, 1.0, True, False)],
    # The same win reached after wandering.
    [(2, 2, 0.0, False, False), (2, 3, 0.0, False, False),
     (8, 3, 1.0, True, False)],
    # A loss.
    [(4, 2, 0.0, False, False), (4, 6, 0.0, False, False),
     (4, 8, -1.0, True, False)],
])
def test_the_total_added_return_is_the_same_constant(script):
    """Ng/Harada/Russell: the shaping sums to ``-Phi(s_0)`` whatever happens in
    between. This is the property that makes it policy-invariant, and the
    property the deleted milestone bonuses did not have.
    """
    shaped = sum(_run(script, gamma=1.0))
    base = sum(step[2] for step in script)
    # s_0 is 2 VP each, so Phi(s_0) is zero and the shaping adds nothing at all.
    assert shaped == pytest.approx(base)


def test_an_uneven_start_shifts_every_episode_by_the_same_constant():
    script = [(5, 2, 0.0, False, False), (8, 2, 1.0, True, False)]
    shaped = sum(_run(script, gamma=1.0, start=(4, 2)))
    # Phi(s_0) = 0.05 * (4 - 2); the shaping pays exactly minus that.
    assert shaped == pytest.approx(1.0 - 0.10)


def test_truncation_is_absorbing_too():
    """A turn-limited game pays 0 and teaches nothing. If Phi were left standing
    there, a policy could bank shaping for a lead it never converted and no
    terminal reward would ever balance the books -- the one way this wrapper
    could stop being policy-invariant."""
    script = [(9, 2, 0.0, False, False), (9, 2, 0.0, False, True)]
    assert sum(_run(script, gamma=1.0)) == pytest.approx(0.0)


def test_zero_weight_leaves_the_reward_untouched():
    script = [(3, 2, 0.0, False, False), (8, 2, 1.0, True, False)]
    assert _run(script, weight=0.0) == [0.0, 1.0]
