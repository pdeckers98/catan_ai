"""Waste telemetry: action-log mining and its aggregation through a match."""

from catanatron import Color
from catanatron.models.actions import Action
from catanatron.models.enums import ActionType

from src.agent.arena import AgentSpec, _action_log_stats, play_match


def _act(color, action_type, value=None):
    return Action(color, action_type, value)


def test_action_log_stats_counts_dev_buys_and_trailing_roads():
    me, them = Color.BLUE, Color.RED
    actions = [
        # Initial placement: two settlements + two roads each, excluded.
        _act(me, ActionType.BUILD_SETTLEMENT, 1),
        _act(me, ActionType.BUILD_ROAD, (0, 1)),
        _act(them, ActionType.BUILD_SETTLEMENT, 40),
        _act(them, ActionType.BUILD_ROAD, (40, 41)),
        _act(me, ActionType.BUILD_SETTLEMENT, 10),
        _act(me, ActionType.BUILD_ROAD, (10, 11)),
        # Mid-game: two roads, then a settlement -- those roads led somewhere.
        _act(me, ActionType.BUILD_ROAD, (11, 12)),
        _act(me, ActionType.BUILD_ROAD, (12, 13)),
        _act(me, ActionType.BUILD_SETTLEMENT, 13),
        _act(me, ActionType.BUY_DEVELOPMENT_CARD),
        # Opponent activity must not count toward the challenger.
        _act(them, ActionType.BUY_DEVELOPMENT_CARD),
        _act(them, ActionType.BUILD_ROAD, (41, 42)),
        # Roads after the last building: trailing.
        _act(me, ActionType.BUILD_ROAD, (13, 14)),
        _act(me, ActionType.BUY_DEVELOPMENT_CARD),
        _act(me, ActionType.BUILD_ROAD, (14, 15)),
    ]
    stats = _action_log_stats(actions, me)
    assert stats["dev_bought"] == 2
    assert stats["trailing_roads"] == 2


def test_action_log_stats_city_resets_trailing_count():
    me = Color.BLUE
    actions = [
        _act(me, ActionType.BUILD_SETTLEMENT, 1),
        _act(me, ActionType.BUILD_ROAD, (0, 1)),
        _act(me, ActionType.BUILD_SETTLEMENT, 10),
        _act(me, ActionType.BUILD_ROAD, (10, 11)),
        _act(me, ActionType.BUILD_ROAD, (11, 12)),
        _act(me, ActionType.BUILD_CITY, 1),
    ]
    assert _action_log_stats(actions, me)["trailing_roads"] == 0


def test_action_log_stats_all_postplacement_roads_trail_without_buildings():
    me = Color.BLUE
    actions = [
        _act(me, ActionType.BUILD_SETTLEMENT, 1),
        _act(me, ActionType.BUILD_ROAD, (0, 1)),
        _act(me, ActionType.BUILD_SETTLEMENT, 10),
        _act(me, ActionType.BUILD_ROAD, (10, 11)),
        _act(me, ActionType.BUILD_ROAD, (11, 12)),
        _act(me, ActionType.BUILD_ROAD, (12, 13)),
    ]
    assert _action_log_stats(actions, me)["trailing_roads"] == 2


def test_play_match_reports_waste_telemetry():
    result = play_match(
        AgentSpec(kind="random"), AgentSpec(kind="random"), num_games=4, seed=7
    )
    for field in ("mean_dev_bought", "mean_dev_unplayed", "mean_end_hand",
                  "mean_final_hand", "mean_trailing_roads",
                  "loss_final_hand", "loss_dev_unplayed", "loss_trailing_roads"):
        value = getattr(result, field)
        assert value >= 0.0
    # Random players end plenty of turns; the sampled hand mean must be real.
    assert result.mean_end_hand > 0.0
    assert "waste:" in result.summary()
