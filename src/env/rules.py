"""Custom 1v1 rule tweaks applied as runtime monkeypatches on Catanatron 3.2.1.

We keep these as monkeypatches (not edits to the installed package) so the change
lives in version control and is reapplied automatically in every process -- including
the fresh interpreters that ``SubprocVecEnv`` spawns for parallel training.

Three things happen here:

1. **Discard threshold raised to 9.** Stock Catanatron makes you discard on a 7
   when you hold *more than 7* cards (``discard_limit=7``). The gym env builds
   ``Game`` without passing it, so we wrap ``Game.__init__`` to inject our value.

2. **Policy-controlled, one-card-at-a-time discard.** Stock Catanatron exposes a
   single ``(DISCARD, None)`` action and then discards a *random* half of the hand
   (see the TODO in ``actions.discard_possibilities``). We replace that with one
   discard action *per resource*: on a 7 the player chooses a single card to drop,
   repeated until ``floor(hand/2)`` cards are gone. This lets the RL policy (and the
   human) decide exactly which cards to discard. It enlarges the gym action space by
   4 (one DISCARD slot -> five), so a model must be (re)trained for the new space.

3. **Correct multi-discarder sequencing.** Because we fully own the discard
   transition now, the buggy upstream re-check (``state.py:489`` hardcoded ``> 7``)
   never runs; we sequence discarders off ``state.discard_limit``.
"""

import gymnasium.spaces as _spaces

import catanatron.game as _game_mod
import catanatron.state as _state_mod
import catanatron.models.actions as _actions_mod
import catanatron_gym.envs.catanatron_env as _gym_env
from catanatron.models.enums import Action, ActionType, ActionPrompt, RESOURCES
from catanatron.models.decks import freqdeck_add
from catanatron.state_functions import (
    player_key, player_num_resource_cards, player_deck_subtract,
)

DISCARD_LIMIT = 9

_PATCH_FLAG = "_catan_rules_patched"


def apply_rule_patches(discard_limit: int = DISCARD_LIMIT) -> None:
    """Idempotently install the custom-rule monkeypatches.

    Safe to call from any module/process; only the first call takes effect.
    """
    if getattr(_game_mod, _PATCH_FLAG, False):
        return
    _patch_discard_limit(discard_limit)
    _patch_sequential_discard()
    _patch_gym_action_space()
    setattr(_game_mod, _PATCH_FLAG, True)


def _patch_discard_limit(discard_limit: int) -> None:
    """Default ``Game``'s ``discard_limit`` to our value when unspecified."""
    orig_init = _game_mod.Game.__init__

    def patched_init(self, *args, **kwargs):
        kwargs.setdefault("discard_limit", discard_limit)
        orig_init(self, *args, **kwargs)

    _game_mod.Game.__init__ = patched_init


# --------------------------------------------------------------------------
# Sequential, choosable discard (engine side)
# --------------------------------------------------------------------------
def _patch_sequential_discard() -> None:
    """Generate per-resource discards and apply them one card at a time."""
    orig_generate = _actions_mod.generate_playable_actions

    def patched_generate(state):
        if state.current_prompt == ActionPrompt.DISCARD:
            return _discard_options(state)
        return orig_generate(state)

    _actions_mod.generate_playable_actions = patched_generate
    _state_mod.generate_playable_actions = patched_generate

    orig_apply = _state_mod.apply_action

    def patched_apply(state, action):
        if action.action_type == ActionType.DISCARD:
            return _apply_single_discard(state, action)
        return orig_apply(state, action)

    _state_mod.apply_action = patched_apply
    _game_mod.apply_action = patched_apply


def _discard_options(state):
    """One DISCARD action per resource the current discarder still holds."""
    color = state.current_color()
    key = player_key(state, color)
    return [
        Action(color, ActionType.DISCARD, resource)
        for resource in RESOURCES
        if state.player_state[f"{key}_{resource}_IN_HAND"] > 0
    ]


def _apply_single_discard(state, action):
    """Drop one chosen card; re-prompt until this player's quota is met."""
    color = action.color
    resource = action.value if action.value is not None else _largest_stack(state, color)

    remaining = getattr(state, "_discard_remaining", None)
    if remaining is None:
        remaining = {}
        state._discard_remaining = remaining
    if color not in remaining:
        # Quota is fixed when the player starts discarding (hand still full).
        remaining[color] = player_num_resource_cards(state, color) // 2

    freqdeck = [0, 0, 0, 0, 0]
    freqdeck[RESOURCES.index(resource)] = 1
    player_deck_subtract(state, color, freqdeck)
    state.resource_freqdeck = freqdeck_add(state.resource_freqdeck, freqdeck)
    remaining[color] -= 1

    if remaining[color] > 0:
        state.current_player_index = state.colors.index(color)
        state.current_prompt = ActionPrompt.DISCARD
        state.is_discarding = True
    else:
        remaining.pop(color, None)
        _advance_after_discarder(state, color)

    state.playable_actions = _state_mod.generate_playable_actions(state)
    return action


def _advance_after_discarder(state, color):
    """Hand off to the next over-limit discarder, or on to MOVE_ROBBER."""
    idx = state.colors.index(color)
    later = [
        player_num_resource_cards(state, c) > state.discard_limit
        for c in state.colors
    ][idx + 1:]
    if any(later):
        state.current_player_index = idx + 1 + later.index(True)
        state.current_prompt = ActionPrompt.DISCARD
        state.is_discarding = True
    else:
        state.current_player_index = state.current_turn_index
        state.current_prompt = ActionPrompt.MOVE_ROBBER
        state.is_discarding = False
        state.is_moving_knight = True


def _largest_stack(state, color):
    """Resource the player holds most of (fallback when no card is specified)."""
    key = player_key(state, color)
    counts = {r: state.player_state[f"{key}_{r}_IN_HAND"] for r in RESOURCES}
    return max(RESOURCES, key=lambda r: (counts[r], -RESOURCES.index(r)))


# --------------------------------------------------------------------------
# Gym action space (one DISCARD slot -> one per resource)
# --------------------------------------------------------------------------
def _patch_gym_action_space() -> None:
    """Expand ACTIONS_ARRAY so the policy can emit a specific discard."""
    array = _gym_env.ACTIONS_ARRAY
    try:
        idx = array.index((ActionType.DISCARD, None))
    except ValueError:
        return  # already expanded

    new_array = (
        array[:idx]
        + [(ActionType.DISCARD, resource) for resource in RESOURCES]
        + array[idx + 1:]
    )
    _gym_env.ACTIONS_ARRAY = new_array
    _gym_env.ACTION_SPACE_SIZE = len(new_array)
    _gym_env.CatanatronEnv.action_space = _spaces.Discrete(len(new_array))

    orig_normalize = _gym_env.normalize_action

    def patched_normalize(action):
        # Keep the resource so (DISCARD, resource) maps to its own slot.
        if action.action_type == ActionType.DISCARD:
            return action
        return orig_normalize(action)

    _gym_env.normalize_action = patched_normalize
