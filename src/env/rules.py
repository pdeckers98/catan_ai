"""Custom 1v1 rule tweaks applied as runtime monkeypatches on Catanatron 3.2.1.

We keep these as monkeypatches (not edits to the installed package) so the change
lives in version control and is reapplied automatically in every process -- including
the fresh interpreters that ``SubprocVecEnv`` spawns for parallel training.

Two things happen here:

1. **Discard threshold raised to 9.** Stock Catanatron makes you discard half your
   hand on a 7 when you hold *more than 7* cards (``discard_limit=7``). The gym env
   constructs ``Game`` without passing ``discard_limit``, so we wrap ``Game.__init__``
   to inject our value. After this, you only discard when holding *more than 9*.

2. **Fix an upstream re-check bug.** ``catanatron.state.apply_action`` triggers the
   discard correctly off ``state.discard_limit`` (state.py:447), but the *re-check*
   that sequences multiple discarders after the first one discards hardcodes ``> 7``
   (state.py:489). With a non-7 limit that wrongly forces a player holding 8-9 cards
   to discard. We recompute that transition correctly using ``state.discard_limit``.

3. **Smart (non-random) discard.** Stock Catanatron has a single ``(DISCARD, None)``
   action and then discards a *random* half of the hand. We instead drop from the
   player's largest stacks (surplus) first, so a bot never throws away scarce cards.
   This keeps the action space (and the trained model) unchanged: the policy still
   just emits "DISCARD"; we pick *which* cards. A caller that supplies an explicit
   ``DISCARD`` value (e.g. a human choosing cards) bypasses this untouched.
"""

import catanatron.game as _game_mod
import catanatron.state as _state_mod
from catanatron.state import (
    generate_playable_actions, player_num_resource_cards, player_key,
)
from catanatron.models.enums import Action, ActionType, ActionPrompt, RESOURCES

DISCARD_LIMIT = 9

_PATCH_FLAG = "_catan_rules_patched"


def apply_rule_patches(discard_limit: int = DISCARD_LIMIT) -> None:
    """Idempotently install the custom-rule monkeypatches.

    Safe to call from any module/process; only the first call takes effect.
    """
    if getattr(_game_mod, _PATCH_FLAG, False):
        return
    _patch_discard_limit(discard_limit)
    _patch_discard_recheck()
    setattr(_game_mod, _PATCH_FLAG, True)


def _patch_discard_limit(discard_limit: int) -> None:
    """Default ``Game``'s ``discard_limit`` to our value when unspecified.

    The gym env calls ``Game(...)`` with keyword args only and never sets
    ``discard_limit``, so ``setdefault`` reliably injects ours without clobbering
    an explicit caller value.
    """
    orig_init = _game_mod.Game.__init__

    def patched_init(self, *args, **kwargs):
        kwargs.setdefault("discard_limit", discard_limit)
        orig_init(self, *args, **kwargs)

    _game_mod.Game.__init__ = patched_init


def _patch_discard_recheck() -> None:
    """Recompute the post-discard transition using ``state.discard_limit``.

    Wraps ``apply_action`` (patching both the ``state`` and ``game`` module bindings,
    since ``game.py`` imported the name directly). The original resource-subtraction
    side effects are kept; only the buggy ``> 7`` discarder re-check is overridden.
    """
    orig_apply = _state_mod.apply_action

    def patched_apply(state, action):
        # Turn a random "DISCARD None" into a chosen-cards discard before applying.
        if action.action_type == ActionType.DISCARD and action.value is None:
            action = _smart_discard_action(state, action)
        result = orig_apply(state, action)
        if action.action_type == ActionType.DISCARD:
            _fix_discard_transition(state, action)
        return result

    _state_mod.apply_action = patched_apply
    _game_mod.apply_action = patched_apply


def _smart_discard_action(state, action):
    """Pick cards to discard: drop from the largest stacks (surplus) first.

    Returns a DISCARD action with an explicit list, so the engine discards those
    exact cards instead of a random half.
    """
    key = player_key(state, action.color)
    counts = {r: state.player_state[f"{key}_{r}_IN_HAND"] for r in RESOURCES}
    num_to_discard = sum(counts.values()) // 2
    discarded = []
    for _ in range(num_to_discard):
        resource = max(RESOURCES, key=lambda r: (counts[r], -RESOURCES.index(r)))
        counts[resource] -= 1
        discarded.append(resource)
    return Action(action.color, ActionType.DISCARD, tuple(discarded))


def _fix_discard_transition(state, action) -> None:
    """Mirror Catanatron's discard sequencing, but keyed on ``state.discard_limit``."""
    idx = state.colors.index(action.color)
    remaining = [
        player_num_resource_cards(state, color) > state.discard_limit
        for color in state.colors
    ][idx + 1:]
    if any(remaining):
        state.current_player_index = idx + 1 + remaining.index(True)
        state.current_prompt = ActionPrompt.DISCARD
        state.is_discarding = True
    else:
        state.current_player_index = state.current_turn_index
        state.current_prompt = ActionPrompt.MOVE_ROBBER
        state.is_discarding = False
        state.is_moving_knight = True
    state.playable_actions = generate_playable_actions(state)
