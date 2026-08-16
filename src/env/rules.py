"""Custom 1v1 rule tweaks applied as runtime monkeypatches on Catanatron 3.2.1.

We keep these as monkeypatches (not edits to the installed package) so the change
lives in version control and is reapplied automatically in every process -- including
the fresh interpreters that ``SubprocVecEnv`` spawns for parallel training.

Seven things happen here:

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

4. **Colonist.io 1v1 robber restrictions.** Two constraints on MOVE_ROBBER:
   - You may only place the robber on a tile that has an opponent building if the
     opponent has placed ≥3 settlements on the board OR built ≥1 city. This prevents
     camping the robber immediately after the initial setup.
   - You may only place the robber on a tile where YOU have a building if you
     yourself have >2 settlements on the board OR ≥1 city. Once you have expanded
     beyond the initial 2 settlements, self-robbing is a legal strategic choice.
   If all tiles are excluded by both rules (degenerate edge case), the filter is
   lifted so the engine always has at least one legal action.

5. **Longest Road awards no victory points -- optionally.** Stock Catanatron
   grants +2 VP to the holder of the longest road. In a short 8-VP 1v1 game that
   single swing is close to a third of the win condition and it rewards exactly
   the degenerate road-spam behaviour we were trying to train away from, so it is
   suppressed by default: ``LONGEST_ROAD_LENGTH`` is still tracked (it stays in
   the observation vector), only the VP award and the ``HAS_ROAD`` flag go.

   Set ``CATAN_LONGEST_ROAD=1`` (see :mod:`src.env.ruleset`) to leave stock
   behaviour alone. The target ruleset for this project -- colonist.io 1v1 at 15
   VP -- has Longest Road enabled, and at that target it stops being a
   distortion: buildings cap at 9 VP (5 settlements, 4 of them upgraded), so 15
   is unreachable without Longest Road, Largest Army *and* VP cards. The patch
   only ever made sense for the short game.

6. **A development card cannot be played on the turn it was bought.** Stock
   ``buy_dev_card`` records only that a card entered the hand, never when, so a
   knight could be bought and played immediately. We count purchases per turn and
   subtract them from the hand when judging playability. The related "one dev card
   per turn" rule needs no patch -- upstream already enforces it through
   ``HAS_PLAYED_DEVELOPMENT_CARD_IN_TURN``.

7. **``_discard_remaining`` survives ``State.copy()``.** Patch 2 stores the
   per-player discard quota on a custom state attribute, but upstream
   ``State.copy`` enumerates the fields it copies explicitly and therefore drops
   it. That is harmless when copies are only taken by accumulators, but MCTS
   copies mid-game constantly -- a search descending through a 7 would re-derive
   the quota from an already-partly-discarded hand and discard too many cards.
"""

import gymnasium.spaces as _spaces

import catanatron.game as _game_mod
import catanatron.state as _state_mod
import catanatron.models.actions as _actions_mod
import catanatron.state_functions as _state_functions_mod
import catanatron_gym.envs.catanatron_env as _gym_env
from catanatron.models.enums import Action, ActionType, ActionPrompt, RESOURCES
from catanatron.models.decks import freqdeck_add
from catanatron.state_functions import (
    player_key, player_num_resource_cards, player_deck_subtract,
)

from src.env.ruleset import LONGEST_ROAD_VP

DISCARD_LIMIT = 9

_PATCH_FLAG = "_catan_rules_patched"


def apply_rule_patches(discard_limit: int = DISCARD_LIMIT,
                       longest_road_vp: bool = LONGEST_ROAD_VP) -> None:
    """Idempotently install the custom-rule monkeypatches.

    Safe to call from any module/process; only the first call takes effect.

    Args:
        discard_limit: hold more than this many cards and a 7 makes you discard.
        longest_road_vp: leave stock Longest Road scoring (+2 VP) in place.
            Defaults to :data:`src.env.ruleset.LONGEST_ROAD_VP`, i.e. off unless
            ``CATAN_LONGEST_ROAD=1`` is set in the environment. Passing it
            explicitly is for tests; a training run sets the variable, because
            the patch has to land the same way in every spawned worker.
    """
    if getattr(_game_mod, _PATCH_FLAG, False):
        return
    _patch_discard_limit(discard_limit)
    _patch_sequential_discard()
    _patch_state_copy()
    _patch_robber_placement()
    if not longest_road_vp:
        _patch_no_longest_road()
    _patch_dev_card_summoning_sickness()
    _patch_gym_action_space()
    setattr(_game_mod, _PATCH_FLAG, True)


def _patch_discard_limit(discard_limit: int) -> None:
    """Default ``Game``'s ``discard_limit`` to our value when unspecified.

    Game.__init__ signature is (players, seed, discard_limit, ...), so
    discard_limit is positional index 2 (excluding self). game.copy() calls
    Game([], None, None, initialize=False), passing None as the third
    positional arg. We must not also inject it as a keyword in that case.
    """
    orig_init = _game_mod.Game.__init__

    def patched_init(self, *args, **kwargs):
        if len(args) <= 2 and "discard_limit" not in kwargs:
            kwargs["discard_limit"] = discard_limit
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


def _patch_state_copy() -> None:
    """Carry the custom ``_discard_remaining`` quota across ``State.copy()``.

    Upstream ``State.copy`` copies a hardcoded list of fields, so our attribute is
    silently dropped. MCTS copies states mid-discard all the time, so without this
    a search that descends through a 7 recomputes the quota from a half-discarded
    hand and over-discards.
    """
    orig_copy = _state_mod.State.copy

    def patched_copy(self):
        state_copy = orig_copy(self)
        remaining = getattr(self, "_discard_remaining", None)
        if remaining:
            state_copy._discard_remaining = dict(remaining)
        return state_copy

    _state_mod.State.copy = patched_copy


def _largest_stack(state, color):
    """Resource the player holds most of (fallback when no card is specified)."""
    key = player_key(state, color)
    counts = {r: state.player_state[f"{key}_{r}_IN_HAND"] for r in RESOURCES}
    return max(RESOURCES, key=lambda r: (counts[r], -RESOURCES.index(r)))


# --------------------------------------------------------------------------
# Colonist.io 1v1 robber placement restrictions
# --------------------------------------------------------------------------
def _patch_robber_placement() -> None:
    """Filter MOVE_ROBBER actions to enforce Colonist.io 1v1 robber rules."""
    orig_robber = _actions_mod.robber_possibilities

    def patched_robber(state, color):
        actions = orig_robber(state, color)

        # Identify the single opponent (1v1 only).
        opponent = next(c for c in state.colors if c != color)

        my_key = player_key(state, color)
        opp_key = player_key(state, opponent)

        # Settlements on the board = pieces placed out (cities return the piece).
        my_settlements = 5 - state.player_state[f"{my_key}_SETTLEMENTS_AVAILABLE"]
        my_cities = 4 - state.player_state[f"{my_key}_CITIES_AVAILABLE"]
        opp_settlements = 5 - state.player_state[f"{opp_key}_SETTLEMENTS_AVAILABLE"]
        opp_cities = 4 - state.player_state[f"{opp_key}_CITIES_AVAILABLE"]

        # Self-robbing is only legal once you've expanded beyond initial setup.
        self_rob_allowed = my_settlements > 2 or my_cities >= 1
        opp_can_be_robbed = opp_settlements >= 3 or opp_cities >= 1

        # Build a set of tile coordinates that are off-limits.
        excluded = set()
        for coord, tile in state.board.map.land_tiles.items():
            has_own = False
            has_opp = False
            for node_id in tile.nodes.values():
                building = state.board.buildings.get(node_id)
                if building is not None:
                    if building[0] == color:
                        has_own = True
                    else:
                        has_opp = True
            if has_own and not self_rob_allowed:
                excluded.add(coord)
            elif has_opp and not opp_can_be_robbed:
                excluded.add(coord)

        filtered = [a for a in actions if a.value[0] not in excluded]
        # Fallback: never leave the engine with zero legal robber moves.
        return filtered if filtered else actions

    _actions_mod.robber_possibilities = patched_robber


# --------------------------------------------------------------------------
# Longest Road grants no victory points
# --------------------------------------------------------------------------
def _patch_no_longest_road() -> None:
    """Track longest-road length but never award the +2 VP (or set HAS_ROAD).

    ``catanatron.state`` imported ``mantain_longest_road`` by name at import time,
    so the binding we must replace is the one in the *state* module's globals; we
    also replace it at its definition site for anything that imports it later.
    """

    def patched_mantain(state, previous_road_color, road_color, road_lengths):
        for color, length in road_lengths.items():
            key = player_key(state, color)
            state.player_state[f"{key}_LONGEST_ROAD_LENGTH"] = length

    _state_mod.mantain_longest_road = patched_mantain
    _state_functions_mod.mantain_longest_road = patched_mantain


# --------------------------------------------------------------------------
# A development card cannot be played on the turn it was bought
# --------------------------------------------------------------------------
def _patch_dev_card_summoning_sickness() -> None:
    """Forbid playing a development card bought in the same turn.

    Stock ``buy_dev_card`` just increments ``{card}_IN_HAND``, with no record of
    *when* a card arrived, so a player could buy a knight and play it
    immediately. We track a per-turn counter and subtract it from the hand when
    deciding playability.

    Note the one-card-per-turn rule is *already* enforced upstream, via
    ``HAS_PLAYED_DEVELOPMENT_CARD_IN_TURN`` (set in ``play_dev_card``, cleared in
    ``player_clean_turn``, checked in ``player_can_play_dev``). We leave it be.

    Storage is a runtime-only key in ``player_state``:

    - ``State.copy()`` does ``player_state.copy()``, so it survives MCTS for free
      -- no repeat of the ``_discard_remaining`` problem.
    - It is deliberately *not* added to ``PLAYER_INITIAL_STATE``. The gym feature
      ordering is derived from a fresh game's sample keys, so growing that
      template would change the 614-dim observation and invalidate every trained
      checkpoint. ``create_vector`` iterates the cached ordering and filters on
      it, so a key that only ever appears at runtime is ignored.

    Victory Point cards are unaffected: they are never "played" (buying one adds
    its VP immediately), and ``player_can_play_dev`` is only consulted for the
    four playable types.
    """
    _bought = "_BOUGHT_THIS_TURN"
    orig_buy = _state_functions_mod.buy_dev_card
    orig_can_play = _state_functions_mod.player_can_play_dev
    orig_clean = _state_functions_mod.player_clean_turn

    def patched_buy(state, color, dev_card):
        orig_buy(state, color, dev_card)
        key = player_key(state, color)
        field = f"{key}_{dev_card}{_bought}"
        state.player_state[field] = state.player_state.get(field, 0) + 1

    def patched_can_play(state, color, dev_card):
        if not orig_can_play(state, color, dev_card):
            return False
        key = player_key(state, color)
        in_hand = state.player_state[f"{key}_{dev_card}_IN_HAND"]
        fresh = state.player_state.get(f"{key}_{dev_card}{_bought}", 0)
        return in_hand - fresh >= 1

    def patched_clean(state, color):
        orig_clean(state, color)
        key = player_key(state, color)
        for field in [k for k in state.player_state
                      if k.startswith(key) and k.endswith(_bought)]:
            state.player_state[field] = 0

    # All three are imported *by name* elsewhere at import time, so rebinding
    # them in state_functions alone would be a silent no-op: state.py calls its
    # own ``buy_dev_card`` (line 428) and ``player_clean_turn`` (line 314), and
    # both state.py and models/actions.py hold their own ``player_can_play_dev``.
    for module in (_state_functions_mod, _state_mod, _actions_mod):
        module.buy_dev_card = patched_buy
        module.player_clean_turn = patched_clean
        module.player_can_play_dev = patched_can_play


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
