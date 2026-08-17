"""Drive a live ``Game`` from an observed stream of actions.

This is how the bridge reconstructs colonist.io's position, and it is
deliberately not a state-poking translator. ``catanatron.state.apply_action``
accepts a *realized* value for every stochastic action -- ``ROLL`` takes the two
dice, ``BUY_DEVELOPMENT_CARD`` the drawn card, ``MOVE_ROBBER`` the stolen
resource (the engine comments that branch ``# for replay functionality``). So an
observed game can be re-applied move for move and the resulting ``State`` is
consistent by construction: piece counts, the bank, the dev deck, longest road,
and every one of the seven patches in :mod:`src.env.rules` all maintain
themselves.

That matters more than convenience. ``MCTSPlayer`` copies and rolls the ``Game``
forward, so a hand-assembled ``State`` that is merely *observation-equivalent*
would search a position the engine could not have produced.

**Desync is the failure mode to fear**, not illegality. A replay that quietly
drifts from the live game looks exactly like a weak agent. So every applied
action is checked against ``playable_actions`` first, modulo the stochastic
value the engine is about to fill in, and a mismatch raises
:class:`DesyncError` immediately.

Hidden information is out of scope here: this module replays what was
*observed*. In 1v1 that is nearly everything -- roll payouts are deterministic
from the board, bank and port trades are public, and both directions of a robber
steal involve us, so we always see the card. The two leaks are the opponent's
discards on a 7 and its dev cards before they are played; filling those in is a
separate belief/determinization step that hands this module a concrete guess.
"""

from catanatron import Color, Game
from catanatron.models.actions import Action
from catanatron.models.enums import ActionType
from catanatron.models.player import RandomPlayer
from catanatron.state import generate_playable_actions

import src.env.catan_env  # noqa: F401  -- applies the custom rule patches
from src.bridge.board import BoardSpec, build_map_from_spec
from src.env.ruleset import VPS_TO_WIN


class DesyncError(RuntimeError):
    """The observed action is not one the reconstructed game allows.

    Either the translation is wrong or colonist.io is playing by different
    rules than :mod:`src.env.rules` installs. Both are fatal; neither should be
    papered over.
    """


def blank_outcome(action: Action) -> Action:
    """Strip the realized chance outcome, leaving the decision that was made.

    ``playable_actions`` offers a roll with no dice and a robber move with no
    stolen card; the log records both filled in. Comparing the blanked forms is
    what lets an observed action be matched against the legal ones.
    """
    if action.action_type == ActionType.ROLL:
        return Action(action.color, action.action_type, None)
    if action.action_type == ActionType.BUY_DEVELOPMENT_CARD:
        return Action(action.color, action.action_type, None)
    if action.action_type == ActionType.MOVE_ROBBER:
        coordinate, robbed_color, _ = action.value
        return Action(action.color, action.action_type,
                      (coordinate, robbed_color, None))
    return action


class GameReplay:
    """A ``Game`` advanced by observed actions rather than by player decisions.

    Args:
        board_spec: the board as dealt.
        colors: seats in turn order. The first entry moves first, which in 1v1
            Catan also means it settles first -- a real edge, so getting this
            backwards is not cosmetic.
        vps_to_win: victory target of the live game. Defaults to the ruleset the
            process was started under; it must match the lobby, or the agent is
            playing to the wrong finish line.

    The players are placeholders and are never asked to decide -- every action
    comes from outside. Ask the agent for a move with
    ``player.decide(replay.game, replay.playable_actions)``.
    """

    def __init__(self, board_spec: BoardSpec, colors=(Color.BLUE, Color.RED),
                 vps_to_win: int = VPS_TO_WIN):
        self.game = Game(
            players=[RandomPlayer(color) for color in colors],
            vps_to_win=vps_to_win,
            catan_map=build_map_from_spec(board_spec),
        )
        self._seat(colors)

    def _seat(self, colors) -> None:
        """Force the seating order instead of accepting the one ``State`` drew.

        ``State.__init__`` calls ``random.sample`` on the players, so the seat
        order is a coin flip. Offline that is a feature -- it is how a benchmark
        stops measuring who got P0. Here it is a bug: seating is *observed*, and
        in 1v1 the first seat settles first, so guessing it wrong means
        reconstructing a materially different game.

        Reordering is safe only because nothing has happened yet: every
        ``player_state`` entry still holds its initial value, and the other
        structures are keyed by color rather than by index. The playable actions
        do have to be regenerated -- they were built for whoever ``sample``
        happened to seat first.
        """
        state = self.game.state
        assert not state.actions and state.num_turns == 0, "reseat before play"

        by_color = {player.color: player for player in state.players}
        state.players = [by_color[color] for color in colors]
        state.colors = tuple(colors)
        state.color_to_index = {color: i for i, color in enumerate(colors)}
        state.playable_actions = generate_playable_actions(state)

    @property
    def state(self):
        return self.game.state

    @property
    def playable_actions(self):
        return self.game.state.playable_actions

    @property
    def current_color(self):
        """Whose decision the game is waiting on."""
        return self.game.state.current_color()

    def winning_color(self):
        return self.game.winning_color()

    def apply(self, action: Action) -> Action:
        """Apply one observed action, fully specified. Returns the logged form."""
        self.check_legal(action)
        # validate_action=False because a filled-in chance outcome is never
        # literally in playable_actions; check_legal has already compared the
        # blanked form, which is the meaningful test.
        return self.game.execute(action, validate_action=False)

    def apply_many(self, actions) -> None:
        for action in actions:
            self.apply(action)

    def check_legal(self, action: Action) -> None:
        """Raise :class:`DesyncError` unless ``action`` is currently allowed."""
        blanked = blank_outcome(action)
        legal = {blank_outcome(a) for a in self.playable_actions}
        if blanked not in legal:
            raise DesyncError(
                f"observed {action} at turn {self.state.num_turns}, prompt "
                f"{self.state.current_prompt}, but the reconstructed game "
                f"offers {sorted(map(str, legal))}"
            )
