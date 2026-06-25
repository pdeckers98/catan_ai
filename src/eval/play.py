"""Play a 1v1 game against the trained agent in a matplotlib window.

You are RED; the trained agent (``checkpoints/agent_final.zip``) is BLUE -- the
side it trained on (P0). Both hands are shown god-mode (resources AND dev cards).

Flow:
- The board labels node ids, so a textual move like ``BUILD_ROAD edge (12, 13)``
  is locatable (an edge is the line between those two nodes).
- On your turn the legal moves are listed numbered in the side panel. Type the
  number into the box and press Enter to play it.
- When you END your turn, click **Next turn** to let the AI play; review the
  result, then it's your turn again.

Run: ``python -m src.eval.play``  (optional ``--model PATH`` / ``--seed N``)
"""

import argparse

import matplotlib.pyplot as plt
from matplotlib.widgets import TextBox, Button
from sb3_contrib import MaskablePPO

import src.env.catan_env  # noqa: F401  -- applies custom rules (discard_limit=9)
from catanatron import Game, Color
from catanatron.models.player import RandomPlayer
from catanatron.models.enums import Action, ActionType, RESOURCES, DEVELOPMENT_CARDS
from src.agent.opponent import PolicyPlayer
from src.env.render import render_board

HUMAN = Color.RED
AI = Color.BLUE

# 2-letter resource prefixes for typing discards (e.g. "wo wo wh").
RESOURCE_PREFIX = {r[:2]: r for r in RESOURCES}

# Controller modes.
HUMAN_TURN = "human"   # waiting for the human to pick a move
REVIEW = "review"      # human ended turn; waiting for "Next turn" click
OVER = "over"          # game finished


def _recent_rolls(state, n=2):
    """The last ``n`` dice rolls as (color, (d1, d2)), most recent first."""
    rolls = []
    for action in reversed(state.actions):
        if action.action_type == ActionType.ROLL and action.value is not None:
            rolls.append((action.color, action.value))
            if len(rolls) == n:
                break
    return rolls


def _dice_lines(state):
    """Two display lines: the latest roll and the previous one."""
    rolls = _recent_rolls(state, 2)
    labels = ["Last roll", "Prev roll"]
    lines = []
    for label, (color, (d1, d2)) in zip(labels, rolls):
        lines.append(f"{label}: {color.value:<4} {d1}+{d2} = {d1 + d2}")
    if not lines:
        lines.append("Last roll: --")
    return lines


def _fmt_action(action, game=None):
    """Compact one-line action label, e.g. 'BUILD_ROAD (12, 13)' or 'END_TURN'.

    MOVE_ROBBER is rendered human-readably (target tile's number + resource and
    whom it robs) instead of raw cube coordinates.
    """
    name = action.action_type.name
    if action.action_type == ActionType.MOVE_ROBBER and game is not None:
        coord, victim, _resource = action.value
        tile = game.state.board.map.land_tiles.get(coord)
        if tile is not None:
            number = tile.number if tile.number is not None else "desert"
            resource = tile.resource if tile.resource is not None else "DESERT"
            label = f"MOVE_ROBBER -> {resource} {number}"
            if victim is not None:
                label += f", rob {victim.value}"
            return label
    if action.value is None or action.value == ():
        return name
    return f"{name} {action.value}"


def _hand_text(state, color, label):
    """God-mode one-block summary of a player's VP, resources and dev cards."""
    key = f"P{state.color_to_index[color]}"
    vp = state.player_state[f"{key}_ACTUAL_VICTORY_POINTS"]
    res = "  ".join(
        f"{r[:2]}:{state.player_state[f'{key}_{r}_IN_HAND']}" for r in RESOURCES
    )
    dev = "  ".join(
        f"{d[:3]}:{state.player_state[f'{key}_{d}_IN_HAND']}"
        for d in DEVELOPMENT_CARDS
    )
    return f"{label} ({color.value})  VP={vp}\n  res  {res}\n  dev  {dev}"


class HumanVsAI:
    """Drives a manual game loop: human (RED) vs frozen policy (BLUE)."""

    def __init__(self, model_path, seed=None):
        model = MaskablePPO.load(model_path, device="cpu")
        self.ai = PolicyPlayer(AI, model)
        self.game = Game(
            players=[RandomPlayer(AI), RandomPlayer(HUMAN)],
            seed=seed,
            vps_to_win=15,
        )
        self.mode = HUMAN_TURN
        self.message = ""
        self._discard_buffer = []  # cards picked so far this discard (one at a time)
        self._build_ui()
        # AI (P0/BLUE) moves first in the opening; advance to the first human
        # decision before handing control over.
        self._run_ai_until_human()
        self._render()

    # ---- UI plumbing -----------------------------------------------------
    def _build_ui(self):
        self.fig = plt.figure(figsize=(15, 9))
        self.ax_board = self.fig.add_axes([0.02, 0.08, 0.62, 0.9])
        self.ax_side = self.fig.add_axes([0.66, 0.08, 0.32, 0.9])
        self.ax_side.axis("off")
        ax_box = self.fig.add_axes([0.10, 0.02, 0.25, 0.045])
        ax_btn = self.fig.add_axes([0.40, 0.02, 0.18, 0.045])
        self.text_box = TextBox(ax_box, "Move # ")
        self.text_box.on_submit(self._on_submit)
        self.next_btn = Button(ax_btn, "Next turn")
        self.next_btn.on_clicked(self._on_next)

    # ---- game progression ------------------------------------------------
    def _run_ai_until_human(self):
        """Execute AI (BLUE) actions until it's the human's decision or game end."""
        while (self.game.winning_color() is None
               and self.game.state.current_color() == AI):
            action = self.ai.decide(self.game, self.game.state.playable_actions)
            self.game.execute(action)
        if self.game.winning_color() is not None:
            self.mode = OVER
        else:
            self.mode = HUMAN_TURN

    def _on_submit(self, text):
        if self.mode != HUMAN_TURN:
            return
        if self._is_human_discard():
            self._submit_discard(text)
            return

        actions = self.game.state.playable_actions
        try:
            idx = int(text.strip())
            action = actions[idx]
        except (ValueError, IndexError):
            self.message = f"Invalid move '{text}'. Enter 0-{len(actions) - 1}."
            self.text_box.set_val("")
            self._render()
            return

        self.text_box.set_val("")
        self.message = ""
        ended_turn = action.action_type == ActionType.END_TURN
        self.game.execute(action)
        self._after_human_action(ended_turn)

    def _after_human_action(self, ended_turn):
        """Shared post-move flow: end game, pause for review, or continue the AI."""
        if self.game.winning_color() is not None:
            self.mode = OVER
        elif self.game.state.current_color() == AI:
            if ended_turn:
                # Pause so the human can review before the AI plays.
                self.mode = REVIEW
                self.message = "Your turn is over. Click 'Next turn' for BLUE."
            else:
                # Control passed to AI mid-turn (e.g. after a forced discard on a
                # 7); continue the AI without an extra click.
                self._run_ai_until_human()
        # else: still the human's decision this turn -> keep prompting.
        self._render()

    # ---- discard (choose your own cards) ---------------------------------
    def _is_human_discard(self):
        actions = self.game.state.playable_actions
        return (self.game.state.current_color() == HUMAN
                and len(actions) == 1
                and actions[0].action_type == ActionType.DISCARD)

    def _human_owned(self):
        state = self.game.state
        key = f"P{state.color_to_index[HUMAN]}"
        return {r: state.player_state[f"{key}_{r}_IN_HAND"] for r in RESOURCES}

    def _discard_need(self):
        return sum(self._human_owned().values()) // 2

    def _discard_menu(self):
        """Resources still available to pick (owned minus what's already buffered)."""
        owned = self._human_owned()
        buf = self._discard_buffer
        return [r for r in RESOURCES if owned[r] - buf.count(r) > 0]

    def _submit_discard(self, text):
        """Pick ONE card to discard; apply once enough have been picked."""
        menu = self._discard_menu()
        token = text.strip()
        if token.isdigit() and int(token) < len(menu):
            res = menu[int(token)]
        else:
            res = RESOURCE_PREFIX.get(token[:2].upper())
        if res is None or res not in menu:
            self._discard_error("Type the number of a resource you still hold.")
            return

        self._discard_buffer.append(res)
        self.text_box.set_val("")
        self.message = ""

        if len(self._discard_buffer) < self._discard_need():
            self._render()  # keep collecting one at a time
            return

        action = Action(HUMAN, ActionType.DISCARD, tuple(self._discard_buffer))
        self._discard_buffer = []
        # Explicit-value discards aren't in playable_actions, so skip validation.
        self.game.execute(action, validate_action=False)
        self._after_human_action(ended_turn=False)

    def _discard_error(self, msg):
        self.message = msg
        self.text_box.set_val("")
        self._render()

    def _on_next(self, _event):
        if self.mode != REVIEW:
            return
        self.message = ""
        self._run_ai_until_human()
        self._render()

    # ---- rendering -------------------------------------------------------
    def _render(self):
        state = self.game.state
        render_board(
            self.game, ax=self.ax_board, label_nodes=True, show_info=False,
            title=f"You are RED  |  turn {state.num_turns}",
        )

        self.ax_side.clear()
        self.ax_side.axis("off")

        header = _dice_lines(state) + [
            "",
            _hand_text(state, AI, "AI"),
            "",
            _hand_text(state, HUMAN, "YOU"),
            "",
            "-" * 36,
        ]
        winner = self.game.winning_color()
        is_discard = winner is None and self.mode == HUMAN_TURN and self._is_human_discard()
        if winner is not None:
            who = "YOU WIN!" if winner == HUMAN else "AI WINS."
            header.append(f"GAME OVER -- {who}")
        elif self.mode == REVIEW:
            header.append(self.message)
        elif is_discard:
            owned = self._human_owned()
            need = sum(owned.values()) // 2
            buf = self._discard_buffer
            remaining = need - len(buf)
            header.append(f"7 ROLLED -- discard {need} cards, one at a time.")
            if buf:
                header.append("Discarded so far: " + " ".join(r[:2] for r in buf))
            header.append(f"Pick card {len(buf) + 1} of {need} "
                          f"({remaining} left) -- type its number:")
            for i, r in enumerate(self._discard_menu()):
                have = owned[r] - buf.count(r)
                header.append(f"  {i}: {r}  (have {have})")
            if self.message:
                header.append(self.message)
        else:
            header.append("Your move -- type its number, press Enter:")
            if self.message:
                header.append(self.message)
        self.ax_side.text(
            0.0, 1.0, "\n".join(header), transform=self.ax_side.transAxes,
            ha="left", va="top", fontsize=8.5, family="monospace",
        )

        # Numbered legal moves, in up to 3 columns so long opening lists fit.
        if self.mode == HUMAN_TURN and winner is None and not is_discard:
            actions = self.game.state.playable_actions
            labels = [f"{i:>2}: {_fmt_action(a, self.game)}" for i, a in enumerate(actions)]
            per_col = 32
            for c in range(0, len(labels), per_col):
                col = labels[c:c + per_col]
                self.ax_side.text(
                    0.0 + 0.40 * (c // per_col), 0.66, "\n".join(col),
                    transform=self.ax_side.transAxes, ha="left", va="top",
                    fontsize=7, family="monospace",
                )
        self.fig.canvas.draw_idle()


def main():
    parser = argparse.ArgumentParser(description="Play 1v1 Catan vs the trained AI.")
    parser.add_argument("--model", default="checkpoints/agent_final.zip",
                        help="Path to the trained MaskablePPO checkpoint.")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    HumanVsAI(args.model, seed=args.seed)
    plt.show()


if __name__ == "__main__":
    main()
