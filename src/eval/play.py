"""Play a 1v1 game against the trained agent in a matplotlib window.

You are RED; the trained agent is BLUE -- the side it trained on (P0). Both hands
are shown god-mode (resources AND dev cards). The ruleset is per-run and must
match what the checkpoint trained under: ``--vps-to-win`` / ``--longest-road``
(see ``src/env/ruleset.py``); the defaults are 8 VP with no Longest Road.

Flow:
- The board labels node ids, so a textual move like ``BUILD_ROAD edge (12, 13)``
  is locatable (an edge is the line between those two nodes).
- On your turn the legal moves are listed numbered in the side panel. Type the
  number into the box and press Enter to play it.
- When you END your turn, click **Next turn** to let the AI play; review the
  result, then it's your turn again.

Run the current agent the way it is meant to be deployed -- search at inference,
opening handed to the placement specialist::

    python -m src.eval.play --vps-to-win 15 --longest-road --max-turns 1500 \\
        --agent ppo-mcts --model checkpoints/archive/ppo-15vp-lr-step400000.zip \\
        --simulations 50 \\
        --placement-model checkpoints/placement/scorer_ppo.pt \\
        --bundle-model    checkpoints/placement/bundle_noroads.pt

(also ``--agent ppo`` to play the same zip directly without search, which is
about 10 points weaker; optional ``--seed N``)
"""

import argparse
import random

# Before any engine import: src.env.rules decides at import time whether Longest
# Road pays its VP, so the ruleset has to be selected first. See src/env/ruleset.py.
from src.env.ruleset import apply_cli_overrides

apply_cli_overrides()

import matplotlib.pyplot as plt
from matplotlib.widgets import TextBox, Button

from catanatron import Color
from catanatron.models.player import RandomPlayer
from catanatron.models.enums import (
    ActionType, ActionPrompt, RESOURCES, DEVELOPMENT_CARDS,
)
from src.agent.arena import build_agent
from src.env import ruleset
from src.env.catan_env import make_1v1_game
from src.env.render import render_board
from src.placement.chooser import PARTNER_RANK

HUMAN = Color.RED
AI = Color.BLUE

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

    def __init__(self, agent_spec, model_path, seed=None, simulations=100,
                 placement_path=None, bundle_path=None,
                 partner_rank=PARTNER_RANK, horizon=None):
        self.ai = build_agent(
            agent_spec, model_path, simulations,
            placement_path=placement_path, bundle_path=bundle_path,
            partner_rank=partner_rank, horizon=horizon,
        )(AI)
        # Placeholder players: this loop drives the engine itself and never calls
        # their decide(). Seating is randomized so the human isn't always second.
        players = [RandomPlayer(AI), RandomPlayer(HUMAN)]
        if random.random() < 0.5:
            players = [RandomPlayer(HUMAN), RandomPlayer(AI)]
        self.game = make_1v1_game(players=players, seed=seed,
                                  vps_to_win=ruleset.VPS_TO_WIN)
        self.mode = HUMAN_TURN
        self.message = ""
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
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)

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

    def _on_key(self, event):
        if event.key == "enter" and self.mode == REVIEW:
            self._on_next(None)

    def _on_submit(self, text):
        if not text.strip():
            if self.mode == REVIEW:
                self._on_next(None)
            return
        if self.mode != HUMAN_TURN:
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

    def _is_discard_prompt(self):
        """True when the human is being asked to discard a card (after a 7)."""
        return self.game.state.current_prompt == ActionPrompt.DISCARD

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
            title=(f"You are RED  |  turn {state.num_turns}  |  "
                   f"first to {ruleset.VPS_TO_WIN} VP"
                   + ("  (longest road +2)" if ruleset.LONGEST_ROAD_VP else "")),
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
        if winner is not None:
            who = "YOU WIN!" if winner == HUMAN else "AI WINS."
            header.append(f"GAME OVER -- {who}")
        elif self.mode == REVIEW:
            header.append(self.message)
        elif self.mode == HUMAN_TURN and self._is_discard_prompt():
            header.append("7 ROLLED -- discard one card; pick its number:")
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
        if self.mode == HUMAN_TURN and winner is None:
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
    parser.add_argument("--agent", default="ppo-mcts",
                        help="Agent spec: ppo-mcts, ppo, mcts, value, weighted.")
    parser.add_argument("--model", default=None,
                        help="MaskablePPO checkpoint (.zip).")
    parser.add_argument("--simulations", type=int, default=100,
                        help="MCTS playouts per move for search-backed agents. "
                             "50 is where search saturates and keeps the wait "
                             "between moves short.")
    parser.add_argument("--horizon", type=int, default=None,
                        help="Cap the search at this many game turns past "
                             "the current position; positions beyond it "
                             "are scored by the value head instead of "
                             "expanded. Default searches as deep as the "
                             "simulation budget reaches.")
    parser.add_argument("--placement-model", default=None,
                        help="PlacementNet checkpoint that plays the AI's "
                             "opening settlements. Every archived checkpoint "
                             "needs one -- without it the AI opens with an "
                             "untrained head.")
    parser.add_argument("--bundle-model", default=None,
                        help="BundleNet checkpoint scoring whole corner pairs. "
                             "Requires --placement-model, which shortlists the "
                             "corners it searches over.")
    parser.add_argument("--partner-rank", type=int, default=PARTNER_RANK,
                        help="Pessimism of the pair search about the first "
                             "seat's second settlement surviving. Needs "
                             "--bundle-model.")
    parser.add_argument("--vps-to-win", type=int, default=ruleset.VPS_TO_WIN,
                        help="Victory points to win. Must match the ruleset the "
                             "checkpoint was trained under to mean anything.")
    parser.add_argument("--longest-road", action=argparse.BooleanOptionalAction,
                        default=ruleset.LONGEST_ROAD_VP,
                        help="Award Longest Road its +2 VP.")
    parser.add_argument("--dev-cards", action=argparse.BooleanOptionalAction,
                        default=ruleset.DEV_CARDS,
                        help="Deal a development deck at all. --no-dev-cards removes "
                             "them from the game entirely: no Largest Army, "
                             "no VP cards.")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    # argparse only re-declares the ruleset flags so --help lists them;
    # apply_cli_overrides already put them in the environment before the engine
    # imported. Print what actually took effect.
    print(f"[Rules] {ruleset.describe()}")

    HumanVsAI(args.agent, args.model, seed=args.seed,
              simulations=args.simulations, horizon=args.horizon,
              placement_path=args.placement_model,
              bundle_path=args.bundle_model,
              partner_rank=args.partner_rank)
    plt.show()


if __name__ == "__main__":
    # Pin PYTHONHASHSEED first, so a seed reproduces the game and not
    # merely the board; this relaunches once when it is unset.
    from src.env.determinism import ensure_hash_seed

    ensure_hash_seed()
    main()
