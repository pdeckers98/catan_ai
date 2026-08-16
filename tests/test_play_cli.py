"""The human-vs-AI entry point has to be able to deploy the agent properly.

That means three things reaching it: the per-run ruleset (the archived
checkpoints are 15 VP with Longest Road), search, and the placement scorer --
without which any checkpoint trained through PlacementWrapper opens with an
untrained head. All three used to be missing here.
"""

import subprocess
import sys

import matplotlib

matplotlib.use("Agg")

from catanatron.models.player import RandomPlayer  # noqa: E402

from src.env import ruleset  # noqa: E402


def test_the_cli_parser_builds_and_offers_the_deployment_flags():
    """argparse %-formats help strings, so a literal % in one crashes the parser.

    It fails at add_argument time, taking the whole entry point down before any
    flag is parsed -- see the same test over src.agent.train.
    """
    result = subprocess.run(
        [sys.executable, "-m", "src.eval.play", "--help"],
        capture_output=True, text=True, timeout=300,
    )
    assert result.returncode == 0, result.stderr
    for flag in ("--placement-model", "--bundle-model", "--simulations",
                 "--vps-to-win", "--longest-road"):
        assert flag in result.stdout


def test_placement_and_search_settings_reach_the_agent(monkeypatch):
    from src.eval import play

    captured = {}

    def fake_build_agent(spec, model_path=None, simulations=100, **kwargs):
        captured.update(spec=spec, model_path=model_path,
                        simulations=simulations, **kwargs)
        return lambda color: RandomPlayer(color)

    monkeypatch.setattr(play, "build_agent", fake_build_agent)
    ui = play.HumanVsAI(
        "ppo-mcts", "model.zip", seed=3, simulations=50,
        placement_path="scorer.pt", bundle_path="bundle.pt",
    )
    play.plt.close(ui.fig)

    assert captured["spec"] == "ppo-mcts"
    assert captured["simulations"] == 50
    assert captured["placement_path"] == "scorer.pt"
    assert captured["bundle_path"] == "bundle.pt"


def test_the_game_is_built_under_the_active_ruleset(monkeypatch):
    """A 15 VP checkpoint played to the default 8 would measure nothing."""
    from src.eval import play

    monkeypatch.setattr(play, "build_agent",
                        lambda *a, **k: (lambda color: RandomPlayer(color)))
    ui = play.HumanVsAI("weighted", None, seed=3)
    play.plt.close(ui.fig)

    assert ui.game.vps_to_win == ruleset.VPS_TO_WIN

# The other half of the ruleset contract -- that --vps-to-win / --longest-road
# reach the environment before the engine imports -- is covered over
# apply_cli_overrides itself in tests/test_rules.py. Re-testing it here would
# mean mutating the ruleset globals mid-suite.
