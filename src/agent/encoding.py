"""Observation and action-mask encoding straight off a live ``Game``.

The gym env builds these internally, but MCTS and AlphaZero self-play drive the
engine directly and need the same encodings without a ``step`` interface.

Everything here is written from the perspective of an explicit ``color``, so a
single network can evaluate positions for either seat: ``create_sample_vector``
relabels the requested color as P0.

``catanatron_gym.envs.catanatron_env`` is imported as a *module* so the expanded
(294-slot) action space installed by ``src.env.rules`` is read at call time rather
than frozen at import.
"""

import numpy as np

import catanatron_gym.envs.catanatron_env as cenv
from catanatron_gym.features import create_sample_vector, get_feature_ordering

import src.env.catan_env  # noqa: F401  -- applies the custom rule patches

_FEATURE_CACHE: dict = {}


def feature_ordering(num_players: int = 2, map_type: str = "BASE") -> list:
    """Cached feature ordering; recomputing it per call is surprisingly costly."""
    key = (num_players, map_type)
    if key not in _FEATURE_CACHE:
        _FEATURE_CACHE[key] = get_feature_ordering(num_players, map_type)
    return _FEATURE_CACHE[key]


def obs_size(num_players: int = 2, map_type: str = "BASE") -> int:
    """Length of the observation vector (614 for a 2-player BASE game)."""
    return len(feature_ordering(num_players, map_type))


def action_size() -> int:
    """Size of the (patched) action space -- 294 once rules.py has run."""
    return cenv.ACTION_SPACE_SIZE


def encode_observation(game, color, map_type: str = "BASE") -> np.ndarray:
    """Feature vector for ``game`` from ``color``'s point of view."""
    features = feature_ordering(len(game.state.colors), map_type)
    return np.asarray(
        create_sample_vector(game, color, features), dtype=np.float32
    )


def legal_action_mask(playable_actions) -> np.ndarray:
    """Boolean mask over the full action space, True where legal."""
    mask = np.zeros(cenv.ACTION_SPACE_SIZE, dtype=bool)
    for action in playable_actions:
        mask[cenv.to_action_space(action)] = True
    return mask


def action_indices(playable_actions) -> list:
    """Action-space index for each playable action, positionally aligned.

    Distinct catanatron Actions can normalize to the same slot (e.g. two
    MOVE_ROBBER actions on the same tile that rob different victims). Search
    treats them as separate children but they share a prior, which is the same
    compromise ``from_action_space`` makes.
    """
    return [cenv.to_action_space(action) for action in playable_actions]
