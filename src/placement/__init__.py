"""Opening-placement specialist.

Initial settlement placement is structurally unlike the rest of Catan: no dice
have been rolled, the board is fully observable, there are ~2 decisions per game,
and those decisions dominate the outcome. It also receives only ~2 of the ~300
gradient samples an episode produces, which is why the main policy learned
"settle where three tiles meet" but never learned that an 8 beats a 3.

This package fixes that by *data*, not by hand-written rules: openings are
explored at random, labelled with the game's actual outcome, and a small model
learns to score nodes from mechanical board facts. Nothing here encodes an
opinion about what makes a placement good -- see :mod:`src.placement.features`
for where that line is drawn, and :mod:`src.placement.heuristic` for the
hand-written scorer, which exists only as an evaluation yardstick.
"""
