# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""The Depth Anything export recipe (``models/depth-anything/export.py``) and its registry rows.

Source scan rather than an import: the recipe is a PEP 723 script whose inline dependencies are
not installed in the test environment.
"""

import ast
from pathlib import Path

from coreai_models.model_registry import UTILITY_PRESETS

_RECIPE = "models/depth-anything/export.py"
_REPO_ROOT = Path(__file__).resolve().parents[4]


def _model_choices(source: str) -> list[str]:
    """The ``choices`` of the recipe's ``--model`` argument."""
    for node in ast.walk(ast.parse(source)):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "add_argument"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and node.args[0].value == "--model"
        ):
            for keyword in node.keywords:
                if keyword.arg == "choices":
                    return ast.literal_eval(keyword.value)
    raise AssertionError(f"{_RECIPE}: no --model argument with choices")


def test_recipe_accepts_every_registered_variant() -> None:
    """``coreai.model.registry`` prints ``--model <hf_id>``, so the recipe has to accept it."""
    registered = {u.hf_id for u in UTILITY_PRESETS if u.export_script == _RECIPE}
    assert registered == {"depth-anything/da3-small", "depth-anything/da3-base"}
    assert registered == set(_model_choices((_REPO_ROOT / _RECIPE).read_text()))
