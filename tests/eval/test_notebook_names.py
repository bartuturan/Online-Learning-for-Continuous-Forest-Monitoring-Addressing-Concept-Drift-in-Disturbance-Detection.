"""Guard against the regression class that modularization keeps producing.

Extracting a helper into `src/` means deleting its notebook definition. Miss one caller
and the notebook raises `NameError` at runtime -- which, in Evaluations.ipynb, is caught
by a bare `except Exception` and silently degrades the output instead of failing loudly.
That is exactly what happened when `_prepare_features_common` was extracted and
`_feature_cache_enabled_for_family` was left behind (commit 0fd01db).

This walks each notebook's cells in order, tracking which names are bound as execution
proceeds, and asserts every name a cell *reads* was bound by an earlier cell, by that
cell, or by an import. It is static, so it needs no data and runs in milliseconds.
"""

import ast
import builtins
import json
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Notebooks whose cells are meant to run top-to-bottom in one namespace.
NOTEBOOKS = [
    'notebooks/evaluation/Evaluations.ipynb',
    'notebooks/training/mlp/MLP_prevyears_and_monthly_features.ipynb',
    'notebooks/training/sgd/SGD Classifier.ipynb',
    'notebooks/training/sgd/SGD Classifier_prevyears.ipynb',
    'notebooks/training/sgd/SGD Classifier_prevyears and monthly features.ipynb',
    'notebooks/training/sgd/SGD Classifier_prevyears-incremental scaler.ipynb',
    'notebooks/training/sgd/SGD Classifier_prevyears and monthly features-incremental scaler.ipynb',
]

BUILTINS = set(dir(builtins))
# Names Jupyter injects, or that only exist inside a live kernel.
NOTEBOOK_RUNTIME_NAMES = {'display', 'get_ipython', 'In', 'Out', 'exit', 'quit', '_'}


def _bound_names(node):
    """Every name this AST node binds into the enclosing namespace."""
    bound = set()
    for child in ast.walk(node):
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            bound.add(child.name)
        elif isinstance(child, (ast.Import, ast.ImportFrom)):
            for alias in child.names:
                bound.add(alias.asname or alias.name.split('.')[0])
        elif isinstance(child, ast.Name) and isinstance(child.ctx, (ast.Store, ast.Del)):
            bound.add(child.id)
        elif isinstance(child, ast.arg):
            bound.add(child.arg)
        elif isinstance(child, ast.alias):
            bound.add(child.asname or child.name.split('.')[0])
        elif isinstance(child, (ast.Global, ast.Nonlocal)):
            bound.update(child.names)
        elif isinstance(child, ast.ExceptHandler) and child.name:
            bound.add(child.name)
        elif isinstance(child, (ast.comprehension,)):
            for sub in ast.walk(child.target):
                if isinstance(sub, ast.Name):
                    bound.add(sub.id)
    return bound


def _read_names(node):
    return {c.id for c in ast.walk(node) if isinstance(c, ast.Name) and isinstance(c.ctx, ast.Load)}


def _code_cells(nb_path):
    nb = json.loads(nb_path.read_text(encoding='utf-8'))
    return [(i, ''.join(c['source'])) for i, c in enumerate(nb['cells'])
            if c.get('cell_type') == 'code']


@pytest.mark.parametrize('rel', NOTEBOOKS, ids=lambda r: Path(r).stem[:40])
def test_no_undefined_names(rel):
    nb_path = PROJECT_ROOT / rel
    if not nb_path.exists():
        pytest.skip(f'{rel} not present')

    available = set(BUILTINS) | NOTEBOOK_RUNTIME_NAMES
    problems = []

    for idx, src in _code_cells(nb_path):
        try:
            tree = ast.parse(src)
        except SyntaxError as exc:
            problems.append(f'cell {idx}: SyntaxError: {exc}')
            continue

        # A cell can use anything it binds itself (defs are hoisted at call time,
        # and later cells may rebind), so union before checking.
        available |= _bound_names(tree)
        undefined = sorted(_read_names(tree) - available)
        if undefined:
            problems.append(f'cell {idx}: undefined {undefined}')

    assert not problems, f'{Path(rel).name}:\n  ' + '\n  '.join(problems)


def test_guard_catches_a_deleted_definition(tmp_path):
    """The guard must actually fail on the shape of bug it exists to catch."""
    nb = {'cells': [
        {'cell_type': 'code', 'source': ['def helper(x):\n', '    return x\n']},
        {'cell_type': 'code', 'source': ['helper(1)\n', 'deleted_helper(2)\n']},
    ]}
    path = tmp_path / 'broken.ipynb'
    path.write_text(json.dumps(nb), encoding='utf-8')

    available = set(BUILTINS)
    undefined_found = []
    for idx, src in _code_cells(path):
        tree = ast.parse(src)
        available |= _bound_names(tree)
        undefined_found += sorted(_read_names(tree) - available)

    assert undefined_found == ['deleted_helper']
