"""Guard against replay sampling regressing back to per-epoch cadence.

Sampling the replay pool once per year (not once per epoch) is a deliberate
design choice -- see the module docstring on src/mlp_replay/replay_strategies.py
for why. Per-epoch resampling silently breaks the replay-ratio axis (the union
of draws across ~15 epochs approaches the whole pool, so RR stops meaning
"distinct examples seen"), confounds early stopping with sampling noise, and
dilutes the model-scored strategies (hard/uncertain/confident mining) by
diluting their "hard" quota with random filler once the model has fit the year.

This walks each notebook's cells, finds the `for epoch in range(MAX_EPOCHS):`
loop, and asserts no replay-sampling call sits inside it. It is static (AST
only, like tests/eval/test_notebook_names.py), so it needs no data and runs in
milliseconds -- and it catches the regression at review time rather than
after an hours-long Kaggle run produces results that quietly aren't comparable
to the rest of the sweep.
"""

import ast
import json
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# These notebooks converged on per-year sampling (see the module docstring).
# The *_reservoir_sampling twins are NOT included here on purpose -- they still
# resample every epoch (out of scope for that change; run_pipeline_stages.py's
# module docstring notes they aren't comparable to these).
NOTEBOOKS = [
    'notebooks/training/mlp/MLP_prevyears_and_monthly_features.ipynb',
    'notebooks/training/mlp/experience_replay/MLP-experience replay.ipynb',
    'notebooks/training/mlp/experience_replay/MLP-experience_replay_hard_example_mining.ipynb',
    'notebooks/training/mlp/experience_replay/MLP-experience_replay_uncertainity_prioritization.ipynb',
    'notebooks/training/mlp/experience_replay/MLP-experience_replay_confidently_correct_memory.ipynb',
    'notebooks/training/mlp/experience_replay/MLP-experience replay_misclassification_buffer.ipynb',
    # er_combined already sampled once per year; it was simply missing from this
    # list. Its samplers are notebook-local copies that happen to share the names
    # below, so the same guard applies to it unchanged.
    'notebooks/training/mlp/experience_replay/MLP_experience_replay_combined.ipynb',
]

# Sampler helpers from src/mlp_replay/replay_strategies.py, plus the raw
# `replay_rng.choice(...)` draw that er_plain and er_misclassification_buffer
# use directly instead of a named helper.
BANNED_SAMPLER_NAMES = {
    'sample_hard_replay_indices',
    'sample_uncertain_replay_indices',
    'sample_confidently_correct_replay_indices',
    'sample_random_replay_indices',
    'sample_grouped_indices',
    'replay_rng.choice',
    # er_combined-only samplers (notebook-local definitions).
    'sample_confident_replay_indices',
    'sample_positive_rate_indices',
    'sample_misclass_replay_indices',
}

REQUIRED_INSIDE_LOOP = 'replay_rng.permutation'


def _code_cells(nb_path):
    nb = json.loads(nb_path.read_text(encoding='utf-8'))
    return [(i, ''.join(c['source'])) for i, c in enumerate(nb['cells'])
            if c.get('cell_type') == 'code']


def _call_name(call_node):
    """Best-effort textual name for a Call's func: 'foo' for a bare Name,
    'a.b' for a single-level Attribute access. Returns None for anything more
    exotic (e.g. a call on a subscript) -- those never match a banned name."""
    func = call_node.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
        return f'{func.value.id}.{func.attr}'
    return None


def _is_epoch_loop(node):
    if not isinstance(node, ast.For):
        return False
    if not isinstance(node.iter, ast.Call):
        return False
    func = node.iter.func
    if not (isinstance(func, ast.Name) and func.id == 'range'):
        return False
    if not node.iter.args:
        return False
    arg0 = node.iter.args[0]
    return isinstance(arg0, ast.Name) and arg0.id == 'MAX_EPOCHS'


def _find_epoch_loops(tree):
    return [node for node in ast.walk(tree) if _is_epoch_loop(node)]


def _call_names_in(node):
    names = []
    for child in ast.walk(node):
        if isinstance(child, ast.Call):
            name = _call_name(child)
            if name is not None:
                names.append(name)
    return names


@pytest.mark.parametrize('rel', NOTEBOOKS, ids=lambda r: Path(r).stem[:40])
def test_no_replay_sampling_inside_epoch_loop(rel):
    nb_path = PROJECT_ROOT / rel
    if not nb_path.exists():
        pytest.skip(f'{rel} not present')

    problems = []
    found_any_epoch_loop = False

    for idx, src in _code_cells(nb_path):
        try:
            tree = ast.parse(src)
        except SyntaxError as exc:
            problems.append(f'cell {idx}: SyntaxError: {exc}')
            continue

        for loop in _find_epoch_loops(tree):
            found_any_epoch_loop = True
            call_names = _call_names_in(loop)

            banned_hits = sorted(set(call_names) & BANNED_SAMPLER_NAMES)
            if banned_hits:
                problems.append(
                    f'cell {idx}: replay sampling call(s) inside `for epoch in range(MAX_EPOCHS):` '
                    f'-- must be sampled once per year, before the epoch loop: {banned_hits}'
                )

            if REQUIRED_INSIDE_LOOP not in call_names:
                problems.append(
                    f'cell {idx}: epoch loop has no {REQUIRED_INSIDE_LOOP} call -- the per-epoch '
                    f'reshuffle of the (fixed) combined batch must stay inside the loop'
                )

    assert found_any_epoch_loop, (
        f'{Path(rel).name}: no `for epoch in range(MAX_EPOCHS):` loop found in any cell -- '
        f'this test may be stale (loop restructured or MAX_EPOCHS renamed)'
    )
    assert not problems, f'{Path(rel).name}:\n  ' + '\n  '.join(problems)


def test_guard_catches_a_sampler_call_left_inside_the_epoch_loop(tmp_path):
    """The guard must actually fail on the shape of regression it exists to catch:
    a sampler call (or a raw replay_rng.choice draw) still inside the loop."""
    nb = {'cells': [
        {'cell_type': 'code', 'source': [
            'MAX_EPOCHS = 15\n',
            'for epoch in range(MAX_EPOCHS):\n',
            '    replay_indices = sample_random_replay_indices(pool_size, target, excluded, replay_rng)\n',
            '    shuffle_idx = replay_rng.permutation(10)\n',
        ]},
    ]}
    path = tmp_path / 'broken.ipynb'
    path.write_text(json.dumps(nb), encoding='utf-8')

    problems = []
    for idx, src in _code_cells(path):
        tree = ast.parse(src)
        for loop in _find_epoch_loops(tree):
            banned_hits = sorted(set(_call_names_in(loop)) & BANNED_SAMPLER_NAMES)
            if banned_hits:
                problems.append((idx, banned_hits))

    assert problems, 'guard failed to flag a sampler call left inside the epoch loop'
    assert problems[0][1] == ['sample_random_replay_indices']


def test_guard_catches_a_raw_replay_rng_choice_left_inside_the_epoch_loop(tmp_path):
    """er_plain and er_misclassification_buffer draw via replay_rng.choice(...)
    directly rather than a named helper -- the guard must catch that shape too."""
    nb = {'cells': [
        {'cell_type': 'code', 'source': [
            'MAX_EPOCHS = 15\n',
            'for epoch in range(MAX_EPOCHS):\n',
            '    replay_indices = replay_rng.choice(pool_size, size=used_size, replace=False)\n',
            '    shuffle_idx = replay_rng.permutation(10)\n',
        ]},
    ]}
    path = tmp_path / 'broken.ipynb'
    path.write_text(json.dumps(nb), encoding='utf-8')

    problems = []
    for idx, src in _code_cells(path):
        tree = ast.parse(src)
        for loop in _find_epoch_loops(tree):
            banned_hits = sorted(set(_call_names_in(loop)) & BANNED_SAMPLER_NAMES)
            if banned_hits:
                problems.append((idx, banned_hits))

    assert problems == [(0, ['replay_rng.choice'])]


def test_guard_catches_missing_permutation_inside_the_epoch_loop(tmp_path):
    """If the reshuffle is accidentally hoisted out alongside the sampling draw,
    every epoch would train on the same (unshuffled) order -- the guard must
    flag an epoch loop with no replay_rng.permutation call in it."""
    nb = {'cells': [
        {'cell_type': 'code', 'source': [
            'MAX_EPOCHS = 15\n',
            'shuffle_idx = replay_rng.permutation(10)\n',  # hoisted out, outside the loop
            'for epoch in range(MAX_EPOCHS):\n',
            '    model.partial_fit(X_chunk, y_chunk)\n',
        ]},
    ]}
    path = tmp_path / 'broken.ipynb'
    path.write_text(json.dumps(nb), encoding='utf-8')

    missing = []
    for idx, src in _code_cells(path):
        tree = ast.parse(src)
        for loop in _find_epoch_loops(tree):
            if REQUIRED_INSIDE_LOOP not in _call_names_in(loop):
                missing.append(idx)

    assert missing == [0]
