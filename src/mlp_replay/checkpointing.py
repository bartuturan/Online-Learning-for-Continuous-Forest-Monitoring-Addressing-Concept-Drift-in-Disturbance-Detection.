import json
import os
import pickle
from datetime import datetime

import numpy as np

TRAINING_HISTORY_BASE_KEYS = [
    'year',
    'train_accuracy',
    'train_precision',
    'train_recall',
    'train_f1',
    'val_accuracy',
    'val_precision',
    'val_recall',
    'val_f1',
    'val_roc_auc',
    'val_pr_auc',
    'replay_pool_size',
    'replay_target_size',
    'replay_used_size',
]


def create_empty_training_history(extra_keys=None):
    history = {key: [] for key in TRAINING_HISTORY_BASE_KEYS}
    for key in extra_keys or []:
        history[key] = []
    return history


def load_all_training_histories(path):
    if path.exists():
        with open(path, 'rb') as f:
            data = pickle.load(f)
        if isinstance(data, dict):
            return data
    return {}


def save_all_training_histories(path, all_histories):
    # Write to a temp file and rename into place -- os.replace is atomic, so a
    # crash mid-write (e.g. the MemoryErrors this pipeline has actually hit)
    # can only ever leave the .tmp file half-written, never the real path.
    tmp_path = path.parent / (path.name + '.tmp')
    with open(tmp_path, 'wb') as f:
        pickle.dump(all_histories, f)
    os.replace(tmp_path, path)


def load_completion_status(path, extra_default_keys=None):
    default_status = {'completed_ratios': [], 'completed_years': {}}
    for key in extra_default_keys or []:
        default_status[key] = {}

    if path.exists():
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if isinstance(data, dict):
            if 'completed_ratios' not in data:
                data['completed_ratios'] = []
            if 'completed_years' not in data:
                data['completed_years'] = {}
            for key in extra_default_keys or []:
                if key not in data:
                    data[key] = {}
            return data
    return default_status


def save_completion_status(path, status):
    status['completed_ratios'] = sorted(list(set(status.get('completed_ratios', []))))
    cleaned_completed_years = {}
    for key, years in status.get('completed_years', {}).items():
        cleaned_completed_years[key] = sorted(list(set(int(y) for y in years)))
    status['completed_years'] = cleaned_completed_years

    tmp_path = path.parent / (path.name + '.tmp')
    with open(tmp_path, 'w', encoding='utf-8') as f:
        json.dump(status, f, indent=2)
    os.replace(tmp_path, path)


def append_training_log(path, message):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with open(path, 'a', encoding='utf-8') as f:
        f.write(f'[{timestamp}] {message}\n')


def format_ratio_key(replay_ratio):
    return f'RR_{replay_ratio:.1f}'


def per_ratio_path(base_path, ratio_key):
    """Derive a ratio-scoped file path from a shared base path.

    Lets 4 replay-ratio processes run concurrently without racing on the same
    checkpoint/history/log file -- each ratio gets its own, named by inserting
    ratio_key before the extension, so whatever suffix/token a notebook already
    bakes into its base filename is preserved untouched.
    """
    return base_path.with_name(f'{base_path.stem}_{ratio_key}{base_path.suffix}')


def format_float_token(value, decimals=3):
    token = f'{float(value):.{decimals}f}'
    token = token.rstrip('0').rstrip('.')
    return token if token else '0'


def sanitize_for_windows_filename(text):
    # Windows forbidden chars: <>:"/\|?*
    forbidden = '<>:"/\\|?*'
    out = ''.join('-' if ch in forbidden else ch for ch in str(text))
    return out.strip(' .')


def format_combined_strategy_suffix(
    hard_example, confidently_correct, uncertainty_prioritization,
    positive_rate, misclassification_buffer, replay_weight_scale,
):
    """Reproduces MLP_experience_replay_combined.ipynb's build_strategy_suffix() exactly --
    the notebook imports this rather than keeping its own copy, so the two can never drift
    (see format_combined_run_key's docstring for why that drift is exactly the bug this exists
    to prevent)."""
    pr_buffer_fraction, pr_target_percent = positive_rate
    token = (
        f'_combined_HE={format_float_token(hard_example)}'
        f'_CC={format_float_token(confidently_correct)}'
        f'_UP={format_float_token(uncertainty_prioritization)}'
        f'_PR=({format_float_token(pr_buffer_fraction)},{format_float_token(pr_target_percent, decimals=1)})'
        f'_MC={format_float_token(misclassification_buffer)}'
        f'_RWS={format_float_token(replay_weight_scale)}'
    )
    return sanitize_for_windows_filename(token)


def format_combined_run_key(
    ratio, hard_example, confidently_correct, uncertainty_prioritization,
    positive_rate, misclassification_buffer, replay_weight_scale,
):
    """Reproduces the exact key MLP_experience_replay_combined.ipynb appends to
    completion_status['completed_ratios'] (its run_key: format_ratio_key(ratio) +
    build_ratio_suffix(ratio)) -- NOT the bare ratio key every other replay notebook uses.
    This notebook's run_key is combo+ratio-composite because CHECKPOINT_DIR is shared across
    every hyperparameter combo (only the filenames inside it are combo-scoped), so a bare
    "RR_0.5" would be ambiguous between combos sharing that ratio. A completion check must
    match against this composite key, or it will report TODO forever even after a genuinely
    successful run."""
    ratio_key = format_ratio_key(ratio)
    suffix = format_combined_strategy_suffix(
        hard_example, confidently_correct, uncertainty_prioritization,
        positive_rate, misclassification_buffer, replay_weight_scale,
    )
    ratio_suffix = sanitize_for_windows_filename(f'{suffix}_RR={format_float_token(ratio)}')
    return f'{ratio_key}{ratio_suffix}'


def get_cached_raw_year(cache, year_idx):
    if year_idx not in cache:
        return np.empty((0, 0), dtype=np.float32), np.empty((0,), dtype=np.int64)
    return cache[year_idx]
