import json
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
    with open(path, 'wb') as f:
        pickle.dump(all_histories, f)


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

    with open(path, 'w', encoding='utf-8') as f:
        json.dump(status, f, indent=2)


def append_training_log(path, message):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with open(path, 'a', encoding='utf-8') as f:
        f.write(f'[{timestamp}] {message}\n')


def format_ratio_key(replay_ratio):
    return f'RR_{replay_ratio:.1f}'


def get_cached_raw_year(cache, year_idx):
    if year_idx not in cache:
        return np.empty((0, 0), dtype=np.float32), np.empty((0,), dtype=np.int64)
    return cache[year_idx]
