"""Scoring metrics and threshold selection for the unified evaluation pipeline.

Moved verbatim out of Evaluations.ipynb cell 3.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_metrics(y_true, y_pred, y_proba):
    out = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1_score': f1_score(y_true, y_pred, zero_division=0),
        'roc_auc': np.nan,
        'pr_auc': np.nan,
    }
    if len(np.unique(y_true)) > 1:
        out['roc_auc'] = roc_auc_score(y_true, y_proba)
        out['pr_auc'] = average_precision_score(y_true, y_proba)
    return out


def best_f1_threshold(y_true, y_score, default_threshold=0.5):
    """Kept separate from src/thresholds.py on purpose.

    src/thresholds.py computes the same PR-curve argmax and, on ordinary input,
    returns the same threshold -- verified by brute-force search, including inputs
    engineered to hit the difference between this function's
    np.argmax(f1_vals[:-1]) and the shared module's full-array argmax with a bounds
    check; no case was found where the two disagree numerically. Where they do
    disagree is the *fallback* paths (empty input, single-class input): this
    function tags them 'fallback' with short reasons, the shared module tags them
    'fallback_global' with longer ones. Those strings are written into the eval
    output tables (threshold_source, threshold_reason), so unifying the two would
    change published columns for no gain -- this helper is not duplicated in any
    other notebook.
    """
    if len(y_true) == 0 or len(np.unique(y_true)) < 2:
        return float(default_threshold), {'source': 'fallback', 'reason': 'insufficient classes', 'n_val_samples': int(len(y_true))}

    precision_vals, recall_vals, thresholds = precision_recall_curve(y_true, y_score)
    if len(thresholds) == 0:
        return float(default_threshold), {'source': 'fallback', 'reason': 'no thresholds', 'n_val_samples': int(len(y_true))}

    f1_vals = 2 * (precision_vals * recall_vals) / (precision_vals + recall_vals + 1e-12)
    best_idx = int(np.argmax(f1_vals[:-1])) if len(f1_vals) > 1 else 0
    threshold = float(thresholds[min(best_idx, len(thresholds) - 1)])
    return threshold, {
        'source': 'computed_pr_f1',
        'reason': 'max F1 on validation PR curve',
        'n_val_samples': int(len(y_true)),
    }
