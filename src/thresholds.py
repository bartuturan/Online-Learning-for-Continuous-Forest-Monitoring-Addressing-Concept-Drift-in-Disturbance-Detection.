"""Shared thresholding utilities.

The PR-curve threshold search and its metadata bookkeeping were duplicated across
the SGD notebooks, the MLP baseline (twice in one notebook — cells 17 and 24, the
second silently shadowing the first), and Evaluations.ipynb. This module is the one
definition.

Scope note: the notebooks' `resolve_dynamic_threshold` / `get_global_dynamic_threshold`
also read `ds`, `model`, `year_values`, `models_dir`, and four threshold maps out of
`globals()`. The state they keep is captured here by ThresholdRegistry; the parts that
load notebook-specific data stay in the notebooks and pass a callback, because turning
those into module functions would mean inventing a config object for every notebook's
paths and is a bigger change than moving code.

No threshold *semantics* change here. The oracle-vs-production and
future-leakage questions in `to do.md` are a separate decision, made tractable by this
logic being in one place, but deliberately not made here.
"""

import numpy as np
from sklearn.metrics import precision_recall_curve


def make_threshold_meta(source, reason, n_val_samples=0):
    return {
        'source': str(source),
        'reason': str(reason),
        'n_val_samples': int(n_val_samples),
    }


def normalize_threshold_meta(
    meta,
    default_source='fallback_global',
    default_reason='best_threshold/default',
):
    if not isinstance(meta, dict):
        return make_threshold_meta(default_source, default_reason, 0)

    return make_threshold_meta(
        meta.get('source', default_source),
        meta.get('reason', default_reason),
        meta.get('n_val_samples', meta.get('n_val_threshold_samples', 0)),
    )


def compute_best_f1_threshold(
    y_true,
    y_score,
    default_threshold=0.5,
    computed_source='computed_pr_f1',
    context_label='validation data',
):
    """Return the PR-curve threshold maximizing F1, plus metadata on which branch was taken."""
    n_samples = int(len(y_true))

    if n_samples == 0:
        return float(default_threshold), make_threshold_meta(
            'fallback_global',
            f'best_threshold/default: no samples available for {context_label}',
            0,
        )

    if len(np.unique(y_true)) <= 1:
        return float(default_threshold), make_threshold_meta(
            'fallback_global',
            f'{context_label} labels contain one class only',
            n_samples,
        )

    precision_vals, recall_vals, thresholds = precision_recall_curve(y_true, y_score)
    f1_vals = 2 * (precision_vals * recall_vals) / (precision_vals + recall_vals + 1e-12)
    best_idx = int(np.argmax(f1_vals))

    if best_idx < len(thresholds):
        return float(thresholds[best_idx]), make_threshold_meta(
            computed_source,
            f'max F1 on {context_label} PR curve',
            n_samples,
        )

    return float(default_threshold), make_threshold_meta(
        'fallback_global',
        f'PR argmax at terminal point without explicit threshold for {context_label}',
        n_samples,
    )


class ThresholdRegistry:
    """Holds per-(model_year, eval_year) and per-model_year thresholds with their metadata.

    Replaces the four loose notebook globals (`model_eval_year_thresholds`,
    `model_year_thresholds`, and their `_meta` twins) plus the coercion pass that
    `_get_or_create_threshold_maps` ran over them on every call.
    """

    def __init__(self, pair_map=None, model_map=None, pair_meta=None, model_meta=None):
        self.pair_map = self._normalize_pair_values(pair_map)
        self.model_map = self._normalize_model_values(model_map)
        self.pair_meta = self._normalize_pair_meta(pair_meta)
        self.model_meta = self._normalize_model_meta(model_meta)

    @staticmethod
    def _normalize_pair_values(raw):
        out = {}
        if not isinstance(raw, dict):
            return out
        for model_key, eval_map in raw.items():
            try:
                model_key = int(model_key)
            except Exception:
                continue
            if not isinstance(eval_map, dict):
                continue
            out[model_key] = {}
            for eval_key, value in eval_map.items():
                try:
                    out[model_key][int(eval_key)] = float(value)
                except Exception:
                    continue
        return out

    @staticmethod
    def _normalize_model_values(raw):
        out = {}
        if not isinstance(raw, dict):
            return out
        for model_key, value in raw.items():
            try:
                out[int(model_key)] = float(value)
            except Exception:
                continue
        return out

    @staticmethod
    def _normalize_pair_meta(raw):
        out = {}
        if not isinstance(raw, dict):
            return out
        for model_key, eval_meta_map in raw.items():
            try:
                model_key = int(model_key)
            except Exception:
                continue
            if not isinstance(eval_meta_map, dict):
                continue
            out[model_key] = {}
            for eval_key, meta in eval_meta_map.items():
                try:
                    eval_key = int(eval_key)
                except Exception:
                    continue
                out[model_key][eval_key] = normalize_threshold_meta(meta)
        return out

    @staticmethod
    def _normalize_model_meta(raw):
        out = {}
        if not isinstance(raw, dict):
            return out
        for model_key, meta in raw.items():
            try:
                model_key = int(model_key)
            except Exception:
                continue
            out[model_key] = normalize_threshold_meta(meta)
        return out

    def resolve(self, model_year_val, eval_year_val, compute_fn=None, default_threshold=0.5, persist=True):
        """Resolve a threshold, preferring saved values over recomputation.

        Fallback order, unchanged from the notebooks:
          1) pair-specific (model_year, eval_year)
          2) model-year threshold
          3) `compute_fn()` -> (threshold, meta), typically a PR curve on eval-year validation
          4) `default_threshold`

        compute_fn is a zero-argument callable so the caller keeps ownership of loading
        its own dataset, model, and scaler.
        """
        model_key = int(model_year_val)
        eval_key = int(eval_year_val)

        if model_key in self.pair_map and eval_key in self.pair_map[model_key]:
            saved_meta = self.pair_meta.get(model_key, {}).get(
                eval_key,
                make_threshold_meta('pair_specific_saved', 'persisted pair-specific threshold', 0),
            )
            return float(self.pair_map[model_key][eval_key]), normalize_threshold_meta(saved_meta)

        if model_key in self.model_map:
            saved_meta = self.model_meta.get(
                model_key,
                make_threshold_meta('model_year_saved', 'persisted model-year threshold', 0),
            )
            return float(self.model_map[model_key]), normalize_threshold_meta(saved_meta)

        if compute_fn is None:
            return float(default_threshold), make_threshold_meta(
                'fallback_global',
                'best_threshold/default: no compute_fn supplied',
                0,
            )

        threshold, threshold_meta = compute_fn()
        threshold_meta = normalize_threshold_meta(threshold_meta)

        if persist:
            self.pair_map.setdefault(model_key, {})[eval_key] = float(threshold)
            self.pair_meta.setdefault(model_key, {})[eval_key] = threshold_meta.copy()
            if model_key == eval_key:
                self.model_map[model_key] = float(threshold)
                self.model_meta[model_key] = threshold_meta.copy()

        return float(threshold), threshold_meta
