import numpy as np
import pytest

from src.eval.metrics import best_f1_threshold, compute_metrics


class TestComputeMetrics:
    def test_multiclass_computes_all_six(self):
        y_true = np.array([0, 0, 1, 1, 1])
        y_pred = np.array([0, 1, 1, 1, 0])
        y_proba = np.array([0.1, 0.6, 0.9, 0.8, 0.3])
        out = compute_metrics(y_true, y_pred, y_proba)
        assert set(out) == {'accuracy', 'precision', 'recall', 'f1_score', 'roc_auc', 'pr_auc'}
        assert not np.isnan(out['roc_auc'])
        assert not np.isnan(out['pr_auc'])
        assert out['accuracy'] == pytest.approx(3 / 5)

    def test_single_class_true_gives_nan_auc(self):
        """roc_auc/pr_auc are undefined with one class present; the other four are not."""
        y_true = np.zeros(5)
        y_pred = np.array([0, 1, 0, 0, 1])
        y_proba = np.array([0.1, 0.6, 0.2, 0.3, 0.7])
        out = compute_metrics(y_true, y_pred, y_proba)
        assert np.isnan(out['roc_auc'])
        assert np.isnan(out['pr_auc'])
        assert not np.isnan(out['accuracy'])
        assert not np.isnan(out['precision'])

    def test_zero_division_does_not_raise(self):
        """No predicted positives at all must return 0, not raise or warn-crash."""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.zeros(4)
        y_proba = np.array([0.1, 0.2, 0.3, 0.4])
        out = compute_metrics(y_true, y_pred, y_proba)
        assert out['precision'] == 0.0
        assert out['recall'] == 0.0
        assert out['f1_score'] == 0.0


class TestBestF1Threshold:
    def test_empty_input_falls_back(self):
        threshold, meta = best_f1_threshold(np.array([]), np.array([]), default_threshold=0.42)
        assert threshold == 0.42
        assert meta == {'source': 'fallback', 'reason': 'insufficient classes', 'n_val_samples': 0}

    def test_single_class_falls_back(self):
        threshold, meta = best_f1_threshold(np.zeros(10), np.linspace(0, 1, 10), default_threshold=0.33)
        assert threshold == 0.33
        assert meta['source'] == 'fallback'
        assert meta['reason'] == 'insufficient classes'
        assert meta['n_val_samples'] == 10

    def test_separable_scores_find_a_threshold_between_classes(self):
        y_true = np.array([0] * 20 + [1] * 20)
        y_score = np.concatenate([np.linspace(0.0, 0.4, 20), np.linspace(0.6, 1.0, 20)])
        threshold, meta = best_f1_threshold(y_true, y_score)
        assert 0.4 <= threshold <= 0.6
        assert meta['source'] == 'computed_pr_f1'
        assert meta['reason'] == 'max F1 on validation PR curve'
        assert meta['n_val_samples'] == 40

    def test_these_exact_fallback_strings_are_published_columns(self):
        """threshold_source / threshold_reason land verbatim in eval output tables."""
        _, meta_empty = best_f1_threshold(np.array([]), np.array([]))
        assert meta_empty['source'] == 'fallback'
        assert meta_empty['reason'] == 'insufficient classes'

        _, meta_single = best_f1_threshold(np.ones(3), np.array([0.1, 0.2, 0.3]))
        assert meta_single['source'] == 'fallback'
        assert meta_single['reason'] == 'insufficient classes'


class TestDivergenceFromSharedThresholds:
    """Pins the real, verified difference from src.thresholds -- and rules out a false one.

    A brute-force search (70,000+ cases, including inputs engineered to hit the gap
    between this module's np.argmax(f1_vals[:-1]) and the shared module's full-array
    argmax with a bounds check) found no case where the two disagree on the numeric
    threshold. They agree on ordinary computed input. They only disagree on the
    fallback paths, in the metadata vocabulary -- and those strings are published
    columns (threshold_source, threshold_reason), which is why the two are kept
    separate rather than unified.
    """

    def test_agree_numerically_on_computed_input(self):
        from src.thresholds import compute_best_f1_threshold

        y_true = np.array([0, 0, 0, 1])
        y_score = np.array([0.1, 0.2, 0.3, 0.9])
        local_threshold, _ = best_f1_threshold(y_true, y_score)
        shared_threshold, _ = compute_best_f1_threshold(y_true, y_score)
        assert local_threshold == shared_threshold

    def test_disagree_in_fallback_vocabulary_on_empty_input(self):
        from src.thresholds import compute_best_f1_threshold

        _, local_meta = best_f1_threshold(np.array([]), np.array([]))
        _, shared_meta = compute_best_f1_threshold(np.array([]), np.array([]))
        assert local_meta['source'] == 'fallback'
        assert shared_meta['source'] == 'fallback_global'
        assert local_meta['reason'] != shared_meta['reason']

    def test_disagree_in_fallback_vocabulary_on_single_class_input(self):
        from src.thresholds import compute_best_f1_threshold

        y_true, y_score = np.zeros(5), np.linspace(0, 1, 5)
        _, local_meta = best_f1_threshold(y_true, y_score)
        _, shared_meta = compute_best_f1_threshold(y_true, y_score)
        assert local_meta['source'] == 'fallback'
        assert shared_meta['source'] == 'fallback_global'
        assert local_meta['reason'] != shared_meta['reason']
