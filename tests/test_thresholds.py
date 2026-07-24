import numpy as np
import pytest

from src.mlp_replay.model import compute_optimal_f1_threshold
from src.thresholds import (
    ThresholdRegistry,
    compute_best_f1_threshold,
    make_threshold_meta,
    normalize_threshold_meta,
)


@pytest.fixture
def separable_scores():
    """Scores where a threshold near 0.5 cleanly separates the classes."""
    y_true = np.array([0] * 50 + [1] * 50)
    y_score = np.concatenate([np.linspace(0.01, 0.45, 50), np.linspace(0.55, 0.99, 50)])
    return y_true, y_score


def test_returns_threshold_that_separates(separable_scores):
    y_true, y_score = separable_scores
    threshold, meta = compute_best_f1_threshold(y_true, y_score)
    assert 0.45 < threshold <= 0.55
    assert meta["source"] == "computed_pr_f1"
    assert meta["n_val_samples"] == 100


def test_empty_input_falls_back(separable_scores):
    threshold, meta = compute_best_f1_threshold(np.array([]), np.array([]), default_threshold=0.42)
    assert threshold == 0.42
    assert meta["source"] == "fallback_global"
    assert meta["n_val_samples"] == 0


def test_single_class_falls_back():
    threshold, meta = compute_best_f1_threshold(
        np.zeros(20), np.linspace(0, 1, 20), default_threshold=0.33)
    assert threshold == 0.33
    assert "one class only" in meta["reason"]
    assert meta["n_val_samples"] == 20


def test_context_label_and_source_reach_the_metadata(separable_scores):
    y_true, y_score = separable_scores
    _, meta = compute_best_f1_threshold(
        y_true, y_score, computed_source="global_computed_pr_f1", context_label="global validation data")
    assert meta["source"] == "global_computed_pr_f1"
    assert "global validation data" in meta["reason"]


def test_mlp_replay_wrapper_agrees_with_the_shared_implementation(separable_scores):
    """compute_optimal_f1_threshold must stay a pure view of compute_best_f1_threshold."""
    y_true, y_score = separable_scores
    expected, _ = compute_best_f1_threshold(y_true, y_score)
    assert compute_optimal_f1_threshold(y_true, y_score) == expected

    for y, s in [(np.array([]), np.array([])), (np.zeros(10), np.linspace(0, 1, 10))]:
        assert compute_optimal_f1_threshold(y, s, default_threshold=0.37) == 0.37


def test_normalize_accepts_the_legacy_sample_count_key():
    meta = normalize_threshold_meta({"source": "s", "reason": "r", "n_val_threshold_samples": 7})
    assert meta["n_val_samples"] == 7


def test_normalize_replaces_non_dict_input():
    meta = normalize_threshold_meta(None, default_source="src", default_reason="why")
    assert meta == make_threshold_meta("src", "why", 0)


class TestThresholdRegistry:
    def test_coerces_string_keys_and_values(self):
        reg = ThresholdRegistry(pair_map={"2019": {"2020": "0.4"}}, model_map={"2019": "0.6"})
        assert reg.pair_map == {2019: {2020: 0.4}}
        assert reg.model_map == {2019: 0.6}

    def test_discards_uncoercible_entries(self):
        reg = ThresholdRegistry(pair_map={"nope": {"2020": 0.4}, "2019": "not-a-dict"},
                                model_map={"2019": "not-a-float"})
        assert reg.pair_map == {}
        assert reg.model_map == {}

    def test_pair_specific_wins_over_model_year(self):
        reg = ThresholdRegistry(pair_map={2019: {2020: 0.4}}, model_map={2019: 0.6})
        threshold, meta = reg.resolve(2019, 2020, compute_fn=lambda: (0.9, {}))
        assert threshold == 0.4
        assert meta["source"] == "pair_specific_saved"

    def test_model_year_used_when_no_pair_entry(self):
        reg = ThresholdRegistry(model_map={2019: 0.6})
        threshold, meta = reg.resolve(2019, 2021, compute_fn=lambda: (0.9, {}))
        assert threshold == 0.6
        assert meta["source"] == "model_year_saved"

    def test_falls_through_to_compute_fn(self):
        reg = ThresholdRegistry()
        threshold, meta = reg.resolve(
            2019, 2021, compute_fn=lambda: (0.77, make_threshold_meta("computed_pr_f1", "r", 5)))
        assert threshold == 0.77
        assert meta["source"] == "computed_pr_f1"

    def test_missing_compute_fn_falls_back_to_default(self):
        reg = ThresholdRegistry()
        threshold, meta = reg.resolve(2019, 2021, default_threshold=0.31)
        assert threshold == 0.31
        assert meta["source"] == "fallback_global"

    def test_persist_records_pair_and_only_self_year_as_model_threshold(self):
        reg = ThresholdRegistry()
        reg.resolve(2019, 2021, compute_fn=lambda: (0.77, {}))
        assert reg.pair_map == {2019: {2021: 0.77}}
        assert reg.model_map == {}, "a cross-year threshold must not become the model-year default"

        reg.resolve(2019, 2019, compute_fn=lambda: (0.55, {}))
        assert reg.model_map == {2019: 0.55}

    def test_persist_false_leaves_registry_untouched(self):
        reg = ThresholdRegistry()
        reg.resolve(2019, 2021, compute_fn=lambda: (0.77, {}), persist=False)
        assert reg.pair_map == {} and reg.model_map == {}

    def test_second_resolve_reuses_the_persisted_value(self):
        reg = ThresholdRegistry()
        calls = []

        def compute():
            calls.append(1)
            return 0.77, {}

        reg.resolve(2019, 2021, compute_fn=compute)
        reg.resolve(2019, 2021, compute_fn=compute)
        assert len(calls) == 1
