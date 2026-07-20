import numpy as np
import pytest

from src.mlp_replay.model import (
    build_mlp_model,
    capture_model_state,
    compute_binary_class_weights,
    compute_optimal_f1_threshold,
    compute_year_positive_rate,
    restore_model_state,
)


def test_build_mlp_model_config():
    model = build_mlp_model()
    assert model.hidden_layer_sizes == (64,)
    assert model.solver == "adam"
    assert model.max_iter == 1
    assert model.warm_start is False


def test_compute_year_positive_rate_default_label():
    assert np.isnan(compute_year_positive_rate(np.array([])))
    assert compute_year_positive_rate(np.array([0, 0, 1, 1])) == 0.5


def test_compute_year_positive_rate_explicit_positive_label():
    # regression test: an earlier version of this function silently dropped
    # positive_label and broke every caller that passed it explicitly.
    assert compute_year_positive_rate(np.array([0, 0, 1, 1]), positive_label=1) == 0.5
    assert compute_year_positive_rate(np.array([2, 2, 5, 5]), positive_label=5) == 0.5
    assert compute_year_positive_rate(np.array([2, 2, 5, 5]), positive_label=2) == 0.5


def test_compute_binary_class_weights_balanced():
    weights, mode = compute_binary_class_weights(np.array([0, 0, 0, 1]))
    assert mode == "balanced"
    assert weights[1] > weights[0]  # minority class gets higher weight


def test_compute_binary_class_weights_single_class_fallback():
    weights, mode = compute_binary_class_weights(np.array([0, 0, 0]))
    assert mode.startswith("smoothed_single_class_present_0")
    assert weights[0] < weights[1]  # absent class (1) gets the higher smoothed weight


def test_compute_binary_class_weights_empty():
    weights, mode = compute_binary_class_weights(np.array([]))
    assert mode == "empty_uniform"
    assert weights == {0: 1.0, 1: 1.0}


def test_compute_binary_class_weights_rejects_non_binary_labels():
    with pytest.raises(ValueError, match="Expected binary labels"):
        compute_binary_class_weights(np.array([0, 1, 2]))


def test_compute_binary_class_weights_rejects_unknown_fallback_mode():
    with pytest.raises(ValueError, match="Unsupported fallback_mode"):
        compute_binary_class_weights(np.array([0, 0, 0]), fallback_mode="bogus")


def test_compute_optimal_f1_threshold_degenerate_inputs():
    assert compute_optimal_f1_threshold(np.array([]), np.array([])) == 0.5
    assert compute_optimal_f1_threshold(np.array([1, 1, 1]), np.array([0.9, 0.8, 0.7])) == 0.5


def test_compute_optimal_f1_threshold_in_range():
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 2, size=200)
    y_proba = rng.random(200)
    threshold = compute_optimal_f1_threshold(y_true, y_proba)
    assert 0.0 <= threshold <= 1.0


def test_compute_optimal_f1_threshold_accepts_legacy_threshold_grid_arg():
    # regression test: the majority of original notebooks called this with a
    # (silently unused) threshold_grid positional argument.
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 2, size=200)
    y_proba = rng.random(200)
    threshold = compute_optimal_f1_threshold(y_true, y_proba, np.linspace(0, 1, 11))
    assert 0.0 <= threshold <= 1.0


def test_capture_and_restore_model_state_roundtrip():
    rng = np.random.default_rng(0)
    model = build_mlp_model()
    X0 = rng.normal(size=(50, 4)).astype(np.float32)
    y0 = rng.integers(0, 2, size=50)
    model.partial_fit(X0, y0, classes=np.array([0, 1]))

    snapshot = capture_model_state(model)
    snapshot_coefs = [c.copy() for c in snapshot["coefs"]]

    X1 = rng.normal(size=(50, 4)).astype(np.float32)
    y1 = rng.integers(0, 2, size=50)
    model.partial_fit(X1, y1)

    assert any(not np.array_equal(a, b) for a, b in zip(model.coefs_, snapshot_coefs))

    restore_model_state(model, snapshot)
    assert all(np.array_equal(a, b) for a, b in zip(model.coefs_, snapshot_coefs))
    if snapshot["optimizer_state"] is not None:
        assert model._optimizer.t == snapshot["optimizer_state"]["t"]
