import numpy as np

from src.mlp_replay.model import build_mlp_model, compute_optimal_f1_threshold
from src.mlp_replay.replay_strategies import (
    build_replay_year_metadata,
    materialize_replay_samples,
    sample_confidently_correct_replay_indices,
    sample_grouped_indices,
    sample_hard_replay_indices,
    sample_random_replay_indices,
    sample_uncertain_replay_indices,
    update_reservoir,
    weighted_choice_without_replacement,
)


def test_weighted_choice_without_replacement_favors_weighted_indices():
    rng = np.random.default_rng(0)
    indices = np.arange(10)
    weights = np.array([0, 0, 5, 5, 0, 0, 0, 0, 0, 0], dtype=float)
    chosen = weighted_choice_without_replacement(indices, weights, sample_size=2, rng=rng)
    assert set(chosen.tolist()) == {2, 3}


def test_weighted_choice_without_replacement_falls_back_to_uniform_when_all_zero():
    rng = np.random.default_rng(0)
    chosen = weighted_choice_without_replacement(np.arange(10), np.zeros(10), sample_size=3, rng=rng)
    assert len(chosen) == 3


def test_weighted_choice_without_replacement_zero_sample_size():
    rng = np.random.default_rng(0)
    chosen = weighted_choice_without_replacement(np.arange(10), np.ones(10), sample_size=0, rng=rng)
    assert len(chosen) == 0


def test_update_reservoir_fills_then_caps_at_capacity():
    rng = np.random.default_rng(0)
    X_res, y_res, seen = None, None, 0

    X_a = np.arange(20).reshape(10, 2).astype(np.float32)
    y_a = np.arange(10)
    X_res, y_res, seen = update_reservoir(X_res, y_res, X_a, y_a, seen, capacity=5, rng=rng)
    assert X_res.shape == (5, 2)
    assert seen == 10

    X_b = np.arange(20, 40).reshape(10, 2).astype(np.float32)
    y_b = np.arange(10, 20)
    X_res, y_res, seen = update_reservoir(X_res, y_res, X_b, y_b, seen, capacity=5, rng=rng)
    assert X_res.shape == (5, 2)  # capacity never exceeded
    assert seen == 20


def test_update_reservoir_empty_batch_is_noop():
    rng = np.random.default_rng(0)
    X_res, y_res, seen = update_reservoir(None, None, np.empty((0, 2)), np.empty(0), 0, capacity=5, rng=rng)
    assert X_res is None
    assert seen == 0


def test_sample_random_replay_indices_excludes_given_indices():
    rng = np.random.default_rng(0)
    chosen = sample_random_replay_indices(
        replay_pool_size=100, random_target_size=10, excluded_indices=np.array([1, 2, 3]), rng=rng,
    )
    assert len(chosen) == 10
    assert not set(chosen.tolist()) & {1, 2, 3}


def test_sample_random_replay_indices_zero_target():
    rng = np.random.default_rng(0)
    chosen = sample_random_replay_indices(replay_pool_size=100, random_target_size=0, excluded_indices=np.array([]), rng=rng)
    assert len(chosen) == 0


def _fit_tiny_model(rng, n=40, n_features=4):
    X = rng.normal(size=(n, n_features)).astype(np.float32)
    y = rng.integers(0, 2, size=n)
    model = build_mlp_model()
    model.partial_fit(X[: n // 2], y[: n // 2], classes=np.array([0, 1]))
    return model, X, y


def test_sample_hard_replay_indices_respects_target_size():
    rng = np.random.default_rng(1)
    model, X_replay, y_replay = _fit_tiny_model(rng, n=40)
    replay_year_spans = [(1, 0, 20), (2, 20, 40)]

    indices, thresholds = sample_hard_replay_indices(
        model, X_replay, y_replay, replay_year_spans, hard_target_size=8, rng=rng,
    )
    assert len(indices) == 8
    assert len(thresholds) == 2


def test_sample_hard_replay_indices_accepts_legacy_threshold_grid():
    rng = np.random.default_rng(1)
    model, X_replay, y_replay = _fit_tiny_model(rng, n=40)
    replay_year_spans = [(1, 0, 20), (2, 20, 40)]

    indices, _ = sample_hard_replay_indices(
        model, X_replay, y_replay, replay_year_spans, hard_target_size=8, rng=rng,
        threshold_grid=np.linspace(0, 1, 11),
    )
    assert len(indices) == 8


def test_sample_uncertain_replay_indices_respects_target_size():
    rng = np.random.default_rng(2)
    model, X_replay, y_replay = _fit_tiny_model(rng, n=40)
    replay_year_spans = [(1, 0, 20), (2, 20, 40)]

    indices, thresholds = sample_uncertain_replay_indices(
        model, X_replay, y_replay, replay_year_spans, uncertain_target_size=8, rng=rng, gaussian_std=0.1,
    )
    assert len(indices) == 8
    assert len(thresholds) == 2


def test_sample_confidently_correct_replay_indices_respects_target_size():
    rng = np.random.default_rng(3)
    model, X_replay, y_replay = _fit_tiny_model(rng, n=40)
    replay_year_spans = [(1, 0, 20), (2, 20, 40)]

    indices, thresholds = sample_confidently_correct_replay_indices(
        model, X_replay, y_replay, replay_year_spans, confident_target_size=8, rng=rng,
    )
    assert len(indices) == 8
    assert len(thresholds) == 2


def test_replay_index_samplers_zero_target_return_empty():
    rng = np.random.default_rng(4)
    model, X_replay, y_replay = _fit_tiny_model(rng, n=40)
    spans = [(1, 0, 20), (2, 20, 40)]

    hard_idx, hard_th = sample_hard_replay_indices(model, X_replay, y_replay, spans, 0, rng)
    assert len(hard_idx) == 0 and hard_th == []

    uncertain_idx, uncertain_th = sample_uncertain_replay_indices(model, X_replay, y_replay, spans, 0, rng, gaussian_std=0.1)
    assert len(uncertain_idx) == 0 and uncertain_th == []

    confident_idx, confident_th = sample_confidently_correct_replay_indices(model, X_replay, y_replay, spans, 0, rng)
    assert len(confident_idx) == 0 and confident_th == []


def test_sample_hard_replay_indices_matches_original_inline_logic_bit_for_bit():
    """Regression guard: the shared _allocate_per_year_quota/_top_up_shortfall
    helpers must reproduce the exact RNG consumption order of the original
    per-notebook inline implementation, or resumed/seeded runs would silently
    diverge from historical results."""

    def original_sample_hard_replay_indices(model, X_replay_pool, y_replay_pool, replay_year_spans, hard_target_size, rng, threshold_grid):
        if hard_target_size <= 0 or len(replay_year_spans) == 0:
            return np.empty((0,), dtype=np.int64), []

        n_years_with_data = len(replay_year_spans)
        base_quota = hard_target_size // n_years_with_data
        remainder = hard_target_size % n_years_with_data

        per_year_targets = []
        for idx, year_span in enumerate(replay_year_spans):
            year_idx, start, end = year_span
            year_target = base_quota + (1 if idx < remainder else 0)
            year_size = end - start
            per_year_targets.append((year_idx, start, end, min(year_target, year_size)))

        selected_indices = []
        threshold_values = []
        for year_idx, start, end, year_target in per_year_targets:
            if year_target <= 0:
                continue
            year_indices = np.arange(start, end, dtype=np.int64)
            y_year = y_replay_pool[start:end]
            y_proba_year = model.predict_proba(X_replay_pool[start:end])[:, 1]
            threshold = compute_optimal_f1_threshold(y_true=y_year, y_proba=y_proba_year, threshold_grid=threshold_grid, default_threshold=0.5)
            threshold_values.append(threshold)
            y_pred_year = (y_proba_year >= threshold).astype(int)
            incorrect_mask = (y_pred_year != y_year)
            hard_weights = np.where(incorrect_mask, np.abs(y_year - y_proba_year), 0.0)
            chosen = weighted_choice_without_replacement(indices=year_indices, weights=hard_weights, sample_size=year_target, rng=rng)
            if len(chosen) > 0:
                selected_indices.append(chosen)

        if selected_indices:
            hard_indices = np.concatenate(selected_indices).astype(np.int64, copy=False)
        else:
            hard_indices = np.empty((0,), dtype=np.int64)

        shortfall = int(hard_target_size) - len(hard_indices)
        if shortfall > 0:
            all_indices = np.arange(len(y_replay_pool), dtype=np.int64)
            remaining_indices = np.setdiff1d(all_indices, hard_indices, assume_unique=False)
            if len(remaining_indices) > 0:
                top_up = rng.choice(remaining_indices, size=min(shortfall, len(remaining_indices)), replace=False)
                hard_indices = np.concatenate([hard_indices, top_up.astype(np.int64, copy=False)])
        return hard_indices, threshold_values

    base_rng = np.random.default_rng(123)
    X_replay = base_rng.normal(size=(60, 5)).astype(np.float32)
    y_replay = base_rng.integers(0, 2, size=60)
    model = build_mlp_model()
    model.partial_fit(X_replay[:30], y_replay[:30], classes=np.array([0, 1]))
    replay_year_spans = [(1, 0, 20), (2, 20, 45), (3, 45, 60)]

    rng_a = np.random.default_rng(999)
    rng_b = np.random.default_rng(999)

    idx_a, th_a = original_sample_hard_replay_indices(model, X_replay, y_replay, replay_year_spans, 25, rng_a, None)
    idx_b, th_b = sample_hard_replay_indices(model, X_replay, y_replay, replay_year_spans, 25, rng_b)

    assert np.array_equal(idx_a, idx_b)
    assert th_a == th_b
    # confirm both RNGs are left in identical internal state (consumption order matches exactly)
    assert np.array_equal(rng_a.random(5), rng_b.random(5))


def test_build_replay_year_metadata_and_sample_and_materialize_pipeline():
    rng = np.random.default_rng(5)
    cache = {
        1: (rng.normal(size=(15, 4)).astype(np.float32), rng.integers(0, 2, size=15)),
        2: (rng.normal(size=(10, 4)).astype(np.float32), rng.integers(0, 2, size=10)),
    }
    metadata, pool_size, pos_size, neg_size = build_replay_year_metadata(cache, current_year_idx=3)
    assert pool_size == 25
    assert pos_size + neg_size == 25

    sources = [(item["year_idx"], item["positive_indices"]) for item in metadata if len(item["positive_indices"]) > 0]
    total_pos = sum(len(idxs) for _, idxs in sources)
    sample_size = min(3, total_pos)
    grouped = sample_grouped_indices(sources, sample_size=sample_size, rng=rng)
    assert sum(len(idxs) for _, idxs in grouped) == sample_size

    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler().fit(np.vstack([cache[1][0], cache[2][0]]))
    X_mat, y_mat = materialize_replay_samples(grouped, cache, scaler)
    assert X_mat.shape[0] == y_mat.shape[0] == sample_size


def test_sample_grouped_indices_empty_inputs():
    rng = np.random.default_rng(0)
    assert sample_grouped_indices([], sample_size=5, rng=rng) == []
    assert sample_grouped_indices([(1, np.array([1, 2]))], sample_size=0, rng=rng) == []


def test_materialize_replay_samples_empty_input():
    X, y = materialize_replay_samples([], cache={}, scaler=None)
    assert X.shape == (0, 0)
    assert y.shape == (0,)
