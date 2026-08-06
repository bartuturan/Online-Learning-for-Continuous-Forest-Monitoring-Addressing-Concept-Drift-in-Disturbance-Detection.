import tracemalloc

import numpy as np
import pytest

from src.mlp_replay.model import build_mlp_model, compute_optimal_f1_threshold
from src.mlp_replay.replay_strategies import (
    build_replay_year_metadata,
    build_replay_year_spans,
    group_global_indices,
    materialize_replay_samples,
    materialize_replay_samples_in_order,
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


# ---------------------------------------------------------------------------
# Streaming replay pool.
#
# The notebooks used to build the replay pool with np.vstack over every prior
# year -- ~3.6GB at the last training year on the real dataset, ~7.2GB counting
# the duplicate, which is what the Kaggle OOM killer was reacting to. The
# streaming path never materializes the pool. These tests exist to prove that
# swapping the access path changes NOTHING about which rows get selected, in
# which order, or how much of the RNG stream is consumed -- the whole change is
# only supposed to be about memory.
# ---------------------------------------------------------------------------

def _multi_year_cache(rng, year_sizes, n_features=6):
    """{year_idx: (X_raw, y)} plus the concatenated pool the old code would build."""
    cache = {}
    for offset, size in enumerate(year_sizes):
        year_idx = offset + 1
        cache[year_idx] = (
            rng.normal(size=(size, n_features)).astype(np.float32),
            rng.integers(0, 2, size=size),
        )
    return cache


def _scaler_for(cache):
    from sklearn.preprocessing import StandardScaler
    return StandardScaler().fit(np.vstack([X for X, _ in cache.values()]))


def _contiguous_pool(cache, scaler, current_year_idx):
    """Exactly what the deleted inline block produced: spans, scaled pool, labels."""
    X_parts, y_parts, spans, cursor = [], [], [], 0
    for year_idx in range(1, current_year_idx):
        X_raw, y_year = cache[year_idx]
        if len(X_raw) == 0:
            continue
        X_parts.append(scaler.transform(X_raw))
        y_parts.append(y_year)
        spans.append((year_idx, cursor, cursor + len(y_year)))
        cursor += len(y_year)
    return spans, np.vstack(X_parts), np.concatenate(y_parts)


SCORED_SAMPLERS = {
    "hard": (sample_hard_replay_indices, {}),
    "uncertain": (sample_uncertain_replay_indices, {"gaussian_std": 0.1}),
    "confident": (sample_confidently_correct_replay_indices, {}),
}


@pytest.mark.parametrize("name", sorted(SCORED_SAMPLERS))
def test_streaming_path_selects_exactly_what_the_contiguous_pool_selects(name):
    sampler, extra = SCORED_SAMPLERS[name]
    fixture_rng = np.random.default_rng(11)
    cache = _multi_year_cache(fixture_rng, [30, 25, 20])
    scaler = _scaler_for(cache)
    spans, X_pool, y_pool = _contiguous_pool(cache, scaler, current_year_idx=4)

    model = build_mlp_model()
    model.partial_fit(X_pool[:40], y_pool[:40], classes=np.array([0, 1]))

    rng_pool = np.random.default_rng(7)
    idx_pool, th_pool = sampler(
        model, X_pool, y_pool, spans, 18, rng_pool, **extra,
    )

    rng_stream = np.random.default_rng(7)
    idx_stream, th_stream = sampler(
        model, None, y_pool, spans, 18, rng_stream, **extra,
        load_scaled_year=lambda year_idx: scaler.transform(cache[year_idx][0]),
    )

    assert np.array_equal(idx_pool, idx_stream)
    assert th_pool == th_stream
    # Same rows AND same RNG consumption -- a streaming run is byte-identical to
    # the run it replaces, so existing checkpoints stay comparable.
    assert np.array_equal(rng_pool.random(5), rng_stream.random(5))


@pytest.mark.parametrize("name", sorted(SCORED_SAMPLERS))
def test_streaming_path_loads_each_year_exactly_once(name):
    """The memory win depends on years being visited one at a time and released.
    A sampler that loaded a year twice would still be correct but would have
    silently reintroduced the cost this change exists to remove."""
    sampler, extra = SCORED_SAMPLERS[name]
    fixture_rng = np.random.default_rng(12)
    cache = _multi_year_cache(fixture_rng, [30, 25, 20])
    scaler = _scaler_for(cache)
    spans, X_pool, y_pool = _contiguous_pool(cache, scaler, current_year_idx=4)

    model = build_mlp_model()
    model.partial_fit(X_pool[:40], y_pool[:40], classes=np.array([0, 1]))

    loads = []

    def load_scaled_year(year_idx):
        loads.append(year_idx)
        return scaler.transform(cache[year_idx][0])

    sampler(
        model, None, y_pool, spans, 18, np.random.default_rng(7), **extra,
        load_scaled_year=load_scaled_year,
    )
    assert sorted(loads) == [1, 2, 3]
    assert len(loads) == len(set(loads))


def test_streaming_sampler_peak_memory_stays_near_one_year():
    """A/B the two access paths on identical work. tracemalloc traces numpy's
    allocations, so the contiguous path's vstack shows up and the streaming
    path's absence of it shows up too.

    The fixture is deliberately feature-heavy (48 columns over 6 years). The real
    dataset is 32 columns over ~5.6M rows per year, where the pool dwarfs the
    index bookkeeping; at toy row counts the fixed costs (_top_up_shortfall's
    np.arange/setdiff1d over the pool index space) would otherwise dominate and
    hide the very thing being measured.
    """
    n_rows, n_features, n_years = 20000, 48, 6
    fixture_rng = np.random.default_rng(13)
    cache = _multi_year_cache(fixture_rng, [n_rows] * n_years, n_features=n_features)
    scaler = _scaler_for(cache)
    spans, X_pool, y_pool = _contiguous_pool(cache, scaler, current_year_idx=n_years + 1)

    model = build_mlp_model()
    model.partial_fit(X_pool[:200], y_pool[:200], classes=np.array([0, 1]))

    def _peak_bytes(fn):
        tracemalloc.start()
        try:
            fn()
            return tracemalloc.get_traced_memory()[1]
        finally:
            tracemalloc.stop()

    def contiguous():
        spans_c, X_c, y_c = _contiguous_pool(cache, scaler, current_year_idx=n_years + 1)
        sample_hard_replay_indices(model, X_c, y_c, spans_c, 3000, np.random.default_rng(3))

    def streaming():
        sample_hard_replay_indices(
            model, None, y_pool, spans, 3000, np.random.default_rng(3),
            load_scaled_year=lambda year_idx: scaler.transform(cache[year_idx][0]),
        )

    peak_contiguous = _peak_bytes(contiguous)
    peak_streaming = _peak_bytes(streaming)

    # StandardScaler.transform preserves float32, so a "year" is 4 bytes/value.
    # The remaining headroom is _top_up_shortfall's np.arange/setdiff1d over the
    # pool index space plus predict_proba's output, none of which scale with the
    # number of years.
    one_year_bytes = n_rows * n_features * 4
    assert peak_streaming < peak_contiguous / 2, (
        f"streaming peak {peak_streaming:,} not meaningfully below contiguous {peak_contiguous:,}"
    )
    assert peak_streaming < one_year_bytes * 4, (
        f"streaming peak {peak_streaming:,} exceeds 4x one year ({one_year_bytes:,}) -- "
        f"something is accumulating across years again"
    )


def test_year_features_accessor_requires_exactly_one_source():
    spans = [(1, 0, 4)]
    y_pool = np.array([0, 1, 0, 1])
    model = build_mlp_model()

    with pytest.raises(ValueError, match="exactly one"):
        sample_hard_replay_indices(model, None, y_pool, spans, 2, np.random.default_rng(0))

    with pytest.raises(ValueError, match="exactly one"):
        sample_hard_replay_indices(
            model, np.zeros((4, 3), dtype=np.float32), y_pool, spans, 2, np.random.default_rng(0),
            load_scaled_year=lambda year_idx: np.zeros((4, 3), dtype=np.float32),
        )


def test_streaming_loader_returning_wrong_row_count_is_rejected():
    """Spans and loader disagreeing would silently mis-attribute rows to years."""
    fixture_rng = np.random.default_rng(14)
    cache = _multi_year_cache(fixture_rng, [12, 10])
    scaler = _scaler_for(cache)
    spans, X_pool, y_pool = _contiguous_pool(cache, scaler, current_year_idx=3)

    model = build_mlp_model()
    model.partial_fit(X_pool, y_pool, classes=np.array([0, 1]))

    with pytest.raises(ValueError, match="spans and the loader disagree"):
        sample_hard_replay_indices(
            model, None, y_pool, spans, 6, np.random.default_rng(0),
            load_scaled_year=lambda year_idx: scaler.transform(cache[year_idx][0])[:-1],
        )


def test_build_replay_year_spans_reproduces_the_vstack_layout():
    fixture_rng = np.random.default_rng(15)
    cache = _multi_year_cache(fixture_rng, [15, 10, 7])
    scaler = _scaler_for(cache)
    expected_spans, _, expected_labels = _contiguous_pool(cache, scaler, current_year_idx=4)

    spans, y_pool, pool_size = build_replay_year_spans(
        lambda year_idx: cache[year_idx][1], current_year_idx=4,
    )
    assert spans == expected_spans
    assert np.array_equal(y_pool, expected_labels)
    assert pool_size == 32


def test_build_replay_year_spans_skips_empty_years_like_the_old_guard():
    cache = {
        1: (np.zeros((5, 3), dtype=np.float32), np.array([0, 1, 0, 1, 0])),
        2: (np.zeros((0, 3), dtype=np.float32), np.array([], dtype=np.int64)),
        3: (np.zeros((4, 3), dtype=np.float32), np.array([1, 1, 0, 0])),
    }
    spans, y_pool, pool_size = build_replay_year_spans(
        lambda year_idx: cache[year_idx][1], current_year_idx=4,
    )
    # Year 2 contributes nothing and year 3 starts where year 1 ended -- no gap.
    assert spans == [(1, 0, 5), (3, 5, 9)]
    assert pool_size == 9
    assert len(y_pool) == 9


def test_build_replay_year_spans_with_no_history():
    spans, y_pool, pool_size = build_replay_year_spans(lambda year_idx: None, current_year_idx=1)
    assert spans == []
    assert pool_size == 0
    assert len(y_pool) == 0


def test_group_global_indices_round_trips_through_spans():
    spans = [(1, 0, 15), (3, 15, 25), (4, 25, 32)]
    global_indices = np.array([31, 0, 16, 14, 25, 15])

    grouped = group_global_indices(global_indices, spans)
    recovered = np.empty(len(global_indices), dtype=np.int64)
    starts = {year_idx: start for year_idx, start, _ in spans}
    for year_idx, local_indices, positions in grouped:
        recovered[positions] = local_indices + starts[year_idx]

    assert np.array_equal(recovered, global_indices)
    assert {year_idx for year_idx, _, _ in grouped} == {1, 3, 4}


def test_group_global_indices_rejects_out_of_range():
    spans = [(1, 0, 10)]
    with pytest.raises(ValueError, match="out of range"):
        group_global_indices(np.array([0, 10]), spans)


def test_group_global_indices_empty_inputs():
    assert group_global_indices(np.array([], dtype=np.int64), [(1, 0, 5)]) == []
    assert group_global_indices(np.array([1]), []) == []


def test_materialize_in_order_matches_indexing_the_contiguous_pool():
    fixture_rng = np.random.default_rng(16)
    cache = _multi_year_cache(fixture_rng, [15, 10, 7])
    scaler = _scaler_for(cache)
    spans, X_pool, y_pool = _contiguous_pool(cache, scaler, current_year_idx=4)

    # Deliberately unsorted and spanning all three years: the caller's order is
    # what the notebooks feed to replay_rng.permutation, so it must survive.
    global_indices = np.array([30, 2, 17, 9, 25, 15, 0])

    X_out, y_out = materialize_replay_samples_in_order(
        global_indices, spans, lambda year_idx: cache[year_idx], scaler,
    )

    assert np.allclose(X_out, X_pool[global_indices])
    assert np.array_equal(y_out, y_pool[global_indices])


def test_materialize_in_order_loads_each_needed_year_once_and_skips_the_rest():
    fixture_rng = np.random.default_rng(17)
    cache = _multi_year_cache(fixture_rng, [15, 10, 7])
    scaler = _scaler_for(cache)
    spans, _, _ = _contiguous_pool(cache, scaler, current_year_idx=4)

    loads = []

    def load_raw_year(year_idx):
        loads.append(year_idx)
        return cache[year_idx]

    # spans are [(1, 0, 15), (2, 15, 25), (3, 25, 32)]
    materialize_replay_samples_in_order(np.array([1, 2, 26]), spans, load_raw_year, scaler)
    assert sorted(loads) == [1, 3]  # year 2 got no picks, so it is never loaded


def test_materialize_in_order_empty_input():
    X, y = materialize_replay_samples_in_order(
        np.array([], dtype=np.int64), [(1, 0, 5)], lambda year_idx: None, scaler=None,
    )
    assert X.shape == (0, 0)
    assert y.shape == (0,)
