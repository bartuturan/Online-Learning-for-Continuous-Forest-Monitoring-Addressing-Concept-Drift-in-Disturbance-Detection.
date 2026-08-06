"""Replay-pool sampling strategies for the MLP experience-replay training loop.

Every sampler here (sample_hard_replay_indices, sample_uncertain_replay_indices,
sample_confidently_correct_replay_indices, sample_random_replay_indices,
sample_grouped_indices) is called ONCE PER YEAR by its notebook, before that
year's epoch loop starts -- never once per epoch. The model-scored strategies
(hard/uncertain/confident) score the replay pool against the model exactly as
it stood at the start of the year; the resulting index set is then held fixed
across all of that year's epochs (only the row ORDER is reshuffled each epoch,
via a separate `replay_rng.permutation` call the notebooks keep inside the
epoch loop -- the composition of the batch does not change).

This is a deliberate choice, not an oversight -- see the "Move replay sampling
from per-epoch to per-year" change for the full rationale. In short: per-epoch
resampling lets a low replay-ratio run's epochs collectively touch nearly the
whole pool (defeating what the ratio is supposed to control), confounds early
stopping (a chunk of the training set changes underneath the val-PR-AUC delta
patience is judging), and dilutes the model-scored strategies -- by the model's
better-fit later epochs, most of what counted as "hard"/"uncertain" earlier in
the year has stopped being so, and the sampler quietly falls back to random
filler (see weighted_choice_without_replacement and _top_up_shortfall below).

tests/mlp_replay/test_replay_sampling_cadence.py statically guards every
in-scope notebook against a sampler call regressing back inside its epoch
loop. The `*_reservoir_sampling` notebook variants are NOT held to this
convention -- they still resample every epoch (out of scope for that change)
and are therefore not comparable to the rest of the sweep; see the note in
scripts/run_pipeline_stages.py's module docstring.
"""

import numpy as np

from .checkpointing import get_cached_raw_year
from .model import compute_optimal_f1_threshold


def weighted_choice_without_replacement(indices, weights, sample_size, rng):
    if sample_size <= 0 or len(indices) == 0:
        return np.empty((0,), dtype=np.int64)

    sample_size = min(int(sample_size), len(indices))
    weights = np.asarray(weights, dtype=np.float64)
    weights = np.where(np.isnan(weights) | (weights < 0), 0.0, weights)
    total_weight = float(weights.sum())

    if total_weight <= 0:
        selected_positions = rng.choice(len(indices), size=sample_size, replace=False)
    else:
        non_zero_mask = weights > 0
        n_non_zero = int(np.sum(non_zero_mask))
        if n_non_zero < sample_size:
            chosen_non_zero = np.where(non_zero_mask)[0]
            remaining_needed = sample_size - n_non_zero
            zero_indices = np.where(~non_zero_mask)[0]
            chosen_zero = rng.choice(zero_indices, size=remaining_needed, replace=False)
            selected_positions = np.concatenate([chosen_non_zero, chosen_zero])
            rng.shuffle(selected_positions)
        else:
            prob = weights / total_weight
            selected_positions = rng.choice(len(indices), size=sample_size, replace=False, p=prob)

    return np.asarray(indices, dtype=np.int64)[selected_positions]


def update_reservoir(X_res, y_res, X_new, y_new, samples_seen, capacity, rng):
    B = len(y_new)
    if B == 0:
        return X_res, y_res, samples_seen
    if X_res is None or len(y_res) == 0:
        if B <= capacity:
            X_res = X_new.copy()
            y_res = y_new.copy()
            samples_seen += B
            return X_res, y_res, samples_seen
        else:
            indices = rng.choice(B, size=capacity, replace=False)
            X_res = X_new[indices]
            y_res = y_new[indices]
            samples_seen += B
            return X_res, y_res, samples_seen
    global_indices = np.arange(samples_seen, samples_seen + B)
    rand_vals = rng.random(B)
    replace_indices = (rand_vals * (global_indices + 1)).astype(np.int64)
    keep_mask = replace_indices < capacity
    if np.any(keep_mask):
        slots = replace_indices[keep_mask]
        X_res[slots] = X_new[keep_mask]
        y_res[slots] = y_new[keep_mask]
    samples_seen += B
    return X_res, y_res, samples_seen


def sample_random_replay_indices(replay_pool_size, random_target_size, excluded_indices, rng):
    if random_target_size <= 0 or replay_pool_size <= 0:
        return np.empty((0,), dtype=np.int64)

    all_indices = np.arange(replay_pool_size, dtype=np.int64)
    available_indices = np.setdiff1d(all_indices, np.asarray(excluded_indices, dtype=np.int64), assume_unique=False)
    if len(available_indices) == 0:
        return np.empty((0,), dtype=np.int64)

    take = min(int(random_target_size), len(available_indices))
    return rng.choice(available_indices, size=take, replace=False).astype(np.int64, copy=False)


def _year_features_accessor(load_scaled_year, X_replay_pool):
    """Resolve the two ways a sampler can reach one past year's SCALED features.

    `load_scaled_year(year_idx) -> X` is the streaming path: the caller hands back
    one year at a time, so the whole history is never resident at once. This is
    what the per-year notebooks use -- at 5.6M pixels x 32 float32 features a
    single year is ~716MB, while the concatenated pool at the last training year
    is ~3.6GB (~7.2GB counting the np.vstack duplicate), which is what was getting
    the Kaggle kernel OOM-killed.

    `X_replay_pool` is the legacy path: one contiguous array indexed by the span's
    [start:end]. Kept because the *_reservoir_sampling notebooks pass a reservoir
    BUFFER here under a synthetic single span [(1, 0, len(pool))] whose "year_idx"
    is not a real year at all, so they cannot use the streaming path.

    Both paths feed the samplers the exact same rows in the exact same order, so
    RNG consumption -- and therefore the selection -- is identical either way.
    """
    if (load_scaled_year is None) == (X_replay_pool is None):
        raise ValueError(
            'Pass exactly one of load_scaled_year= (streaming) or X_replay_pool= (legacy contiguous pool).'
        )

    if X_replay_pool is not None:
        return lambda year_idx, start, end: X_replay_pool[start:end]

    def accessor(year_idx, start, end):
        X_year = load_scaled_year(year_idx)
        if len(X_year) != end - start:
            raise ValueError(
                f'load_scaled_year({year_idx}) returned {len(X_year)} rows but its span covers '
                f'{end - start} -- the spans and the loader disagree about this year.'
            )
        return X_year

    return accessor


def _allocate_per_year_quota(replay_year_spans, target_size):
    n_years_with_data = len(replay_year_spans)
    base_quota = target_size // n_years_with_data
    remainder = target_size % n_years_with_data

    per_year_targets = []
    for idx, year_span in enumerate(replay_year_spans):
        year_idx, start, end = year_span
        year_target = base_quota + (1 if idx < remainder else 0)
        year_size = end - start
        per_year_targets.append((year_idx, start, end, min(year_target, year_size)))
    return per_year_targets


def _top_up_shortfall(indices, target_size, pool_size, rng):
    shortfall = int(target_size) - len(indices)
    if shortfall > 0:
        all_indices = np.arange(pool_size, dtype=np.int64)
        remaining_indices = np.setdiff1d(all_indices, indices, assume_unique=False)
        if len(remaining_indices) > 0:
            top_up = rng.choice(remaining_indices, size=min(shortfall, len(remaining_indices)), replace=False)
            indices = np.concatenate([indices, top_up.astype(np.int64, copy=False)])
    return indices


def sample_hard_replay_indices(
    model,
    X_replay_pool,
    y_replay_pool,
    replay_year_spans,
    hard_target_size,
    rng,
    threshold_grid=None,
    *,
    load_scaled_year=None,
):
    """Pass EITHER X_replay_pool (legacy contiguous pool) or load_scaled_year=
    (streaming, one year at a time) -- see _year_features_accessor."""
    if hard_target_size <= 0 or len(replay_year_spans) == 0:
        return np.empty((0,), dtype=np.int64), []

    get_year_X = _year_features_accessor(load_scaled_year, X_replay_pool)
    per_year_targets = _allocate_per_year_quota(replay_year_spans, hard_target_size)

    selected_indices = []
    threshold_values = []

    for year_idx, start, end, year_target in per_year_targets:
        if year_target <= 0:
            continue

        year_indices = np.arange(start, end, dtype=np.int64)
        y_year = y_replay_pool[start:end]
        y_proba_year = model.predict_proba(get_year_X(year_idx, start, end))[:, 1]
        threshold = compute_optimal_f1_threshold(
            y_true=y_year,
            y_proba=y_proba_year,
            threshold_grid=threshold_grid,
            default_threshold=0.5,
        )
        threshold_values.append(threshold)

        y_pred_year = (y_proba_year >= threshold).astype(int)
        incorrect_mask = (y_pred_year != y_year)

        # weights are confidence of incorrect prediction for incorrect samples, 0 for correct
        hard_weights = np.where(incorrect_mask, np.abs(y_year - y_proba_year), 0.0)

        chosen = weighted_choice_without_replacement(
            indices=year_indices,
            weights=hard_weights,
            sample_size=year_target,
            rng=rng,
        )
        if len(chosen) > 0:
            selected_indices.append(chosen)

    if selected_indices:
        hard_indices = np.concatenate(selected_indices).astype(np.int64, copy=False)
    else:
        hard_indices = np.empty((0,), dtype=np.int64)

    hard_indices = _top_up_shortfall(hard_indices, hard_target_size, len(y_replay_pool), rng)
    return hard_indices, threshold_values


def sample_uncertain_replay_indices(
    model,
    X_replay_pool,
    y_replay_pool,
    replay_year_spans,
    uncertain_target_size,
    rng,
    gaussian_std,
    threshold_grid=None,
    *,
    load_scaled_year=None,
):
    """Pass EITHER X_replay_pool (legacy contiguous pool) or load_scaled_year=
    (streaming, one year at a time) -- see _year_features_accessor."""
    EPSILON = 1e-12
    if uncertain_target_size <= 0 or len(replay_year_spans) == 0:
        return np.empty((0,), dtype=np.int64), []

    get_year_X = _year_features_accessor(load_scaled_year, X_replay_pool)
    per_year_targets = _allocate_per_year_quota(replay_year_spans, uncertain_target_size)

    selected_indices = []
    threshold_values = []

    for year_idx, start, end, year_target in per_year_targets:
        if year_target <= 0:
            continue

        year_indices = np.arange(start, end, dtype=np.int64)
        y_year = y_replay_pool[start:end]
        y_proba_year = model.predict_proba(get_year_X(year_idx, start, end))[:, 1]
        threshold = compute_optimal_f1_threshold(
            y_true=y_year,
            y_proba=y_proba_year,
            threshold_grid=threshold_grid,
            default_threshold=0.5,
        )
        threshold_values.append(threshold)

        distances = np.abs(y_proba_year - threshold)
        if len(distances) > 0 and year_target > 0:
            k_idx = min(int(year_target) - 1, len(distances) - 1)
            d_k = np.partition(distances, k_idx)[k_idx]
            std = max(float(gaussian_std), d_k / 4.8)
        else:
            std = float(gaussian_std)

        gaussian_weights = np.exp(-(distances ** 2) / (2.0 * (std ** 2) + EPSILON))
        chosen = weighted_choice_without_replacement(
            indices=year_indices,
            weights=gaussian_weights,
            sample_size=year_target,
            rng=rng,
        )
        if len(chosen) > 0:
            selected_indices.append(chosen)

    if selected_indices:
        uncertain_indices = np.concatenate(selected_indices).astype(np.int64, copy=False)
    else:
        uncertain_indices = np.empty((0,), dtype=np.int64)

    uncertain_indices = _top_up_shortfall(uncertain_indices, uncertain_target_size, len(y_replay_pool), rng)
    return uncertain_indices, threshold_values


def sample_confidently_correct_replay_indices(
    model,
    X_replay_pool,
    y_replay_pool,
    replay_year_spans,
    confident_target_size,
    rng,
    threshold_grid=None,
    *,
    load_scaled_year=None,
):
    """Pass EITHER X_replay_pool (legacy contiguous pool) or load_scaled_year=
    (streaming, one year at a time) -- see _year_features_accessor."""
    if confident_target_size <= 0 or len(replay_year_spans) == 0:
        return np.empty((0,), dtype=np.int64), []

    get_year_X = _year_features_accessor(load_scaled_year, X_replay_pool)
    per_year_targets = _allocate_per_year_quota(replay_year_spans, confident_target_size)

    selected_indices = []
    threshold_values = []

    for year_idx, start, end, year_target in per_year_targets:
        if year_target <= 0:
            continue

        year_indices = np.arange(start, end, dtype=np.int64)
        y_year = y_replay_pool[start:end]
        y_proba_year = model.predict_proba(get_year_X(year_idx, start, end))[:, 1]
        threshold = compute_optimal_f1_threshold(
            y_true=y_year,
            y_proba=y_proba_year,
            threshold_grid=threshold_grid,
            default_threshold=0.5,
        )
        threshold_values.append(threshold)

        y_pred_year = (y_proba_year >= threshold).astype(int)
        correct_mask = (y_pred_year == y_year)

        # weights are confidence of correct prediction (probability of true class) for correct samples, 0 for incorrect
        confident_weights = np.where(correct_mask, 1.0 - np.abs(y_year - y_proba_year), 0.0)

        chosen = weighted_choice_without_replacement(
            indices=year_indices,
            weights=confident_weights,
            sample_size=year_target,
            rng=rng,
        )
        if len(chosen) > 0:
            selected_indices.append(chosen)

    if selected_indices:
        confident_indices = np.concatenate(selected_indices).astype(np.int64, copy=False)
    else:
        confident_indices = np.empty((0,), dtype=np.int64)

    confident_indices = _top_up_shortfall(confident_indices, confident_target_size, len(y_replay_pool), rng)
    return confident_indices, threshold_values


def build_replay_year_spans(load_year_labels, current_year_idx):
    """Labels-only twin of the `np.vstack(replay_X_parts)` pool build.

    Returns (replay_year_spans, y_replay_pool, replay_pool_size) where spans are
    the same (year_idx, start, end) triples the samplers already take, laid out in
    the same order and with the same offsets the concatenated pool would have had.
    Features are never touched: the caller hands back only that year's labels, so
    a 27.9M-row history costs ~223MB of int64 here instead of ~3.6GB of float32
    features (~7.2GB with the vstack duplicate).

    load_year_labels(year_idx) returns that year's label array, or an empty array
    for a year with no usable samples -- empty years are skipped, exactly as the
    `if len(X_past_raw) > 0` guard in the old inline pool build did, so the span
    layout is unchanged.
    """
    spans = []
    label_parts = []
    cursor = 0

    for past_year_idx in range(1, current_year_idx):
        y_past = load_year_labels(past_year_idx)
        if y_past is None or len(y_past) == 0:
            continue
        span_len = len(y_past)
        spans.append((past_year_idx, cursor, cursor + span_len))
        label_parts.append(y_past)
        cursor += span_len

    if label_parts:
        y_replay_pool = np.concatenate(label_parts)
    else:
        y_replay_pool = np.empty((0,), dtype=np.int64)

    return spans, y_replay_pool, cursor


def group_global_indices(global_indices, replay_year_spans):
    """Map global pool offsets back to the year that owns them.

    Returns [(year_idx, local_indices, positions)]: `local_indices` are offsets
    within that year's own array, and `positions` are where those rows sat in the
    caller's `global_indices`. Carrying `positions` is what lets
    materialize_replay_samples_in_order rebuild the caller's exact row order --
    which matters because the notebooks feed the materialized batch straight into
    a per-epoch `replay_rng.permutation`, so a different starting order is a
    different training run.

    Spans are contiguous by construction (build_replay_year_spans lays each year
    down at the previous year's end), so one searchsorted resolves every index.
    """
    global_indices = np.asarray(global_indices, dtype=np.int64)
    if len(replay_year_spans) == 0 or len(global_indices) == 0:
        return []

    starts = np.array([start for _, start, _ in replay_year_spans], dtype=np.int64)
    ends = np.array([end for _, _, end in replay_year_spans], dtype=np.int64)
    year_ids = [year_idx for year_idx, _, _ in replay_year_spans]

    if global_indices.min() < starts[0] or global_indices.max() >= ends[-1]:
        raise ValueError(
            f'global index out of range: got [{global_indices.min()}, {global_indices.max()}], '
            f'spans cover [{starts[0]}, {ends[-1]})'
        )

    span_of = np.searchsorted(ends, global_indices, side='right')

    grouped = []
    for span_pos in np.unique(span_of):
        span_pos = int(span_pos)
        positions = np.flatnonzero(span_of == span_pos)
        local_indices = global_indices[positions] - starts[span_pos]
        grouped.append((year_ids[span_pos], local_indices, positions))
    return grouped


def materialize_replay_samples_in_order(global_indices, replay_year_spans, load_raw_year, scaler):
    """Fetch exactly the selected rows, scale only those, in the caller's order.

    The point of the whole streaming design: each needed year is loaded once,
    immediately reduced to its selected rows, and released -- so peak memory is one
    year's features plus the (much smaller) output, never the whole history. The
    output buffer is preallocated at its final size, so there is no vstack
    duplicate either.

    load_raw_year(year_idx) -> (X_raw, y) matches get_cached_raw_year's contract.
    """
    global_indices = np.asarray(global_indices, dtype=np.int64)
    grouped = group_global_indices(global_indices, replay_year_spans)
    if not grouped:
        return np.empty((0, 0), dtype=np.float32), np.empty((0,), dtype=np.int64)

    X_out = None
    y_out = None
    for year_idx, local_indices, positions in grouped:
        X_raw, y_year = load_raw_year(year_idx)
        if len(X_raw) == 0:
            continue
        X_selected = scaler.transform(X_raw[local_indices])
        y_selected = y_year[local_indices]
        if X_out is None:
            X_out = np.empty((len(global_indices), X_selected.shape[1]), dtype=X_selected.dtype)
            y_out = np.empty((len(global_indices),), dtype=y_selected.dtype)
        X_out[positions] = X_selected
        y_out[positions] = y_selected

    if X_out is None:
        return np.empty((0, 0), dtype=np.float32), np.empty((0,), dtype=np.int64)
    return X_out, y_out


def build_replay_year_metadata(cache, current_year_idx, positive_label=1):
    metadata = []
    replay_pool_size = 0
    replay_positive_pool_size = 0
    replay_negative_pool_size = 0

    for past_year_idx in range(1, current_year_idx):
        X_past_raw, y_past = get_cached_raw_year(cache, past_year_idx)
        if len(X_past_raw) == 0:
            continue

        unique_replay_labels = set(np.unique(y_past).tolist())
        if not unique_replay_labels.issubset({0, 1}):
            raise ValueError(f'Expected binary replay labels in {{0, 1}}, got {sorted(unique_replay_labels)}')

        positive_indices = np.flatnonzero(y_past == positive_label)
        negative_indices = np.flatnonzero(y_past != positive_label)

        metadata.append(
            {
                'year_idx': past_year_idx,
                'positive_indices': positive_indices,
                'negative_indices': negative_indices,
            }
        )

        replay_pool_size += len(y_past)
        replay_positive_pool_size += len(positive_indices)
        replay_negative_pool_size += len(negative_indices)

    return metadata, replay_pool_size, replay_positive_pool_size, replay_negative_pool_size


def sample_grouped_indices(sources, sample_size, rng):
    if sample_size <= 0 or not sources:
        return []

    source_counts = np.array([len(indices) for _, indices in sources], dtype=np.int64)
    total_count = int(source_counts.sum())
    if total_count == 0:
        return []

    sample_size = int(min(sample_size, total_count))
    picked_global = np.sort(rng.choice(total_count, size=sample_size, replace=False))

    cumulative = np.cumsum(source_counts)
    source_ids = np.searchsorted(cumulative, picked_global, side='right')
    cumulative_prev = np.concatenate(([0], cumulative[:-1]))

    grouped = []
    for source_id in np.unique(source_ids):
        source_id_int = int(source_id)
        source_mask = source_ids == source_id_int
        local_offsets = picked_global[source_mask] - cumulative_prev[source_id_int]
        year_idx, year_local_indices = sources[source_id_int]
        sampled_local_indices = year_local_indices[local_offsets]
        grouped.append((year_idx, sampled_local_indices))

    return grouped


def materialize_replay_samples(grouped_indices, cache, scaler):
    """Rows come back grouped by year, in `grouped_indices` order. Use
    materialize_replay_samples_in_order instead when the caller's own row order
    has to be preserved."""
    return materialize_grouped_replay_samples(
        grouped_indices, lambda year_idx: get_cached_raw_year(cache, year_idx), scaler
    )


def materialize_grouped_replay_samples(grouped_indices, load_raw_year, scaler):
    if not grouped_indices:
        return np.empty((0, 0), dtype=np.float32), np.empty((0,), dtype=np.int64)

    replay_X_chunks = []
    replay_y_chunks = []
    feature_dim = 0

    for past_year_idx, sampled_local_indices in grouped_indices:
        if len(sampled_local_indices) == 0:
            continue

        X_past_raw, y_past = load_raw_year(past_year_idx)
        if len(X_past_raw) == 0:
            continue

        X_sampled_raw = X_past_raw[sampled_local_indices]
        y_sampled = y_past[sampled_local_indices]

        X_sampled_scaled = scaler.transform(X_sampled_raw)
        replay_X_chunks.append(X_sampled_scaled)
        replay_y_chunks.append(y_sampled)
        feature_dim = X_sampled_scaled.shape[1]

    if not replay_X_chunks:
        return np.empty((0, feature_dim), dtype=np.float32), np.empty((0,), dtype=np.int64)

    X_replay_sampled = np.vstack(replay_X_chunks)
    y_replay_sampled = np.concatenate(replay_y_chunks)
    return X_replay_sampled, y_replay_sampled
