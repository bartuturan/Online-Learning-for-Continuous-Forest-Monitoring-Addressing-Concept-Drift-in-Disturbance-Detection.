import numpy as np


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
