"""Joint-vs-marginal discriminator test for real concept drift (a P(Y|X) shift).

The other drift notebooks in this repo train a discriminator on `s2_bands` alone to
tell two periods apart. That is a two-sample test on the marginal P(X): it detects
covariate shift and is blind to a change in P(Y|X) unless that change also moves X.
`concept_drift_detection-class-conditional-hgb.ipynb` gets at P(Y|X) indirectly, by
running one X-only discriminator per class and reading the asymmetry between them.

This module implements the direct version. Factor the joint density ratio between
period A and period B:

    P_B(X,Y) / P_A(X,Y) = [P_B(X) / P_A(X)] * [P_B(Y|X) / P_A(Y|X)]
                           covariate shift     real concept drift

Fit two discriminators of "which period did this row come from" on *identical rows*:
one on `X` alone, one on `[X, y]`. The first can only exploit the left factor; the
second sees both. Their separability gap is the real-drift signal.

Two constraints make the gap mean that, and both are enforced here rather than left
to the caller:

1. **Matched priors.** Disturbance rates in the current split run roughly 1.5%-3.1%
   across years. Left alone the joint discriminator wins by reading `y` and reports
   prior shift under a new name. Every side of every comparison is drawn to the *same*
   (n_class1, n_class0) design, so P(y=1) is identical on both sides by construction.
   Prior shift is real drift, but it is a different kind and belongs in its own table.

   With priors matched at pi*, the matched conditional on each side depends only on
   (pi*, P(x|y)), and pi* is shared - so the two matched conditionals are equal iff
   both class-conditionals are equal. The gap therefore fires on class-*asymmetric*
   movement of P(x|y), which is exactly what a P(Y|X) change is.

2. **Shared rows.** The marginal and joint discriminators must see the same rows,
   differing only by the label column, or the gap conflates drift with the luck of
   two different draws. `joint_marginal_gap` builds the rows once and fits both.

Sizes are held fixed across pair fits and null fits (see `design_sizes`), because AUC
variance and HGB's overfitting behaviour are both strongly n-dependent - a null
measured at a different row count is not a floor for the pair.
"""

from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import f1_score, log_loss, roc_auc_score

DEFAULT_HGB_PARAMS = dict(
    loss="log_loss",
    learning_rate=0.1,
    max_iter=200,
    max_leaf_nodes=31,
    l2_regularization=0.0,
    class_weight="balanced",
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=10,
    random_state=42,
)


@dataclass(frozen=True)
class YearPools:
    """Row indices for one (split, year), partitioned by class and by use.

    `holdout_*` is carved off before anything else and is never handed to a
    discriminator. Section 8 of the notebook fits its label model h on it, so that h's
    residuals are out-of-sample on *both* sides of every pair - otherwise h scores its
    own training rows on the year_prev side only, and that asymmetry alone would
    inflate the gap with no drift present.
    """

    pool_class1: np.ndarray
    pool_class0: np.ndarray
    holdout_class1: np.ndarray
    holdout_class0: np.ndarray


def carve_pools(y, rng, holdout_fraction=0.0):
    """Split one year's rows into per-class discriminator pools and an h-fitting holdout."""
    idx1 = np.flatnonzero(y == 1)
    idx0 = np.flatnonzero(y == 0)
    idx1 = idx1[rng.permutation(len(idx1))]
    idx0 = idx0[rng.permutation(len(idx0))]

    n_hold1 = int(round(len(idx1) * holdout_fraction))
    n_hold0 = int(round(len(idx0) * holdout_fraction))

    return YearPools(
        pool_class1=idx1[n_hold1:],
        pool_class0=idx0[n_hold0:],
        holdout_class1=idx1[:n_hold1],
        holdout_class0=idx0[:n_hold0],
    )


def design_sizes(pools, class0_per_class1, class0_cap=None):
    """Pick the (n_class1, n_class0) every side of every comparison will be drawn to.

    `pools` is the collection of YearPools for one split, across all usable years. Sizes
    are halved so that a same-year null can draw two *disjoint* sides of the same size a
    real pair uses.

    class0 is capped relative to class1 (`class0_per_class1`) rather than left at the
    natural ~50:1 ratio: the design prior is shared by both sides either way, so validity
    is unaffected, but a near-balanced pi* is where the label column carries the most
    information and the gap has the most power. It does mean the marginal AUC here is not
    directly comparable to `x_only_auc` in the class-conditional notebook, which uses the
    natural class mix.
    """
    pools = list(pools)
    if not pools:
        raise ValueError("design_sizes needs at least one YearPools")

    n_class1 = min(len(p.pool_class1) for p in pools) // 2
    n_class0 = min(len(p.pool_class0) for p in pools) // 2
    n_class0 = min(n_class0, class0_per_class1 * n_class1)
    if class0_cap is not None:
        n_class0 = min(n_class0, class0_cap)

    if n_class1 < 1 or n_class0 < 1:
        raise ValueError(
            f"degenerate design: n_class1={n_class1}, n_class0={n_class0} - "
            "some year has too few rows of one class"
        )
    return int(n_class1), int(n_class0)


def draw_side(pool, n_class1, n_class0, rng):
    """Draw one side of a comparison: exactly n_class1 + n_class0 row indices."""
    take1 = rng.choice(pool.pool_class1, size=n_class1, replace=False)
    take0 = rng.choice(pool.pool_class0, size=n_class0, replace=False)
    return np.concatenate([take1, take0])


def draw_disjoint_sides(pool, n_class1, n_class0, rng):
    """Draw two non-overlapping sides from one year - the same-year null's two halves."""
    perm1 = rng.permutation(pool.pool_class1)
    perm0 = rng.permutation(pool.pool_class0)
    if len(perm1) < 2 * n_class1 or len(perm0) < 2 * n_class0:
        raise ValueError("pool too small for two disjoint sides at this design size")

    side_a = np.concatenate([perm1[:n_class1], perm0[:n_class0]])
    side_b = np.concatenate([perm1[n_class1:2 * n_class1], perm0[n_class0:2 * n_class0]])
    return side_a, side_b


def attach(X, extra):
    """Append a channel (label, residual, ...) to a feature matrix. `None` returns X."""
    if extra is None:
        return X
    extra = np.asarray(extra)
    if extra.ndim == 1:
        extra = extra[:, None]
    return np.column_stack([X, extra.astype(X.dtype, copy=False)])


def build_pair(X_side_a, X_side_b):
    """Stack two matrices; the target is which side a row came from (0/1)."""
    if X_side_a is None or X_side_b is None or len(X_side_a) == 0 or len(X_side_b) == 0:
        return None, None
    X_pair = np.vstack([X_side_a, X_side_b])
    side_label = np.concatenate([
        np.zeros(len(X_side_a), dtype=np.uint8),
        np.ones(len(X_side_b), dtype=np.uint8),
    ])
    return X_pair, side_label


def fit_discriminator(pair_train, pair_val, pair_test, hgb_params=None):
    """Fit one side-discriminator and score it on the held-out val/test pairs.

    Each `pair_*` is `(X_side_a, X_side_b)` with any extra channels already attached.
    Returns None when a split is empty or single-class.
    """
    hgb_params = DEFAULT_HGB_PARAMS if hgb_params is None else hgb_params

    X_train, y_train = build_pair(*pair_train)
    X_val, y_val = build_pair(*pair_val)
    X_test, y_test = build_pair(*pair_test)

    if any(v is None for v in [X_train, y_train, X_val, y_val, X_test, y_test]):
        return None
    if any(len(np.unique(v)) < 2 for v in [y_train, y_val, y_test]):
        return None

    model = HistGradientBoostingClassifier(**hgb_params)
    model.fit(X_train, y_train)

    val_proba = model.predict_proba(X_val)[:, 1]
    test_proba = model.predict_proba(X_test)[:, 1]

    return {
        "n_train": int(len(y_train)),
        "n_val": int(len(y_val)),
        "n_test": int(len(y_test)),
        "val_auc": float(roc_auc_score(y_val, val_proba)),
        "test_auc": float(roc_auc_score(y_test, test_proba)),
        "val_f1": float(f1_score(y_val, (val_proba >= 0.5).astype(int))),
        "test_f1": float(f1_score(y_test, (test_proba >= 0.5).astype(int))),
        "val_logloss": float(log_loss(y_val, val_proba, labels=[0, 1])),
        "test_logloss": float(log_loss(y_test, test_proba, labels=[0, 1])),
    }


def joint_marginal_gap(split_sides, hgb_params=None, extra_channel=None):
    """Fit the marginal (X) and joint ([X, channel]) discriminators on identical rows.

    `split_sides` maps each of "train"/"val"/"test" to `(X_a, y_a, X_b, y_b)`. The joint
    channel defaults to `y`; pass `extra_channel` as `{"train": (e_a, e_b), ...}` to use
    something else (Section 8 passes the residual `y - h(X)`).

    Returned `gap_auc` is the headline statistic. It is only interpretable against a
    same-year null of the same quantity at the same design size - a gap of 0.02 can be
    ordinary HGB noise at these row counts.
    """
    marg_pairs, joint_pairs = {}, {}
    for split_name, (X_a, y_a, X_b, y_b) in split_sides.items():
        if extra_channel is None:
            e_a, e_b = y_a, y_b
        else:
            e_a, e_b = extra_channel[split_name]
        marg_pairs[split_name] = (X_a, X_b)
        joint_pairs[split_name] = (attach(X_a, e_a), attach(X_b, e_b))

    marginal = fit_discriminator(
        marg_pairs["train"], marg_pairs["val"], marg_pairs["test"], hgb_params
    )
    joint = fit_discriminator(
        joint_pairs["train"], joint_pairs["val"], joint_pairs["test"], hgb_params
    )
    if marginal is None or joint is None:
        return None

    row = {f"marg_{k}": v for k, v in marginal.items()}
    row.update({f"joint_{k}": v for k, v in joint.items()})
    row["gap_auc"] = joint["test_auc"] - marginal["test_auc"]
    row["gap_val_auc"] = joint["val_auc"] - marginal["val_auc"]
    # Log-loss falls when the joint model is genuinely better calibrated about which
    # period a row came from, so the reduction is a second, threshold-free view of the
    # same gap. Reported alongside, never as the headline.
    row["gap_logloss_reduction"] = marginal["test_logloss"] - joint["test_logloss"]
    return row


def flip_labels_in_region(y, region_mask, fraction, rng):
    """Swap labels symmetrically inside a feature-space region.

    Equal numbers of class-1 and class-0 rows inside the region trade labels, so P(X) is
    untouched (no row moves), P(y=1) is untouched (counts are preserved), and only
    P(y|x) changes, inside the region. That is real concept drift with the covariate and
    prior channels held fixed - the injection the gap statistic is supposed to catch.
    """
    y_out = np.asarray(y).copy()
    in_region_1 = np.flatnonzero(region_mask & (y_out == 1))
    in_region_0 = np.flatnonzero(region_mask & (y_out == 0))

    n_flip = int(round(fraction * min(len(in_region_1), len(in_region_0))))
    if n_flip == 0:
        return y_out, 0

    pick1 = rng.choice(in_region_1, size=n_flip, replace=False)
    pick0 = rng.choice(in_region_0, size=n_flip, replace=False)
    y_out[pick1] = 0
    y_out[pick0] = 1
    return y_out, n_flip


def importance_resample(score, alpha, keep_fraction, rng):
    """Row indices for a covariate-shift-only control: reweight P(X), leave P(Y|X) alone.

    Rows are *selected* with probability proportional to `sigmoid(alpha * z(score))`,
    never modified, and `score` must be a function of X only - so P(y|x) is preserved
    exactly for every x. Selection uses the Gumbel-top-k trick, which is weighted
    sampling without replacement (duplicated rows would just get memorised).

    The obvious alternative - adding a constant to every band of the later year - is not
    a covariate-only control: moving x while leaving y behind makes P_B(y|x) =
    P_A(y|x-delta), which is itself a conditional change, so it would move the gap and
    prove nothing.
    """
    score = np.asarray(score, dtype=np.float64)
    std = score.std()
    z = (score - score.mean()) / std if std > 0 else np.zeros_like(score)
    log_w = -np.logaddexp(0.0, -alpha * z)  # log sigmoid(alpha * z)

    n_keep = max(1, int(round(len(score) * keep_fraction)))
    gumbel = -np.log(-np.log(rng.random(len(score))))
    return np.argsort(-(log_w + gumbel))[:n_keep]


def fit_label_model(X, y, hgb_params=None):
    """Fit h: X -> P(y=1), the label model behind the residual channel."""
    hgb_params = DEFAULT_HGB_PARAMS if hgb_params is None else hgb_params
    model = HistGradientBoostingClassifier(**hgb_params)
    model.fit(X, y)
    return model


def residual_channel(model, X, y):
    """`y - h(X)`, the residual fed to the joint discriminator in Section 8.

    Because h(X) is a deterministic function of X and X is already in the feature set,
    `[X, y - h(X)]` carries exactly the same information as `[X, y]`. The variant is a
    finite-sample bet, not an information gain: the residual concentrates the conditional
    signal into one column that a depth-limited tree can split on directly, instead of
    making it reconstruct h internally.
    """
    return np.asarray(y, dtype=np.float64) - model.predict_proba(X)[:, 1]


def classify_gap(gap_elevated, marginal_elevated):
    """Turn the two null-calibrated verdicts into the reading for a pair."""
    if gap_elevated and marginal_elevated:
        return "real concept drift + covariate shift"
    if gap_elevated:
        return "real concept drift (P(Y|X) shift)"
    if marginal_elevated:
        return "covariate shift only (P(Y|X) stable)"
    return "no drift beyond prior shift"
