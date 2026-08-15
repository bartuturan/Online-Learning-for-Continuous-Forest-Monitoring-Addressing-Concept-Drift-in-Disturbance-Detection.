"""Tests for the joint-vs-marginal real-drift discriminator.

The behavioural tests at the bottom are the point of this file: they run the two-way
injection control on synthetic data where the ground truth is known by construction, and
assert that the gap statistic fires on a P(Y|X) change and stays flat under a pure P(X)
change. That is the same check Sections 6 and 7 of
`concept_drift_detection-real-joint-discriminator.ipynb` run on real S2 bands, at a size
that fits in a test suite.
"""

import numpy as np
import pytest

from src.drift.joint import (
    attach,
    build_pair,
    carve_pools,
    classify_gap,
    design_sizes,
    draw_disjoint_sides,
    draw_side,
    fit_label_model,
    flip_labels_in_region,
    importance_resample,
    joint_marginal_gap,
    residual_channel,
)

# Small and shallow: these tests check the statistic's direction, not its precision.
FAST_HGB = dict(
    loss="log_loss",
    learning_rate=0.1,
    max_iter=60,
    max_leaf_nodes=15,
    class_weight="balanced",
    early_stopping=False,
    random_state=0,
)

N_ROWS = 9000
N_FEATURES = 4


def make_period(rng):
    """X ~ N(0, I) with a genuine conditional P(y=1|x) = sigmoid(2 * x0)."""
    X = rng.normal(size=(N_ROWS, N_FEATURES)).astype(np.float32)
    p = 1.0 / (1.0 + np.exp(-2.0 * X[:, 0]))
    y = (rng.random(N_ROWS) < p).astype(np.uint8)
    return X, y


def split_blocks(X, y, seed):
    """Disjoint train/val/test blocks, standing in for the notebook's pixel splits.

    Disjointness matters more than it looks: drawing all three from one pool lets the
    same row land in train and test with the same side label, and the discriminator
    memorises it - which reads as separability that is not there.
    """
    rng = np.random.default_rng(seed)
    cuts = np.array_split(rng.permutation(len(y)), 3)
    return {name: (X[c], y[c]) for name, c in zip(["train", "val", "test"], cuts)}


def build_sides(blocks_a, blocks_b, n_class1, n_class0, seed=10):
    sides = {}
    for offset, split in enumerate(["train", "val", "test"]):
        rng = np.random.default_rng(seed + offset)
        X_a, y_a = blocks_a[split]
        X_b, y_b = blocks_b[split]
        idx_a = draw_side(carve_pools(y_a, rng), n_class1, n_class0, rng)
        idx_b = draw_side(carve_pools(y_b, rng), n_class1, n_class0, rng)
        sides[split] = (X_a[idx_a], y_a[idx_a], X_b[idx_b], y_b[idx_b])
    return sides


def sizes_for(*block_sets):
    pools = [
        carve_pools(y, np.random.default_rng(7))
        for blocks in block_sets
        for _, y in blocks.values()
    ]
    return design_sizes(pools, class0_per_class1=1)


@pytest.fixture(scope="module")
def periods():
    rng = np.random.default_rng(0)
    X_a, y_a = make_period(rng)
    X_b, y_b = make_period(rng)
    return split_blocks(X_a, y_a, 100), split_blocks(X_b, y_b, 101)


# --- pool carving and design sizing ---------------------------------------------------

def test_carve_pools_partitions_every_row():
    y = np.array([0, 1, 0, 1, 0, 1, 0, 1], dtype=np.uint8)
    pools = carve_pools(y, np.random.default_rng(0), holdout_fraction=0.5)

    all_idx = np.concatenate([
        pools.pool_class1, pools.pool_class0, pools.holdout_class1, pools.holdout_class0
    ])
    assert sorted(all_idx.tolist()) == list(range(len(y)))
    assert set(pools.pool_class1) & set(pools.holdout_class1) == set()
    assert all(y[i] == 1 for i in pools.pool_class1)
    assert all(y[i] == 0 for i in pools.pool_class0)


def test_carve_pools_holdout_fraction_splits_each_class():
    y = np.concatenate([np.ones(100, dtype=np.uint8), np.zeros(200, dtype=np.uint8)])
    pools = carve_pools(y, np.random.default_rng(0), holdout_fraction=0.25)

    assert len(pools.holdout_class1) == 25
    assert len(pools.pool_class1) == 75
    assert len(pools.holdout_class0) == 50
    assert len(pools.pool_class0) == 150


def test_design_sizes_halves_and_takes_the_smallest_year():
    pools = [
        carve_pools(np.concatenate([np.ones(400), np.zeros(4000)]).astype(np.uint8), np.random.default_rng(0)),
        carve_pools(np.concatenate([np.ones(100), np.zeros(9000)]).astype(np.uint8), np.random.default_rng(1)),
    ]
    n_class1, n_class0 = design_sizes(pools, class0_per_class1=10)

    assert n_class1 == 50  # min(400, 100) // 2 - halved so a null can draw two sides
    assert n_class0 == 500  # ratio-capped at 10 * 50, well under min(4000, 9000) // 2


def test_design_sizes_respects_absolute_cap():
    pools = [carve_pools(np.concatenate([np.ones(1000), np.zeros(9000)]).astype(np.uint8), np.random.default_rng(0))]
    _, n_class0 = design_sizes(pools, class0_per_class1=100, class0_cap=300)
    assert n_class0 == 300


def test_design_sizes_rejects_a_degenerate_design():
    pools = [carve_pools(np.concatenate([np.ones(1), np.zeros(50)]).astype(np.uint8), np.random.default_rng(0))]
    with pytest.raises(ValueError, match="degenerate design"):
        design_sizes(pools, class0_per_class1=4)


# --- drawing sides --------------------------------------------------------------------

def test_draw_side_hits_the_design_exactly():
    y = np.concatenate([np.ones(300, dtype=np.uint8), np.zeros(900, dtype=np.uint8)])
    pools = carve_pools(y, np.random.default_rng(0))
    idx = draw_side(pools, 100, 400, np.random.default_rng(1))

    assert len(idx) == 500
    assert (y[idx] == 1).sum() == 100
    assert (y[idx] == 0).sum() == 400
    assert len(np.unique(idx)) == len(idx)


def test_draw_disjoint_sides_do_not_overlap():
    y = np.concatenate([np.ones(300, dtype=np.uint8), np.zeros(900, dtype=np.uint8)])
    pools = carve_pools(y, np.random.default_rng(0))
    side_a, side_b = draw_disjoint_sides(pools, 100, 400, np.random.default_rng(1))

    assert set(side_a.tolist()) & set(side_b.tolist()) == set()
    assert (y[side_a] == 1).sum() == (y[side_b] == 1).sum() == 100
    assert (y[side_a] == 0).sum() == (y[side_b] == 0).sum() == 400


def test_draw_disjoint_sides_refuses_an_oversized_design():
    y = np.concatenate([np.ones(50, dtype=np.uint8), np.zeros(900, dtype=np.uint8)])
    pools = carve_pools(y, np.random.default_rng(0))
    with pytest.raises(ValueError, match="too small for two disjoint sides"):
        draw_disjoint_sides(pools, 40, 100, np.random.default_rng(1))


def test_matched_design_equalises_the_prior_across_sides():
    rng = np.random.default_rng(0)
    y_rare = (rng.random(6000) < 0.02).astype(np.uint8)
    y_common = (rng.random(6000) < 0.20).astype(np.uint8)

    pools = [carve_pools(y_rare, rng), carve_pools(y_common, rng)]
    n_class1, n_class0 = design_sizes(pools, class0_per_class1=4)

    rate_rare = y_rare[draw_side(pools[0], n_class1, n_class0, rng)].mean()
    rate_common = y_common[draw_side(pools[1], n_class1, n_class0, rng)].mean()
    assert rate_rare == pytest.approx(rate_common)


# --- matrix assembly ------------------------------------------------------------------

def test_attach_appends_a_channel_and_none_is_a_no_op():
    X = np.arange(6, dtype=np.float32).reshape(3, 2)
    assert attach(X, None) is X

    out = attach(X, np.array([1, 0, 1], dtype=np.uint8))
    assert out.shape == (3, 3)
    assert out[:, -1].tolist() == [1.0, 0.0, 1.0]
    assert out.dtype == X.dtype


def test_build_pair_labels_rows_by_side():
    X_pair, side = build_pair(np.zeros((3, 2)), np.ones((4, 2)))
    assert X_pair.shape == (7, 2)
    assert side.tolist() == [0, 0, 0, 1, 1, 1, 1]


def test_build_pair_returns_none_on_an_empty_side():
    assert build_pair(np.zeros((0, 2)), np.ones((4, 2))) == (None, None)
    assert build_pair(None, np.ones((4, 2))) == (None, None)


# --- injection helpers ----------------------------------------------------------------

def test_flip_preserves_class_counts_and_leaves_x_alone():
    rng = np.random.default_rng(0)
    y = (rng.random(2000) < 0.4).astype(np.uint8)
    region = rng.random(2000) < 0.5

    y_flipped, n_flip = flip_labels_in_region(y, region, 0.5, rng)

    assert n_flip > 0
    assert y_flipped.sum() == y.sum()  # prior untouched: symmetric swap
    assert np.array_equal(y_flipped[~region], y[~region])  # only the region moves
    assert (y_flipped != y).sum() == 2 * n_flip


def test_flip_fraction_zero_is_a_no_op():
    rng = np.random.default_rng(0)
    y = (rng.random(500) < 0.4).astype(np.uint8)
    y_out, n_flip = flip_labels_in_region(y, np.ones(500, dtype=bool), 0.0, rng)

    assert n_flip == 0
    assert np.array_equal(y_out, y)


def test_importance_resample_selects_distinct_rows_biased_by_score():
    rng = np.random.default_rng(0)
    score = rng.normal(size=5000)

    keep = importance_resample(score, alpha=3.0, keep_fraction=0.5, rng=rng)

    assert len(keep) == 2500
    assert len(np.unique(keep)) == len(keep)  # without replacement
    assert score[keep].mean() > score.mean() + 0.3


def test_importance_resample_with_zero_alpha_is_unbiased():
    rng = np.random.default_rng(0)
    score = rng.normal(size=5000)
    keep = importance_resample(score, alpha=0.0, keep_fraction=0.5, rng=rng)
    assert score[keep].mean() == pytest.approx(score.mean(), abs=0.1)


def test_residual_channel_is_signed_error():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(400, 3))
    y = (X[:, 0] > 0).astype(np.uint8)

    model = fit_label_model(X, y, FAST_HGB)
    residual = residual_channel(model, X, y)

    assert residual.shape == (400,)
    assert np.all(residual[y == 1] > -1.0) and np.all(residual[y == 1] <= 1.0)
    assert residual[y == 1].mean() > residual[y == 0].mean()


# --- reading the verdict --------------------------------------------------------------

@pytest.mark.parametrize("gap_hot,marg_hot,expected", [
    (False, False, "no drift beyond prior shift"),
    (False, True, "covariate shift only (P(Y|X) stable)"),
    (True, False, "real concept drift (P(Y|X) shift)"),
    (True, True, "real concept drift + covariate shift"),
])
def test_classify_gap(gap_hot, marg_hot, expected):
    assert classify_gap(gap_hot, marg_hot) == expected


# --- the two-way injection control ----------------------------------------------------

def test_null_gap_is_flat_when_both_sides_share_a_generator(periods):
    blocks_a, blocks_b = periods
    n_class1, n_class0 = sizes_for(blocks_a, blocks_b)

    row = joint_marginal_gap(build_sides(blocks_a, blocks_b, n_class1, n_class0), FAST_HGB)

    assert row["marg_test_auc"] == pytest.approx(0.5, abs=0.03)
    assert abs(row["gap_auc"]) < 0.02


def test_conditional_drift_lifts_the_gap_and_not_the_marginal(periods):
    """Label flips inside a region: P(X) and P(y) fixed, only P(y|x) moves."""
    blocks_a, blocks_b = periods
    n_class1, n_class0 = sizes_for(blocks_a, blocks_b)

    gaps = []
    for fraction in [0.0, 0.5, 1.0]:
        flipped = {}
        for split, (X_split, y_split) in blocks_b.items():
            y_flip, _ = flip_labels_in_region(
                y_split, X_split[:, 1] > 0, fraction, np.random.default_rng(3)
            )
            flipped[split] = (X_split, y_flip)

        row = joint_marginal_gap(build_sides(blocks_a, flipped, n_class1, n_class0), FAST_HGB)
        gaps.append(row["gap_auc"])
        # X is byte-identical on both sides, so the marginal must stay blind throughout
        assert row["marg_test_auc"] == pytest.approx(0.5, abs=0.03)

    assert gaps[0] < gaps[1] < gaps[2], f"gap not monotone in flip fraction: {gaps}"
    assert gaps[2] > 0.10


def test_covariate_shift_lifts_the_marginal_and_not_the_gap(periods):
    """Importance resampling on X: P(X) moves a lot, P(y|x) is preserved by construction."""
    blocks_a, blocks_b = periods

    shifted = {}
    for split, (X_split, y_split) in blocks_b.items():
        keep = importance_resample(X_split[:, 1], 3.0, 0.5, np.random.default_rng(4))
        shifted[split] = (X_split[keep], y_split[keep])

    n_class1, n_class0 = sizes_for(blocks_a, shifted)
    row = joint_marginal_gap(build_sides(blocks_a, shifted, n_class1, n_class0), FAST_HGB)

    assert row["marg_test_auc"] > 0.60, "control did not actually induce covariate shift"
    assert abs(row["gap_auc"]) < 0.03, "gap fired on a pure P(X) shift"


def residual_run(blocks_a, blocks_b, n_class1, n_class0):
    """Run the gap with the residual channel, observing the h-holdout discipline.

    h is fit on a holdout carved out of side A's train rows, and every row it later
    scores comes from the disjoint pool. Fitting h on rows that also land in the
    discriminator's train pair would give side A optimistically small residuals and side
    B honest ones - an asymmetry that reads as drift when there is none, which is what
    `test_residual_channel_stays_flat_under_the_null` pins down.
    """
    X_train_a, y_train_a = blocks_a["train"]
    holdout_pools = carve_pools(y_train_a, np.random.default_rng(11), holdout_fraction=0.5)
    holdout_idx = np.concatenate([holdout_pools.holdout_class1, holdout_pools.holdout_class0])
    label_model = fit_label_model(X_train_a[holdout_idx], y_train_a[holdout_idx], FAST_HGB)

    pool_idx = np.concatenate([holdout_pools.pool_class1, holdout_pools.pool_class0])
    blocks_a_pooled = dict(blocks_a, train=(X_train_a[pool_idx], y_train_a[pool_idx]))
    sides = build_sides(blocks_a_pooled, blocks_b, n_class1, n_class0)

    extra = {
        split: (
            residual_channel(label_model, X_a, y_a),
            residual_channel(label_model, X_b, y_b),
        )
        for split, (X_a, y_a, X_b, y_b) in sides.items()
    }
    return joint_marginal_gap(sides, FAST_HGB, extra_channel=extra)


def test_residual_channel_tracks_the_label_channel(periods):
    """The residual variant is a finite-sample bet, so it should agree in direction."""
    blocks_a, blocks_b = periods
    n_class1, n_class0 = sizes_for(blocks_a, blocks_b)

    flipped = {}
    for split, (X_split, y_split) in blocks_b.items():
        y_flip, _ = flip_labels_in_region(
            y_split, X_split[:, 1] > 0, 1.0, np.random.default_rng(3)
        )
        flipped[split] = (X_split, y_flip)

    assert residual_run(blocks_a, flipped, n_class1, n_class0)["gap_auc"] > 0.10


def test_residual_channel_stays_flat_under_the_null(periods):
    """No drift: h's residuals must not manufacture a gap on their own."""
    blocks_a, blocks_b = periods
    n_class1, n_class0 = sizes_for(blocks_a, blocks_b)

    row = residual_run(blocks_a, blocks_b, n_class1, n_class0)
    assert abs(row["gap_auc"]) < 0.03
