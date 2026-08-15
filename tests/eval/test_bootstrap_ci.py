import numpy as np
import pytest
from sklearn.metrics import average_precision_score, f1_score

from src.eval.bootstrap_ci import (
    DEFAULT_BASELINE_FAMILY,
    DEFAULT_STRATEGY_PATTERN,
    _average_precision_from_bin_counts,
    _bin_and_cube_counts,
    _weighted_average_precision,
    benjamini_hochberg,
    bootstrap_f1_ci,
    bootstrap_pr_auc_ci,
    cube_confusion_counts,
    draw_cube_resamples,
    f1_from_counts,
    paired_bootstrap_margin,
    select_families,
)


class TestSelectFamilies:
    def test_matches_replay_and_combined_but_not_baseline_itself(self):
        families = {
            DEFAULT_BASELINE_FAMILY: {},
            'mlp_prevyears_monthly_features_incremental_scaler_experience_replay_RR_0.2': {},
            'mlp_prevyears_monthly_features_incremental_scaler_missclassification_buffer_RR_0.3': {},
            'mlp_combined_HE=0_CC=0.2_UP=0.2_PR=(0.1,10)_RR=0.5': {},
            'baseline': {},
            'huber': {},
        }
        baseline_id, strategy_ids = select_families(families)

        assert baseline_id == DEFAULT_BASELINE_FAMILY
        assert DEFAULT_BASELINE_FAMILY not in strategy_ids
        assert len(strategy_ids) == 3
        # The SGD families must not be swept in by the MLP-scoped pattern.
        assert 'baseline' not in strategy_ids
        assert 'huber' not in strategy_ids

    def test_missing_baseline_raises(self):
        with pytest.raises(KeyError, match='not in the registry'):
            select_families({'huber': {}}, baseline_id='nope')

    def test_pattern_matching_nothing_raises_before_any_inference(self):
        """A silent empty comparison set would waste a full inference sweep."""
        families = {DEFAULT_BASELINE_FAMILY: {}}
        with pytest.raises(ValueError, match='matched no families'):
            select_families(families, strategy_pattern=r'^will_not_match')


class TestCubeConfusionCounts:
    def test_counts_partition_the_rows(self):
        y_true = np.array([1, 1, 0, 0, 1, 0])
        y_pred = np.array([1, 0, 1, 0, 1, 0])
        cube_idx = np.array([7, 7, 7, 9, 9, 9])

        tp, fp, fn, cube_ids = cube_confusion_counts(y_true, y_pred, cube_idx)

        assert list(cube_ids) == [7, 9]
        assert list(tp) == [1, 1]
        assert list(fp) == [1, 0]
        assert list(fn) == [1, 0]

    def test_summed_counts_reproduce_sklearn_f1(self):
        rng = np.random.default_rng(0)
        y_true = rng.integers(0, 2, 500)
        y_pred = rng.integers(0, 2, 500)
        cube_idx = rng.integers(0, 20, 500)

        tp, fp, fn, _ = cube_confusion_counts(y_true, y_pred, cube_idx)

        assert f1_from_counts(tp.sum(), fp.sum(), fn.sum()) == pytest.approx(
            f1_score(y_true, y_pred, zero_division=0))

    def test_all_zero_counts_give_zero_not_nan(self):
        assert f1_from_counts(0, 0, 0) == 0.0

    def test_misaligned_inputs_raise(self):
        with pytest.raises(ValueError, match='must be aligned'):
            cube_confusion_counts([1, 0], [1, 0, 1], [1, 1, 1])

    def test_cube_id_absent_from_explicit_ids_raises(self):
        with pytest.raises(ValueError, match='absent from cube_ids'):
            cube_confusion_counts([1, 0], [1, 0], [3, 99], cube_ids=np.array([3, 4]))


class TestDrawCubeResamples:
    def test_each_resample_draws_exactly_n_cubes(self):
        weights = draw_cube_resamples(n_cubes=10, n_bootstrap=50, rng=np.random.default_rng(1))
        assert weights.shape == (50, 10)
        assert np.all(weights.sum(axis=1) == 10)

    def test_same_seed_gives_same_draws(self):
        a = draw_cube_resamples(8, 20, np.random.default_rng(7))
        b = draw_cube_resamples(8, 20, np.random.default_rng(7))
        assert np.array_equal(a, b)


class TestWeightedAveragePrecision:
    def test_unweighted_matches_sklearn(self):
        """The weighted sweep must reduce to sklearn's average_precision_score."""
        rng = np.random.default_rng(3)
        y_true = rng.integers(0, 2, 300)
        y_score = rng.random(300)

        order = np.argsort(-y_score, kind='stable')
        got = _weighted_average_precision(y_true[order], np.ones(300))[0]

        assert got == pytest.approx(average_precision_score(y_true, y_score))

    def test_integer_weights_match_physically_duplicating_rows(self):
        """A weight of k must be identical to the row appearing k times."""
        rng = np.random.default_rng(4)
        y_true = rng.integers(0, 2, 40)
        y_score = rng.random(40)
        weights = rng.integers(0, 4, 40)

        order = np.argsort(-y_score, kind='stable')
        weighted = _weighted_average_precision(y_true[order], weights[order])[0]

        expanded_true = np.repeat(y_true, weights)
        expanded_score = np.repeat(y_score, weights)
        assert weighted == pytest.approx(
            average_precision_score(expanded_true, expanded_score))

    def test_resample_with_no_positives_gives_zero_not_nan(self):
        y_true = np.array([1, 0, 0])
        # Weight the only positive row out of the resample entirely.
        got = _weighted_average_precision(y_true, np.array([0.0, 1.0, 1.0]))
        assert got[0] == 0.0


def _two_cube_fixture(n_per_cube=60, n_cubes=12, seed=5):
    """Labels/scores where cube membership genuinely carries signal.

    Each cube gets its own positive rate and its own score offset, so pixels within a
    cube are correlated -- which is the property that makes cube-level and pixel-level
    bootstraps disagree.
    """
    rng = np.random.default_rng(seed)
    cube_idx = np.repeat(np.arange(n_cubes), n_per_cube)
    per_cube_rate = rng.uniform(0.05, 0.8, n_cubes)
    y_true = rng.random(len(cube_idx)) < per_cube_rate[cube_idx]
    per_cube_offset = rng.uniform(-0.3, 0.3, n_cubes)
    y_score = np.clip(rng.random(len(cube_idx)) * 0.5 + y_true * 0.4
                      + per_cube_offset[cube_idx], 0, 1)
    return y_true.astype(int), y_score, cube_idx


class TestBootstrapF1:
    def test_point_estimate_matches_sklearn(self):
        y_true, y_score, cube_idx = _two_cube_fixture()
        y_pred = (y_score >= 0.5).astype(int)

        result = bootstrap_f1_ci(y_true, y_pred, cube_idx, n_bootstrap=200,
                                 rng=np.random.default_rng(0))

        assert result['point_estimate'] == pytest.approx(
            f1_score(y_true, y_pred, zero_division=0))
        assert result['ci_lower'] <= result['point_estimate'] <= result['ci_upper']

    def test_cube_bootstrap_is_wider_than_pixel_bootstrap(self):
        """The reason this module resamples cubes at all.

        Pixels inside a cube are correlated, so treating each as an independent draw
        understates the spread. If this ever inverts, the resampling unit is wrong.
        """
        y_true, y_score, cube_idx = _two_cube_fixture()
        y_pred = (y_score >= 0.5).astype(int)

        cube_level = bootstrap_f1_ci(y_true, y_pred, cube_idx, n_bootstrap=400,
                                     rng=np.random.default_rng(0))
        # Same machinery, but every pixel is its own "cube" -- i.e. a pixel bootstrap.
        pixel_level = bootstrap_f1_ci(y_true, y_pred, np.arange(len(y_true)),
                                      n_bootstrap=400, rng=np.random.default_rng(0))

        assert cube_level['bootstrap_std'] > 2 * pixel_level['bootstrap_std']

    def test_bca_returns_a_valid_interval(self):
        y_true, y_score, cube_idx = _two_cube_fixture()
        y_pred = (y_score >= 0.5).astype(int)

        result = bootstrap_f1_ci(y_true, y_pred, cube_idx, n_bootstrap=300,
                                 method='bca', rng=np.random.default_rng(0))

        assert result['ci_method'] == 'bca'
        assert result['ci_lower'] < result['ci_upper']
        assert 0.0 <= result['ci_lower'] <= 1.0


class TestBootstrapPrAuc:
    def test_point_estimate_matches_sklearn(self):
        y_true, y_score, cube_idx = _two_cube_fixture()

        result = bootstrap_pr_auc_ci(y_true, y_score, cube_idx, n_bootstrap=100,
                                     rng=np.random.default_rng(0))

        assert result['point_estimate'] == pytest.approx(
            average_precision_score(y_true, y_score))

    def test_binning_at_full_resolution_is_exact(self):
        """With at least as many bins as rows, the binned path must be the exact sweep.

        This is what makes the optimisation safe to default on: it degrades to the
        original computation rather than approximating, whenever it can afford to.
        """
        y_true, y_score, cube_idx = _two_cube_fixture()
        weights = draw_cube_resamples(12, 32, np.random.default_rng(2))

        positions = np.searchsorted(np.unique(cube_idx), cube_idx)
        order = np.argsort(-y_score, kind='stable')
        exact = _weighted_average_precision(
            y_true[order], weights[:, positions[order]])

        pos_counts, all_counts = _bin_and_cube_counts(
            y_true[order].astype(bool), positions[order], 12, n_bins=len(y_true))
        binned = _average_precision_from_bin_counts(weights, pos_counts, all_counts)

        assert binned == pytest.approx(exact)

    def test_coarse_binning_tracks_the_exact_sweep(self):
        """Far fewer bins than rows must still reproduce the interval closely.

        The real run bins 1.28M rows; if coarse binning moved the answer materially
        the speedup would be buying a different statistic.
        """
        y_true, y_score, cube_idx = _two_cube_fixture(n_per_cube=400, n_cubes=20)

        exact = bootstrap_pr_auc_ci(y_true, y_score, cube_idx, n_bootstrap=200,
                                    rng=np.random.default_rng(2), n_bins=10 ** 9)
        coarse = bootstrap_pr_auc_ci(y_true, y_score, cube_idx, n_bootstrap=200,
                                     rng=np.random.default_rng(2), n_bins=512)

        assert coarse['ci_lower'] == pytest.approx(exact['ci_lower'], abs=2e-3)
        assert coarse['ci_upper'] == pytest.approx(exact['ci_upper'], abs=2e-3)

    def test_point_estimate_is_exact_regardless_of_binning(self):
        """The published number must never come from the approximation."""
        y_true, y_score, cube_idx = _two_cube_fixture()

        coarse = bootstrap_pr_auc_ci(y_true, y_score, cube_idx, n_bootstrap=20,
                                     rng=np.random.default_rng(2), n_bins=16)

        assert coarse['point_estimate'] == pytest.approx(
            average_precision_score(y_true, y_score))


class TestPairedBootstrapMargin:
    @pytest.mark.parametrize('metric', ['f1', 'pr_auc'])
    def test_identical_arms_give_exactly_zero_margin(self, metric):
        """The core guarantee of pairing: shared draws cancel exactly.

        If the two arms were resampled independently, this would produce a wide
        interval around zero instead of a point mass at it.
        """
        y_true, y_score, cube_idx = _two_cube_fixture()

        result = paired_bootstrap_margin(
            strategy=(y_true, y_score),
            baseline=(y_true, y_score),
            cube_idx=cube_idx,
            metric=metric,
            strategy_threshold=0.5,
            baseline_threshold=0.5,
            n_bootstrap=200,
            rng=np.random.default_rng(0),
        )

        assert result['point_estimate'] == pytest.approx(0.0, abs=1e-12)
        assert result['ci_lower'] == pytest.approx(0.0, abs=1e-12)
        assert result['ci_upper'] == pytest.approx(0.0, abs=1e-12)
        assert result['bootstrap_std'] == pytest.approx(0.0, abs=1e-12)

    def test_paired_margin_is_tighter_than_differencing_independent_arms(self):
        """Why the paired design is worth the extra plumbing.

        Both arms see the same cubes, so their errors are correlated and the variance
        of the difference is much smaller than the variance either arm carries alone.
        """
        y_true, baseline_score, cube_idx = _two_cube_fixture()
        rng = np.random.default_rng(11)
        # A strategy that differs from baseline only slightly, as the real ones do.
        strategy_score = np.clip(baseline_score + rng.normal(0, 0.02, len(baseline_score)), 0, 1)

        paired = paired_bootstrap_margin(
            strategy=(y_true, strategy_score),
            baseline=(y_true, baseline_score),
            cube_idx=cube_idx,
            metric='f1',
            strategy_threshold=0.5,
            baseline_threshold=0.5,
            n_bootstrap=400,
            rng=np.random.default_rng(0),
        )
        arm = bootstrap_f1_ci(y_true, (strategy_score >= 0.5).astype(int), cube_idx,
                              n_bootstrap=400, rng=np.random.default_rng(0))

        assert paired['bootstrap_std'] < arm['bootstrap_std']

    def test_point_margin_equals_difference_of_arm_point_estimates(self):
        y_true, baseline_score, cube_idx = _two_cube_fixture()
        rng = np.random.default_rng(12)
        strategy_score = np.clip(baseline_score + rng.normal(0, 0.05, len(baseline_score)), 0, 1)

        result = paired_bootstrap_margin(
            strategy=(y_true, strategy_score),
            baseline=(y_true, baseline_score),
            cube_idx=cube_idx,
            metric='pr_auc',
            n_bootstrap=100,
            rng=np.random.default_rng(0),
        )

        assert result['point_estimate'] == pytest.approx(
            average_precision_score(y_true, strategy_score)
            - average_precision_score(y_true, baseline_score))
        assert result['strategy_point_estimate'] == pytest.approx(
            average_precision_score(y_true, strategy_score))

    def test_mismatched_rows_raise_rather_than_compare_different_pixels(self):
        y_true, y_score, cube_idx = _two_cube_fixture()
        shuffled = y_true.copy()
        shuffled[:10] = 1 - shuffled[:10]

        with pytest.raises(ValueError, match='identical rows'):
            paired_bootstrap_margin(
                strategy=(y_true, y_score),
                baseline=(shuffled, y_score),
                cube_idx=cube_idx,
                metric='f1',
                strategy_threshold=0.5,
            baseline_threshold=0.5,
                n_bootstrap=10,
            )

    def test_unknown_metric_raises(self):
        y_true, y_score, cube_idx = _two_cube_fixture()
        with pytest.raises(ValueError, match='Unknown metric'):
            paired_bootstrap_margin((y_true, y_score), (y_true, y_score), cube_idx,
                                    metric='roc_auc', n_bootstrap=10)


class TestBenjaminiHochberg:
    def test_known_values(self):
        p = np.array([0.01, 0.02, 0.03, 0.04, 0.05])
        q = benjamini_hochberg(p)
        assert q == pytest.approx([0.05, 0.05, 0.05, 0.05, 0.05])

    def test_preserves_input_order(self):
        p = np.array([0.9, 0.001, 0.5])
        q = benjamini_hochberg(p)
        assert q[1] < q[2] < q[0]

    def test_adjusted_values_never_exceed_one(self):
        q = benjamini_hochberg(np.array([0.8, 0.9, 0.95]))
        assert np.all(q <= 1.0)

    def test_empty_input(self):
        assert len(benjamini_hochberg(np.array([]))) == 0
