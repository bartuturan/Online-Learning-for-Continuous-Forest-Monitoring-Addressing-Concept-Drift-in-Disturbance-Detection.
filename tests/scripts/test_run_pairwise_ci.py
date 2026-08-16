"""Tests for the pairwise / mean-column aggregation.

The statistics primitives are covered in tests/eval/test_bootstrap_ci.py. What is
specific here is the aggregation contract: that the mean is taken *within* a resample
rather than across independently-drawn ones, that the strategy-vs-baseline margins are
scoped and floored correctly, that the cross-objective slices really are different
trained models, and -- most importantly -- that the hardcoded config registry still
points at the models the thesis tables actually print.

Strategy-vs-strategy pairs are tested in tests/scripts/test_run_group_ci.py, which
covers the one remaining pair table; TestPairTableRemoved below pins the fact that this
module no longer produces a second, redundant one.
"""
import itertools

import numpy as np
import pandas as pd
import pytest

import scripts.run_pairwise_ci as pw
from src.eval.bootstrap_ci import draw_cube_resamples


class TestConfigRegistry:
    def test_seven_strategies_plus_baseline_in_each_panel(self):
        for rows in (pw.FORECAST_ROWS, pw.RETENTION_ROWS):
            assert len(rows) == 8
            assert rows[0][0] == pw.BASELINE_LABEL
            assert len({family_id for _, family_id in rows}) == 8

    def test_twenty_one_pairs_among_the_seven_strategies(self):
        strategies = [label for label, _ in pw.FORECAST_ROWS if label != pw.BASELINE_LABEL]
        assert len(strategies) == 7
        assert len(list(itertools.combinations(strategies, 2))) == 21

    def test_all_seven_retention_configs_differ_from_forecast(self):
        """The two panels are different trained models, not the same models re-scored.

        Misclassification Buffer used to be the one exception, shared between the
        panels because it ran at a single ratio and so was never selected for either
        objective. It now sweeps RR 0.2-0.5 and selects RR=0.4 forecasting, RR=0.5
        retaining, so every strategy differs between the panels.
        """
        forecast = {f for _, f in pw.FORECAST_ROWS[1:]}
        retention = {f for _, f in pw.RETENTION_ROWS[1:]}
        assert not (forecast & retention)

    def test_slice_cells_match_the_printed_year_columns(self):
        assert [c[1] for c in pw.FORECAST_CELLS] == [2018, 2019, 2020, 2021, 2022]
        assert [c[1] for c in pw.RETENTION_CELLS] == [2017, 2018, 2019, 2020, 2021, 2022]
        # Forecast is next-year: each model scored one year ahead of its training year.
        assert all(eval_year - model_year == 1 for model_year, eval_year in pw.FORECAST_CELLS)
        # Retention is the final 2022 model back-tested.
        assert all(model_year == 2022 for model_year, _ in pw.RETENTION_CELLS)

    def test_config_rows_match_published_tables(self):
        """Every registered family_id must exist in the published table for its slice.

        The registry was resolved by matching printed values; this keeps it honest if
        a family is ever renamed or a table regenerated.
        """
        published = pd.read_csv(pw.PUBLISHED_TABLE, float_precision='round_trip')
        for slice_name, spec in pw.SLICES.items():
            for _, family_id in spec['rows']:
                for model_year, eval_year in spec['cells']:
                    match = published[
                        (published.family_id == family_id)
                        & (published.table_type == spec['table_type'])
                        & (published.model_year == model_year)
                        & (published.eval_year == eval_year)
                    ]
                    assert len(match) == 1, (
                        f'{slice_name}: {family_id} model={model_year} '
                        f'eval={eval_year} -> {len(match)} rows')


class TestSharedResampleMean:
    def test_elementwise_mean_equals_score_then_average_within_a_draw(self):
        """The identity the aggregation relies on.

        Averaging per-year sample vectors elementwise is only equal to "resample once,
        score every year on that draw, then average" because each year sees the same
        weights row. This asserts that equality directly.
        """
        rng = np.random.default_rng(0)
        weights = draw_cube_resamples(20, 128, np.random.default_rng(42))
        # Three years of per-cube counts, scored on the shared draws.
        per_year = [rng.integers(1, 40, (20,)).astype(float) for _ in range(3)]
        per_year_samples = [weights @ counts for counts in per_year]

        elementwise = np.mean(per_year_samples, axis=0)
        within_draw = np.array([
            np.mean([weights[b] @ counts for counts in per_year])
            for b in range(len(weights))
        ])
        assert elementwise == pytest.approx(within_draw)

    def test_independent_per_year_draws_understate_the_spread(self):
        """Why the shared draw is required, not merely convenient.

        Cubes recur across years, so a hard cube is hard in most years it appears in.
        Independent draws let that variation cancel when averaging, shrinking the
        spread of the mean and producing an over-confident interval.
        """
        rng = np.random.default_rng(1)
        n_cubes, n_boot = 25, 2000
        # A per-cube effect shared across years -- the correlation being preserved.
        cube_effect = rng.normal(10, 4, n_cubes)
        per_year = [cube_effect + rng.normal(0, 0.3, n_cubes) for _ in range(6)]

        shared = draw_cube_resamples(n_cubes, n_boot, np.random.default_rng(7))
        shared_mean = np.mean([shared @ c for c in per_year], axis=0)

        independent = np.mean(
            [draw_cube_resamples(n_cubes, n_boot, np.random.default_rng(100 + i)) @ c
             for i, c in enumerate(per_year)], axis=0)

        assert shared_mean.std() > 1.5 * independent.std()


class TestPairTableRemoved:
    """strategy_pairs.csv was dropped as a duplicate of simultaneous_pairs.csv.

    Pinned as a test because the redundancy is easy to reintroduce: the pair margins
    are three lines of itertools away, and a second multiplicity story for the same
    21 comparisons is exactly what was removed.
    """

    def test_two_sided_p_helper_is_gone(self):
        assert not hasattr(pw, 'two_sided_p')

    def test_run_returns_only_mean_and_baseline_frames(self):
        import inspect
        source = inspect.getsource(pw.run)
        assert 'strategy_pairs.csv' not in source


@pytest.fixture(scope='module')
def pairwise_result(tmp_path_factory):
    """One real run(), shared by every class that needs its output.

    Reads the real cached predictions (read-only, safe) but must never write back into
    experiments/bootstrap_ci/pairwise/ -- that directory holds the real 2000-resample
    results this analysis is reported from, and a stray test write previously clobbered
    it with 64-resample numbers. pytest's regular `monkeypatch` fixture is
    function-scoped and cannot be used here, so the MonkeyPatch object is managed
    directly.

    Small n_bootstrap and n_bins: these tests exercise shape and plumbing, not interval
    accuracy. Point estimates are unaffected by n_bins (pr_auc_bin_table always
    computes them on the exact unbinned curve, and F1 is exact), so the assertions that
    compare against published numbers stay meaningful while the run drops from minutes
    to seconds. Module-scoped rather than class-scoped so the 154 reductions happen
    once for the whole file instead of once per class.
    """
    tmp_dir = tmp_path_factory.mktemp('pairwise_ci_test')
    mp = pytest.MonkeyPatch()
    mp.setattr(pw, 'PAIRWISE_DIR', tmp_dir)
    mp.setattr(pw, 'SAMPLES_PATH', tmp_dir / 'arm_samples.npz')
    yield pw.run(n_bootstrap=64, seed=42, alpha=0.05, f1_tolerance=1e-4, n_bins=1024)
    mp.undo()


class TestRunOutputs:

    def test_baseline_present_in_mean_column(self, pairwise_result):
        mean_df, _ = pairwise_result
        assert pw.BASELINE_LABEL in set(mean_df.row_label)

    def test_mean_column_reproduces_published_per_year_mean(self, pairwise_result):
        """The point estimate must equal the Mean column the thesis prints."""
        mean_df, _ = pairwise_result
        published = pd.read_csv(pw.PUBLISHED_TABLE, float_precision='round_trip')
        for slice_name in ('forecast_mean', 'retention_mean'):
            spec = pw.SLICES[slice_name]
            for label, family_id in spec['rows']:
                expected = np.mean([
                    published[(published.family_id == family_id)
                              & (published.table_type == spec['table_type'])
                              & (published.model_year == my)
                              & (published.eval_year == ey)].iloc[0].f1_score
                    for my, ey in spec['cells']])
                got = mean_df[(mean_df['slice'] == slice_name)
                              & (mean_df.row_label == label)
                              & (mean_df.metric == 'f1_score')].iloc[0].point_estimate
                assert got == pytest.approx(expected, abs=1e-6), f'{slice_name}/{label}'

    def test_retention_2017_is_a_single_year_not_a_mean(self, pairwise_result):
        mean_df, _ = pairwise_result
        rows = mean_df[mean_df['slice'] == 'retention_2017']
        assert set(rows.n_years_averaged) == {1}

    def test_sample_vectors_persisted_for_reuse(self, pairwise_result):
        assert pw.SAMPLES_PATH.exists()
        with np.load(pw.SAMPLES_PATH) as payload:
            assert len(payload.files) == len(pw.SLICES) * 8 * len(pw.METRICS)
            assert all(payload[k].shape == (64,) for k in payload.files)


class TestBaselineMargins:
    """The strategy-vs-baseline mean margin.

    This is the comparison Chapter 4 argues from most ("every strategy improves on the
    unmitigated baseline"), and before baseline_margins.csv existed it had no
    mean-level interval anywhere in the outputs.
    """

    def test_one_row_per_strategy_excluding_the_baseline_itself(self, pairwise_result):
        _, baseline_df = pairwise_result
        counts = baseline_df.groupby(['slice', 'metric']).size()
        assert set(counts.values) == {7}
        assert len(counts) == len(pw.SLICES) * len(pw.METRICS)
        assert pw.BASELINE_LABEL not in set(baseline_df.row_label)

    def test_margin_equals_difference_of_the_two_point_estimates(self, pairwise_result):
        _, baseline_df = pairwise_result
        expected = baseline_df.strategy_point_estimate - baseline_df.baseline_point_estimate
        assert expected.values == pytest.approx(baseline_df.point_estimate.values)

    def test_every_row_shares_one_baseline_family(self, pairwise_result):
        _, baseline_df = pairwise_result
        assert baseline_df.baseline_family_id.nunique() == 1

    def test_p_value_is_floored_and_capped(self, pairwise_result):
        """A strategy winning every resample must not report a bare zero."""
        _, baseline_df = pairwise_result
        n_boot = int(baseline_df.n_bootstrap.iloc[0])
        assert (baseline_df.p_value_one_sided >= 1 / (n_boot + 1) - 1e-12).all()
        assert (baseline_df.p_value_one_sided <= 1.0).all()

    def test_holm_is_scoped_within_slice_and_metric(self, pairwise_result):
        _, baseline_df = pairwise_result
        assert set(baseline_df.holm_family_size) == {7}
        assert baseline_df.holm_family.nunique() == len(pw.SLICES) * len(pw.METRICS)
        # Holm corrects the two-sided tail, and an adjustment can only inflate a p.
        assert set(baseline_df.holm_p_side) == {'two_sided'}
        assert (baseline_df.p_value_holm >= baseline_df.p_value_two_sided - 1e-12).all()

    def test_written_to_disk_with_json_twin(self, pairwise_result):
        assert (pw.PAIRWISE_DIR / 'baseline_margins.csv').exists()
        assert (pw.PAIRWISE_DIR / 'baseline_margins.json').exists()


class TestCrossObjectiveSlices:
    """Each configuration scored on the objective it was *not* selected for.

    These are Table 4.9's off-diagonal columns. Section 5.6's single-model
    recommendation rests on one of them, so they need intervals like anything else.
    """

    def test_cross_slices_reuse_the_other_slices_cells(self):
        assert pw.SLICES['forecast_at_retention_config']['cells'] == pw.FORECAST_CELLS
        assert pw.SLICES['retention_at_forecast_config']['cells'] == pw.RETENTION_CELLS

    def test_cross_slices_carry_the_other_panels_configurations(self):
        assert pw.SLICES['forecast_at_retention_config']['rows'] == pw.RETENTION_ROWS
        assert pw.SLICES['retention_at_forecast_config']['rows'] == pw.FORECAST_ROWS

    def test_table_type_matches_the_cells_not_the_rows(self):
        """Thresholds must come from the panel the cells belong to."""
        assert pw.SLICES['forecast_at_retention_config']['table_type'] == 'next_year'
        assert pw.SLICES['retention_at_forecast_config']['table_type'] == 'prior_years'

    def test_cross_slice_is_not_a_relabelled_copy_of_the_diagonal(self):
        """All seven strategies change configuration between objectives.

        Only the baseline is the same trained model in both panels, since it is a
        single configuration and so is never selected. Misclassification Buffer was
        the second shared row until it was swept over four ratios. If this ever grew
        to cover a strategy, the cross slices would be scoring the same models twice
        and their intervals would say nothing about the trade-off.
        """
        forecast_families = set(dict(pw.FORECAST_ROWS).values())
        retention_families = set(dict(pw.RETENTION_ROWS).values())
        shared = forecast_families & retention_families
        assert shared == {pw.FORECAST_ROWS[0][1]}
        assert pw.SLICES['forecast_at_retention_config']['rows'] == pw.RETENTION_ROWS
        # 7 strategies on each side, disjoint; only the baseline is common.
        assert len(forecast_families ^ retention_families) == 14
