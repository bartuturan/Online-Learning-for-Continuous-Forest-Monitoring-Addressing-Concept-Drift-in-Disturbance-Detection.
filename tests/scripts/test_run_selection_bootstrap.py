"""Tests for the selection-procedure bootstrap.

The arithmetic that matters here is not the resampling (tests/eval/test_bootstrap_ci.py
covers that) but three claims specific to bootstrapping an argmax:

  * a sweep is classified correctly, including the two id patterns that overlap;
  * the winner is chosen per resample on the selection metric and then read off on
    *both* metrics, so F1 follows the configuration PR-AUC picked;
  * the reported bias isolates the winner's curse from the ordinary resampling bias
    every arm carries -- a sweep of one must report exactly zero selection bias, or a
    single-configuration strategy gets charged for a selection it never made.

Most of this is hermetic. TestAgainstRealCache runs the real pipeline over one small
sweep, because the guard that the observed argmax reproduces the configuration the
thesis prints can only be exercised against real predictions.
"""
import numpy as np
import pandas as pd
import pytest

import scripts.run_selection_bootstrap as sb


class TestClassify:
    def test_baseline_is_not_a_swept_strategy(self):
        assert sb.classify('mlp_prevyears_monthly_features_incremental_scaler') is None

    @pytest.mark.parametrize('family_id,expected', [
        ('mlp_prevyears_monthly_features_incremental_scaler_experience_replay_RR_0.4',
         'Uniform Random'),
        ('mlp_prevyears_monthly_features_incremental_scaler_experience_replay_RR_0.2_PR_0.10',
         'PR Stratification'),
        ('mlp_prevyears_monthly_features_incremental_scaler_experience_replay_confidently_correct_memory_RR_0.2',
         'Confidently Correct'),
        ('mlp_prevyears_monthly_features_incremental_scaler_experience_replay_hard_example_mining_RR_0.5',
         'Hard Example Mining'),
        ('mlp_prevyears_monthly_features_incremental_scaler_experience_replay_uncertainity_prioritization_RR_0.4',
         'Uncertainty Prioritization'),
        ('mlp_prevyears_monthly_features_incremental_scaler_missclassification_buffer_RR_0.3',
         'Misclassification Buffer'),
        ('mlp_combined_HE=0_CC=0.2_UP=0.2_PR=(0.1,10)_RR=0.5', 'Combined'),
    ])
    def test_each_family_maps_to_its_criterion(self, family_id, expected):
        assert sb.classify(family_id) == expected

    def test_pr_stratification_is_not_swallowed_by_uniform_random(self):
        """Both ids contain 'experience_replay_RR'; only the anchored pattern separates
        them, so a reordering of STRATEGY_PATTERNS would silently merge two sweeps."""
        pr = 'mlp_prevyears_monthly_features_incremental_scaler_experience_replay_RR_0.2_PR_0.10'
        uniform = 'mlp_prevyears_monthly_features_incremental_scaler_experience_replay_RR_0.2'
        assert sb.classify(pr) == 'PR Stratification'
        assert sb.classify(uniform) == 'Uniform Random'

    @pytest.mark.parametrize('family_id', [
        'prevyears_monthly_features_incremental_scaler_experience_replay_RR_0.2',
        'prevyears_monthly_features_incremental_scaler_experience_replay_RR_0.5',
        'lagged_features_incremental_scaler',
        'baseline',
    ])
    def test_sgd_families_are_out_of_scope(self, family_id):
        """The Table 3.1 SGD ladder shares the MLP id suffixes minus the 'mlp_' prefix.
        Unanchored patterns pulled four of them into the Uniform Random sweep, which
        would have taken the argmax over configurations the thesis never swept -- and
        which have no cached predictions at all."""
        assert sb.classify(family_id) is None

    def test_scope_matches_the_rest_of_the_pipeline(self):
        """classify() must admit exactly what DEFAULT_STRATEGY_PATTERN admits, or the
        selection bootstrap and the per-cell bootstrap describe different populations."""
        import re
        from src.eval.bootstrap_ci import DEFAULT_STRATEGY_PATTERN
        for family_id in [
            'mlp_combined_HE=0_CC=0.2_UP=0.2_PR=(0.1,10)_RR=0.5',
            'mlp_prevyears_monthly_features_incremental_scaler_experience_replay_RR_0.4',
            'mlp_prevyears_monthly_features_incremental_scaler_missclassification_buffer_RR_0.3',
            'prevyears_monthly_features_incremental_scaler_experience_replay_RR_0.4',
            'lagged_features',
        ]:
            in_scope = bool(re.search(DEFAULT_STRATEGY_PATTERN, family_id))
            assert (sb.classify(family_id) is not None) == in_scope, family_id


class TestBuildSweeps:
    @staticmethod
    def _published(rows):
        return pd.DataFrame(rows, columns=['family_id', 'table_type', 'model_year', 'eval_year'])

    def test_families_missing_a_cell_are_excluded(self):
        cells = [(2020, 2021), (2021, 2022)]
        complete = 'mlp_prevyears_monthly_features_incremental_scaler_experience_replay_RR_0.4'
        partial = 'mlp_prevyears_monthly_features_incremental_scaler_experience_replay_RR_0.5'
        published = self._published(
            [(complete, 'next_year', my, ey) for my, ey in cells]
            + [(partial, 'next_year', 2020, 2021)])
        sweeps = sb.build_sweeps(published, 'next_year', cells)
        assert sweeps == {'Uniform Random': [complete]}

    def test_baseline_never_enters_a_sweep(self):
        cells = [(2020, 2021)]
        published = self._published(
            [(sb.DEFAULT_BASELINE_FAMILY, 'next_year', 2020, 2021)])
        assert sb.build_sweeps(published, 'next_year', cells) == {}


class TestSelectWithinSweep:
    @staticmethod
    def _arms(levels, n_boot=200, seed=0):
        """Arms whose pr_auc ordering varies across resamples but f1 tracks pr_auc."""
        rng = np.random.default_rng(seed)
        arms = {}
        for name, level in levels.items():
            pr = level + rng.normal(0, 0.02, n_boot)
            arms[name] = {'pr_auc': pr, 'f1_score': pr + 0.1,
                          'point_pr_auc': level, 'point_f1_score': level + 0.1}
        return arms

    def test_selected_pr_auc_is_the_per_resample_maximum(self):
        arms = self._arms({'a': 0.30, 'b': 0.31, 'c': 0.29})
        ids = ['a', 'b', 'c']
        selected, winner, _ = sb.select_within_sweep(arms, ids)
        expected = np.max(np.stack([arms[i]['pr_auc'] for i in ids]), axis=0)
        assert np.allclose(selected['pr_auc'], expected)

    def test_f1_follows_the_configuration_pr_auc_chose(self):
        """Selection is on PR-AUC; F1 must be the winner's F1, not F1's own maximum."""
        arms = self._arms({'a': 0.30, 'b': 0.31, 'c': 0.29})
        ids = ['a', 'b', 'c']
        selected, winner, _ = sb.select_within_sweep(arms, ids)
        expected = np.array([arms[ids[w]]['f1_score'][i] for i, w in enumerate(winner)])
        assert np.allclose(selected['f1_score'], expected)

    def test_observed_winner_uses_point_estimates_not_resamples(self):
        arms = self._arms({'a': 0.30, 'b': 0.31, 'c': 0.29})
        _, _, observed = sb.select_within_sweep(arms, ['a', 'b', 'c'])
        assert observed == 1

    def test_single_configuration_sweep_always_wins(self):
        arms = self._arms({'only': 0.30})
        selected, winner, observed = sb.select_within_sweep(arms, ['only'])
        assert (winner == 0).all() and observed == 0
        assert np.array_equal(selected['pr_auc'], arms['only']['pr_auc'])

    def test_win_fractions_sum_to_one(self):
        arms = self._arms({'a': 0.30, 'b': 0.305, 'c': 0.29})
        ids = ['a', 'b', 'c']
        _, winner, _ = sb.select_within_sweep(arms, ids)
        assert sum(np.mean(winner == j) for j in range(len(ids))) == pytest.approx(1.0)

    def test_a_dominated_configuration_never_wins(self):
        arms = self._arms({'a': 0.30, 'b': 0.31})
        arms['dominated'] = {'pr_auc': np.full_like(arms['a']['pr_auc'], -1.0),
                             'f1_score': np.full_like(arms['a']['f1_score'], -1.0),
                             'point_pr_auc': -1.0, 'point_f1_score': -1.0}
        _, winner, _ = sb.select_within_sweep(arms, ['a', 'b', 'dominated'])
        assert np.mean(winner == 2) == 0.0


class TestAgainstRealCache:
    """One real sweep end to end. Hard Example Mining is chosen because it has four
    configurations (so selection is live) and is small enough to stay quick."""

    @staticmethod
    @pytest.fixture(scope='class')
    def result(tmp_path_factory):
        tmp_dir = tmp_path_factory.mktemp('selection_bootstrap_test')
        mp = pytest.MonkeyPatch()
        mp.setattr(sb, 'OUTPUT_DIR', tmp_dir)
        mp.setattr(sb, 'OBJECTIVES', {'forecast': sb.OBJECTIVES['forecast']})
        real_build = sb.build_sweeps
        mp.setattr(sb, 'build_sweeps', lambda pub, tt, cells: {
            k: v for k, v in real_build(pub, tt, cells)
            .items() if k in ('Hard Example Mining', 'Misclassification Buffer')})
        yield sb.run(n_bootstrap=64, seed=42, alpha=0.05, f1_tolerance=1e-4, n_bins=1024)
        mp.undo()

    @staticmethod
    @pytest.fixture(scope='class')
    def single_arm_result(tmp_path_factory):
        """The same run with Misclassification Buffer trimmed to a single arm.

        No real strategy is a sweep of one any more -- Misclassification Buffer was
        the last, and it now sweeps RR 0.2-0.5 -- so the k=1 guarantees below would
        otherwise have nothing to assert against. Trimming a real sweep keeps them
        exercising the real decomposition path rather than a synthetic stand-in.

        It must be trimmed to the arm FORECAST_ROWS reports (RR=0.4), not an arbitrary
        one: run() raises when the observed argmax is not the reported configuration,
        and with k=1 the argmax is whichever arm is left.
        """
        tmp_dir = tmp_path_factory.mktemp('selection_bootstrap_single_arm_test')
        mp = pytest.MonkeyPatch()
        mp.setattr(sb, 'OUTPUT_DIR', tmp_dir)
        mp.setattr(sb, 'OBJECTIVES', {'forecast': sb.OBJECTIVES['forecast']})
        real_build = sb.build_sweeps

        def one_arm(pub, tt, cells):
            sweeps = real_build(pub, tt, cells)
            trimmed = [f for f in sweeps['Misclassification Buffer'] if f.endswith('RR_0.4')]
            assert len(trimmed) == 1, 'RR_0.4 arm missing from the real sweep'
            return {'Misclassification Buffer': trimmed}

        mp.setattr(sb, 'build_sweeps', one_arm)
        yield sb.run(n_bootstrap=64, seed=42, alpha=0.05, f1_tolerance=1e-4, n_bins=1024)
        mp.undo()

    def test_observed_argmax_reproduces_the_reported_configuration(self, result):
        """run() raises if it does not, so reaching here is the assertion; this pins
        which configuration that was."""
        arm_df, _, _ = result
        selected = arm_df[arm_df.strategy == 'Hard Example Mining'].selected_label.unique()
        assert list(selected) == ['Hard Example Mining (RR=0.4)']

    def test_single_config_sweep_reports_exactly_zero_selection_bias(self, single_arm_result):
        """The bug this decomposition exists to prevent: a strategy swept at one
        configuration must not be charged a winner's curse."""
        arm_df, _, margin_df = single_arm_result
        single = arm_df[arm_df.strategy == 'Misclassification Buffer']
        assert (single.selection_bias == 0.0).all()
        assert (single.bias_corrected_point == single.point_estimate).all()
        assert (margin_df[margin_df.strategy == 'Misclassification Buffer']
                .selection_bias == 0.0).all()

    def test_single_config_sweep_has_identical_naive_and_selection_intervals(self, single_arm_result):
        arm_df, _, _ = single_arm_result
        single = arm_df[arm_df.strategy == 'Misclassification Buffer']
        assert (single.naive_ci_lower == single.selection_ci_lower).all()
        assert (single.naive_ci_upper == single.selection_ci_upper).all()

    def test_multi_config_sweep_has_nonzero_selection_bias(self, result):
        arm_df, _, _ = result
        for strategy in ('Hard Example Mining', 'Misclassification Buffer'):
            multi = arm_df[arm_df.strategy == strategy]
            assert (multi.selection_bias != 0.0).all(), strategy

    def test_resampling_bias_is_reported_separately_and_is_nonzero(self, result):
        """Present for every arm including the k=1 sweep -- that is the whole point of
        splitting it out of selection_bias."""
        arm_df, _, _ = result
        assert (arm_df.resampling_bias != 0.0).all()

    def test_win_fractions_sum_to_one_within_each_sweep(self, result):
        _, stability_df, _ = result
        totals = stability_df.groupby(['objective', 'strategy']).win_fraction.sum()
        assert totals.values == pytest.approx(1.0)

    def test_exactly_one_reported_config_per_sweep(self, result):
        _, stability_df, _ = result
        counts = stability_df.groupby(['objective', 'strategy']).is_reported_config.sum()
        assert set(counts.values) == {1}

    def test_margin_point_estimate_is_naive_not_selection_aware(self, result):
        """The point estimate stays the published number; only the interval and the
        explicit correction column change."""
        _, _, margin_df = result
        expected = margin_df.point_estimate + margin_df.baseline_point_estimate
        arm_df, _, _ = result
        for row in margin_df.itertuples():
            arm = arm_df[(arm_df.strategy == row.strategy) & (arm_df.metric == row.metric)].iloc[0]
            assert row.point_estimate == pytest.approx(
                arm.point_estimate - row.baseline_point_estimate)
        assert len(expected) == len(margin_df)

    def test_tables_written_with_json_twins(self, result):
        for name in ('selected_arm_ci', 'selection_stability', 'selection_margins'):
            assert (sb.OUTPUT_DIR / f'{name}.csv').exists()
            assert (sb.OUTPUT_DIR / f'{name}.json').exists()

    def test_corrected_interval_is_the_selection_interval_shifted_by_the_curse(self, result):
        arm_df, _, margin_df = result
        for df in (arm_df, margin_df):
            assert df.corrected_ci_lower.values == pytest.approx(
                (df.selection_ci_lower - df.selection_bias).values)
            assert df.corrected_ci_upper.values == pytest.approx(
                (df.selection_ci_upper - df.selection_bias).values)

    def test_selection_interval_dominates_naive_on_the_selection_metric(self, result):
        """max_j is taken on PR-AUC, so the resampled maximum is >= the fixed winner's
        score on every draw. The interval can therefore only move up, which is why
        `selection_excludes_zero` must never be read as a significance test: an effect
        could gain significance purely by being re-selected."""
        _, _, margin_df = result
        pr = margin_df[margin_df.metric == sb.SELECTION_METRIC]
        assert (pr.selection_ci_lower >= pr.naive_ci_lower - 1e-12).all()
        assert (pr.selection_ci_upper >= pr.naive_ci_upper - 1e-12).all()
        assert (pr.selection_bias >= -1e-12).all()

    def test_f1_is_not_dominated_because_it_is_not_the_selection_metric(self, result):
        """F1 is read off whichever configuration PR-AUC picked, and that configuration
        can have worse F1 than the fixed winner -- so no dominance and the curse may be
        negative. Asserted as a property of the construction, not of these numbers."""
        arm_df, _, _ = result
        multi = arm_df[(arm_df.metric == 'f1_score') & (arm_df.sweep_size > 1)]
        assert len(multi) > 0
        # Nothing forces a sign; the test is that the code does not assume one.
        assert multi.selection_bias.notna().all()
