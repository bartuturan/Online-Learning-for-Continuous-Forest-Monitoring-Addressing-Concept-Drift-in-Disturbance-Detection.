"""Tests for the planned group-contrast aggregation.

Hermetic in the same way as tests/scripts/test_run_group_ci.py: a small synthetic
arm_samples.npz + mean_column_ci.csv is built behind monkeypatched paths rather than
depending on scripts/run_pairwise_ci.py having run.

What is worth testing here is not the bootstrap math (tests/eval/test_bootstrap_ci.py
covers that) but the claims the module docstring makes about the *grouping*: that the
partition is total and unambiguous, that a group score is the unweighted mean of its
members, that one critical value is genuinely shared across a contrast family, and
that `resolution` bounds every half-width in the band it summarises.

One test is deliberately not hermetic -- TestRealPartition checks the real
run_pairwise_ci.SLICES against the real GROUPS, because the failure this guards
against is exactly the two drifting apart when a row is renamed.
"""
import itertools

import numpy as np
import pandas as pd
import pytest

import scripts.run_group_contrasts as gc
import scripts.run_pairwise_ci as pw

LABELS = [
    'Baseline (No Replay)',
    'Uniform Random (RR=0.4)',
    'PR Stratification (RR=0.2, PR=0.10)',
    'Confidently Correct (RR=0.4)',
    'Uncertainty Prioritization (RR=0.5)',
    'Hard Example Mining (RR=0.4)',
    'Misclassification Buffer (RR=0.4)',
    'Combined (RR=0.5, HE/CC/UP/PR=.1/.1/.1/.2)',
]


@pytest.fixture
def synthetic(tmp_path, monkeypatch):
    """One (slice, metric) with all eight arms and a known group ordering.

    Members share a common noise term, as the real pipeline guarantees by scoring
    every arm on the identical draw sequence; a construction with independent noise
    would let the max-T family look less correlated than it really is.
    """
    rng = np.random.default_rng(0)
    n_boot = 500
    shared = rng.normal(0, 0.01, n_boot)
    level = {
        'Baseline (No Replay)': 0.30,
        'Uniform Random (RR=0.4)': 0.38,
        'PR Stratification (RR=0.2, PR=0.10)': 0.39,
        'Confidently Correct (RR=0.4)': 0.39,
        'Uncertainty Prioritization (RR=0.5)': 0.40,
        'Hard Example Mining (RR=0.4)': 0.32,
        'Misclassification Buffer (RR=0.4)': 0.31,
        'Combined (RR=0.5, HE/CC/UP/PR=.1/.1/.1/.2)': 0.385,
    }
    samples = {label: level[label] + shared + rng.normal(0, 0.004, n_boot)
               for label in LABELS}
    points = {label: float(np.mean(samples[label])) for label in LABELS}

    rows = [(label, f'family_{label}') for label in LABELS]
    slices = {'demo': dict(rows=rows, cells=[(2021, 2022)], table_type='next_year')}

    tmp_dir = tmp_path / 'pairwise'
    tmp_dir.mkdir()
    np.savez_compressed(
        tmp_dir / 'arm_samples.npz',
        **{f'demo|{label}|f1_score': samples[label] for label in LABELS})
    pd.DataFrame([
        {'slice': 'demo', 'row_label': label, 'metric': 'f1_score',
         'point_estimate': points[label]}
        for label in LABELS
    ]).to_csv(tmp_dir / 'mean_column_ci.csv', index=False)

    monkeypatch.setattr(gc, 'PAIRWISE_DIR', tmp_dir)
    monkeypatch.setattr(gc, 'SAMPLES_PATH', tmp_dir / 'arm_samples.npz')
    monkeypatch.setattr(gc, 'MEAN_CI_PATH', tmp_dir / 'mean_column_ci.csv')
    monkeypatch.setattr(gc, 'GROUP_MEANS_PATH', tmp_dir / 'group_means.csv')
    monkeypatch.setattr(gc, 'CONTRASTS_PATH', tmp_dir / 'group_contrasts.csv')
    monkeypatch.setattr(gc, 'HOMOGENEITY_PATH', tmp_dir / 'band_homogeneity.csv')
    monkeypatch.setattr(gc, 'SLICES', slices)
    monkeypatch.setattr(gc, 'METRICS', ('f1_score',))

    return {'samples': samples, 'points': points, 'tmp_dir': tmp_dir}


class TestAssignGroups:
    def test_every_arm_lands_in_exactly_one_group(self):
        membership = gc.assign_groups(LABELS)
        assigned = sorted(itertools.chain.from_iterable(membership.values()))
        assert assigned == sorted(LABELS)

    def test_group_sizes_match_the_documented_partition(self):
        membership = gc.assign_groups(LABELS)
        assert {name: len(members) for name, members in membership.items()} == {
            'G0_no_replay': 1, 'G1_uncurated': 1, 'G2_curated_non_error': 3,
            'G3_error_driven': 2, 'G4_multi_criterion': 1,
        }

    def test_unclassified_arm_raises_rather_than_being_dropped(self):
        with pytest.raises(ValueError, match='matched 0 groups'):
            gc.assign_groups(LABELS + ['Some New Strategy (RR=0.6)'])

    def test_missing_group_raises(self):
        with pytest.raises(ValueError, match='No arms found for group'):
            gc.assign_groups([label for label in LABELS if not label.startswith('Combined')])

    def test_ambiguous_prefix_raises(self, monkeypatch):
        monkeypatch.setitem(gc.GROUPS, 'G5_overlapping', ('Uniform',))
        with pytest.raises(ValueError, match='matched 2 groups'):
            gc.assign_groups(LABELS)


class TestGroupSamples:
    def test_group_score_is_the_unweighted_member_mean(self, synthetic):
        membership = gc.assign_groups(LABELS)
        grouped, points = gc.group_samples(
            synthetic['samples'], synthetic['points'], membership)
        members = membership['G2_curated_non_error']
        expected = np.mean([synthetic['samples'][label] for label in members], axis=0)
        assert np.allclose(grouped['G2_curated_non_error'], expected)
        assert points['G2_curated_non_error'] == pytest.approx(
            float(np.mean([synthetic['points'][label] for label in members])))

    def test_singleton_group_is_its_member(self, synthetic):
        membership = gc.assign_groups(LABELS)
        grouped, _ = gc.group_samples(
            synthetic['samples'], synthetic['points'], membership)
        assert np.array_equal(grouped['G0_no_replay'],
                              synthetic['samples']['Baseline (No Replay)'])


class TestMaxTFamily:
    def test_critical_value_dominates_every_contrasts_own_quantile(self):
        rng = np.random.default_rng(1)
        diffs = rng.normal(0, 1, (400, 4)) + np.array([0.0, 0.5, -0.3, 0.1])
        points = diffs.mean(axis=0)
        critical_value, se, _, _, _ = gc.max_t_family(diffs, points, 0.05)
        own = np.percentile(np.abs((diffs - points) / se), 95, axis=0)
        assert np.all(critical_value >= own - 1e-12)

    def test_one_shared_critical_value_and_half_width_scales_with_se(self):
        rng = np.random.default_rng(2)
        diffs = rng.normal(0, 1, (400, 3)) * np.array([1.0, 2.0, 4.0])
        points = diffs.mean(axis=0)
        critical_value, se, half_width, _, _ = gc.max_t_family(diffs, points, 0.05)
        assert np.allclose(half_width, critical_value * se)

    def test_p_value_never_exactly_zero(self):
        rng = np.random.default_rng(3)
        diffs = rng.normal(0, 0.001, (200, 2)) + 10.0
        _, _, _, _, p_value = gc.max_t_family(diffs, diffs.mean(axis=0), 0.05)
        assert np.all(p_value >= 1 / 201)

    def test_identical_groups_collapse_without_division_by_zero(self):
        diffs = np.zeros((100, 2))
        critical_value, se, half_width, significant, _ = gc.max_t_family(
            diffs, np.zeros(2), 0.05)
        assert np.all(se == 0) and np.all(half_width == 0) and not significant.any()
        assert np.isfinite(critical_value)


class TestContrastRows:
    def test_one_row_per_planned_contrast_sharing_a_critical_value(self, synthetic):
        membership = gc.assign_groups(LABELS)
        grouped, points = gc.group_samples(
            synthetic['samples'], synthetic['points'], membership)
        rows = gc.contrast_rows(grouped, points, 0.05)
        assert len(rows) == len(gc.CONTRASTS)
        assert len({row['critical_value_maxt'] for row in rows}) == 1

    def test_point_estimate_equals_difference_of_group_point_estimates(self, synthetic):
        membership = gc.assign_groups(LABELS)
        grouped, points = gc.group_samples(
            synthetic['samples'], synthetic['points'], membership)
        for row in gc.contrast_rows(grouped, points, 0.05):
            assert row['point_estimate'] == pytest.approx(
                points[row['group_a']] - points[row['group_b']])
            assert row['a_point_estimate'] == pytest.approx(points[row['group_a']])

    def test_significant_contrast_has_simultaneous_ci_excluding_zero(self, synthetic):
        membership = gc.assign_groups(LABELS)
        grouped, points = gc.group_samples(
            synthetic['samples'], synthetic['points'], membership)
        rows = gc.contrast_rows(grouped, points, 0.05)
        for row in rows:
            if row['sim_significant']:
                assert row['sim_ci_lower'] > 0 or row['sim_ci_upper'] < 0

    def test_fewer_contrasts_than_pairs_gives_a_smaller_critical_value(self, synthetic):
        """The reason for preferring contrasts over the 21-pair family."""
        membership = gc.assign_groups(LABELS)
        grouped, points = gc.group_samples(
            synthetic['samples'], synthetic['points'], membership)
        contrast_crit = gc.contrast_rows(grouped, points, 0.05)[0]['critical_value_maxt']

        strategies = [label for label in LABELS if not label.startswith('Baseline')]
        pairs = list(itertools.combinations(strategies, 2))
        diffs = np.stack([synthetic['samples'][a] - synthetic['samples'][b]
                          for a, b in pairs], axis=1)
        pair_points = np.array([synthetic['points'][a] - synthetic['points'][b]
                                for a, b in pairs])
        pair_crit, *_ = gc.max_t_family(diffs, pair_points, 0.05)

        assert len(pairs) == 21
        assert contrast_crit < pair_crit


class TestBandHomogeneity:
    def test_ten_pairs_among_the_five_non_error_driven_arms(self, synthetic):
        membership = gc.assign_groups(LABELS)
        rows = gc.homogeneity_rows(
            synthetic['samples'], synthetic['points'], membership, 0.05)
        assert len(rows) == 10
        assert all(row['band_size'] == 5 for row in rows)

    def test_band_excludes_error_driven_arms_and_the_baseline(self, synthetic):
        membership = gc.assign_groups(LABELS)
        rows = gc.homogeneity_rows(
            synthetic['samples'], synthetic['points'], membership, 0.05)
        appearing = {row['strategy_a'] for row in rows} | {row['strategy_b'] for row in rows}
        assert not any(label.startswith(('Baseline', 'Hard Example', 'Misclassification'))
                       for label in appearing)

    def test_resolution_bounds_every_half_width_in_the_band(self, synthetic):
        membership = gc.assign_groups(LABELS)
        rows = gc.homogeneity_rows(
            synthetic['samples'], synthetic['points'], membership, 0.05)
        resolution = rows[0]['resolution']
        for row in rows:
            assert row['sim_ci_upper'] - row['point_estimate'] <= resolution + 1e-12

    def test_no_ordering_resolved_agrees_with_the_pair_count(self, synthetic):
        membership = gc.assign_groups(LABELS)
        rows = gc.homogeneity_rows(
            synthetic['samples'], synthetic['points'], membership, 0.05)
        n_significant = sum(row['sim_significant'] for row in rows)
        assert rows[0]['n_significant_pairs'] == n_significant
        assert rows[0]['no_ordering_resolved'] == (n_significant == 0)

    def test_reports_no_equivalence_verdict(self, synthetic):
        """Absence of a significant pair must not be surfaced as 'homogeneous'."""
        membership = gc.assign_groups(LABELS)
        rows = gc.homogeneity_rows(
            synthetic['samples'], synthetic['points'], membership, 0.05)
        assert 'homogeneous' not in rows[0]

    def test_spread_over_resolution_is_the_documented_ratio(self, synthetic):
        membership = gc.assign_groups(LABELS)
        rows = gc.homogeneity_rows(
            synthetic['samples'], synthetic['points'], membership, 0.05)
        assert rows[0]['spread_over_resolution'] == pytest.approx(
            rows[0]['observed_spread'] / rows[0]['resolution'])

    def test_observed_spread_is_the_band_point_estimate_range(self, synthetic):
        membership = gc.assign_groups(LABELS)
        band = [label for group in gc.BAND_GROUPS for label in membership[group]]
        rows = gc.homogeneity_rows(
            synthetic['samples'], synthetic['points'], membership, 0.05)
        band_points = [synthetic['points'][label] for label in band]
        assert rows[0]['observed_spread'] == pytest.approx(
            max(band_points) - min(band_points))


class TestRun:
    def test_writes_all_three_tables_with_json_twins(self, synthetic):
        gc.run(0.05)
        for name in ('group_means', 'group_contrasts', 'band_homogeneity'):
            assert (synthetic['tmp_dir'] / f'{name}.csv').exists()
            assert (synthetic['tmp_dir'] / f'{name}.json').exists()

    def test_group_means_cover_every_group_once(self, synthetic):
        means_df, _, _ = gc.run(0.05)
        assert sorted(means_df.group) == sorted(gc.GROUPS)
        assert means_df.n_members.sum() == len(LABELS)

    def test_group_mean_ci_contains_its_point_estimate(self, synthetic):
        means_df, _, _ = gc.run(0.05)
        assert ((means_df.ci_lower <= means_df.point_estimate)
                & (means_df.point_estimate <= means_df.ci_upper)).all()

    def test_missing_samples_file_raises_clearly(self, synthetic):
        (synthetic['tmp_dir'] / 'arm_samples.npz').unlink()
        with pytest.raises(FileNotFoundError, match='run_pairwise_ci'):
            gc.run(0.05)


class TestRealPartition:
    """Guards the one thing the hermetic fixture cannot: that GROUPS still covers the
    real reported rows. A renamed row in run_pairwise_ci.SLICES silently changing what
    a group averages is the failure mode worth catching in CI."""

    @pytest.mark.parametrize('slice_name', sorted(pw.SLICES))
    def test_every_reported_row_is_classified(self, slice_name):
        labels = [label for label, _ in pw.SLICES[slice_name]['rows']]
        membership = gc.assign_groups(labels)
        assert sum(len(members) for members in membership.values()) == len(labels)

    @pytest.mark.parametrize('slice_name', sorted(pw.SLICES))
    def test_band_is_five_arms_in_every_slice(self, slice_name):
        labels = [label for label, _ in pw.SLICES[slice_name]['rows']]
        membership = gc.assign_groups(labels)
        band = [label for group in gc.BAND_GROUPS for label in membership[group]]
        assert len(band) == 5
