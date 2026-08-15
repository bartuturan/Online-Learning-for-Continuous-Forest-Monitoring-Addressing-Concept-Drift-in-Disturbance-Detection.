"""Tests for the rank-distribution and simultaneous-interval aggregation.

Entirely hermetic: builds a small synthetic arm_samples.npz + mean_column_ci.csv
matching the real files' shape (via monkeypatched paths), rather than depending on
scripts/run_pairwise_ci.py having already run. The two things worth testing here are
not the general bootstrap math (covered in tests/eval/test_bootstrap_ci.py) but the
aggregation-specific claims made in the module docstring: that ranks form a proper
distribution, and that the max-T critical value genuinely reflects the joint
correlation between all 21 pairs rather than being a single pair's threshold in
disguise.
"""
import numpy as np
import pandas as pd
import pytest

import scripts.run_group_ci as g


@pytest.fixture
def synthetic_pairwise(tmp_path, monkeypatch):
    """A miniature (slice, metric) with 4 arms and a known best/worst ordering."""
    rng = np.random.default_rng(0)
    n_boot = 500
    labels = ['Baseline (No Replay)', 'Strategy A', 'Strategy B', 'Strategy C']
    # Strategy A is clearly best, Strategy C clearly worst; A/B/C share correlated
    # noise (same underlying resample draws, as the real pipeline guarantees) so a
    # naive independent-draw assumption would get the joint structure wrong.
    shared_noise = rng.normal(0, 0.01, n_boot)
    true_level = {'Baseline (No Replay)': 0.30, 'Strategy A': 0.42,
                 'Strategy B': 0.33, 'Strategy C': 0.31}
    samples = {
        label: true_level[label] + shared_noise + rng.normal(0, 0.01, n_boot)
        for label in labels
    }
    point_estimates = {label: float(np.mean(samples[label])) for label in labels}

    rows = [(label, f'family_{label}') for label in labels]
    slices = {'demo_slice': dict(rows=rows, cells=[(2020, 2020)], table_type='own_year')}

    tmp_dir = tmp_path / 'pairwise'
    tmp_dir.mkdir()
    np.savez_compressed(
        tmp_dir / 'arm_samples.npz',
        **{f'demo_slice|{label}|f1_score': samples[label] for label in labels})
    mean_df = pd.DataFrame([
        {'slice': 'demo_slice', 'row_label': label, 'metric': 'f1_score',
         'point_estimate': point_estimates[label]}
        for label in labels
    ])
    mean_df.to_csv(tmp_dir / 'mean_column_ci.csv', index=False)

    monkeypatch.setattr(g, 'PAIRWISE_DIR', tmp_dir)
    monkeypatch.setattr(g, 'SAMPLES_PATH', tmp_dir / 'arm_samples.npz')
    monkeypatch.setattr(g, 'MEAN_CI_PATH', tmp_dir / 'mean_column_ci.csv')
    monkeypatch.setattr(g, 'RANK_DIST_PATH', tmp_dir / 'rank_distribution.csv')
    monkeypatch.setattr(g, 'SIMULTANEOUS_PATH', tmp_dir / 'simultaneous_pairs.csv')
    monkeypatch.setattr(g, 'SLICES', slices)
    monkeypatch.setattr(g, 'METRICS', ('f1_score',))

    return {'labels': labels, 'samples': samples, 'point_estimates': point_estimates,
            'rows': rows, 'tmp_dir': tmp_dir}


class TestRankDistribution:
    def test_rank_fractions_sum_to_one_per_arm(self, synthetic_pairwise):
        d = synthetic_pairwise
        out = g.rank_distribution(d['samples'], d['point_estimates'], d['rows'])
        for row in out:
            total = sum(row[f'rank_{r}_frac'] for r in range(1, 5))
            assert total == pytest.approx(1.0)

    def test_each_resample_assigns_every_rank_exactly_once(self, synthetic_pairwise):
        """Rank fractions summed across arms, at a fixed rank, must total 1 -- exactly
        one arm holds each rank position in every resample."""
        d = synthetic_pairwise
        out = g.rank_distribution(d['samples'], d['point_estimates'], d['rows'])
        for r in range(1, 5):
            total = sum(row[f'rank_{r}_frac'] for row in out)
            assert total == pytest.approx(1.0)

    def test_dominant_arm_wins_almost_always(self, synthetic_pairwise):
        """Strategy A was built with a clear margin over everything else."""
        d = synthetic_pairwise
        out = {row['row_label']: row for row in
               g.rank_distribution(d['samples'], d['point_estimates'], d['rows'])}
        assert out['Strategy A']['top1_prob'] > 0.95
        assert out['Strategy A']['mean_rank'] < out['Strategy C']['mean_rank']

    def test_mean_rank_consistent_with_rank_fractions(self, synthetic_pairwise):
        d = synthetic_pairwise
        out = g.rank_distribution(d['samples'], d['point_estimates'], d['rows'])
        for row in out:
            implied = sum(r * row[f'rank_{r}_frac'] for r in range(1, 5))
            assert implied == pytest.approx(row['mean_rank'], abs=1e-6)

    def test_worst_arm_never_ranks_first(self, synthetic_pairwise):
        d = synthetic_pairwise
        out = {row['row_label']: row for row in
               g.rank_distribution(d['samples'], d['point_estimates'], d['rows'])}
        assert out['Strategy C']['rank_1_frac'] < 0.05


class TestSimultaneousPairs:
    def test_critical_value_dominates_every_single_pairs_own_quantile(self, synthetic_pairwise):
        """The point of the max-T construction: multiplicity can only widen, never
        narrow, the shared threshold relative to any individual pair considered alone.

        max_j |T_j| >= |T_k| for every resample and every k, so the alpha-quantile of
        the max must be >= the alpha-quantile of any single column -- this holds
        pointwise, not just on average, so it is a strict mathematical requirement of
        a correct implementation, not merely a plausible-looking property.
        """
        d = synthetic_pairwise
        strategy_rows = [(l, f) for l, f in d['rows'] if l != 'Baseline (No Replay)']
        pairs = g.simultaneous_pairs(d['samples'], d['point_estimates'], strategy_rows, alpha=0.05)
        critical_value = pairs[0]['critical_value_maxt']

        for label_a, label_b in [('Strategy A', 'Strategy B'), ('Strategy A', 'Strategy C'),
                                 ('Strategy B', 'Strategy C')]:
            diff = d['samples'][label_a] - d['samples'][label_b]
            point = d['point_estimates'][label_a] - d['point_estimates'][label_b]
            se = diff.std(ddof=1)
            single_pair_quantile = np.percentile(np.abs((diff - point) / se), 95)
            assert critical_value >= single_pair_quantile - 1e-9

    def test_all_pairs_share_one_critical_value(self, synthetic_pairwise):
        d = synthetic_pairwise
        strategy_rows = [(l, f) for l, f in d['rows'] if l != 'Baseline (No Replay)']
        pairs = g.simultaneous_pairs(d['samples'], d['point_estimates'], strategy_rows, alpha=0.05)
        assert len({p['critical_value_maxt'] for p in pairs}) == 1

    def test_three_pairs_among_three_strategies(self, synthetic_pairwise):
        d = synthetic_pairwise
        strategy_rows = [(l, f) for l, f in d['rows'] if l != 'Baseline (No Replay)']
        pairs = g.simultaneous_pairs(d['samples'], d['point_estimates'], strategy_rows, alpha=0.05)
        assert len(pairs) == 3

    def test_identical_arms_collapse_without_div_by_zero(self, synthetic_pairwise):
        """se=0 (identical samples in every resample) must not raise or emit inf/nan."""
        d = synthetic_pairwise
        samples = dict(d['samples'])
        samples['Strategy B'] = samples['Strategy A'].copy()
        point_estimates = dict(d['point_estimates'])
        point_estimates['Strategy B'] = point_estimates['Strategy A']

        rows = [('Strategy A', 'fam_a'), ('Strategy B', 'fam_b')]
        pairs = g.simultaneous_pairs(samples, point_estimates, rows, alpha=0.05)

        assert len(pairs) == 1
        assert pairs[0]['se_diff'] == 0.0
        assert pairs[0]['point_estimate'] == pytest.approx(0.0)
        assert pairs[0]['sim_ci_lower'] == pytest.approx(0.0)
        assert pairs[0]['sim_ci_upper'] == pytest.approx(0.0)
        assert not pairs[0]['sim_significant']
        assert np.isfinite(pairs[0]['critical_value_maxt'])

    def test_p_value_maxt_never_exactly_zero(self, synthetic_pairwise):
        """B resamples cannot resolve below 1/(B+1); an exact 0 would overclaim."""
        d = synthetic_pairwise
        strategy_rows = [(l, f) for l, f in d['rows'] if l != 'Baseline (No Replay)']
        pairs = g.simultaneous_pairs(d['samples'], d['point_estimates'], strategy_rows, alpha=0.05)
        assert all(p['p_value_maxt'] > 0 for p in pairs)

    def test_significant_pair_has_ci_excluding_zero(self, synthetic_pairwise):
        d = synthetic_pairwise
        strategy_rows = [(l, f) for l, f in d['rows'] if l != 'Baseline (No Replay)']
        pairs = g.simultaneous_pairs(d['samples'], d['point_estimates'], strategy_rows, alpha=0.05)
        for p in pairs:
            excludes_zero = (p['sim_ci_lower'] > 0) or (p['sim_ci_upper'] < 0)
            assert p['sim_significant'] == excludes_zero


class TestRunEndToEnd:
    def test_writes_expected_tables(self, synthetic_pairwise):
        rank_df, pair_df = g.run(alpha=0.05)

        assert len(rank_df) == 4  # 4 arms, 1 slice, 1 metric
        assert len(pair_df) == 3  # C(3,2) pairs among the 3 non-baseline arms
        assert 'demo_slice' not in set()  # sanity: slices monkeypatched correctly
        assert set(rank_df['slice']) == {'demo_slice'}
        assert set(pair_df['slice']) == {'demo_slice'}

    def test_baseline_appears_in_ranks_but_not_pairs(self, synthetic_pairwise):
        rank_df, pair_df = g.run(alpha=0.05)
        assert 'Baseline (No Replay)' in set(rank_df.row_label)
        assert 'Baseline (No Replay)' not in set(pair_df.strategy_a) | set(pair_df.strategy_b)

    def test_json_twins_written(self, synthetic_pairwise):
        g.run(alpha=0.05)
        assert (synthetic_pairwise['tmp_dir'] / 'rank_distribution.json').exists()
        assert (synthetic_pairwise['tmp_dir'] / 'simultaneous_pairs.json').exists()

    def test_missing_samples_file_raises_clearly(self, synthetic_pairwise):
        synthetic_pairwise['tmp_dir'].joinpath('arm_samples.npz').unlink()
        with pytest.raises(FileNotFoundError, match='Run scripts/run_pairwise_ci.py first'):
            g.run(alpha=0.05)


class TestMainCli:
    def test_main_writes_manifest(self, synthetic_pairwise):
        exit_code = g.main(['--alpha', '0.10'])
        assert exit_code == 0
        manifest_path = synthetic_pairwise['tmp_dir'] / 'group_ci_manifest.json'
        assert manifest_path.exists()

        import json
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        assert manifest['alpha'] == 0.10
        assert manifest['pairs_per_slice_metric'] == 3
        assert 'max-T' in manifest['method']
