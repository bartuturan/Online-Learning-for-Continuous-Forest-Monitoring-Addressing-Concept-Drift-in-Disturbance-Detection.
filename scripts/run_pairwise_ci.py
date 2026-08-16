"""Pairwise strategy comparisons and mean-column CIs for the thesis result tables.

scripts/run_bootstrap_ci.py answers "does this strategy beat the no-replay baseline?"
per (model_year, eval_year) cell for every configuration in the registry. This script
works at the level the thesis actually argues at -- the Mean column -- restricted to
the rows it prints:

  * **Mean-column CIs.** Tables 4.3/4.4/4.6/4.7 each end in a Mean column, and the
    selection rule in Section 3.8 maximises it, but it carries no uncertainty. The
    mean here is the unweighted mean of per-year scores -- the same estimator the
    tables print -- *not* a pooled score over all years' pixels. That distinction is
    load-bearing: the disturbance rate swings 1.55%-3.05% across years, so pooling
    would silently weight disturbance-heavy years more and put a CI on a different
    number than the one on the page.

  * **Strategy vs baseline** (`baseline_margins.csv`). One row per reported strategy,
    differenced against the no-replay baseline *inside each resample*. This is the
    comparison Chapter 4 leans on hardest -- "every strategy improves on the
    unmitigated baseline" -- and it is deliberately absent from the pair table below,
    so before this existed it had no mean-level interval anywhere in the outputs.
    One-sided, because a direction is hypothesised.

Strategy-vs-strategy pairs are deliberately **not** produced here. They used to be,
as `strategy_pairs.csv`: all 21 pairs among the seven reported strategies, each
summarised on its own and reassembled with a Benjamini-Hochberg correction. That table
duplicated `simultaneous_pairs.csv` from scripts/run_group_ci.py, which covers the same
21 pairs by single-step max-T over the same shared draws. Checked across all 210 pair
rows, the two nested exactly: every max-T-significant pair was also BH-significant and
none the other way round, BH simply being more permissive (135 vs 106). Keeping both
meant two multiplicity stories for one comparison, so the max-T one was kept -- it
carries a family-wise guarantee rather than an average-false-discovery-rate one, and it
shares its construction with scripts/run_group_contrasts.py, which is now the primary
test for the claims Chapter 4 makes about *kinds* of curation. Any pairwise contrast
remains recoverable from `arm_samples.npz` by arithmetic.

Five slices. The first three score each configuration on the objective it was selected
for; the last two score the same configurations on the other objective, which is what
Table 4.9 prints in its off-diagonal columns:

  forecast_mean                 next-year forecast, mean over five transitions (2018-2022)
  retention_mean                final 2022 model back-tested, mean over 2017-2022
  retention_2017                the same model on 2017 alone -- the most-forgotten year
  forecast_at_retention_config  retention-optimal configurations, forecasting
  retention_at_forecast_config  forecast-optimal configurations, retaining

The cross slices cost no extra inference -- every (family, cell) they need is already
in the prediction cache -- and they are what puts an interval on Section 5.6's
single-model recommendation, which previously rested on a bare 0.010 PR-AUC gap.

Holm-Bonferroni is applied to the two-sided tail proportion within each (slice,
metric), in families of 7 -- chosen over BH because the prose makes claims about
*named* individual strategies ("PR Stratification's clears zero"), which is what a
family-wise guarantee is for, rather than an average-false-discovery-rate one over the
whole family; Holm also needs no independence/PRDS assumption and is uniformly at
least as powerful as a flat Bonferroni bar. Two-sided because these families contain
negative margins as well as positive ones, and a one-sided p cannot flag a shortfall --
see `baseline_margin_rows`.

Why the shared resample matters, and why it is already guaranteed:
every year is scored on the *same* 297 physical test cubes, and a cube that is hard
(cloud contamination, ambiguous disturbance boundary) is plausibly hard in most years
it appears in. Drawing an independent resample per year would let favourable and
unfavourable draws partially cancel when averaging across years, understating the
uncertainty of the mean. `draw_cube_resamples` is called with a fixed seed against a
fixed 297-cube set, so every year, family and metric sees the identical draw sequence
-- which makes averaging per-year sample vectors elementwise exactly equal to
"resample once, score every year on that draw, then average".

Usage:
    python -u scripts/run_pairwise_ci.py
    python -u scripts/run_pairwise_ci.py --n-bootstrap 2000 --seed 42
    python -u scripts/run_pairwise_ci.py --n-bins 16384   # ~10x faster while iterating
"""
import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.run_bootstrap_ci import (  # noqa: E402
    PUBLISHED_TABLE,
    load_prediction,
)
from src.eval.bootstrap_ci import (  # noqa: E402
    DEFAULT_PR_AUC_BINS,
    _average_precision_from_bin_counts,
    cube_confusion_counts,
    draw_cube_resamples,
    f1_from_counts,
    holm_bonferroni,
    margin_tail_p_values,
    pr_auc_bin_table,
    summarise,
)
from src.eval.tables import save_table  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_ROOT = PROJECT_ROOT / 'experiments' / 'bootstrap_ci'
PAIRWISE_DIR = OUTPUT_ROOT / 'pairwise'
SAMPLES_PATH = PAIRWISE_DIR / 'arm_samples.npz'

BASELINE_LABEL = 'Baseline (No Replay)'

#: The rows Tables 4.3/4.4 print, in table order. Each was resolved to its family_id
#: by matching the table's own printed per-year F1 values against
#: all_families_combined.csv -- not by parsing the configuration string -- so a
#: renamed family or a mis-transcribed configuration cannot silently point this at a
#: different model. Verified by test_config_rows_match_published_tables.
#:
#: The two Misclassification Buffer rows are the exception: they postdate those tables,
#: which were printed while that strategy was pinned to RR=0.3 and MISCLASS_FRACTION=0.5.
#: It now sweeps RR 0.2-0.5 at MISCLASS_FRACTION=0.3 like every other strategy, so its
#: row was resolved the way the selection procedure resolves one -- the PR-AUC argmax
#: over the sweep on each objective's own slice (Section 3.8), giving RR=0.4 forecasting
#: and RR=0.5 retaining. That rule reproduces all ten of the other panel rows exactly,
#: which is what justifies applying it here; run_selection_bootstrap.run() re-derives the
#: same argmax from the prediction cache and raises if it disagrees with these rows.
FORECAST_ROWS = [
    (BASELINE_LABEL, 'mlp_prevyears_monthly_features_incremental_scaler'),
    ('Uniform Random (RR=0.4)', 'mlp_prevyears_monthly_features_incremental_scaler_experience_replay_RR_0.4'),
    ('PR Stratification (RR=0.2, PR=0.10)', 'mlp_prevyears_monthly_features_incremental_scaler_experience_replay_RR_0.2_PR_0.10'),
    ('Confidently Correct (RR=0.4)', 'mlp_prevyears_monthly_features_incremental_scaler_experience_replay_confidently_correct_memory_RR_0.4'),
    ('Uncertainty Prioritization (RR=0.5)', 'mlp_prevyears_monthly_features_incremental_scaler_experience_replay_uncertainity_prioritization_RR_0.5'),
    ('Hard Example Mining (RR=0.4)', 'mlp_prevyears_monthly_features_incremental_scaler_experience_replay_hard_example_mining_RR_0.4'),
    ('Misclassification Buffer (RR=0.4)', 'mlp_prevyears_monthly_features_incremental_scaler_missclassification_buffer_RR_0.4'),
    ('Combined (RR=0.5, HE/CC/UP/PR=.1/.1/.1/.2)', 'mlp_combined_HE=0.1_CC=0.1_UP=0.1_PR=(0.2,10)_RR=0.5'),
]

#: The rows Tables 4.6/4.7 print. All seven differ from FORECAST_ROWS -- these are
#: different trained models, not the same models re-scored.
RETENTION_ROWS = [
    (BASELINE_LABEL, 'mlp_prevyears_monthly_features_incremental_scaler'),
    ('Uniform Random (RR=0.5)', 'mlp_prevyears_monthly_features_incremental_scaler_experience_replay_RR_0.5'),
    ('PR Stratification (RR=0.4, PR=0.15)', 'mlp_prevyears_monthly_features_incremental_scaler_experience_replay_RR_0.4_PR_0.15'),
    ('Confidently Correct (RR=0.2)', 'mlp_prevyears_monthly_features_incremental_scaler_experience_replay_confidently_correct_memory_RR_0.2'),
    ('Uncertainty Prioritization (RR=0.4)', 'mlp_prevyears_monthly_features_incremental_scaler_experience_replay_uncertainity_prioritization_RR_0.4'),
    ('Hard Example Mining (RR=0.5)', 'mlp_prevyears_monthly_features_incremental_scaler_experience_replay_hard_example_mining_RR_0.5'),
    ('Misclassification Buffer (RR=0.5)', 'mlp_prevyears_monthly_features_incremental_scaler_missclassification_buffer_RR_0.5'),
    ('Combined (RR=0.5, CC/UP/PR=.2/.2/.1)', 'mlp_combined_HE=0_CC=0.2_UP=0.2_PR=(0.1,10)_RR=0.5'),
]

#: Which (model_year, eval_year) cells each slice averages over, and which published
#: table_type supplies the thresholds. 'retention' is prior_years at model_year=2022:
#: the 2022 model back-tested on each year, confirmed by matching the baseline's six
#: printed F1 values exactly (final_model_each_year holds no rows for these families).
FORECAST_CELLS = [(y - 1, y) for y in (2018, 2019, 2020, 2021, 2022)]
RETENTION_CELLS = [(2022, y) for y in (2017, 2018, 2019, 2020, 2021, 2022)]

#: The three slices above score each configuration on the objective it was selected
#: for. The two below are the same configurations scored on the *other* objective --
#: what Table 4.9 prints as its off-diagonal columns, and what Section 5.6 leans on
#: when it recommends PR Stratification as a single-model default because its
#: retention-optimal configuration "still forecasts above the baseline on average".
#: Those numbers carried no interval before. Crossing each row list against the other
#: slice's cells supplies one at no inference cost: every (family, cell) here is
#: already cached, because run_bootstrap_ci.py's triples_for_family covers next_year
#: and prior_years for every family in the registry.
SLICES = {
    'forecast_mean': dict(rows=FORECAST_ROWS, cells=FORECAST_CELLS, table_type='next_year'),
    'retention_mean': dict(rows=RETENTION_ROWS, cells=RETENTION_CELLS, table_type='prior_years'),
    'retention_2017': dict(rows=RETENTION_ROWS, cells=[(2022, 2017)], table_type='prior_years'),
    'forecast_at_retention_config': dict(rows=RETENTION_ROWS, cells=FORECAST_CELLS,
                                         table_type='next_year'),
    'retention_at_forecast_config': dict(rows=FORECAST_ROWS, cells=RETENTION_CELLS,
                                         table_type='prior_years'),
}

METRICS = ('f1_score', 'pr_auc')


def per_year_samples(family_id, model_year, eval_year, threshold, weights,
                     n_bins=DEFAULT_PR_AUC_BINS):
    """Bootstrap sample vectors for one (family, cell), one entry per resample.

    Returns {'f1_score': (n_boot,), 'pr_auc': (n_boot,)} plus the exact point
    estimates. Both metrics are evaluated on the *same* `weights` rows, which is what
    lets the caller average across years and difference across families afterwards.
    """
    loaded = load_prediction(family_id, model_year, eval_year)
    if loaded is None:
        raise FileNotFoundError(
            f'No cached predictions for {family_id} model={model_year} eval={eval_year}. '
            f'Run scripts/run_bootstrap_ci.py --stage predict first.')
    y_proba, y_true, cube_idx = loaded

    tp, fp, fn, _ = cube_confusion_counts(y_true, y_proba >= threshold, cube_idx)
    f1_samples = f1_from_counts(weights @ tp, weights @ fp, weights @ fn)

    table = pr_auc_bin_table(y_true, y_proba, cube_idx, n_bins=n_bins)
    pr_samples = _average_precision_from_bin_counts(
        weights, table['pos_counts'], table['all_counts'])

    return {
        'f1_score': f1_samples,
        'pr_auc': pr_samples,
        'point_f1_score': float(f1_from_counts(tp.sum(), fp.sum(), fn.sum())),
        'point_pr_auc': table['point_estimate'],
        'n_cubes': int(len(np.unique(cube_idx))),
    }


def build_arm_samples(published, weights, f1_tolerance, n_bins=DEFAULT_PR_AUC_BINS):
    """Per-(slice, row, metric) sample vectors, averaged across the slice's years.

    The average is taken elementwise over resample index, which is identical to
    "draw once, score every year on that draw, then average" because every year is
    scored against the same fixed 297-cube set with the same seed. Doing it this way
    means each (family, cell) is reduced exactly once even though the retention_mean
    and retention_2017 slices overlap.
    """
    cell_cache = {}
    arms = {}

    for slice_name, spec in SLICES.items():
        for label, family_id in spec['rows']:
            per_year = []
            for model_year, eval_year in spec['cells']:
                key = (family_id, model_year, eval_year)
                if key not in cell_cache:
                    row = published[
                        (published.family_id == family_id)
                        & (published.table_type == spec['table_type'])
                        & (published.model_year == model_year)
                        & (published.eval_year == eval_year)
                    ]
                    if len(row) != 1:
                        raise ValueError(
                            f'Expected exactly one published row for {family_id} '
                            f'{spec["table_type"]} model={model_year} eval={eval_year}, '
                            f'found {len(row)}.')
                    row = row.iloc[0]
                    result = per_year_samples(
                        family_id, model_year, eval_year,
                        float(row.threshold_used), weights, n_bins=n_bins)

                    # Same guard the per-year pipeline applies: if re-inference no
                    # longer reproduces the printed number, every CI built on it
                    # describes a different model than the thesis reports.
                    if not np.isclose(result['point_f1_score'], float(row.f1_score),
                                      atol=f1_tolerance):
                        raise ValueError(
                            f'Re-inference disagrees with the published table for '
                            f'{family_id} model={model_year} eval={eval_year}: '
                            f'{result["point_f1_score"]:.6f} vs {row.f1_score:.6f}.')
                    cell_cache[key] = result
                per_year.append(cell_cache[key])

            for metric in METRICS:
                arms[(slice_name, label, metric)] = {
                    'samples': np.mean([c[metric] for c in per_year], axis=0),
                    'point_estimate': float(np.mean([c[f'point_{metric}'] for c in per_year])),
                    'family_id': family_id,
                    'n_years': len(per_year),
                }

    return arms, cell_cache


def baseline_margin_rows(arms, slice_name, spec, metric, alpha):
    """Each reported strategy minus the no-replay baseline, on this slice's mean.

    The pair table deliberately excludes the baseline, so the comparison the thesis
    argues from most -- "does replay beat not replaying?" -- had no mean-level
    interval anywhere in the outputs, at any slice. The margin is differenced inside
    each resample, which preserves the pairing that makes a strategy-vs-baseline
    margin resolvable at all; differencing two separately-summarised intervals would
    not.

    Both tail proportions are reported, floored at 1/(B+1) for the same reason
    paired_bootstrap_margin is. The one-sided value keeps the hypothesised direction
    (the claim under test is that replay helps) and is what the earlier revisions of
    this table corrected; Holm-Bonferroni is now applied by the caller to the
    *two-sided* value within the (slice, metric) block, because four of these seven
    strategies post negative margins on forecasting and a one-sided p cannot flag
    those at all -- it returned p ~ 1 for a Misclassification Buffer row whose own
    interval excluded zero. Two-sided Holm makes the adjusted-p column and the
    interval column agree on every row.
    """
    baseline = arms[(slice_name, BASELINE_LABEL, metric)]
    rows = []
    for label, family_id in spec['rows']:
        if label == BASELINE_LABEL:
            continue
        arm = arms[(slice_name, label, metric)]
        margin = arm['samples'] - baseline['samples']
        point = arm['point_estimate'] - baseline['point_estimate']
        rows.append({
            'slice': slice_name,
            'metric': metric,
            'row_label': label,
            'family_id': family_id,
            'baseline_family_id': baseline['family_id'],
            'strategy_point_estimate': arm['point_estimate'],
            'baseline_point_estimate': baseline['point_estimate'],
            **summarise(margin, point, alpha=alpha),
            **margin_tail_p_values(margin),
        })
    return rows


def run(n_bootstrap, seed, alpha, f1_tolerance, n_bins=DEFAULT_PR_AUC_BINS):
    if not PUBLISHED_TABLE.exists():
        raise FileNotFoundError(f'Missing {PUBLISHED_TABLE}.')
    published = pd.read_csv(PUBLISHED_TABLE, float_precision='round_trip')
    PAIRWISE_DIR.mkdir(parents=True, exist_ok=True)

    # One draw sequence, shared by every year, family, metric and slice. This is the
    # whole basis for both the mean and the pairing.
    weights = draw_cube_resamples(297, n_bootstrap, np.random.default_rng(seed))

    print(f'Reducing arms ({n_bootstrap} resamples over 297 cubes, seed {seed})...')
    arms, cell_cache = build_arm_samples(published, weights, f1_tolerance, n_bins=n_bins)
    print(f'  {len(cell_cache)} distinct (family, cell) reductions -> {len(arms)} arms')

    mean_rows = []
    baseline_rows = []

    for slice_name, spec in SLICES.items():
        for metric in METRICS:
            for label, family_id in spec['rows']:
                arm = arms[(slice_name, label, metric)]
                mean_rows.append({
                    'slice': slice_name,
                    'row_label': label,
                    'family_id': family_id,
                    'metric': metric,
                    'n_years_averaged': arm['n_years'],
                    **summarise(arm['samples'], arm['point_estimate'], alpha=alpha),
                })

            baseline_block = baseline_margin_rows(arms, slice_name, spec, metric, alpha)
            p_holm = holm_bonferroni([r['p_value_two_sided'] for r in baseline_block])
            for row, p_value in zip(baseline_block, p_holm):
                row['p_value_holm'] = float(p_value)
                row['holm_family'] = f'{slice_name}:{metric}'
                row['holm_family_size'] = len(baseline_block)
                row['holm_p_side'] = 'two_sided'
            baseline_rows.extend(baseline_block)

    mean_df = pd.DataFrame(mean_rows)
    baseline_df = pd.DataFrame(baseline_rows)
    save_table(mean_df, PAIRWISE_DIR / 'mean_column_ci.csv')
    save_table(baseline_df, PAIRWISE_DIR / 'baseline_margins.csv')

    # Persist the reduced sample vectors so any further contrast is arithmetic rather
    # than another pass over 3.7 GB of cached predictions.
    np.savez_compressed(
        SAMPLES_PATH,
        **{f'{s}|{label}|{m}': arms[(s, label, m)]['samples'] for (s, label, m) in arms},
    )

    print(f'  wrote {len(mean_df)} mean-column rows and '
          f'{len(baseline_df)} baseline-margin rows')
    return mean_df, baseline_df


def write_manifest(args):
    try:
        commit = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=PROJECT_ROOT,
                                capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        commit = None
    manifest = {
        'created_at': datetime.now(timezone.utc).isoformat(timespec='seconds').replace('+00:00', 'Z'),
        'git_commit': commit,
        'n_bootstrap': args.n_bootstrap,
        'seed': args.seed,
        'ci_alpha': args.alpha,
        'resampling_unit': 'cube',
        'n_cubes': 297,
        'mean_estimator': 'unweighted mean of per-year scores, averaged within each resample',
        'slices': {name: {'cells': spec['cells'], 'table_type': spec['table_type']}
                   for name, spec in SLICES.items()},
        'pr_auc_bins': args.n_bins,
        'n_strategies': len(FORECAST_ROWS) - 1,
        'holm_scope': f'baseline margins, within each (slice, metric); '
                      f'{len(SLICES) * len(METRICS)} families of {len(FORECAST_ROWS) - 1}',
        'strategy_pairs': 'not produced here; see pairwise/simultaneous_pairs.csv '
                          '(scripts/run_group_ci.py)',
    }
    with open(PAIRWISE_DIR / 'run_manifest.json', 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2)
        f.write('\n')


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--n-bootstrap', type=int, default=2000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--alpha', type=float, default=0.05)
    parser.add_argument('--n-bins', type=int, default=DEFAULT_PR_AUC_BINS,
                        help=(f'Score-rank bins for the resampled PR curves (default '
                              f'{DEFAULT_PR_AUC_BINS}). 16384 runs ~10x faster and moves '
                              f'the interval bounds by <2e-4; lower it while iterating, '
                              f'leave it at the default for numbers that go in the thesis.'))
    parser.add_argument('--f1-tolerance', type=float, default=1e-4)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    run(args.n_bootstrap, args.seed, args.alpha, args.f1_tolerance, n_bins=args.n_bins)
    write_manifest(args)
    print('\nDone.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
