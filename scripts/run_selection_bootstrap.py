"""Bootstrap the configuration-selection procedure, not just the selected model.

Every other bootstrap in this repo holds the configuration fixed and resamples cubes.
That answers "how stable is *this* model's score?" -- but it is not the question the
thesis's tables raise. Section 3.8 reports each strategy at its own argmax over 4-8
swept configurations, and Section 3.8 also records that no held-out data remained for
that selection, so the argmax was taken on mean PR-AUC over the *same test years* the
intervals are computed from. A fixed-configuration interval is therefore centred on a
maximum-of-k statistic and inherits its upward bias while saying nothing about it: the
winner's curse.

How large is it? Max-minus-mean across each sweep runs +0.006 to +0.021 PR-AUC, which
invites the guess that the curse is of that order and therefore comparable to the
forecast effects being claimed (+0.006 to +0.016). Measured, it is not: the winner's
curse comes out between 0.0000 and +0.0044 across all 28 (objective, strategy, metric)
margins. The reason is correlation. Configurations within one sweep differ only in
replay budget, so a resample that is kind to one is kind to all of them; they rise and
fall together and the maximum gains little over any single member. Max-minus-mean
measures how far apart the configurations truly are, which is not the same quantity at
all. The correction is therefore small -- but it had to be computed to be known, and
"small" is a result, not an assumption.

The fix is to resample the whole procedure. On each of the 2000 draws:

  1. score every configuration in a strategy's sweep on that draw,
  2. take the argmax of mean PR-AUC *within that strategy* -- the same rule Section 3.8
     applies, on resampled data,
  3. record the winner's score, on both metrics.

The resulting distribution describes "sweep k configurations on these test years and
take the best", which is the procedure that actually produced the numbers on the page.
That is a better estimand than any single configuration's score, because no one chose
a configuration a priori -- the data chose it.

Three things come out of it:

  selected_arm_ci        Naive (fixed-configuration) interval beside the
                         selection-aware one, plus a two-way split of the bias:
                         `resampling_bias` is the ordinary bootstrap bias every arm
                         carries (resample mean minus point estimate, present even
                         for a sweep of one), and `selection_bias` is what taking a
                         maximum adds on top of it (mean of resampled maxima minus
                         mean of the fixed winner's resamples). Only the second is
                         the winner's curse, and only the second is corrected in
                         `bias_corrected_point`. Reporting their sum instead would
                         charge a single-configuration strategy like Misclassification
                         Buffer for a selection it never made. The correction is
                         partial -- it does not manufacture a held-out selection set --
                         but it moves the estimate in the right direction and makes
                         the size of the problem visible instead of implicit.

  selection_stability    How often each configuration in a sweep actually wins across
                         resamples. Within-band resolution measures ~0.013-0.021 while
                         sweep spreads run ~0.005, so the argmax is expected to be
                         unstable; a strategy whose reported configuration wins 20% of
                         the time is not reporting a configuration, it is reporting a
                         coin flip.

  selection_margins      Selected arm minus the no-replay baseline, differenced within
                         each resample. This is the selection-aware counterpart of
                         baseline_margins.csv, and the honest version of the comparison
                         Chapter 4 argues from: the baseline was swept at one
                         configuration and so carries no selection advantage at all,
                         which is exactly what makes the naive margin flattering.

**Read `corrected_*`, not `selection_*`, when asking whether an effect is real.** On
the selection metric the resampled maximum is by construction at least as large as the
fixed winner's score on *every* draw, so the raw selection interval is stochastically
above the naive one and its bounds can only move up. Treating `selection_excludes_zero`
as a significance verdict would mean an effect can gain significance purely by being
re-selected, which is circular -- and it is exactly what happens for two forecast
PR-AUC margins here. `corrected_ci_lower/upper` shift that interval down by the
estimated curse and are what survive scrutiny; of those two, one holds and one does not.

The asymmetry between metrics is worth knowing when reading the table: selection runs
on PR-AUC, so dominance holds there and every PR-AUC selection bias is non-negative.
F1 is merely read off whichever configuration PR-AUC chose, and that configuration can
have *worse* F1 than the fixed winner -- 5 of 14 F1 margins carry a negative curse.

Selection is on PR-AUC even when the F1 figure is what gets reported, because that is
what Section 3.8 specifies; the winning configuration is chosen once per resample and
then read off on both metrics. Uses only the cached predictions -- no inference.

Usage:
    python -u scripts/run_selection_bootstrap.py
    python -u scripts/run_selection_bootstrap.py --n-bins 16384    # ~10x faster
"""
import argparse
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.run_pairwise_ci import (  # noqa: E402
    FORECAST_CELLS,
    FORECAST_ROWS,
    PUBLISHED_TABLE,
    RETENTION_CELLS,
    RETENTION_ROWS,
    per_year_samples,
)
from src.eval.bootstrap_ci import (  # noqa: E402
    DEFAULT_BASELINE_FAMILY,
    DEFAULT_PR_AUC_BINS,
    DEFAULT_STRATEGY_PATTERN,
    draw_cube_resamples,
    summarise,
)
from src.eval.tables import save_table  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / 'experiments' / 'bootstrap_ci' / 'selection'

METRICS = ('f1_score', 'pr_auc')
SELECTION_METRIC = 'pr_auc'

#: family_id -> strategy, by the curation criterion its id encodes. Ordered: the
#: PR-Stratification pattern must be tried before the plain experience_replay one,
#: because a PR-stratified family id contains 'experience_replay_RR' too.
STRATEGY_PATTERNS = [
    ('Combined', r'^mlp_combined'),
    ('Confidently Correct', r'confidently_correct'),
    ('Hard Example Mining', r'hard_example_mining'),
    ('Uncertainty Prioritization', r'uncertainity_prioritization'),
    ('Misclassification Buffer', r'missclassification_buffer'),
    ('PR Stratification', r'experience_replay_RR_[0-9.]+_PR_'),
    ('Uniform Random', r'experience_replay_RR_[0-9.]+$'),
]

OBJECTIVES = {
    'forecast': dict(cells=FORECAST_CELLS, table_type='next_year', reported=FORECAST_ROWS),
    'retention': dict(cells=RETENTION_CELLS, table_type='prior_years', reported=RETENTION_ROWS),
}


def classify(family_id):
    """Strategy a swept family belongs to, or None for anything out of scope.

    Gated on DEFAULT_STRATEGY_PATTERN first, which anchors at '^mlp_'. Without that
    gate the SGD families from the Table 3.1 feature-engineering ladder -- whose ids
    end in 'experience_replay_RR_0.4' exactly like the MLP ones, minus the prefix --
    fall into the Uniform Random sweep. They are different classifiers with no cached
    predictions, so the argmax would be taken over a sweep the thesis never swept.
    Reusing the same pattern the rest of the pipeline scopes on keeps the two from
    drifting apart.
    """
    if family_id == DEFAULT_BASELINE_FAMILY:
        return None
    if not re.search(DEFAULT_STRATEGY_PATTERN, family_id):
        return None
    for strategy, pattern in STRATEGY_PATTERNS:
        if re.search(pattern, family_id):
            return strategy
    return None


def build_sweeps(published, table_type, cells):
    """{strategy: [family_id, ...]} for every family with a complete set of cells.

    Built from the published table rather than from a pasted list, so a sweep that
    gains a configuration is picked up automatically -- the selection procedure being
    bootstrapped here is "argmax over whatever was swept", and hardcoding the sweep
    would silently freeze it at today's contents.
    """
    sweeps = {}
    for family_id in sorted(published.family_id.unique()):
        strategy = classify(family_id)
        if strategy is None:
            continue
        complete = all(
            len(published[(published.family_id == family_id)
                          & (published.table_type == table_type)
                          & (published.model_year == model_year)
                          & (published.eval_year == eval_year)]) == 1
            for model_year, eval_year in cells)
        if complete:
            sweeps.setdefault(strategy, []).append(family_id)
    return sweeps


def reduce_family(published, family_id, table_type, cells, weights, f1_tolerance, n_bins):
    """Mean-over-cells sample vectors and point estimates for one configuration."""
    per_cell = []
    for model_year, eval_year in cells:
        row = published[(published.family_id == family_id)
                        & (published.table_type == table_type)
                        & (published.model_year == model_year)
                        & (published.eval_year == eval_year)]
        if len(row) != 1:
            raise ValueError(
                f'Expected one published row for {family_id} {table_type} '
                f'model={model_year} eval={eval_year}, found {len(row)}.')
        row = row.iloc[0]
        result = per_year_samples(family_id, model_year, eval_year,
                                  float(row.threshold_used), weights, n_bins=n_bins)
        # Same guard the rest of the pipeline applies: if re-inference no longer
        # reproduces the printed number, every interval built on it is about a
        # different model than the thesis reports.
        if not np.isclose(result['point_f1_score'], float(row.f1_score), atol=f1_tolerance):
            raise ValueError(
                f'Re-inference disagrees with the published table for {family_id} '
                f'model={model_year} eval={eval_year}: '
                f'{result["point_f1_score"]:.6f} vs {row.f1_score:.6f}.')
        per_cell.append(result)

    return {
        **{m: np.mean([c[m] for c in per_cell], axis=0) for m in METRICS},
        **{f'point_{m}': float(np.mean([c[f'point_{m}'] for c in per_cell])) for m in METRICS},
    }


def select_within_sweep(arms, family_ids):
    """Per-resample argmax over a sweep, plus the winner's score on both metrics.

    Returns the selected sample vectors, the winner index per resample, and the
    observed (unresampled) argmax. Ties go to the lowest index via argmax's own rule;
    with continuous scores they do not occur in practice.
    """
    selection = np.stack([arms[f][SELECTION_METRIC] for f in family_ids], axis=0)
    winner = np.argmax(selection, axis=0)

    selected = {}
    for metric in METRICS:
        stacked = np.stack([arms[f][metric] for f in family_ids], axis=0)
        selected[metric] = stacked[winner, np.arange(stacked.shape[1])]

    observed_points = [arms[f][f'point_{SELECTION_METRIC}'] for f in family_ids]
    observed_winner = int(np.argmax(observed_points))
    return selected, winner, observed_winner


def run(n_bootstrap, seed, alpha, f1_tolerance, n_bins):
    published = pd.read_csv(PUBLISHED_TABLE, float_precision='round_trip')
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # The same fixed 297-cube draw sequence every other script uses, so a margin taken
    # here against the baseline is paired exactly as it is in baseline_margins.csv.
    weights = draw_cube_resamples(297, n_bootstrap, np.random.default_rng(seed))

    arm_rows, stability_rows, margin_rows = [], [], []

    for objective, spec in OBJECTIVES.items():
        cells, table_type = spec['cells'], spec['table_type']
        sweeps = build_sweeps(published, table_type, cells)
        reported = dict((fid, label) for label, fid in spec['reported'])

        n_configs = sum(len(v) for v in sweeps.values())
        print(f'\n=== {objective}: {len(sweeps)} strategies, {n_configs} configurations '
              f'over {len(cells)} cells ===')

        baseline = reduce_family(published, DEFAULT_BASELINE_FAMILY, table_type, cells,
                                 weights, f1_tolerance, n_bins)

        for strategy, family_ids in sorted(sweeps.items()):
            arms = {f: reduce_family(published, f, table_type, cells,
                                     weights, f1_tolerance, n_bins)
                    for f in family_ids}
            selected, winner, observed_winner = select_within_sweep(arms, family_ids)
            observed_family = family_ids[observed_winner]

            # The observed argmax must be the configuration the thesis prints for this
            # objective. If it is not, either the selection rule stated in Section 3.8
            # is not the rule that produced the tables, or the sweep has drifted.
            if observed_family not in reported:
                raise ValueError(
                    f'{objective}/{strategy}: argmax on mean {SELECTION_METRIC} selects '
                    f'{observed_family}, which is not the configuration reported for '
                    f'this objective ({sorted(set(reported) & set(family_ids))}).')

            print(f'  {strategy:28s} k={len(family_ids)}  selected {reported[observed_family]}'
                  f'  wins {np.mean(winner == observed_winner):.1%} of resamples')

            for j, family_id in enumerate(family_ids):
                stability_rows.append({
                    'objective': objective,
                    'strategy': strategy,
                    'family_id': family_id,
                    'is_reported_config': family_id == observed_family,
                    'sweep_size': len(family_ids),
                    f'point_{SELECTION_METRIC}': arms[family_id][f'point_{SELECTION_METRIC}'],
                    'win_fraction': float(np.mean(winner == j)),
                })

            for metric in METRICS:
                naive = arms[observed_family][metric]
                naive_point = arms[observed_family][f'point_{metric}']
                aware = selected[metric]

                # Measured against the fixed-configuration resample mean, NOT against
                # the point estimate: every bootstrap here carries an ordinary
                # resampling bias (bootstrap_mean - point) that has nothing to do with
                # selection and is present even for a sweep of one. Differencing the
                # two resample means cancels it, leaving only what taking a maximum
                # adds -- which is the winner's curse and the only part worth
                # correcting. A k=1 sweep gets exactly 0 by construction.
                resampling_bias = float(naive.mean() - naive_point)
                selection_bias = float(aware.mean() - naive.mean())

                naive_ci = summarise(naive, naive_point, alpha=alpha)
                aware_ci = summarise(aware, naive_point, alpha=alpha)
                # The whole aware distribution shifted down by the estimated curse.
                # A constant shift, so shifting the two percentiles is exact.
                arm_rows.append({
                    'objective': objective,
                    'strategy': strategy,
                    'metric': metric,
                    'selected_family_id': observed_family,
                    'selected_label': reported[observed_family],
                    'sweep_size': len(family_ids),
                    'point_estimate': naive_point,
                    'naive_ci_lower': naive_ci['ci_lower'],
                    'naive_ci_upper': naive_ci['ci_upper'],
                    'selection_ci_lower': aware_ci['ci_lower'],
                    'selection_ci_upper': aware_ci['ci_upper'],
                    'corrected_ci_lower': aware_ci['ci_lower'] - selection_bias,
                    'corrected_ci_upper': aware_ci['ci_upper'] - selection_bias,
                    'naive_width': naive_ci['ci_upper'] - naive_ci['ci_lower'],
                    'selection_width': aware_ci['ci_upper'] - aware_ci['ci_lower'],
                    'resampling_bias': resampling_bias,
                    'selection_bias': selection_bias,
                    'bias_corrected_point': naive_point - selection_bias,
                    'reported_config_win_fraction': float(np.mean(winner == observed_winner)),
                    'n_bootstrap': n_bootstrap,
                    'ci_alpha': alpha,
                })

                naive_margin = naive - baseline[metric]
                aware_margin = aware - baseline[metric]
                naive_point_margin = naive_point - baseline[f'point_{metric}']
                # Same decomposition as above: the baseline arm is common to both
                # margins and cancels, so this is purely the selected arm's curse.
                margin_bias = float(aware_margin.mean() - naive_margin.mean())
                naive_m = summarise(naive_margin, naive_point_margin, alpha=alpha)
                aware_m = summarise(aware_margin, naive_point_margin, alpha=alpha)
                margin_rows.append({
                    'objective': objective,
                    'strategy': strategy,
                    'metric': metric,
                    'selected_label': reported[observed_family],
                    'baseline_point_estimate': baseline[f'point_{metric}'],
                    'point_estimate': naive_point_margin,
                    'naive_ci_lower': naive_m['ci_lower'],
                    'naive_ci_upper': naive_m['ci_upper'],
                    'naive_excludes_zero': bool(naive_m['ci_lower'] > 0 or naive_m['ci_upper'] < 0),
                    # Raw, for transparency -- but see the module docstring: on the
                    # selection metric this can only move up, so it is not a test.
                    'selection_ci_lower': aware_m['ci_lower'],
                    'selection_ci_upper': aware_m['ci_upper'],
                    'selection_excludes_zero': bool(aware_m['ci_lower'] > 0 or aware_m['ci_upper'] < 0),
                    # The one to quote.
                    'corrected_ci_lower': aware_m['ci_lower'] - margin_bias,
                    'corrected_ci_upper': aware_m['ci_upper'] - margin_bias,
                    'corrected_excludes_zero': bool((aware_m['ci_lower'] - margin_bias) > 0
                                                    or (aware_m['ci_upper'] - margin_bias) < 0),
                    'selection_bias': margin_bias,
                    'bias_corrected_margin': naive_point_margin - margin_bias,
                    'p_value_one_sided_selection': float(
                        (1 + np.sum(aware_margin <= 0)) / (len(aware_margin) + 1)),
                    'n_bootstrap': n_bootstrap,
                    'ci_alpha': alpha,
                })

    arm_df = pd.DataFrame(arm_rows)
    stability_df = pd.DataFrame(stability_rows)
    margin_df = pd.DataFrame(margin_rows)
    save_table(arm_df, OUTPUT_DIR / 'selected_arm_ci.csv')
    save_table(stability_df, OUTPUT_DIR / 'selection_stability.csv')
    save_table(margin_df, OUTPUT_DIR / 'selection_margins.csv')
    print(f'\nwrote {len(arm_df)} arm rows, {len(stability_df)} stability rows, '
          f'{len(margin_df)} margin rows')
    return arm_df, stability_df, margin_df


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
        'pr_auc_bins': args.n_bins,
        'resampling_unit': 'cube',
        'selection_metric': SELECTION_METRIC,
        'selection_rule': 'argmax of mean PR-AUC within each strategy sweep, re-run '
                          'inside every resample',
        'bias_correction': 'selection_bias = mean(resampled maxima) - mean(fixed '
                           'winner resamples), isolating the winner\'s curse from the '
                           'ordinary resampling bias; partial, does not substitute for '
                           'a held-out selection split',
        'quote_column': 'corrected_ci_lower/upper (the raw selection interval is '
                        'stochastically above the naive one on the selection metric '
                        'and is not a significance test)',
        'objectives': {name: {'cells': spec['cells'], 'table_type': spec['table_type']}
                       for name, spec in OBJECTIVES.items()},
    }
    with open(OUTPUT_DIR / 'run_manifest.json', 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2)
        f.write('\n')


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--n-bootstrap', type=int, default=2000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--alpha', type=float, default=0.05)
    parser.add_argument('--f1-tolerance', type=float, default=1e-4)
    parser.add_argument('--n-bins', type=int, default=DEFAULT_PR_AUC_BINS,
                        help=f'Score-rank bins for resampled PR curves (default '
                             f'{DEFAULT_PR_AUC_BINS}); 16384 is ~10x faster.')
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    run(args.n_bootstrap, args.seed, args.alpha, args.f1_tolerance, args.n_bins)
    write_manifest(args)
    print('\nDone.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
