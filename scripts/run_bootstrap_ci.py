"""Cube-level bootstrap confidence intervals for the reported model comparisons.

The published tables under experiments/evaluation/eval_outputs/unified_eval/ give one
F1 and one PR-AUC per (family, model_year, eval_year) and nothing about how stable
those numbers are. Margins between replay strategies there run as small as ~0.006,
which is not obviously distinguishable from the noise of having sampled one
particular set of forest cubes. This script answers "would this margin survive a
different sample of forest?" by re-running inference once per configuration, then
resampling the 297 test *cubes* (see src/eval/bootstrap_ci.py for why cubes and not
pixels) and recomputing the metrics on each resample.

Two stages, because they have completely different costs and failure modes:

  predict    Load each checkpoint, run one inference pass over the test split, save
             the per-pixel probabilities. Hours. Resumable at per-configuration
             granularity -- an existing .npz is trusted and skipped, so an
             interrupted run costs only the configuration it died on.
  bootstrap  Read those probabilities back and do the resampling. Minutes. Cheap to
             re-run with different --n-bootstrap or --ci-method.

Outputs live under experiments/bootstrap_ci/, which collides with nothing: the large
per-pixel .npz files are already covered by .gitignore's blanket `*.npz`, while the
small summary CSVs are picked up by its `!experiments/**/*.csv` override, matching how
every other evaluation output in this repo is tracked.

Usage:
    python -u scripts/run_bootstrap_ci.py --stage all
    python -u scripts/run_bootstrap_ci.py --stage predict
    python -u scripts/run_bootstrap_ci.py --stage bootstrap --n-bootstrap 2000
    python -u scripts/run_bootstrap_ci.py --stage all --n-bootstrap 100 \
        --strategy-pattern 'RR=0\\.2$' --table-types own_year   # quick smoke run
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

from src.eval.artifacts import (  # noqa: E402
    infer_years_from_models,
    load_model,
    load_scaler,
    resolve_scaler_year,
    validate_model_features,
    validate_scaler_features,
)
from src.eval.bootstrap_ci import (  # noqa: E402
    DEFAULT_BASELINE_FAMILY,
    DEFAULT_PR_AUC_BINS,
    DEFAULT_STRATEGY_PATTERN,
    benjamini_hochberg,
    bootstrap_f1_ci,
    bootstrap_pr_auc_ci,
    cube_confusion_counts,
    f1_from_counts,
    paired_bootstrap_margin,
    pr_auc_bin_table,
    select_families,
)
from src.eval.families import build_families, open_family_dataset  # noqa: E402
from src.eval.tables import save_table  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_ROOT = PROJECT_ROOT / 'experiments' / 'bootstrap_ci'
PREDICTIONS_DIR = OUTPUT_ROOT / 'predictions'
CONTEXT_DIR = PREDICTIONS_DIR / '_context'
CI_RESULTS_DIR = OUTPUT_ROOT / 'ci_results'
PUBLISHED_TABLE = (PROJECT_ROOT / 'experiments' / 'evaluation' / 'eval_outputs'
                   / 'unified_eval' / 'all_families_combined.csv')
SPLIT_PATH = PROJECT_ROOT / 'data_split.npz'

DEFAULT_TABLE_TYPES = ('own_year', 'prior_years', 'next_year')

#: The feature flags each prep_kind implies. Mirrors PREP_FN_BY_KIND in
#: notebooks/evaluation/Evaluations.ipynb cell 3 -- the same four kinds with the same
#: flags, expressed as data because this script calls the underlying feature builder
#: directly (it needs the row mask, which the notebook's wrappers do not surface).
PREP_KIND_FLAGS = {
    'baseline_notebook': dict(include_last_year=False, include_monthly=False,
                              include_neighbourhood=False, impute_window='expanding'),
    'prevyears': dict(include_last_year=True, include_monthly=False,
                      include_neighbourhood=False, impute_window='all_years'),
    'monthly': dict(include_last_year=True, include_monthly=True,
                    include_neighbourhood=False, impute_window='all_years'),
    'neighbourhood': dict(include_last_year=False, include_monthly=False,
                          include_neighbourhood=True, impute_window='all_years'),
}


def triples_for_family(years):
    """Every (model_year, eval_year) this analysis needs for one family.

    The union of what own_year, prior_years and next_year ask for, copied from
    evaluate_family in notebooks/evaluation/Evaluations.ipynb cell 4 rather than
    reinvented -- own_year is (y, y), prior_years is every eval_year <= model_year
    (which already contains own_year), and next_year is model_year + 1 when that year
    exists. Deduplicated, because the same pair appears in more than one table and
    inference for it only needs doing once.
    """
    triples = set()
    for model_year in years:
        for eval_year in years:
            if eval_year <= model_year:
                triples.add((model_year, eval_year))
        if model_year + 1 in years:
            triples.add((model_year, model_year + 1))
    return sorted(triples)


def context_path(family_cfg):
    """Where the shared (y_true, cube_idx) for a dataset/prep_kind/year is cached.

    Labels and cube membership depend only on the dataset, the feature flags and the
    eval year -- never on which model is being scored. Storing them once per group
    instead of once per configuration keeps ~900 prediction files from each carrying
    their own duplicate copy of the same 1.28M-element arrays.
    """
    dataset_stem = Path(family_cfg['dataset_path']).stem
    return f"{dataset_stem}__{family_cfg['prep_kind']}"


def prediction_path(family_id, model_year, eval_year):
    return PREDICTIONS_DIR / family_id / f'model_{model_year}_eval_{eval_year}.npz'


def run_predict_stage(families, family_ids, test_pixel_indices, verify_manifest=True):
    """Inference pass per (family, model_year, eval_year), grouped to share work.

    Grouped by (dataset, prep_kind) then by eval year, because the *raw* features for
    a year are identical across every family that shares those two things -- which is
    all ~36 families in the default scope. Computing them once per year rather than
    once per configuration is the difference between ~6 feature builds and ~900.
    """
    from src.cube.manifest import verify_split_matches_dataset
    from src.eval.features import apply_scaler
    from src.mlp_replay.data import prepare_raw_features_for_year

    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    CONTEXT_DIR.mkdir(parents=True, exist_ok=True)

    dataset_cache = {}
    groups = {}
    for family_id in family_ids:
        family_cfg = families[family_id]
        groups.setdefault(
            (str(family_cfg['dataset_path']), family_cfg['prep_kind']), []
        ).append(family_id)

    written = skipped = 0

    for (dataset_path, prep_kind), group_family_ids in sorted(groups.items()):
        ds = open_family_dataset(families[group_family_ids[0]], dataset_cache)

        if verify_manifest:
            # The whole analysis joins predictions to cubes positionally. If the pixel
            # population were re-sampled since the split was built, that join would be
            # silently wrong rather than absent, so fail here instead.
            verify_split_matches_dataset(SPLIT_PATH, ds)

        year_to_idx = {int(y): i for i, y in enumerate(ds.year.values)}
        cube_idx_all = ds['cube_idx'].values[test_pixel_indices]

        # Which (family, model_year) pairs need each eval year.
        work_by_eval_year = {}
        family_years = {}
        for family_id in group_family_ids:
            family_cfg = families[family_id]
            years = infer_years_from_models(family_cfg['models_dir'], family_cfg['model_template'])
            if not years:
                print(f'  WARN no model years on disk for {family_id}, skipping')
                continue
            family_years[family_id] = years
            for model_year, eval_year in triples_for_family(years):
                work_by_eval_year.setdefault(eval_year, []).append((family_id, model_year))

        flags = PREP_KIND_FLAGS[prep_kind]

        for eval_year in sorted(work_by_eval_year):
            pending = [
                (family_id, model_year)
                for family_id, model_year in work_by_eval_year[eval_year]
                if not prediction_path(family_id, model_year, eval_year).exists()
            ]
            skipped += len(work_by_eval_year[eval_year]) - len(pending)
            if not pending:
                continue

            print(f'[{prep_kind}] eval_year={eval_year}: building features for '
                  f'{len(pending)} pending configuration(s)')
            X_raw, y_true, row_mask = prepare_raw_features_for_year(
                ds, test_pixel_indices, year_to_idx[int(eval_year)],
                dtype=None, return_row_mask=True, **flags)

            if len(X_raw) == 0:
                print(f'  no usable rows for eval_year={eval_year}, skipping')
                continue

            cube_idx = cube_idx_all[row_mask]
            context_file = CONTEXT_DIR / f'{context_path(families[pending[0][0]])}__eval_{eval_year}.npz'
            if not context_file.exists():
                np.savez_compressed(
                    context_file,
                    y_true=y_true.astype(np.int8),
                    cube_idx=cube_idx.astype(np.int32),
                    n_test_samples=np.int64(len(y_true)),
                )

            for family_id, model_year in pending:
                family_cfg = families[family_id]
                model_obj = load_model(family_cfg['models_dir'], family_cfg['model_template'], model_year)
                if model_obj is None:
                    print(f'  WARN missing model {family_id} year {model_year}')
                    continue

                scaler_year, _ = resolve_scaler_year(family_cfg, model_year, eval_year)
                scaler_obj = load_scaler(family_cfg['models_dir'], family_cfg['scaler_template'], scaler_year)

                X, _ = apply_scaler(X_raw, scaler_obj, 'none' if scaler_obj is None else 'transform')
                validate_scaler_features(X, scaler_obj, family_cfg['label'], model_year, eval_year, 'test')
                validate_model_features(X, model_obj, family_cfg['label'], model_year, eval_year, 'test')

                y_proba = model_obj.predict_proba(X)[:, 1]

                out_path = prediction_path(family_id, model_year, eval_year)
                out_path.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(
                    out_path,
                    # float32 is ~1e-7 relative precision on a probability -- far below
                    # any threshold margin that changes a prediction, and it halves the
                    # footprint of the largest artifact this script produces.
                    y_proba=y_proba.astype(np.float32),
                    context=context_file.name,
                    scaler_year=np.int64(scaler_year),
                )
                written += 1
                print(f'  saved {family_id} model={model_year} eval={eval_year}')

            del X_raw

    print(f'\npredict stage: {written} written, {skipped} already present')
    return written, skipped


def load_prediction(family_id, model_year, eval_year):
    """Return (y_proba, y_true, cube_idx) for one configuration, or None if absent."""
    path = prediction_path(family_id, model_year, eval_year)
    if not path.exists():
        return None
    with np.load(path, allow_pickle=False) as payload:
        y_proba = payload['y_proba']
        context_name = str(payload['context'])
    with np.load(CONTEXT_DIR / context_name, allow_pickle=False) as context:
        y_true = context['y_true']
        cube_idx = context['cube_idx']
    if len(y_proba) != len(y_true):
        raise ValueError(
            f'{path.name} has {len(y_proba)} probabilities but its context has '
            f'{len(y_true)} rows. The prediction and the labels it is scored against '
            f'came from different feature builds; regenerate both.'
        )
    return y_proba, y_true, cube_idx


def run_bootstrap_stage(baseline_id, strategy_ids, table_types, n_bootstrap, seed,
                        alpha, ci_method, fdr_correction, f1_tolerance,
                        n_bins=DEFAULT_PR_AUC_BINS):
    """Resample, score, and write the CI and margin tables."""
    if not PUBLISHED_TABLE.exists():
        raise FileNotFoundError(
            f'Missing {PUBLISHED_TABLE}. The bootstrap reuses the thresholds the '
            f'evaluation pipeline already chose; run that pipeline first.')

    published = pd.read_csv(PUBLISHED_TABLE, float_precision='round_trip')
    CI_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    baseline_rows = published[published['family_id'] == baseline_id]
    baseline_lookup = {
        (row.table_type, int(row.model_year), int(row.eval_year)): row
        for row in baseline_rows.itertuples()
    }

    # Keyed by (model_year, eval_year); holds one entry at a time because the scope
    # is iterated in published-table order, so a key is finished before the next.
    baseline_table_cache = {}

    for table_type in table_types:
        ci_rows = []
        margin_rows = []
        scope = published[(published['table_type'] == table_type)
                          & (published['family_id'].isin([baseline_id, *strategy_ids]))]
        # Sorted so every family sharing a (model_year, eval_year) is visited
        # consecutively -- that is what lets one baseline bin table serve all of them
        # instead of being rebuilt per row. The published table is ordered by family,
        # which would defeat the cache entirely.
        scope = scope.sort_values(['model_year', 'eval_year', 'family_id'])

        print(f'\n=== {table_type}: {len(scope)} published rows in scope ===')

        for row in scope.itertuples():
            model_year, eval_year = int(row.model_year), int(row.eval_year)
            loaded = load_prediction(row.family_id, model_year, eval_year)
            if loaded is None:
                print(f'  MISSING predictions for {row.family_id} '
                      f'model={model_year} eval={eval_year}; run --stage predict')
                continue
            y_proba, y_true, cube_idx = loaded
            threshold = float(row.threshold_used)

            # Re-inference must reproduce the published number, or the resampling is
            # describing a distribution the thesis never reported.
            tp, fp, fn, _ = cube_confusion_counts(y_true, y_proba >= threshold, cube_idx)
            recomputed_f1 = float(f1_from_counts(tp.sum(), fp.sum(), fn.sum()))
            if not np.isclose(recomputed_f1, float(row.f1_score), atol=f1_tolerance):
                raise ValueError(
                    f'Re-inference disagrees with the published table for '
                    f'{row.family_id} model={model_year} eval={eval_year}: '
                    f'recomputed F1 {recomputed_f1:.6f} vs published {row.f1_score:.6f} '
                    f'(tolerance {f1_tolerance}). The saved probabilities do not '
                    f'reproduce the reported result, so any CI from them would be '
                    f'about a different model than the one in the thesis.'
                )

            rng = np.random.default_rng(seed)
            f1_result = bootstrap_f1_ci(y_true, y_proba >= threshold, cube_idx,
                                        n_bootstrap=n_bootstrap, alpha=alpha,
                                        method=ci_method, rng=rng)
            # Reducing this arm to its bin table is the expensive step; build it once
            # and reuse it for both this interval and the paired margin below.
            arm_table = pr_auc_bin_table(y_true, y_proba, cube_idx, n_bins=n_bins)
            rng = np.random.default_rng(seed)
            pr_result = bootstrap_pr_auc_ci(n_bootstrap=n_bootstrap, alpha=alpha,
                                            method=ci_method, rng=rng, table=arm_table)

            base = {
                'family_id': row.family_id,
                'family_label': row.family_label,
                'table_type': table_type,
                'model_year': model_year,
                'eval_year': eval_year,
                'is_baseline': row.family_id == baseline_id,
                'n_test_samples': int(len(y_true)),
                'n_cubes': int(len(np.unique(cube_idx))),
                'threshold_used': threshold,
                'published_f1_score': float(row.f1_score),
                'recomputed_f1_score': recomputed_f1,
            }
            ci_rows.append({**base, 'metric': 'f1_score', **f1_result})
            ci_rows.append({**base, 'metric': 'pr_auc', **pr_result})

            if row.family_id == baseline_id:
                continue

            baseline_row = baseline_lookup.get((table_type, model_year, eval_year))
            if baseline_row is None:
                print(f'  no baseline row for {table_type} model={model_year} '
                      f'eval={eval_year}; margin skipped')
                continue
            baseline_loaded = load_prediction(baseline_id, model_year, eval_year)
            if baseline_loaded is None:
                print(f'  MISSING baseline predictions for model={model_year} '
                      f'eval={eval_year}; margin skipped')
                continue
            baseline_proba, baseline_true, _ = baseline_loaded

            # Every strategy at this (model_year, eval_year) is compared against the
            # same baseline arm, so its table is built once and reused across all of
            # them rather than rebuilt ~31 times.
            baseline_key = (model_year, eval_year)
            if baseline_key not in baseline_table_cache:
                baseline_table_cache.clear()
                baseline_table_cache[baseline_key] = pr_auc_bin_table(
                    baseline_true, baseline_proba, cube_idx, n_bins=n_bins)
            baseline_table = baseline_table_cache[baseline_key]

            for metric in ('f1_score', 'pr_auc'):
                rng = np.random.default_rng(seed)
                margin = paired_bootstrap_margin(
                    strategy=(y_true, y_proba),
                    baseline=(baseline_true, baseline_proba),
                    cube_idx=cube_idx,
                    metric='f1' if metric == 'f1_score' else 'pr_auc',
                    strategy_threshold=threshold,
                    baseline_threshold=float(baseline_row.threshold_used),
                    n_bootstrap=n_bootstrap,
                    alpha=alpha,
                    method=ci_method,
                    rng=rng,
                    strategy_table=arm_table,
                    baseline_table=baseline_table,
                    n_bins=n_bins,
                )
                margin.pop('metric', None)
                margin_rows.append({
                    'family_id': row.family_id,
                    'family_label': row.family_label,
                    'baseline_family_id': baseline_id,
                    'table_type': table_type,
                    'model_year': model_year,
                    'eval_year': eval_year,
                    'metric': metric,
                    'strategy_threshold': threshold,
                    'baseline_threshold': float(baseline_row.threshold_used),
                    **margin,
                })

        ci_df = pd.DataFrame(ci_rows)
        margin_df = pd.DataFrame(margin_rows)

        if fdr_correction and len(margin_df) > 0:
            # Corrected within (table_type, metric): that is the family of comparisons
            # a claim like "these strategies beat baseline on F1" is actually made over.
            margin_df['q_value_bh'] = np.nan
            for metric, block in margin_df.groupby('metric'):
                # Two-sided: a one-sided p in the "strategy wins" direction returns
                # ~1 for any negative margin, however large, and so cannot flag a
                # shortfall its own interval already excludes zero on.
                margin_df.loc[block.index, 'q_value_bh'] = benjamini_hochberg(
                    block['p_value_two_sided'].to_numpy())

        save_table(ci_df, CI_RESULTS_DIR / f'{table_type}.csv')
        save_table(margin_df, CI_RESULTS_DIR / f'{table_type}_margins.csv')
        print(f'  wrote {len(ci_df)} CI rows and {len(margin_df)} margin rows')

    return True


def _display_path(path):
    """Repo-relative when it is inside the repo, absolute otherwise."""
    try:
        return str(Path(path).relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def write_run_manifest(args, baseline_id, strategy_ids, table_types):
    """Record exactly what produced these numbers, so the thesis can cite it."""
    try:
        commit = subprocess.run(
            ['git', 'rev-parse', 'HEAD'], cwd=PROJECT_ROOT,
            capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        commit = None

    manifest = {
        'created_at': datetime.now(timezone.utc).isoformat(timespec='seconds').replace('+00:00', 'Z'),
        'git_commit': commit,
        'n_bootstrap': args.n_bootstrap,
        'seed': args.seed,
        'ci_method': args.ci_method,
        'ci_alpha': args.alpha,
        'fdr_correction': bool(args.fdr_correction),
        'resampling_unit': 'cube',
        'baseline_family_id': baseline_id,
        'strategy_pattern': args.strategy_pattern,
        'strategy_family_ids': strategy_ids,
        'table_types': list(table_types),
        'pr_auc_bins': args.n_bins,
        'threshold_source': _display_path(PUBLISHED_TABLE),
    }
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_ROOT / 'run_manifest.json', 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2)
        f.write('\n')
    return manifest


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--stage', choices=['predict', 'bootstrap', 'all'], default='all')
    parser.add_argument('--n-bootstrap', type=int, default=2000,
                        help='Resamples per interval (default 2000).')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--alpha', type=float, default=0.05,
                        help='1 - alpha is the interval coverage (default 0.05 -> 95%%).')
    parser.add_argument('--ci-method', choices=['percentile', 'bca'], default='percentile')
    parser.add_argument('--table-types', default=','.join(DEFAULT_TABLE_TYPES),
                        help='Comma-separated subset of own_year,prior_years,next_year.')
    parser.add_argument('--baseline', default=DEFAULT_BASELINE_FAMILY)
    parser.add_argument('--strategy-pattern', default=DEFAULT_STRATEGY_PATTERN,
                        help='Regex over family ids selecting the strategy arms.')
    parser.add_argument('--fdr-correction', action='store_true',
                        help='Add Benjamini-Hochberg q-values to the margin tables.')
    parser.add_argument('--n-bins', type=int, default=DEFAULT_PR_AUC_BINS,
                        help=(f'Score-rank bins for the resampled PR curves (default '
                              f'{DEFAULT_PR_AUC_BINS}). Cost is dominated by a '
                              f'(n_bootstrap x n_cubes) @ (n_cubes x n_bins) matmul, so '
                              f'this is the speed dial: 16384 runs ~10x faster and moves '
                              f'the interval bounds by <2e-4 on the real test split. Lower '
                              f'it while iterating, leave it at the default for a run '
                              f'whose numbers go in the thesis.'))
    parser.add_argument('--f1-tolerance', type=float, default=1e-4,
                        help='How far recomputed F1 may sit from the published value.')
    parser.add_argument('--skip-manifest-check', action='store_true',
                        help='Skip the pixel-identity check (only for synthetic tests).')
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    table_types = [t.strip() for t in args.table_types.split(',') if t.strip()]

    families = build_families(PROJECT_ROOT)
    baseline_id, strategy_ids = select_families(
        families, baseline_id=args.baseline, strategy_pattern=args.strategy_pattern)

    print(f'Baseline: {baseline_id}')
    print(f'Strategies ({len(strategy_ids)}):')
    for family_id in strategy_ids:
        print(f'  - {family_id}')
    print(f'Table types: {table_types}')

    if args.stage in ('predict', 'all'):
        split = np.load(SPLIT_PATH)
        test_pixel_indices = split['test_pixel_indices']
        print(f'\nTest split: {len(test_pixel_indices):,} pixels')
        run_predict_stage(families, [baseline_id, *strategy_ids], test_pixel_indices,
                          verify_manifest=not args.skip_manifest_check)

    if args.stage in ('bootstrap', 'all'):
        run_bootstrap_stage(
            baseline_id=baseline_id,
            strategy_ids=strategy_ids,
            table_types=table_types,
            n_bootstrap=args.n_bootstrap,
            seed=args.seed,
            alpha=args.alpha,
            ci_method=args.ci_method,
            fdr_correction=args.fdr_correction,
            f1_tolerance=args.f1_tolerance,
            n_bins=args.n_bins,
        )
        write_run_manifest(args, baseline_id, strategy_ids, table_types)

    print('\nDone.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
