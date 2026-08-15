"""Joint-resample group comparisons: rank distributions and simultaneous intervals.

scripts/run_pairwise_ci.py reduces the cached predictions to one shared 2000-resample
experiment: every arm (baseline and all 7 strategies) scored on the identical
resampled cube indices, persisted as arm_samples.npz. The obvious way to compare
strategies from that is 21 separate pairwise margins reassembled with a
Benjamini-Hochberg correction -- and that is what this pipeline used to do, until the
two tables were found to nest exactly (every max-T-significant pair was also
BH-significant, never the reverse) and the BH one was dropped as redundant. BH would
treat those 21 summaries as loosely related anyway; it does not use the fact that they
are 21 different functions of the *same* 2000 draws. This script uses that shared draw
directly, two ways:

  rank distribution   On each resample, rank all 8 arms best-to-worst by score. The
                       resulting per-arm rank histogram answers "which one wins"
                       directly, with uncertainty in the ranking itself -- something
                       21 separate pairwise verdicts don't obviously combine into.

  simultaneous pairs  A bootstrap analogue of Tukey's HSD / single-step max-T
                       (Westfall & Young, 1993): studentise every one of the 21
                       pairwise differences by that pair's own bootstrap standard
                       deviation, take the *max* |studentised difference| across all
                       21 pairs on each resample, and use the alpha-quantile of that
                       max -- across all resamples -- as one shared critical value.
                       A pair is significant if its own studentised difference
                       exceeds that shared threshold. Because the threshold is
                       calibrated from the joint resampled distribution of all 21
                       pairs together, it reflects their actual correlation exactly,
                       rather than approximating it the way BH's independence/PRDS
                       assumption does -- and it delivers a family-wise guarantee
                       ("at least (1-alpha) of the time, every flagged pair is a real
                       difference") rather than BH's average-false-discovery-rate
                       guarantee. This is now the only strategy-vs-strategy table the
                       pipeline produces; it is the exhaustive backing for
                       scripts/run_group_contrasts.py, which tests the claims Chapter 4
                       actually makes about *kinds* of curation and is the headline.

Single-step, not step-down: this uses one shared critical value for every pair rather
than the more powerful iterative step-down variant, trading a little power for a
construction simple enough to state and verify in one paragraph.

Reads experiments/bootstrap_ci/pairwise/arm_samples.npz and mean_column_ci.csv,
written by scripts/run_pairwise_ci.py -- run that first. Does not re-touch any
prediction cache or the published evaluation tables.

Usage:
    python -u scripts/run_group_ci.py
    python -u scripts/run_group_ci.py --alpha 0.10
"""
import argparse
import itertools
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.run_pairwise_ci import (  # noqa: E402
    BASELINE_LABEL,
    METRICS,
    PAIRWISE_DIR,
    SAMPLES_PATH,
    SLICES,
)
from src.eval.tables import save_table  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MEAN_CI_PATH = PAIRWISE_DIR / 'mean_column_ci.csv'
RANK_DIST_PATH = PAIRWISE_DIR / 'rank_distribution.csv'
SIMULTANEOUS_PATH = PAIRWISE_DIR / 'simultaneous_pairs.csv'


def load_arm_samples():
    if not SAMPLES_PATH.exists():
        raise FileNotFoundError(
            f'Missing {SAMPLES_PATH}. Run scripts/run_pairwise_ci.py first -- this '
            f'script only reduces its output, it does not read predictions itself.')
    with np.load(SAMPLES_PATH) as payload:
        return {key: payload[key] for key in payload.files}


def rank_distribution(samples, point_estimates, rows):
    """Per-arm rank histogram from ranking every resample once.

    Higher score is better in both metrics used here (F1, PR-AUC), so rank 1 is the
    highest value in a resample. Ties are broken by 'average' rank, which matters
    only in the vanishingly unlikely event two arms produce bit-identical resampled
    scores.
    """
    from scipy.stats import rankdata

    matrix = np.stack([samples[label] for label, _ in rows], axis=1)  # (n_boot, n_arms)
    n_arms = matrix.shape[1]
    ranks = np.apply_along_axis(lambda row: rankdata(-row, method='average'), 1, matrix)

    out = []
    for i, (label, family_id) in enumerate(rows):
        arm_ranks = ranks[:, i]
        row = {
            'row_label': label,
            'family_id': family_id,
            'point_estimate': point_estimates[label],
            'mean_rank': float(arm_ranks.mean()),
            'median_rank': float(np.median(arm_ranks)),
            'top1_prob': float(np.mean(arm_ranks <= 1.0)),
        }
        for r in range(1, n_arms + 1):
            row[f'rank_{r}_frac'] = float(np.mean(np.round(arm_ranks) == r))
        out.append(row)
    return out


def simultaneous_pairs(samples, point_estimates, strategy_rows, alpha):
    """Single-step max-T simultaneous intervals for all pairs among strategy_rows.

    Returns one row per pair, all sharing one `critical_value` per (slice, metric)
    call -- that shared value is what gives the family-wise coverage guarantee.
    """
    n_boot = len(next(iter(samples.values())))
    pairs = list(itertools.combinations(strategy_rows, 2))

    diffs = np.empty((n_boot, len(pairs)), dtype=np.float64)
    point_diffs = np.empty(len(pairs), dtype=np.float64)
    for j, ((label_a, _), (label_b, _)) in enumerate(pairs):
        diffs[:, j] = samples[label_a] - samples[label_b]
        point_diffs[j] = point_estimates[label_a] - point_estimates[label_b]

    # Single overall SE per pair (not re-estimated within each resample -- that would
    # need a nested bootstrap). se==0 only when a pair's samples are literally
    # identical in every resample, in which case point_diffs is also 0 and the
    # studentised statistic is defined as 0 rather than 0/0.
    se = diffs.std(axis=0, ddof=1)
    se_safe = np.where(se > 0, se, 1.0)

    studentized = (diffs - point_diffs) / se_safe
    max_abs_t_per_resample = np.max(np.abs(studentized), axis=1)
    critical_value = float(np.percentile(max_abs_t_per_resample, 100 * (1 - alpha)))

    own_stat = np.where(se > 0, np.abs(point_diffs) / se_safe, 0.0)
    # Monte Carlo tail proportion, floored at 1/(B+1) so no p-value claims more
    # resolution than B resamples can provide.
    p_value_maxt = np.array([
        (1 + np.sum(max_abs_t_per_resample >= stat)) / (n_boot + 1)
        for stat in own_stat
    ])

    rows = []
    for j, ((label_a, family_a), (label_b, family_b)) in enumerate(pairs):
        half_width = critical_value * se[j]
        rows.append({
            'strategy_a': label_a,
            'strategy_b': label_b,
            'family_id_a': family_a,
            'family_id_b': family_b,
            'point_estimate': float(point_diffs[j]),
            'se_diff': float(se[j]),
            'critical_value_maxt': critical_value,
            'sim_ci_lower': float(point_diffs[j] - half_width),
            'sim_ci_upper': float(point_diffs[j] + half_width),
            'sim_significant': bool(se[j] > 0 and (own_stat[j] > critical_value)),
            'p_value_maxt': float(p_value_maxt[j]),
        })
    return rows


def run(alpha):
    samples = load_arm_samples()
    mean_df = pd.read_csv(MEAN_CI_PATH, float_precision='round_trip')
    PAIRWISE_DIR.mkdir(parents=True, exist_ok=True)

    rank_rows = []
    pair_rows = []

    for slice_name, spec in SLICES.items():
        strategy_rows = [(label, fid) for label, fid in spec['rows'] if label != BASELINE_LABEL]

        for metric in METRICS:
            point_estimates = {
                label: float(mean_df[(mean_df['slice'] == slice_name)
                                     & (mean_df.row_label == label)
                                     & (mean_df.metric == metric)].iloc[0].point_estimate)
                for label, _ in spec['rows']
            }
            arm_samples = {
                label: samples[f'{slice_name}|{label}|{metric}']
                for label, _ in spec['rows']
            }

            for row in rank_distribution(arm_samples, point_estimates, spec['rows']):
                rank_rows.append({'slice': slice_name, 'metric': metric, **row})

            for row in simultaneous_pairs(arm_samples, point_estimates, strategy_rows, alpha):
                pair_rows.append({'slice': slice_name, 'metric': metric, **row})

    rank_df = pd.DataFrame(rank_rows)
    pair_df = pd.DataFrame(pair_rows)
    save_table(rank_df, RANK_DIST_PATH)
    save_table(pair_df, SIMULTANEOUS_PATH)

    print(f'  wrote {len(rank_df)} rank rows and {len(pair_df)} simultaneous-pair rows')
    return rank_df, pair_df


def _display_path(path):
    """Repo-relative when it is inside the repo, absolute otherwise."""
    try:
        return str(Path(path).relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def write_manifest(args, n_pairs):
    try:
        commit = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=PROJECT_ROOT,
                                capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        commit = None
    manifest = {
        'created_at': datetime.now(timezone.utc).isoformat(timespec='seconds').replace('+00:00', 'Z'),
        'git_commit': commit,
        'alpha': args.alpha,
        'method': 'single-step max-T (Westfall-Young), studentised, one shared '
                  'critical value per (slice, metric)',
        'source': _display_path(SAMPLES_PATH),
        'pairs_per_slice_metric': n_pairs,
    }
    with open(PAIRWISE_DIR / 'group_ci_manifest.json', 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2)
        f.write('\n')


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--alpha', type=float, default=0.05)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    _, pair_df = run(args.alpha)
    write_manifest(args, int(len(pair_df) / (len(SLICES) * len(METRICS))))
    print('\nDone.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
