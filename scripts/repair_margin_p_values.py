"""One-off repair: put the (1+k)/(B+1) floor on already-written margin p-values.

`paired_bootstrap_margin` used to report `p_value_one_sided` as k/B, where k is the
number of resamples in which the strategy failed to beat the baseline. A strategy that
won every resample therefore printed a bare `0.0`, and Benjamini-Hochberg propagated
that to `q = 0.0`. 579 of the 1984 margin rows in experiments/bootstrap_ci/ci_results/
carry such a zero. B resamples cannot resolve a tail finer than 1/(B+1), so a printed
zero claims a precision the experiment does not have -- not a number to defend in a
viva.

src/eval/bootstrap_ci.py now emits the floored form, so a fresh
`run_bootstrap_ci.py --stage bootstrap` would produce correct files. That stage is the
slow one (~1024 configurations), and re-running it is not necessary here: k is exactly
recoverable from what was written, because p was k/B with B known from the same row.
So this script recomputes p = (1 + round(p*B)) / (B + 1) and re-runs BH within each
(table_type, metric) block, which is arithmetic on the CSV and reproduces exactly what
the fixed code would have written. The recovery is asserted to round-trip before
anything is overwritten.

Idempotent: a file already in the floored form is detected and skipped, so running
this twice is harmless. Safe to delete once the bootstrap stage has been re-run from
the fixed source.

SUPERSEDED on the BH step. The published tables now correct `p_value_two_sided`
(see repair_margin_two_sided_q.py for why), and the BH re-run below still uses
`p_value_one_sided`. CI_RESULTS_DIR below points at a path that no longer exists, so
this script is inert as written; do not repoint it at the live pairwise outputs without
switching that column, or it will quietly reinstate the one-sided q-values.

Usage:
    python -u scripts/repair_margin_p_values.py --dry-run
    python -u scripts/repair_margin_p_values.py
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.eval.bootstrap_ci import benjamini_hochberg  # noqa: E402
from src.eval.tables import save_table  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CI_RESULTS_DIR = PROJECT_ROOT / 'experiments' / 'bootstrap_ci' / 'ci_results'
MARGIN_TABLES = ('own_year_margins', 'prior_years_margins', 'next_year_margins')


def is_unfloored(p_values, n_bootstrap):
    """True when every p is an exact multiple of 1/B -- the old, unfloored form.

    (1 + k) / (B + 1) is essentially never an exact multiple of 1/B, so this cleanly
    separates the two forms without needing a marker column.
    """
    scaled = np.asarray(p_values, dtype=np.float64) * n_bootstrap
    return bool(np.all(np.abs(scaled - np.round(scaled)) < 1e-9))


def repair_frame(df):
    """Return (repaired_df, n_zeros_fixed, max_abs_q_change), or None if already fixed."""
    n_bootstrap = int(df['n_bootstrap'].iloc[0])
    if df['n_bootstrap'].nunique() != 1:
        raise ValueError(f'Mixed n_bootstrap in one table: {sorted(df.n_bootstrap.unique())}')
    if not is_unfloored(df['p_value_one_sided'], n_bootstrap):
        return None

    losses = np.round(df['p_value_one_sided'].to_numpy() * n_bootstrap).astype(int)
    # Refuse to write anything unless k is recovered exactly.
    if not np.allclose(losses / n_bootstrap, df['p_value_one_sided'].to_numpy(), atol=1e-12):
        raise ValueError('p_value_one_sided does not round-trip as k/B; refusing to repair.')

    repaired = df.copy()
    repaired['p_value_one_sided'] = (1 + losses) / (n_bootstrap + 1)

    if 'q_value_bh' in repaired.columns:
        # .copy() is load-bearing: to_numpy() can hand back a view onto the frame's
        # block, and the .loc assignment below writes through it -- which silently
        # made every reported delta come out as exactly zero.
        old_q = repaired['q_value_bh'].to_numpy(dtype=np.float64).copy()
        # Same scope the bootstrap stage uses: within (table_type, metric). The frame
        # is one table_type, so grouping by metric alone reproduces it.
        for _, block in repaired.groupby('metric'):
            repaired.loc[block.index, 'q_value_bh'] = benjamini_hochberg(
                block['p_value_one_sided'].to_numpy())
        q_change = float(np.nanmax(np.abs(repaired['q_value_bh'].to_numpy() - old_q)))
        flips = int(((old_q < 0.05) != (repaired['q_value_bh'].to_numpy() < 0.05)).sum())
    else:
        q_change, flips = 0.0, 0

    return repaired, int((df['p_value_one_sided'] == 0).sum()), q_change, flips


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--dry-run', action='store_true',
                        help='Report what would change without writing.')
    args = parser.parse_args(argv)

    touched = 0
    for stem in MARGIN_TABLES:
        path = CI_RESULTS_DIR / f'{stem}.csv'
        if not path.exists():
            print(f'{stem}: absent, skipped')
            continue

        df = pd.read_csv(path, float_precision='round_trip')
        result = repair_frame(df)
        if result is None:
            print(f'{stem}: already floored, skipped')
            continue

        repaired, n_zeros, q_change, flips = result
        print(f'{stem}: {n_zeros} zero p-values -> {1 / (int(df.n_bootstrap.iloc[0]) + 1):.2e}, '
              f'max |dq| = {q_change:.2e}, significance flips at 0.05 = {flips}')
        if not args.dry_run:
            save_table(repaired, path)
            touched += 1

    if args.dry_run:
        print('\nDry run; nothing written.')
    else:
        print(f'\nRewrote {touched} table(s) (CSV + JSON twin).')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
