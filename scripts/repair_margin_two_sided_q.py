"""Recompute the baseline-margin q-values against the two-sided tail proportion.

`baseline_margins.csv` originally corrected `p_value_one_sided` -- the share of
resamples in which the strategy failed to beat the baseline. That is the right
direction for a family where every margin is hypothesised positive, but four of the
seven forecasting strategies post *negative* margins, and for those the one-sided p
approaches 1 by construction no matter how far from zero the margin sits. The visible
symptom was Misclassification Buffer's forecasting F1 row: q = 0.998 printed beside a
95% interval of [-0.031, -0.007] that excludes zero. The q column and the interval
column disagreed on that row while claiming to answer the same question.

This recomputes `p_value_two_sided` for every margin row and re-runs BH within each
(slice, metric) family of seven. It works off `arm_samples.npz`, which holds the same
shared resample vectors the table was built from, so the result is exact rather than
reconstructed from the rounded p-values already in the CSV -- and the stored
`p_value_one_sided` is re-derived from those draws first as a check that the two
artefacts still correspond.

Rerunning run_pairwise_ci.py end to end produces the same numbers; this exists so the
correction does not require re-scoring every cached prediction. Idempotent.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.eval.bootstrap_ci import benjamini_hochberg, margin_tail_p_values  # noqa: E402
from src.eval.tables import save_table  # noqa: E402

PAIRWISE_DIR = Path(__file__).resolve().parent / 'bootstrap_ci' / 'pairwise'
MARGINS_PATH = PAIRWISE_DIR / 'baseline_margins.csv'
SAMPLES_PATH = PAIRWISE_DIR / 'arm_samples.npz'
BASELINE_LABEL = 'Baseline (No Replay)'
ALPHA = 0.05


def repair(df, samples):
    df = df.copy()
    if 'p_value_two_sided' not in df.columns:
        df.insert(df.columns.get_loc('p_value_one_sided') + 1, 'p_value_two_sided', np.nan)

    old_q = df['q_value_bh'].to_numpy(dtype=np.float64).copy()

    for (slice_name, metric), block in df.groupby(['slice', 'metric'], sort=False):
        baseline = samples[f'{slice_name}|{BASELINE_LABEL}|{metric}']
        tails = []
        for idx, row in block.iterrows():
            margin = samples[f'{slice_name}|{row["row_label"]}|{metric}'] - baseline
            tail = margin_tail_p_values(margin)
            # The draws must still be the ones this table was summarised from.
            if not np.isclose(tail['p_value_one_sided'], row['p_value_one_sided'], atol=1e-12):
                raise ValueError(
                    f'{slice_name}/{metric}/{row["row_label"]}: one-sided p re-derived from '
                    f'arm_samples.npz is {tail["p_value_one_sided"]:.6f} but the CSV stores '
                    f'{row["p_value_one_sided"]:.6f}. The two artefacts are out of sync; '
                    f'rerun run_pairwise_ci.py rather than repairing.'
                )
            tails.append((idx, tail['p_value_two_sided']))

        idxs = [i for i, _ in tails]
        p_two = np.array([p for _, p in tails])
        df.loc[idxs, 'p_value_two_sided'] = p_two
        df.loc[idxs, 'q_value_bh'] = benjamini_hochberg(p_two)
        df.loc[idxs, 'bh_p_side'] = 'two_sided'

    new_q = df['q_value_bh'].to_numpy(dtype=np.float64)
    p_two = df['p_value_two_sided'].to_numpy(dtype=np.float64)
    ci_sig = (df['ci_lower'].to_numpy() > 0) | (df['ci_upper'].to_numpy() < 0)

    # A row where BH lifts q above alpha although the raw interval excludes zero is
    # not a contradiction -- that is the multiplicity correction doing its job across
    # seven comparisons. The defect being repaired is narrower: the *uncorrected* p
    # disagreeing with its own interval, which can only happen when a one-sided tail is
    # blind to the margin's sign. Both are percentile quantities on the same draws, so
    # after the switch they must agree row by row.
    contradictions = df.index[ci_sig & (p_two >= ALPHA)].tolist()

    return df, {
        'flips': int(((old_q < ALPHA) != (new_q < ALPHA)).sum()),
        'max_q_change': float(np.nanmax(np.abs(new_q - old_q))),
        'sign_blind_before': int(((df['p_value_one_sided'].to_numpy() >= ALPHA) & ci_sig).sum()),
        'sign_blind_after': len(contradictions),
        'bh_lifted': int((ci_sig & (p_two < ALPHA) & (new_q >= ALPHA)).sum()),
        'contradiction_rows': contradictions,
    }


def main():
    if not MARGINS_PATH.exists() or not SAMPLES_PATH.exists():
        raise FileNotFoundError(f'Need both {MARGINS_PATH} and {SAMPLES_PATH}.')

    df = pd.read_csv(MARGINS_PATH, float_precision='round_trip')
    samples = np.load(SAMPLES_PATH, allow_pickle=True)

    repaired, stats = repair(df, samples)
    save_table(repaired, MARGINS_PATH)

    print(f'Rewrote {MARGINS_PATH.name}: {len(repaired)} rows, BH now over two-sided p.')
    print(f'  verdict flips at alpha={ALPHA}:      {stats["flips"]}')
    print(f'  largest q change:                 {stats["max_q_change"]:.4f}')
    print(f'  sign-blind rows (p vs own CI):    {stats["sign_blind_before"]} -> '
          f'{stats["sign_blind_after"]}')
    print(f'  rows where BH alone lifts q past alpha (legitimate): {stats["bh_lifted"]}')
    if stats['sign_blind_after']:
        raise SystemExit(
            f'Rows {stats["contradiction_rows"]} still report p >= {ALPHA} beside an '
            f'interval excluding zero; investigate before publishing.')


if __name__ == '__main__':
    main()
