"""Switch the baseline-margin multiplicity correction from BH to Holm-Bonferroni.

`baseline_margins.csv` flagged each strategy's margin over the no-replay baseline with
a Benjamini-Hochberg q-value, controlling the average false-discovery rate across each
(slice, metric) family of seven. The thesis prose makes claims about *named* individual
strategies ("PR Stratification's clears zero"), which is what a family-wise error
guarantee is for -- "with >=95% probability, every flagged margin in the family is
real" -- rather than BH's average-proportion one. Holm-Bonferroni gives that guarantee,
needs no independence/PRDS assumption on the p-values (unlike BH), and is uniformly at
least as powerful as a flat Bonferroni bar.

Holm is a pure function of the `p_value_two_sided` column already sitting in this file,
so this recomputes it directly from the CSV rather than re-deriving anything from
`arm_samples.npz` or re-touching the prediction cache -- checked by hand during
planning that this reproduces exactly what a fresh `run_pairwise_ci.py` run now
produces (that script was switched to `holm_bonferroni` first). Renames the BH-era
columns to match the file's existing `p_value_one_sided` / `p_value_two_sided`
convention: `q_value_bh` -> `p_value_holm`, `bh_family` -> `holm_family`,
`bh_family_size` -> `holm_family_size`, `bh_p_side` -> `holm_p_side`.

Idempotent: a file already carrying `p_value_holm` just gets it recomputed in place.

Usage:
    python -u scripts/repair_margin_holm.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.eval.bootstrap_ci import holm_bonferroni  # noqa: E402
from src.eval.tables import save_table  # noqa: E402

PAIRWISE_DIR = Path(__file__).resolve().parent / 'bootstrap_ci' / 'pairwise'
MARGINS_PATH = PAIRWISE_DIR / 'baseline_margins.csv'
MANIFEST_PATH = PAIRWISE_DIR / 'run_manifest.json'
ALPHA = 0.05


def repair(df):
    df = df.copy()
    had_bh = 'q_value_bh' in df.columns
    old_sig = (df['q_value_bh'].to_numpy(dtype=np.float64) < ALPHA) if had_bh else None

    if 'p_value_holm' not in df.columns:
        insert_at = df.columns.get_loc('q_value_bh') if had_bh else \
            df.columns.get_loc('p_value_two_sided') + 1
        df.insert(insert_at, 'p_value_holm', np.nan)

    flips = []
    for (slice_name, metric), block in df.groupby(['slice', 'metric'], sort=False):
        p_holm = holm_bonferroni(block['p_value_two_sided'].to_numpy(dtype=np.float64))
        df.loc[block.index, 'p_value_holm'] = p_holm
        df.loc[block.index, 'holm_family'] = f'{slice_name}:{metric}'
        df.loc[block.index, 'holm_family_size'] = len(block)
        df.loc[block.index, 'holm_p_side'] = 'two_sided'

        if had_bh:
            new_sig = p_holm < ALPHA
            block_old_sig = old_sig[df.index.get_indexer(block.index)]
            for idx, was_sig, is_sig in zip(block.index, block_old_sig, new_sig):
                if was_sig != is_sig:
                    row = df.loc[idx]
                    flips.append(
                        f'{slice_name}/{metric}/{row["row_label"]}: '
                        f'BH q={row["q_value_bh"]:.4f} ({"sig" if was_sig else "n.s."}) -> '
                        f'Holm p={row["p_value_holm"]:.4f} ({"sig" if is_sig else "n.s."})'
                    )

    df['holm_family_size'] = df['holm_family_size'].astype(int)
    df = df.drop(columns=[c for c in ('q_value_bh', 'bh_family', 'bh_family_size', 'bh_p_side')
                          if c in df.columns])
    return df, flips


def main():
    if not MARGINS_PATH.exists():
        raise FileNotFoundError(f'Missing {MARGINS_PATH}; run run_pairwise_ci.py first.')

    df = pd.read_csv(MARGINS_PATH, float_precision='round_trip')
    repaired, flips = repair(df)
    save_table(repaired, MARGINS_PATH)

    print(f'Rewrote {MARGINS_PATH.name}: {len(repaired)} rows, BH -> Holm-Bonferroni.')
    print(f'  verdict flips at alpha={ALPHA}: {len(flips)}')
    for line in flips:
        print(f'    {line}')

    if MANIFEST_PATH.exists():
        import json
        manifest = json.loads(MANIFEST_PATH.read_text(encoding='utf-8'))
        if 'bh_scope' in manifest:
            manifest['holm_scope'] = manifest.pop('bh_scope')
            MANIFEST_PATH.write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
            print(f'  patched {MANIFEST_PATH.name}: bh_scope -> holm_scope')


if __name__ == '__main__':
    main()
