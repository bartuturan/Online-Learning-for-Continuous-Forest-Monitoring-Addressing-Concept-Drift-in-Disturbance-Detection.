"""Reading and writing the evaluation result tables.

Every table is written twice: a CSV for convenience and a JSON twin carrying the same
rows plus metadata. The JSON is the archival copy -- `.gitignore` excludes `*.csv`, so
the JSONs are the only version-controlled record of what each evaluation produced.

The JSON deliberately carries **no timestamp**. It used to embed
`generated_at_utc`, which meant re-running the pipeline rewrote every tracked JSON with
a one-line diff even when the underlying table was unchanged and skipped by the resume
guard. That made `git status` useless as a check on whether a run had side effects. Git
already records when a file changed and by whom; `row_count` and `columns` are the
metadata that carries information about the content.
"""

import json

import pandas as pd


def save_table(df, csv_path):
    """Write `df` as CSV plus a JSON twin holding the same rows and light metadata."""
    df.to_csv(csv_path, index=False)
    payload = {
        'metadata': {
            'row_count': int(len(df)),
            'columns': list(df.columns),
        },
        'rows': df.to_dict(orient='records'),
    }
    with open(csv_path.with_suffix('.json'), 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2)


def load_or_run_table(table_csv_path, fn_compute):
    """Return an existing table, or compute and save it.

    Returns `(df, was_skipped)`. This is the resume guard that lets an interrupted
    evaluation pick up where it stopped: an existing CSV is trusted and `fn_compute`
    is never called.
    """
    if table_csv_path.exists():
        print(f'SKIP existing table: {table_csv_path.name}')
        return pd.read_csv(table_csv_path), True

    table_df = fn_compute()
    save_table(table_df, table_csv_path)
    print(f'SAVED table: {table_csv_path.name} ({len(table_df):,} rows)')
    return table_df, False
