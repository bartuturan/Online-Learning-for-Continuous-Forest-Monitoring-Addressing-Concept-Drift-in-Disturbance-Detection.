import json

import pandas as pd
import pytest

from src.eval.tables import load_or_run_table, save_table


@pytest.fixture
def frame():
    return pd.DataFrame({'family_id': ['a', 'b'], 'f1_score': [0.5, 0.75], 'eval_year': [2018, 2019]})


def test_writes_csv_and_json_twin(tmp_path, frame):
    csv_path = tmp_path / 'table.csv'
    save_table(frame, csv_path)

    assert csv_path.exists()
    json_path = tmp_path / 'table.json'
    assert json_path.exists()

    payload = json.loads(json_path.read_text(encoding='utf-8'))
    assert payload['metadata']['row_count'] == 2
    assert payload['metadata']['columns'] == ['family_id', 'f1_score', 'eval_year']
    assert payload['rows'][1] == {'family_id': 'b', 'f1_score': 0.75, 'eval_year': 2019}


def test_json_carries_no_timestamp(tmp_path, frame):
    """A timestamp would make every re-run dirty every tracked JSON."""
    save_table(frame, tmp_path / 'table.csv')
    payload = json.loads((tmp_path / 'table.json').read_text(encoding='utf-8'))
    assert 'generated_at_utc' not in payload['metadata']


def test_repeated_save_is_byte_identical(tmp_path, frame):
    """The property the timestamp removal buys: re-running does not touch the file."""
    csv_path = tmp_path / 'table.csv'
    save_table(frame, csv_path)
    first = (tmp_path / 'table.json').read_bytes()
    save_table(frame, csv_path)
    assert (tmp_path / 'table.json').read_bytes() == first


def test_load_or_run_computes_and_saves_when_absent(tmp_path, frame):
    calls = []

    def compute():
        calls.append(1)
        return frame

    df, skipped = load_or_run_table(tmp_path / 'table.csv', compute)
    assert len(calls) == 1
    assert skipped is False
    assert len(df) == 2
    assert (tmp_path / 'table.csv').exists() and (tmp_path / 'table.json').exists()


def test_load_or_run_skips_compute_when_csv_exists(tmp_path, frame):
    csv_path = tmp_path / 'table.csv'
    save_table(frame, csv_path)

    calls = []

    def compute():
        calls.append(1)
        return pd.DataFrame([])

    df, skipped = load_or_run_table(csv_path, compute)
    assert calls == [], 'the resume guard must not recompute an existing table'
    assert skipped is True
    assert len(df) == 2


def test_empty_table_round_trips_without_raising(tmp_path):
    """A table that legitimately computed to zero rows must survive a re-run.

    pd.DataFrame([]).to_csv() writes a bare newline, which pd.read_csv rejects with
    EmptyDataError. Before this was handled, the 8 families whose
    final_model_each_year table is empty crashed on every second run, and the
    caller's bare `except` reported it as __family_failure__.
    """
    csv_path = tmp_path / 'empty.csv'
    save_table(pd.DataFrame([]), csv_path)
    assert csv_path.stat().st_size <= 2, 'expected a headerless CSV'

    df, skipped = load_or_run_table(csv_path, lambda: pytest.fail('must not recompute'))
    assert skipped is True
    assert len(df) == 0


def test_empty_table_json_twin_is_well_formed(tmp_path):
    save_table(pd.DataFrame([]), tmp_path / 'empty.csv')
    payload = json.loads((tmp_path / 'empty.json').read_text(encoding='utf-8'))
    assert payload['metadata']['row_count'] == 0
    assert payload['metadata']['columns'] == []
    assert payload['rows'] == []


def test_resumed_load_preserves_float_bits(tmp_path):
    """A resumed run must reproduce the archived JSON byte-for-byte.

    The combined table is rebuilt from tables reloaded off disk. pandas' default CSV
    float parser is not correctly-rounded and loses ~1 ULP, so without
    float_precision='round_trip' a resume rewrote every archived JSON with
    last-digit-different values -- noise indistinguishable from a real change.
    """
    values = [0.11661426235693624, 0.39998274448902116, 0.9756937344993515]
    original = pd.DataFrame({'family_id': list('abc'), 'f1_score': values})

    csv_path = tmp_path / 'table.csv'
    save_table(original, csv_path)
    reloaded, skipped = load_or_run_table(csv_path, lambda: pytest.fail('must not recompute'))

    assert skipped is True
    assert list(reloaded['f1_score']) == values, 'float bits changed on the round trip'

    # And the JSON rewritten from the reloaded frame must match the original byte-for-byte.
    first = (tmp_path / 'table.json').read_bytes()
    save_table(reloaded, csv_path)
    assert (tmp_path / 'table.json').read_bytes() == first
