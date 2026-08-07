import json

import pandas as pd
import pytest

from scripts.run_pipeline_stages import read_eval_state, write_eval_state
from scripts.verify_eval_tables import (
    find_corrupt_tables,
    maybe_downgrade_eval_state,
    validate_csv,
    validate_json_twin,
)


def _write_table(csv_path, json_path, rows):
    """A well-formed save_table()-style pair -- rows is a list of dicts."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)
    payload = {
        "metadata": {"row_count": len(rows), "columns": list(df.columns) if rows else []},
        "rows": rows,
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# validate_csv
# ---------------------------------------------------------------------------

def test_validate_csv_well_formed(tmp_path):
    path = tmp_path / "t.csv"
    pd.DataFrame([{"a": 1, "b": 2}, {"a": 3, "b": 4}]).to_csv(path, index=False)
    ok, row_count = validate_csv(path)
    assert ok is True
    assert row_count == 2


def test_validate_csv_legitimately_empty_is_not_corrupt(tmp_path):
    """save_table's own documented shape for a genuinely-empty table: a bare
    newline that raises EmptyDataError on read. Must not be flagged."""
    path = tmp_path / "t.csv"
    path.write_text("\n", encoding="utf-8")
    ok, row_count = validate_csv(path)
    assert ok is True
    assert row_count == 0


def test_validate_csv_truncated_mid_row_is_corrupt(tmp_path):
    path = tmp_path / "t.csv"
    # An unterminated quoted field -- genuinely unparseable, unlike a short
    # row (which pandas just NaN-fills) or a clean empty file.
    path.write_bytes(b'"unterminated quote,1,2\n3,4,5')
    ok, row_count = validate_csv(path)
    assert ok is False
    assert row_count is None


# ---------------------------------------------------------------------------
# validate_json_twin
# ---------------------------------------------------------------------------

def test_validate_json_twin_consistent(tmp_path):
    json_path = tmp_path / "t.json"
    json_path.write_text(json.dumps({
        "metadata": {"row_count": 2, "columns": ["a"]},
        "rows": [{"a": 1}, {"a": 2}],
    }), encoding="utf-8")
    assert validate_json_twin(json_path, expected_row_count=2) is True


def test_validate_json_twin_missing_file(tmp_path):
    assert validate_json_twin(tmp_path / "missing.json", expected_row_count=0) is False


def test_validate_json_twin_malformed_json(tmp_path):
    json_path = tmp_path / "t.json"
    json_path.write_text("{not valid json", encoding="utf-8")
    assert validate_json_twin(json_path, expected_row_count=0) is False


def test_validate_json_twin_row_count_mismatch(tmp_path):
    json_path = tmp_path / "t.json"
    json_path.write_text(json.dumps({
        "metadata": {"row_count": 5, "columns": ["a"]},  # claims 5
        "rows": [{"a": 1}, {"a": 2}],                     # only has 2
    }), encoding="utf-8")
    assert validate_json_twin(json_path, expected_row_count=2) is False


def test_validate_json_twin_rows_not_a_list(tmp_path):
    json_path = tmp_path / "t.json"
    json_path.write_text(json.dumps({"metadata": {"row_count": 0}, "rows": "oops"}), encoding="utf-8")
    assert validate_json_twin(json_path, expected_row_count=0) is False


def test_validate_json_twin_legitimately_empty(tmp_path):
    json_path = tmp_path / "t.json"
    json_path.write_text(json.dumps({
        "metadata": {"row_count": 0, "columns": []}, "rows": [],
    }), encoding="utf-8")
    assert validate_json_twin(json_path, expected_row_count=0) is True


# ---------------------------------------------------------------------------
# find_corrupt_tables
# ---------------------------------------------------------------------------

def test_find_corrupt_tables_identifies_exactly_the_broken_ones(tmp_path):
    eval_dir = tmp_path / "unified_eval"

    # A healthy family table -- must NOT be flagged.
    _write_table(
        eval_dir / "baseline" / "baseline_combined.csv",
        eval_dir / "baseline" / "baseline_combined.json",
        [{"model_year": 2017, "f1": 0.5}],
    )

    # A family table whose CSV is truncated garbage.
    bad_csv = eval_dir / "baseline" / "baseline_broken.csv"
    bad_csv.parent.mkdir(parents=True, exist_ok=True)
    bad_csv.write_bytes(b'"unterminated,1\n2,3')

    # A family table whose CSV is fine but the .json twin is simply missing
    # (e.g. process killed between the two writes).
    orphan_csv = eval_dir / "baseline" / "baseline_orphan.csv"
    pd.DataFrame([{"a": 1}]).to_csv(orphan_csv, index=False)

    # A family table whose .json twin disagrees with the CSV's row count.
    stale_csv = eval_dir / "baseline" / "baseline_stale_twin.csv"
    stale_json = eval_dir / "baseline" / "baseline_stale_twin.json"
    pd.DataFrame([{"a": 1}, {"a": 2}]).to_csv(stale_csv, index=False)
    stale_json.write_text(json.dumps({
        "metadata": {"row_count": 1, "columns": ["a"]}, "rows": [{"a": 1}],
    }), encoding="utf-8")

    # A summary_matrices CSV with no twin at all -- expected shape, must NOT
    # be flagged even though it looks identical to the orphan case above.
    summary_csv = eval_dir / "summary_matrices" / "metric_matrix_overview.csv"
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"family": "baseline", "f1": 0.5}]).to_csv(summary_csv, index=False)

    # A legitimately-empty family table (bare newline CSV + row_count:0 twin).
    empty_csv = eval_dir / "baseline" / "baseline_final_model_each_year.csv"
    empty_json = eval_dir / "baseline" / "baseline_final_model_each_year.json"
    empty_csv.write_text("\n", encoding="utf-8")
    empty_json.write_text(json.dumps({"metadata": {"row_count": 0, "columns": []}, "rows": []}),
                           encoding="utf-8")

    corrupt = find_corrupt_tables(eval_dir)
    corrupt_names = {csv_path.name for csv_path, _json_path, _reason in corrupt}

    assert corrupt_names == {"baseline_broken.csv", "baseline_orphan.csv", "baseline_stale_twin.csv"}


def test_find_corrupt_tables_all_healthy_returns_empty(tmp_path):
    eval_dir = tmp_path / "unified_eval"
    _write_table(eval_dir / "a.csv", eval_dir / "a.json", [{"x": 1}])
    assert find_corrupt_tables(eval_dir) == []


# ---------------------------------------------------------------------------
# maybe_downgrade_eval_state
# ---------------------------------------------------------------------------

def test_maybe_downgrade_eval_state_from_complete(tmp_path):
    write_eval_state(tmp_path, signature={"stage": "x"}, status="complete")
    assert maybe_downgrade_eval_state(tmp_path) is True

    state = read_eval_state(tmp_path)
    assert state["status"] == "in_progress"
    assert state["signature"] == {"stage": "x"}  # untouched


def test_maybe_downgrade_eval_state_leaves_in_progress_alone(tmp_path):
    write_eval_state(tmp_path, signature={"stage": "x"}, status="in_progress")
    assert maybe_downgrade_eval_state(tmp_path) is False
    assert read_eval_state(tmp_path)["status"] == "in_progress"


def test_maybe_downgrade_eval_state_no_state_file(tmp_path):
    assert maybe_downgrade_eval_state(tmp_path) is False
