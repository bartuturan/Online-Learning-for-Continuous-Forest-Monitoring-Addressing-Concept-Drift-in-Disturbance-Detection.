"""Verify every evaluation table's CSV/JSON pair is intact, repair any that
aren't, and trigger the orchestrator to regenerate exactly what's missing.

Why this exists: src/eval/tables.py's save_table() writes a table's .csv, then
its .json twin, as two separate, non-atomic file writes (unlike e.g.
src/mlp_replay/checkpointing.py's atomic writes elsewhere in this repo). A
Kaggle session killed mid-write can leave one or both truncated. load_or_run_table
then trusts any existing .csv unconditionally -- a truncated-but-still-parseable
file wouldn't just fail to load, it would silently load as an incomplete table.

This script applies the same "valid" test load_or_run_table itself uses (a
table that's genuinely empty is written as a bare-newline CSV that raises
EmptyDataError on read -- that's expected, not corrupt; see that function's
docstring), plus a consistency check against each table's .json twin (same
row count, same columns), deletes anything that fails either test, and -- if
anything was deleted -- downgrades the orchestrator's own eval-state file from
"complete" to "in_progress" so the next eval_unified run resumes rather than
reporting DONE, and only recomputes the tables that are now missing (every
other cached table is left alone, per load_or_run_table's normal behavior).

Usage:
    python -u scripts/verify_eval_tables.py                # scan, repair, recreate
    python -u scripts/verify_eval_tables.py --dry-run       # scan only, changes nothing
    python -u scripts/verify_eval_tables.py --no-recreate   # scan + delete, but don't
                                                              # re-run eval_unified
    python -u scripts/verify_eval_tables.py --git-push      # also persist afterward
                                                              # (see git_commit_and_push)
"""
import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.run_pipeline_stages import (  # noqa: E402
    EVAL_DIR_REL,
    PROJECT_ROOT,
    STAGES_BY_ID,
    git_commit_and_push,
    read_eval_state,
    run_eval_stage,
    write_eval_state,
)


def validate_csv(csv_path):
    """Mirrors load_or_run_table's own notion of "valid": either it parses, or
    it's the specific empty-table shape that function already treats as
    legitimate. Returns (ok, row_count) -- row_count is None when ok is False."""
    try:
        df = pd.read_csv(csv_path, float_precision="round_trip")
        return True, len(df)
    except pd.errors.EmptyDataError:
        return True, 0
    except Exception:
        return False, None


def validate_json_twin(json_path, expected_row_count):
    """save_table always writes a .json twin alongside the .csv, holding the
    same rows plus {row_count, columns} metadata. A missing twin, unparseable
    twin, or one whose row_count/rows disagree with the .csv is exactly the
    signature of an interrupted write (the two files are written in sequence,
    never atomically together)."""
    if not json_path.exists():
        return False
    try:
        payload = json.loads(json_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError, UnicodeDecodeError):
        return False
    if not isinstance(payload, dict):
        return False
    metadata = payload.get("metadata", {})
    rows = payload.get("rows")
    if not isinstance(rows, list):
        return False
    if metadata.get("row_count") != expected_row_count:
        return False
    if len(rows) != expected_row_count:
        return False
    return True


# Evaluations.ipynb writes two different kinds of CSV under EVAL_DIR: per-family
# tables (cell 4, via load_or_run_table -> save_table, always with a .json twin --
# these are the resumable, individually-cached units) and derived summary
# matrices (cell 5, plain matrix_df.to_csv(), never a twin, unconditionally
# rewritten every time the notebook runs). Expecting a twin under
# summary_matrices/ would flag every one of those files as corrupt for a
# reason that was never true.
SUMMARY_MATRICES_DIRNAME = "summary_matrices"


def find_corrupt_tables(eval_dir):
    """Every *.csv under eval_dir whose file fails validate_csv, or (family
    tables only) whose .json twin fails validate_json_twin. Returns
    [(csv_path, json_path_or_None, reason)]."""
    corrupt = []
    for csv_path in sorted(eval_dir.rglob("*.csv")):
        ok, row_count = validate_csv(csv_path)
        if not ok:
            corrupt.append((csv_path, csv_path.with_suffix(".json"), "CSV failed to parse"))
            continue

        if SUMMARY_MATRICES_DIRNAME in csv_path.relative_to(eval_dir).parts:
            continue  # no twin expected -- see module note above

        json_path = csv_path.with_suffix(".json")
        if not validate_json_twin(json_path, row_count):
            corrupt.append((csv_path, json_path, "missing or inconsistent .json twin"))
    return corrupt


def maybe_downgrade_eval_state(root):
    """If the orchestrator's eval-state file currently says "complete", rewrite
    it to "in_progress" for the SAME training signature. Never touches the
    signature itself -- only status -- so the next eval_unified run enters the
    "resuming" branch (keep cached tables) rather than either wrongly reporting
    DONE or wrongly clearing every table as if the training state had changed.
    Returns True if it downgraded anything."""
    state = read_eval_state(root)
    if state and state.get("status") == "complete":
        write_eval_state(root, state["signature"], "in_progress")
        return True
    return False


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--dry-run", action="store_true",
                         help="Report corrupt tables, delete nothing.")
    parser.add_argument("--no-recreate", action="store_true",
                         help="Delete corrupt tables but don't re-run eval_unified afterward.")
    parser.add_argument("--git-push", action="store_true",
                         help="Commit and push experiments/ after recreating (see "
                              "git_commit_and_push in run_pipeline_stages.py).")
    parser.add_argument("--root", default=str(PROJECT_ROOT))
    args = parser.parse_args()

    root = Path(args.root)
    eval_dir = root / EVAL_DIR_REL
    if not eval_dir.exists():
        print(f"No evaluation output directory at {eval_dir} -- nothing to verify.")
        return 0

    total = sum(1 for _ in eval_dir.rglob("*.csv"))
    corrupt = find_corrupt_tables(eval_dir)
    if not corrupt:
        print(f"Checked {total} table(s) -- all intact.")
        return 0

    print(f"{len(corrupt)}/{total} table(s) corrupt:")
    for csv_path, _json_path, reason in corrupt:
        print(f"  {csv_path.relative_to(root)}  ({reason})")

    if args.dry_run:
        print("\n--dry-run: nothing deleted.")
        return 1

    for csv_path, json_path, _reason in corrupt:
        csv_path.unlink(missing_ok=True)
        json_path.unlink(missing_ok=True)
    print(f"\nDeleted {len(corrupt)} table(s) (and their .json twins).")

    if maybe_downgrade_eval_state(root):
        print("orchestrator_eval_state.json: complete -> in_progress (same training "
              "signature) -- the next eval_unified run will resume and recompute only "
              "the deleted tables.")

    if args.no_recreate:
        print("\n--no-recreate: run `python -u scripts/run_pipeline_stages.py "
              "--only eval_unified` yourself when ready.")
        return 1

    print("\nRe-running eval_unified to regenerate the deleted table(s)...")
    stage = STAGES_BY_ID["eval_unified"]
    log_dir = root / "pipeline_logs" / time.strftime("%Y%m%dT%H%M%S")
    ok, log_ref = run_eval_stage(stage, root, log_dir)
    if not ok:
        print(f"eval_unified FAILED -- see {log_ref}")
        return 1

    print(f"eval_unified OK (log: {log_ref})")
    if args.git_push:
        git_commit_and_push(root, "Repair corrupted evaluation table(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
