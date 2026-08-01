"""Run an MLP experience-replay notebook's REPLAY_RATIOS sweep as parallel processes.

Each ratio is launched as its own `jupyter nbconvert --execute` subprocess, with
MLP_REPLAY_RATIO_OVERRIDE set so the notebook trains only that one ratio (see the
env-var check right after each in-scope notebook's REPLAY_RATIOS assignment).
Safe because every replay-ratio process now reads/writes its own checkpoint files
(per_ratio_path in src/mlp_replay/checkpointing.py) rather than one shared file --
see the "Parallelize the MLP experience-replay ratio sweep" plan for why that
matters. sklearn's MLPClassifier.partial_fit doesn't benefit from extra BLAS
threads (benchmarked separately), so each subprocess is pinned to a single BLAS
thread to avoid 4 processes each also spawning multiple threads and
oversubscribing the machine.

After all ratio subprocesses succeed, one more nbconvert pass runs with no ratio
override. Since every ratio is by then already marked complete in its own
completion_status file, this pass skips all training almost instantly and
deterministically writes a combined CSV covering all ratios -- this sidesteps any
timing-dependent race in which of the 4 parallel processes' own trailing merge
cell happens to finish last.

Memory, not CPU, is what limits concurrency here: precompute_yearly_raw_cache
holds every year of the train+val split in memory at once (~5GB/process for this
dataset), so running all 4 ratios at once needs ~20GB+ and will MemoryError on a
16GB machine (confirmed: 2 of 4 crashed in exactly this way during verification).
--max-parallel batches ratios instead of launching all of them at once; the
default (2) was chosen for a 16GB machine with headroom for the OS and other
apps -- raise it on a bigger machine (e.g. Kaggle, a cloud VM) for closer to the
full N-way speedup.

Usage:
    python scripts/run_replay_sweep_parallel.py [notebook ...] [--ratios 0.2 0.3 0.4 0.5] [--max-parallel 2]

With no notebook arguments, runs all 5 in-scope notebooks.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ER_DIR = PROJECT_ROOT / "notebooks" / "training" / "mlp" / "experience_replay"

DEFAULT_NOTEBOOKS = [
    "MLP-experience replay.ipynb",
    "MLP-experience replay-target-positive-rate.ipynb",
    "MLP-experience_replay_confidently_correct_memory.ipynb",
    "MLP-experience_replay_hard_example_mining.ipynb",
    "MLP-experience_replay_uncertainity_prioritization.ipynb",
]
DEFAULT_RATIOS = [0.2, 0.3, 0.4, 0.5]
DEFAULT_MAX_PARALLEL = 2

SINGLE_THREAD_BLAS_ENV = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
}


def resolve_notebook_path(name_or_path):
    p = Path(name_or_path)
    if p.exists():
        return p
    candidate = ER_DIR / name_or_path
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"Notebook not found: {name_or_path} (looked in {ER_DIR})")


def nbconvert_command(notebook_path, output_path):
    return [
        sys.executable, "-m", "jupyter", "nbconvert",
        "--to", "notebook", "--execute",
        "--ExecutePreprocessor.timeout=-1",
        "--ExecutePreprocessor.kernel_name=python3",
        f"--output={output_path.name}",
        f"--output-dir={output_path.parent}",
        str(notebook_path),
    ]


def run_ratio_subprocess(notebook_path, ratio, log_dir):
    log_dir.mkdir(parents=True, exist_ok=True)
    ratio_tag = f"RR_{ratio:.1f}"
    output_path = notebook_path.with_name(f"{notebook_path.stem}__{ratio_tag}.ipynb")
    log_path = log_dir / f"{notebook_path.stem}__{ratio_tag}.log"

    env = {**os.environ, **SINGLE_THREAD_BLAS_ENV, "MLP_REPLAY_RATIO_OVERRIDE": str(ratio)}
    log_file = open(log_path, "w", encoding="utf-8")
    proc = subprocess.Popen(
        nbconvert_command(notebook_path, output_path),
        cwd=PROJECT_ROOT, env=env, stdout=log_file, stderr=subprocess.STDOUT,
    )
    return proc, log_file, log_path


def run_verification_pass(notebook_path, log_dir):
    """No ratio override: with every ratio already marked complete, this just
    skips training and writes a definitive, race-free combined CSV."""
    log_dir.mkdir(parents=True, exist_ok=True)
    output_path = notebook_path.with_name(f"{notebook_path.stem}__merge_verify.ipynb")
    log_path = log_dir / f"{notebook_path.stem}__merge_verify.log"

    env = {**os.environ, **SINGLE_THREAD_BLAS_ENV}
    with open(log_path, "w", encoding="utf-8") as log_file:
        result = subprocess.run(
            nbconvert_command(notebook_path, output_path),
            cwd=PROJECT_ROOT, env=env, stdout=log_file, stderr=subprocess.STDOUT,
        )
    return result.returncode, log_path


def _batches(items, batch_size):
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]


def run_sweep_for_notebook(notebook_path, ratios, log_dir, max_parallel):
    print(f"\n=== {notebook_path.name}: {len(ratios)} ratio(s), up to {max_parallel} at a time ===")
    failures = []

    for batch in _batches(ratios, max_parallel):
        print(f"  Launching batch: {[f'RR_{r:.1f}' for r in batch]}")
        running = [run_ratio_subprocess(notebook_path, ratio, log_dir) for ratio in batch]

        for (proc, log_file, log_path), ratio in zip(running, batch):
            returncode = proc.wait()
            log_file.close()
            status = "OK" if returncode == 0 else "FAILED"
            print(f"  RR_{ratio:.1f}: {status} (log: {log_path})")
            if returncode != 0:
                failures.append((ratio, log_path))

        if failures:
            # Stop launching further batches -- no point starting more ratio
            # processes once one has already failed (e.g. MemoryError).
            print(f"  {len(failures)} ratio(s) failed for {notebook_path.name} -- not launching further batches.")
            break

    if failures:
        print(f"  Skipping merge verification pass for {notebook_path.name}.")
        return False

    print(f"  All {len(ratios)} ratios succeeded. Running final merge/verification pass...")
    returncode, log_path = run_verification_pass(notebook_path, log_dir)
    if returncode != 0:
        print(f"  Merge verification pass FAILED (log: {log_path})")
        return False
    print(f"  Merge verification pass OK (log: {log_path})")
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("notebooks", nargs="*", default=DEFAULT_NOTEBOOKS,
                         help="Notebook filename(s) (resolved against experience_replay/ if not a full path)")
    parser.add_argument("--ratios", nargs="+", type=float, default=DEFAULT_RATIOS)
    parser.add_argument("--max-parallel", type=int, default=DEFAULT_MAX_PARALLEL,
                         help="Max ratio processes to run at once (default 2; see module docstring for why)")
    parser.add_argument("--log-dir", default=str(PROJECT_ROOT / "replay_sweep_logs"))
    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    overall_ok = True
    for name in args.notebooks:
        notebook_path = resolve_notebook_path(name)
        ok = run_sweep_for_notebook(notebook_path, args.ratios, log_dir, args.max_parallel)
        overall_ok = overall_ok and ok

    if not overall_ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
