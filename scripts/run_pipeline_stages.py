"""Stage-chaining orchestrator for the training pipeline (see the "Stage-chaining
orchestrator for the training pipeline on Kaggle" plan for the full design writeup).

Problem this solves: Kaggle sessions cap out around 9-12h, but the training stage
(data prep + 5 SGD notebooks + the MLP baseline + all 13 experience-replay
notebooks) takes far longer. Re-running this script in every new session runs
only whatever isn't finished yet -- completion is derived entirely from the
artifacts each notebook already writes (final model files, completion_status.json
files, zarr variable presence), the same way src/eval/tables.py's
load_or_run_table and src/mlp_replay/checkpointing.py's completion_status.json
already work. There is no separate orchestrator state file to fall out of sync
with reality.

Scope: data prep (3 notebooks), 5 SGD notebooks, the MLP baseline, and all 13
experience-replay notebooks under notebooks/training/mlp/experience_replay/.
Explicitly NOT covered: XGBoost (removed from the active pipeline in commit
48ec425; only exists at old_notebooks/XGBoost.ipynb, which is gitignored -- the
README's references to notebooks/training/xgboost/XGBoost.ipynb are stale), the
2 SGD experience-replay notebooks (blocking input() resume prompt that fires
exactly in the chaining case -- run those by hand), evaluation, and drift
detection (separate stages, not this script).

=== Kaggle runbook ===

Data prep runs LOCALLY (this avoids uploading the 11GB + 21GB raw source zarrs):

    python -u scripts/run_pipeline_stages.py --group dataprep

Then upload training_data_with_features_plus_monthly_indices.zarr (4.6GB) +
data_split.npz as one Kaggle dataset (via `kaggle datasets create` -- a zarr
store's thousands of chunk files make the web UI impractical).

On Kaggle, only the training groups run, so the ~20GB working quota is
comfortable. The rule that keeps it that way: anything a stage WRITES must be a
real copy into /kaggle/working; anything a stage only READS may be a symlink
into the read-only /kaggle/input (symlinks don't consume the quota).

Session 1:
    !git clone --depth 1 --branch pipeline-rerun <repo-url> /kaggle/working/repo
    %cd /kaggle/working/repo
    # symlink the attached dataset in as the two files training_data_with_features_plus_monthly_indices.zarr
    # and data_split.npz (read-only source -- symlink, don't copy)
    import os
    os.environ['MLP_FEATURE_CACHE_DIR'] = '/kaggle/working/feature_cache_disk'
    !python -u scripts/run_pipeline_stages.py --list
    !python -u scripts/run_pipeline_stages.py --group sgd mlp_baseline er_parallel er_sequential --time-budget 8.0

Session N: identical, plus one cell before the run that carries the previous
session's *committed* output forward -- symlink the dataset again, and
copytree() experiments/ (dirs_exist_ok=True, since the fresh clone already has
tracked files there) and feature_cache_disk/ from the previous session's output
dataset. Then the same run command resumes: DONE stages are skipped, PARTIAL
ones pick up mid-sweep.

Usage:
    python -u scripts/run_pipeline_stages.py --list
    python -u scripts/run_pipeline_stages.py --dry-run
    python -u scripts/run_pipeline_stages.py --group sgd mlp_baseline
    python -u scripts/run_pipeline_stages.py --only er_hard_example_mining --max-parallel 4
"""

import argparse
import difflib
import json
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_replay_sweep_parallel import (  # noqa: E402
    SINGLE_THREAD_BLAS_ENV,
    nbconvert_command,
    run_sweep_for_notebook,
)
from src.mlp_replay.checkpointing import (  # noqa: E402
    format_ratio_key,
    load_completion_status,
    per_ratio_path,
)

MONTHLY_INDEX_VARS = (
    "ndvi_cv_year", "ndvi_max_m2m_drop_year", "ndvi_max_year", "ndvi_min_year", "ndvi_std_year",
    "ndwi_cv_year", "ndwi_max_m2m_drop_year", "ndwi_max_year", "ndwi_min_year", "ndwi_std_year",
)

BYTES_PER_REPLAY_PROCESS = int(5.5 * 1024 ** 3)
DEFAULT_MAX_PARALLEL_FALLBACK = 2


# ---------------------------------------------------------------------------
# Stage spec
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class StageStatus:
    state: str            # "DONE" | "PARTIAL" | "TODO" | "BLOCKED"
    detail: str = ""       # one line, shown in the status table
    remaining: tuple = ()  # ratios still to do (all of them, if TODO) -- fed straight to the sweep runner
    remedy: str = ""       # BLOCKED only: a copy-pasteable command
    note: str = ""         # advisory, doesn't change state (e.g. legacy-completion-file warning)


@dataclass(frozen=True)
class Stage:
    id: str
    group: str             # dataprep | sgd | mlp_baseline | er_parallel | er_sequential
    notebook: str          # PROJECT_ROOT-relative
    check: "object"        # Callable[[Path, "Stage"], StageStatus]
    runner: str            # "nbconvert" | "ratio_sweep"
    ratios: tuple = ()
    requires: tuple = ()
    inputs: tuple = ()     # PROJECT_ROOT-relative paths that must exist before this can run at all
    default_selected: bool = True
    notes: str = ""


# ---------------------------------------------------------------------------
# Completion-check helpers -- pure, unit-testable, no subprocess/import cost
# ---------------------------------------------------------------------------

def artifacts_exist(root, rel_paths):
    rel_paths = tuple(rel_paths)
    present = [p for p in rel_paths if (root / p).exists()]
    missing = [p for p in rel_paths if p not in present]
    if not missing:
        return StageStatus("DONE", detail=f"{len(present)}/{len(rel_paths)} artifact(s) present")
    if not present:
        return StageStatus("TODO", detail=f"missing: {', '.join(missing)}")
    return StageStatus("PARTIAL", detail=f"missing: {', '.join(missing)}", remaining=tuple(missing))


def _zarr_variable_names(store):
    """Read a zarr v3 store's inline consolidated_metadata; None if unreadable
    (caller falls back to per-variable directory existence)."""
    zarr_json = store / "zarr.json"
    if not zarr_json.exists():
        return None
    try:
        meta = json.loads(zarr_json.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    consolidated = meta.get("consolidated_metadata")
    if not isinstance(consolidated, dict):
        return None
    var_meta = consolidated.get("metadata")
    if not isinstance(var_meta, dict):
        return None
    return set(var_meta.keys())


def zarr_has_vars(root, rel_store, required_vars):
    store = root / rel_store
    if not store.exists():
        return StageStatus("TODO", detail=f"{rel_store} does not exist")

    var_names = _zarr_variable_names(store)
    if var_names is not None:
        present = [v for v in required_vars if v in var_names]
    else:
        present = [v for v in required_vars if (store / v).exists()]
    missing = [v for v in required_vars if v not in present]

    if not missing:
        return StageStatus("DONE", detail=f"{rel_store} has all {len(required_vars)} required vars")

    # Exists but incomplete (or matches none of the expected vars at all) -- never
    # auto-run, never auto-delete. A partially-written store looks identical to a
    # freshly-started one from the outside; only a human should decide to remove it.
    delete_cmd = f'rm -rf "{store}"'
    rerun_cmd = f"python -u scripts/run_pipeline_stages.py --from dataprep_merge_monthly"
    remedy = (
        f"Delete the incomplete store, then re-run:\n"
        f"  {delete_cmd}\n"
        f"  {rerun_cmd}"
    )
    return StageStatus(
        "BLOCKED",
        detail=f"{rel_store} exists but incomplete ({len(present)}/{len(required_vars)} vars, "
               f"missing: {', '.join(missing)})",
        remedy=remedy,
    )


def ratios_complete_shared(root, rel_json, ratios):
    status = load_completion_status(root / rel_json)
    completed = set(status.get("completed_ratios", []))
    keys = [format_ratio_key(r) for r in ratios]
    missing = [r for r, k in zip(ratios, keys) if k not in completed]

    if not missing:
        return StageStatus("DONE", detail=f"{len(ratios)}/{len(ratios)} ratios complete ({rel_json})")
    n_done = len(ratios) - len(missing)
    state = "TODO" if n_done == 0 else "PARTIAL"
    return StageStatus(state, detail=f"{n_done}/{len(ratios)} ratios complete", remaining=tuple(missing))


def ratios_complete_per_ratio(root, rel_json_base, ratios):
    base = root / rel_json_base
    missing, done = [], []
    for r in ratios:
        key = format_ratio_key(r)
        status = load_completion_status(per_ratio_path(base, key))
        (done if key in status.get("completed_ratios", []) else missing).append(r)

    note = ""
    if missing and base.exists():
        # Pre-refactor notebooks wrote one shared completion file across all ratios;
        # the refactored notebook only ever reads its own per-ratio files, so a
        # legacy shared file claiming completion is silently ignored by the
        # notebook itself -- surface that here so a TODO doesn't look like a bug.
        legacy_completed = set(load_completion_status(base).get("completed_ratios", []))
        if legacy_completed:
            note = (
                f"legacy shared completion file lists {sorted(legacy_completed)} as done, but the "
                f"refactored notebook only reads its own per-ratio files -- ignored, will retrain."
            )

    if not missing:
        return StageStatus("DONE", detail=f"{len(ratios)}/{len(ratios)} ratios complete (per-ratio files)", note=note)
    state = "TODO" if not done else "PARTIAL"
    return StageStatus(state, detail=f"{len(done)}/{len(ratios)} ratios complete",
                        remaining=tuple(missing), note=note)


def artifacts_check(*rels):
    return lambda root, stage: artifacts_exist(root, rels)


def zarr_vars_check(rel_store, required_vars):
    return lambda root, stage: zarr_has_vars(root, rel_store, required_vars)


def shared_check(rel_json):
    return lambda root, stage: ratios_complete_shared(root, rel_json, stage.ratios)


def per_ratio_check(rel_json_base):
    return lambda root, stage: ratios_complete_per_ratio(root, rel_json_base, stage.ratios)


def er_combined_check(root, stage):
    """Same as shared_check, plus a warning if sibling completion files using a
    DIFFERENT strategy-hyperparameter suffix exist in the same directory -- this
    notebook's completion filename is config-coupled (encodes 6 tunable
    hyperparameters), so it's easy to tune a value and have this stage silently
    report TODO forever without realizing old runs under other configs exist."""
    rel_json = ("experiments/mlp/combined/training_checkpoints_mlp_experience_replay_combined/"
                "mlp_replay_completion_status_combined_HE=0_CC=0.2_UP=0.1_PR=(0.2,10)_MC=0_RWS=1.json")
    status = ratios_complete_shared(root, rel_json, stage.ratios)

    combined_dir = root / "experiments/mlp/combined/training_checkpoints_mlp_experience_replay_combined"
    if combined_dir.exists():
        current_name = Path(rel_json).name
        siblings = sorted(
            p.name for p in combined_dir.glob("mlp_replay_completion_status_combined_*.json")
            if p.name != current_name
        )
        if siblings:
            extra = (f"{len(siblings)} completion file(s) for OTHER strategy hyperparameters exist "
                      f"in this dir (not used by the current config): {siblings}")
            status = replace(status, note=(status.note + " " + extra).strip())
    return status


# ---------------------------------------------------------------------------
# Stage table
# ---------------------------------------------------------------------------

ER = "notebooks/training/mlp/experience_replay"
ERC = "experiments/mlp/experience_replay"
SGD_RATIOS = ()  # SGD notebooks have no ratio sweep

STAGES = (
    # --- data prep --------------------------------------------------------
    Stage(
        id="dataprep_base", group="dataprep",
        notebook="notebooks/data_prep/Data-prep.ipynb",
        inputs=("full_dataset_resizedv2.zarr",),
        check=artifacts_check("training_data_with_features.zarr", "data_split.npz"),
        runner="nbconvert",
        notes="Unconditionally overwrites its output (shutil.rmtree + mode='w') -- no internal skip.",
    ),
    Stage(
        id="dataprep_merge_monthly", group="dataprep",
        notebook="notebooks/data_prep/merge monthly features.ipynb",
        requires=("dataprep_base",),
        inputs=("training_data_with_features.zarr", "full_dataset_20m_monthly_with_indices.zarr"),
        check=zarr_vars_check("training_data_with_features_plus_monthly_indices.zarr", MONTHLY_INDEX_VARS),
        runner="nbconvert",
        notes=("Raises FileExistsError if its output already exists (does not skip gracefully); "
               "builds the store via copytree + per-feature append, so a crash mid-run leaves an "
               "EXISTING but INCOMPLETE store. This is why its completion check reads variable "
               "names rather than just checking the store exists."),
    ),
    Stage(
        id="dataprep_lagged_monthly", group="dataprep",
        notebook="notebooks/data_prep/feature_prep_lagged_monthly.ipynb",
        requires=("dataprep_base",),
        inputs=("training_data_with_features.zarr", "data_split.npz"),
        check=artifacts_check(
            "feature_cache_lagged_monthly.zarr",
            "feature_cache_lagged_monthly_metadata.json",
            "feature_cache_lagged_monthly_validation.json",
        ),
        runner="nbconvert",
        default_selected=False,
        notes="Leaf: nothing in the active pipeline reads feature_cache_lagged_monthly.zarr. "
              "Not selected by default -- run explicitly via --only dataprep_lagged_monthly.",
    ),

    # --- SGD ---------------------------------------------------------------
    Stage(
        id="sgd_baseline", group="sgd",
        notebook="notebooks/training/sgd/SGD Classifier.ipynb",
        requires=("dataprep_base",),
        check=artifacts_check("experiments/sgd/models/model_year_2022.pkl"),
        runner="nbconvert",
        notes="No resume logic -- a rerun retrains all 6 years (2017-2022) from scratch.",
    ),
    Stage(
        id="sgd_prevyears", group="sgd",
        notebook="notebooks/training/sgd/SGD Classifier_prevyears.ipynb",
        requires=("dataprep_base",),
        check=artifacts_check("experiments/sgd/models_lagged_features/model_year_2022_lagged_features.pkl"),
        runner="nbconvert",
        notes="No resume logic.",
    ),
    Stage(
        id="sgd_prevyears_incr", group="sgd",
        notebook="notebooks/training/sgd/SGD Classifier_prevyears-incremental scaler.ipynb",
        requires=("dataprep_base",),
        check=artifacts_check(
            "experiments/sgd/models_lagged_features_incremental_scaler/"
            "scaler_final_lagged_features_incremental_scaler.pkl"
        ),
        runner="nbconvert",
        notes="No resume logic.",
    ),
    Stage(
        id="sgd_prevyears_monthly", group="sgd",
        notebook="notebooks/training/sgd/SGD Classifier_prevyears and monthly features.ipynb",
        requires=("dataprep_merge_monthly",),
        check=artifacts_check(
            "experiments/sgd/models_prevyears_monthly_features/model_year_2022_prevyears_monthly_features.pkl"
        ),
        runner="nbconvert",
        notes="No resume logic.",
    ),
    Stage(
        id="sgd_prevyears_monthly_incr", group="sgd",
        notebook="notebooks/training/sgd/SGD Classifier_prevyears and monthly features-incremental scaler.ipynb",
        requires=("dataprep_merge_monthly",),
        check=artifacts_check(
            "experiments/sgd/models_prevyears_monthly_features_incremental_scaler/"
            "scaler_final_prevyears_monthly_features_incremental_scaler.pkl"
        ),
        runner="nbconvert",
        notes="No resume logic.",
    ),

    # --- MLP baseline (has its own ratio sweep, but shares one checkpoint file
    #     across ratios like the unrefactored ER notebooks -- never parallelize) --
    Stage(
        id="mlp_baseline", group="mlp_baseline",
        notebook="notebooks/training/mlp/MLP_prevyears_and_monthly_features.ipynb",
        requires=("dataprep_merge_monthly",), ratios=(0.2, 0.3, 0.4, 0.5),
        check=shared_check("experiments/mlp/baseline/training_checkpoints_mlp_replay_ratios/mlp_replay_completion_status.json"),
        runner="nbconvert",
        notes="Has its own internal replay-ratio sweep with full resume logic, but a shared "
              "completion file across ratios -- run as a single sequential process, never parallelized.",
    ),

    # --- experience replay: parallel-safe (per-ratio checkpoint files) ------
    Stage(
        id="er_plain", group="er_parallel",
        notebook=f"{ER}/MLP-experience replay.ipynb",
        requires=("dataprep_merge_monthly",), ratios=(0.2, 0.3, 0.4, 0.5),
        check=per_ratio_check(f"{ERC}/training_checkpoints_mlp_experience_replay/mlp_replay_completion_status.json"),
        runner="ratio_sweep",
    ),
    Stage(
        id="er_target_positive_rate", group="er_parallel",
        notebook=f"{ER}/MLP-experience replay-target-positive-rate.ipynb",
        requires=("dataprep_merge_monthly",), ratios=(0.2, 0.3, 0.4, 0.5),
        check=per_ratio_check(f"{ERC}/training_checkpoints_mlp_experience_replay_PR_0.10/mlp_replay_completion_status.json"),
        runner="ratio_sweep",
    ),
    Stage(
        id="er_confidently_correct", group="er_parallel",
        notebook=f"{ER}/MLP-experience_replay_confidently_correct_memory.ipynb",
        requires=("dataprep_merge_monthly",), ratios=(0.2, 0.3, 0.4, 0.5),
        check=per_ratio_check(
            f"{ERC}/training_checkpoints_mlp_experience_replay_confidently_correct_memory/"
            "mlp_replay_completion_status_confidently_correct_memory.json"
        ),
        runner="ratio_sweep",
    ),
    Stage(
        id="er_hard_example_mining", group="er_parallel",
        notebook=f"{ER}/MLP-experience_replay_hard_example_mining.ipynb",
        requires=("dataprep_merge_monthly",), ratios=(0.2, 0.3, 0.4, 0.5),
        check=per_ratio_check(
            f"{ERC}/training_checkpoints_mlp_experience_replay_hard_example_mining/"
            "mlp_replay_completion_status_hard_example_mining.json"
        ),
        runner="ratio_sweep",
    ),
    Stage(
        id="er_uncertainty_prioritization", group="er_parallel",
        notebook=f"{ER}/MLP-experience_replay_uncertainity_prioritization.ipynb",
        requires=("dataprep_merge_monthly",), ratios=(0.2, 0.3, 0.4, 0.5),
        check=per_ratio_check(
            f"{ERC}/training_checkpoints_mlp_experience_replay_uncertainity_prioritization/"
            "mlp_replay_completion_status_uncertainity_prioritization.json"
        ),
        runner="ratio_sweep",
    ),

    # --- experience replay: NOT refactored, shared completion file across
    #     ratios -- must run as a single sequential process each, never through
    #     the parallel launcher (would clobber each other's progress) ---------
    Stage(
        id="er_plain_reservoir", group="er_sequential",
        notebook=f"{ER}/MLP-experience replay_reservoir_sampling.ipynb",
        requires=("dataprep_merge_monthly",), ratios=(0.2, 0.3, 0.4, 0.5),
        check=shared_check(f"{ERC}/training_checkpoints_mlp_experience_replay_reservoir_sampling/mlp_replay_completion_status.json"),
        runner="nbconvert",
    ),
    Stage(
        id="er_target_positive_rate_reservoir", group="er_sequential",
        notebook=f"{ER}/MLP-experience replay-target-positive-rate_reservoir_sampling.ipynb",
        requires=("dataprep_merge_monthly",), ratios=(0.2, 0.3, 0.4, 0.5),
        check=shared_check(
            f"{ERC}/training_checkpoints_mlp_experience_replay_reservoir_sampling_PR_0.10/"
            "mlp_replay_completion_status.json"
        ),
        runner="nbconvert",
        notes="Note the token order: _reservoir_sampling_PR_0.10 (token after, unlike its non-reservoir twin).",
    ),
    Stage(
        id="er_confidently_correct_reservoir", group="er_sequential",
        notebook=f"{ER}/MLP-experience_replay_confidently_correct_memory_reservoir_sampling.ipynb",
        requires=("dataprep_merge_monthly",), ratios=(0.2, 0.3, 0.4, 0.5),
        check=shared_check(
            f"{ERC}/training_checkpoints_mlp_experience_replay_confidently_correct_memory_reservoir_sampling/"
            "mlp_replay_completion_status_confidently_correct_memory.json"
        ),
        runner="nbconvert",
    ),
    Stage(
        id="er_hard_example_mining_reservoir", group="er_sequential",
        notebook=f"{ER}/MLP-experience_replay_hard_example_mining_reservoir_sampling.ipynb",
        requires=("dataprep_merge_monthly",), ratios=(0.2, 0.3, 0.4, 0.5),
        check=shared_check(
            f"{ERC}/training_checkpoints_mlp_experience_replay_hard_example_mining_reservoir_sampling/"
            "mlp_replay_completion_status_hard_example_mining.json"
        ),
        runner="nbconvert",
    ),
    Stage(
        id="er_uncertainty_prioritization_reservoir", group="er_sequential",
        notebook=f"{ER}/MLP-experience_replay_uncertainity_prioritization_reservoir_sampling.ipynb",
        requires=("dataprep_merge_monthly",), ratios=(0.2, 0.3, 0.4, 0.5),
        check=shared_check(
            f"{ERC}/training_checkpoints_mlp_experience_replay_uncertainity_prioritization_reservoir_sampling/"
            "mlp_replay_completion_status_uncertainity_prioritization.json"
        ),
        runner="nbconvert",
    ),
    Stage(
        id="er_misclassification_buffer", group="er_sequential",
        notebook=f"{ER}/MLP-experience replay_misclassification_buffer.ipynb",
        requires=("dataprep_merge_monthly",), ratios=(0.3,),
        check=shared_check(
            f"{ERC}/training_checkpoints_mlp_missclassification_buffer/"
            "mlp_missclassification_buffer_completion_status.json"
        ),
        runner="nbconvert",
        notes="Single fixed ratio (0.3) -- nothing to parallelize even before the shared-file risk.",
    ),
    Stage(
        id="er_misclassification_buffer_reservoir", group="er_sequential",
        notebook=f"{ER}/MLP-experience replay_misclassification_buffer_reservoir_sampling.ipynb",
        requires=("dataprep_merge_monthly",), ratios=(0.3,),
        check=shared_check(
            f"{ERC}/training_checkpoints_mlp_missclassification_buffer_reservoir_sampling/"
            "mlp_missclassification_buffer_completion_status.json"
        ),
        runner="nbconvert",
        notes="Single fixed ratio (0.3).",
    ),
    Stage(
        id="er_combined", group="er_sequential",
        notebook=f"{ER}/MLP_experience_replay_combined.ipynb",
        requires=("dataprep_merge_monthly",), ratios=(0.5,),
        check=er_combined_check,
        runner="nbconvert",
        notes="Sweeps 1 ratio while combining multiple replay strategies at fixed fractions; "
              "its completion filename encodes those fractions (HE/CC/UP/PR/MC/RWS), hardcoded "
              "here for the notebook's current default values. Writes to experiments/mlp/combined, "
              "not experiments/mlp/experience_replay.",
    ),
)

STAGES_BY_ID = {s.id: s for s in STAGES}


# ---------------------------------------------------------------------------
# Status computation, selection, dependency resolution
# ---------------------------------------------------------------------------

def compute_status(stage, root):
    missing_inputs = [p for p in stage.inputs if not (root / p).exists()]
    if missing_inputs:
        joined = ", ".join(missing_inputs)
        return StageStatus(
            "BLOCKED",
            detail=f"missing input(s): {joined}",
            remedy=f"Ensure these exist under {root} before running this stage: {joined}",
        )
    return stage.check(root, stage)


def select_stages(stages, only=None, groups=None, from_id=None, skip=None):
    by_id = {s.id: s for s in stages}
    known_groups = {s.group for s in stages}

    for sid in list(only or []) + list(skip or []) + ([from_id] if from_id else []):
        if sid not in by_id:
            close = difflib.get_close_matches(sid, by_id.keys(), n=3)
            hint = f" Did you mean: {close}?" if close else ""
            raise ValueError(f"Unknown stage id '{sid}'.{hint}")
    if groups:
        unknown = sorted(set(groups) - known_groups)
        if unknown:
            raise ValueError(f"Unknown group(s): {unknown}. Known groups: {sorted(known_groups)}")

    if only:
        only_set = set(only)
        selected = [s for s in stages if s.id in only_set]
    elif groups:
        group_set = set(groups)
        selected = [s for s in stages if s.group in group_set]
    else:
        selected = [s for s in stages if s.default_selected]

    if from_id:
        idx = next(i for i, s in enumerate(stages) if s.id == from_id)
        from_ids = {s.id for s in stages[idx:]}
        selected = [s for s in selected if s.id in from_ids]

    if skip:
        skip_set = set(skip)
        selected = [s for s in selected if s.id not in skip_set]

    order = {s.id: i for i, s in enumerate(stages)}
    return sorted(selected, key=lambda s: order[s.id])


def resolve_cancelled(stages, seed_ids):
    """Every stage whose transitive `requires` closure touches seed_ids, excluding
    the seeds themselves. seed_ids is either "stages currently BLOCKED" (at listing
    time) or "stages BLOCKED or FAILED so far this run" (at execution time)."""
    seed_ids = set(seed_ids)
    cancelled = set(seed_ids)
    changed = True
    while changed:
        changed = False
        for s in stages:
            if s.id in cancelled:
                continue
            if any(dep in cancelled for dep in s.requires):
                cancelled.add(s.id)
                changed = True
    return cancelled - seed_ids


def auto_max_parallel(available_bytes):
    n = available_bytes // BYTES_PER_REPLAY_PROCESS
    return max(1, min(4, int(n)))


def resolve_max_parallel(arg_value):
    if arg_value != "auto":
        return int(arg_value), f"--max-parallel {arg_value} (explicit)"
    try:
        import psutil
        available = psutil.virtual_memory().available
        n = auto_max_parallel(available)
        gib = available / 1024 ** 3
        return n, f"auto: {n} ({gib:.1f} GiB available / {BYTES_PER_REPLAY_PROCESS / 1024**3:.1f} GiB per replay process)"
    except ImportError:
        return DEFAULT_MAX_PARALLEL_FALLBACK, f"auto: psutil unavailable, falling back to {DEFAULT_MAX_PARALLEL_FALLBACK}"


# ---------------------------------------------------------------------------
# Runners
# ---------------------------------------------------------------------------

def run_nbconvert_stage(stage, root, log_dir):
    log_dir.mkdir(parents=True, exist_ok=True)
    output_path = log_dir / "executed" / f"{stage.id}.ipynb"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{stage.id}.log"

    import os
    import subprocess
    env = {**os.environ, **SINGLE_THREAD_BLAS_ENV}
    with open(log_path, "w", encoding="utf-8") as log_file:
        result = subprocess.run(
            nbconvert_command(root / stage.notebook, output_path),
            cwd=root, env=env, stdout=log_file, stderr=subprocess.STDOUT,
        )
    return result.returncode == 0, log_path


def run_sweep_stage(stage, root, log_dir, max_parallel, remaining_ratios):
    stage_log_dir = log_dir / stage.id
    ok = run_sweep_for_notebook(root / stage.notebook, list(remaining_ratios), stage_log_dir, max_parallel)
    return ok, stage_log_dir


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _tail(path, n=40):
    try:
        lines = Path(path).read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return []
    return lines[-n:]


def print_status_table(stages, statuses, selected_ids, root):
    print(f"Pipeline stage status   PROJECT_ROOT={root}")
    print()
    header = f"  {'#':>2}  {'STAGE':<42} {'GROUP':<14} {'STATUS':<8} DETAIL"
    print(header)
    for i, s in enumerate(stages, 1):
        st = statuses[s.id]
        marker = "*" if s.id in selected_ids else " "
        print(f"{marker} {i:>2}  {s.id:<42} {s.group:<14} {st.state:<8} {st.detail}")
        if st.note:
            print(f"       {'':<42} {'':<14} {'':<8} (!) {st.note}")
    counts = {}
    for st in statuses.values():
        counts[st.state] = counts.get(st.state, 0) + 1
    summary = ", ".join(f"{n} {state}" for state, n in sorted(counts.items()))
    print(f"\nSummary: {len(stages)} stages -- {summary}")
    print("(* = selected for this run)")

    blocked_ids = {s.id for s in stages if statuses[s.id].state == "BLOCKED"}
    if blocked_ids:
        print("\nBLOCKED -- manual action required before these can run:")
        for s in stages:
            if s.id not in blocked_ids:
                continue
            st = statuses[s.id]
            print(f"\n  {s.id}")
            print(f"    {st.detail}")
            if st.remedy:
                for line in st.remedy.splitlines():
                    print(f"    {line}")
        cancelled = resolve_cancelled(stages, blocked_ids)
        cancelled_selected = cancelled & set(selected_ids)
        if cancelled_selected:
            print(f"\n  Stages cancelled by the above (depend on a BLOCKED stage): {sorted(cancelled_selected)}")


def print_run_summary(results, unfinished, elapsed):
    """unfinished = selected stages neither run nor already-DONE this session --
    cancelled (upstream BLOCKED/failed), not-run (time budget), or stopped-early."""
    print(f"\n=== Run summary   elapsed {elapsed/3600:.1f}h ===")
    n_ok = n_failed = 0
    for stage_id, ok, log_path, duration in results:
        status = "OK" if ok else "FAILED"
        n_ok += ok
        n_failed += not ok
        print(f"  {status:<8} {stage_id:<42} {duration/3600:5.1f}h   {log_path}")
        if not ok:
            print("  --- last 40 lines of the failing log ---")
            for line in _tail(log_path):
                print(f"  {line}")
            print("  ----------------------------------------")
    for stage_id in sorted(unfinished):
        print(f"  NOT RUN  {stage_id:<40} (see per-stage message above for why)")
    print(f"\n{len(results) + len(unfinished)} selected: {n_ok} ok, {n_failed} failed, {len(unfinished)} not run.")
    print("Re-run the same command next session to resume -- DONE work is skipped automatically.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_arg_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--list", action="store_true", help="Print status table and exit. No side effects.")
    parser.add_argument("--dry-run", action="store_true", help="Print status + execution plan, don't run anything.")
    parser.add_argument("--only", nargs="+", metavar="ID")
    parser.add_argument("--group", nargs="+", metavar="GROUP",
                         help="dataprep | sgd | mlp_baseline | er_parallel | er_sequential")
    parser.add_argument("--from", dest="from_id", metavar="ID")
    parser.add_argument("--skip", nargs="+", metavar="ID")
    parser.add_argument("--max-parallel", default="auto", help="Integer, or 'auto' (default; sizes from available RAM)")
    parser.add_argument("--time-budget", type=float, default=None, metavar="HOURS")
    parser.add_argument("--log-dir", default=None)
    parser.add_argument("--force", action="store_true", help="Re-run DONE stages (refused for dataprep_merge_monthly)")
    parser.add_argument("--on-failure", choices=("continue", "stop"), default="continue")
    parser.add_argument("--no-auto-downgrade", action="store_true",
                         help="Don't drop --max-parallel to 1 after a sweep stage MemoryErrors")
    return parser


def main(argv=None):
    args = build_arg_parser().parse_args(argv)
    root = PROJECT_ROOT

    try:
        selected = select_stages(STAGES, only=args.only, groups=args.group, from_id=args.from_id, skip=args.skip)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    statuses = {s.id: compute_status(s, root) for s in STAGES}
    selected_ids = [s.id for s in selected]
    print_status_table(STAGES, statuses, selected_ids, root)

    if args.list:
        return 0

    max_parallel, reasoning = resolve_max_parallel(args.max_parallel)
    print(f"\nmax-parallel: {reasoning}")

    if args.dry_run:
        print("\nExecution plan:")
        for s in selected:
            st = statuses[s.id]
            action = "skip (DONE)" if st.state == "DONE" and not args.force else f"run ({s.runner})"
            print(f"  {s.id:<42} {action}")
        return 0

    log_dir = Path(args.log_dir) if args.log_dir else root / "pipeline_logs" / time.strftime("%Y%m%dT%H%M%S")
    log_dir.mkdir(parents=True, exist_ok=True)

    cancelled = set()   # stages whose upstream is BLOCKED or has FAILED this run
    not_run = set()     # selected stages never attempted (time budget, or run stopped early)
    results = []        # (stage_id, ok, log_ref, duration) for stages actually executed
    start = time.monotonic()
    downgraded = False
    stopped_early = False

    for i, stage in enumerate(selected, 1):
        st = statuses[stage.id]

        if stage.id in cancelled:
            continue
        if any(dep in cancelled or statuses[dep].state == "BLOCKED" for dep in stage.requires):
            cancelled |= {stage.id} | resolve_cancelled(STAGES, {stage.id})
            print(f"\n[{i}/{len(selected)}] {stage.id}: CANCELLED (depends on a BLOCKED or failed stage)")
            continue
        if st.state == "BLOCKED":
            print(f"\n[{i}/{len(selected)}] {stage.id}: BLOCKED -- see status table above, skipping")
            cancelled |= resolve_cancelled(STAGES, {stage.id})
            continue
        if st.state == "DONE" and not args.force:
            print(f"\n[{i}/{len(selected)}] {stage.id}: already DONE, skipping")
            continue
        if stage.id == "dataprep_merge_monthly" and st.state == "DONE" and args.force:
            print(f"\n[{i}/{len(selected)}] {stage.id}: --force refused (FileExistsError makes forcing pointless; "
                  f"delete the store yourself first)")
            not_run.add(stage.id)
            continue

        if stopped_early or (args.time_budget is not None and (time.monotonic() - start) / 3600 >= args.time_budget):
            reason = "run stopped early" if stopped_early else f"time budget {args.time_budget}h exhausted"
            print(f"\n[{i}/{len(selected)}] {stage.id}: NOT RUN ({reason})")
            not_run.add(stage.id)
            continue

        print(f"\n[{i}/{len(selected)}] {stage.id} ({stage.group})"
              + (f"  ratios {[format_ratio_key(r) for r in st.remaining]}" if st.remaining else ""))
        stage_start = time.monotonic()

        if stage.runner == "ratio_sweep":
            effective_parallel = 1 if downgraded else max_parallel
            ratios_to_run = st.remaining if st.remaining else stage.ratios
            ok, stage_log_dir = run_sweep_stage(stage, root, log_dir, effective_parallel, ratios_to_run)
            log_ref = stage_log_dir
            if not ok and not args.no_auto_downgrade and not downgraded:
                logs = list(stage_log_dir.glob("*.log")) if stage_log_dir.exists() else []
                if any("MemoryError" in "\n".join(_tail(p, 200)) for p in logs):
                    downgraded = True
                    print("  (!) MemoryError detected -- downgrading max-parallel to 1 for remaining sweep stages")
        else:
            ok, log_ref = run_nbconvert_stage(stage, root, log_dir)

        duration = time.monotonic() - stage_start
        results.append((stage.id, ok, log_ref, duration))
        print(f"[{i}/{len(selected)}] {stage.id}  {'OK' if ok else 'FAILED'}  {duration/3600:.1f}h")

        if not ok:
            cancelled |= resolve_cancelled(STAGES, {stage.id})
            if args.on_failure == "stop":
                print("--on-failure stop: halting run.")
                stopped_early = True

    elapsed = time.monotonic() - start
    unfinished = (cancelled | not_run) & set(selected_ids)
    print_run_summary(results, unfinished, elapsed)

    if any(not ok for _, ok, _, _ in results):
        return 1
    if unfinished:
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
