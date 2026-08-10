"""Stage-chaining orchestrator for the training pipeline (see the "Stage-chaining
orchestrator for the training pipeline on Kaggle" plan for the full design writeup).

Problem this solves: Kaggle sessions cap out around 9-12h, but the full pipeline
(data prep + 5 SGD notebooks + the MLP baseline + the experience-replay notebooks
+ evaluation) takes far longer. Re-running this script in every new session runs
only whatever isn't finished yet -- completion is derived entirely from the
artifacts each notebook already writes (final model files, completion_status.json
files, zarr variable presence), the same way src/eval/tables.py's
load_or_run_table and src/mlp_replay/checkpointing.py's completion_status.json
already work. There is no separate orchestrator state file to fall out of sync
with reality.

Order: data prep -> SGD -> MLP baseline -> experience replay -> evaluation.

Scope: data prep (3 notebooks), 5 SGD notebooks, the MLP baseline, the 13
experience-replay notebooks under notebooks/training/mlp/experience_replay/,
and the unified evaluation (notebooks/evaluation/Evaluations.ipynb).

Not selected by default (still runnable via --only or --group): the 6 *_reservoir_sampling
experience-replay stages, excluded by user decision -- src/eval/families.py
defines 50 evaluation families and none are reservoir, so skipping them costs
the evaluation nothing; dataprep_lagged_monthly, whose output nothing reads; and
the 5 "er_combined_<id>" stages (group="combined_strategies") -- these regenerate
historical combined-replay-strategy hyperparameter combos explored before this
pipeline-rerun effort (see COMBINED_STRATEGY_CONFIGS), not part of the primary
training set "er_combined" already covers. Run them with --group combined_strategies.

Explicitly NOT covered at all: XGBoost (removed from the active pipeline in
commit 48ec425; only exists at old_notebooks/XGBoost.ipynb, which is gitignored
-- the README's references to notebooks/training/xgboost/XGBoost.ipynb are
stale), the 2 SGD experience-replay notebooks (blocking input() resume prompt
that fires exactly in the chaining case -- run those by hand), the visualization
notebooks, and drift detection.

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

IMPORTANT -- /kaggle/working is wiped when a session ends; nothing survives
unless it's pushed somewhere durable before that happens. Pass --git-push so
the orchestrator commits and pushes experiments/ to the git remote after EVERY
stage (not just at the end) -- that way a session dying mid-run only loses
whatever was still in-flight, not everything trained so far. This needs the
git remote's URL to already carry push credentials (e.g. a token embedded in
the remote URL, set up once per session since kernel restarts don't persist
`git remote set-url`), and a `user.email`/`user.name` configured -- both
belong in your session-setup cell alongside cloning, not in this script.

kaggle/fonda_pipeline_runner.ipynb is the ready-made version of all of this --
upload it once and Run All each session. The equivalent by hand:

Session 1:
    !git clone --depth 1 --branch pipeline-rerun <repo-url> /kaggle/working/repo
    %cd /kaggle/working/repo
    !pip install -q "zarr>=3" "scikit-learn>=1.7"   # Kaggle's images lag both
    # symlink the attached dataset in as the two files training_data_with_features_plus_monthly_indices.zarr
    # and data_split.npz (read-only source -- symlink, don't copy)
    import os
    os.environ['MLP_FEATURE_CACHE_DIR'] = '/kaggle/working/feature_cache_disk'
    !python -u scripts/run_pipeline_stages.py --list
    !python -u scripts/run_pipeline_stages.py --time-budget 8.0 --git-push

Session N: identical, same command. --git-push means the fresh clone already
has the previous session's trained artifacts (they're on the git remote), so
nothing needs copying forward: DONE stages skip, PARTIAL ones resume mid-sweep,
and an interrupted evaluation picks up from its cached tables. Only
feature_cache_disk/ (never committed -- a disk cache, not an artifact) is lost
between sessions, and it simply rebuilds itself.

=== Multi-seed runs ===

--seed N reads data_split_seed_N.npz, trains with seed N, and writes everything
under experiments/seed_N/. The original experiments/ tree (seed 42) is never read
or written by a seeded run, so seeds cannot overwrite each other or the baseline.

Generate the split first -- it is cheap (one variable read, no feature
computation), so it runs in the Kaggle session rather than being uploaded:

    !python -u scripts/make_seed_split.py --seed 1
    !python -u scripts/run_pipeline_stages.py --seed 1 --time-budget 8.0 --git-push

Without the split file, dataprep_base reports PARTIAL and every training stage
BLOCKS -- a deliberate stop, so a missing split can't silently train seed N
against seed 42's partition.

One seed per session is the intended cadence: the disk feature cache is keyed by
a hash of the pixel-index array, so each seed's split rebuilds it from scratch,
and two seeds in one session would hold two full caches against the ~20GB quota.

After the seeds finish, aggregate to mean +/- std:

    python scripts/aggregate_seed_results.py

Note that varying the split makes this Monte-Carlo cross-validation -- each seed
scores a different test set, so the spread covers data-partition variance as well
as training stochasticity. Say which protocol you ran when reporting the numbers.

Usage:
    python -u scripts/run_pipeline_stages.py --list
    python -u scripts/run_pipeline_stages.py --dry-run
    python -u scripts/run_pipeline_stages.py --group sgd mlp_baseline
    python -u scripts/run_pipeline_stages.py --only er_hard_example_mining --max-parallel 4
    python -u scripts/run_pipeline_stages.py --only eval_unified   # evaluation on its own
    python -u scripts/run_pipeline_stages.py --seed 1 --list       # seed 1's tree

    # Invalidate a stage's checkpoints/models/history after a code change that
    # changes what they mean (e.g. replay sampling moved from per-epoch to
    # per-year) -- otherwise the stage still looks DONE and never re-runs:
    python -u scripts/run_pipeline_stages.py --reset-stage er_hard_example_mining   # preview only
    python -u scripts/run_pipeline_stages.py --reset-stage er_hard_example_mining --yes
    # Only wired up for the 5 non-reservoir experience-replay stages (er_plain,
    # er_hard_example_mining, er_uncertainty_prioritization, er_confidently_correct,
    # er_misclassification_buffer) -- the ones whose `artifacts` are declared. The
    # *_reservoir_sampling stages still resample every epoch (out of scope for that
    # change) and are not comparable to these five.
"""

import argparse
import difflib
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field, replace
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
    format_combined_run_key,
    format_combined_strategy_suffix,
    format_float_token,
    format_ratio_key,
    load_completion_status,
    per_ratio_path,
)
from src.seed_run import SEED_ENV_VAR  # noqa: E402

MONTHLY_INDEX_VARS = (
    "ndvi_cv_year", "ndvi_max_m2m_drop_year", "ndvi_max_year", "ndvi_min_year", "ndvi_std_year",
    "ndwi_cv_year", "ndwi_max_m2m_drop_year", "ndwi_max_year", "ndwi_min_year", "ndwi_std_year",
)

# Peak RSS of one replay-sweep worker, measured against the real dataset
# (5,597,776 train / 1,273,437 val pixels x 32 float32 features = 716MB per
# training year, 163MB per validation year):
#
#   4.3GB  train_feature_cache -- precompute_yearly_raw_cache holds all 6 years
#   1.0GB  val_feature_cache   -- same, for the validation split
#   0.2GB  y_replay_pool       -- labels for the whole history (features are NOT
#                                 materialized; see replay_strategies.py)
#   0.7GB  one year's scaled features, transient, while the samplers stream it
#   0.4GB  the selected replay sample itself (<=2.8M rows at RR_0.5)
#   2.1GB  X_combined_base + the per-epoch shuffled copy of it
#   ~1GB   headroom: _top_up_shortfall's np.arange/setdiff1d over the global
#          pool index space, model/scaler state, BLAS scratch
#
# The old value (5.5GB) counted roughly the two feature caches and nothing else,
# so `auto` picked 4 on Kaggle's 30GiB CPU box and the workers were OOM-killed.
BYTES_PER_REPLAY_PROCESS = int(12 * 1024 ** 3)
DEFAULT_MAX_PARALLEL_FALLBACK = 2

# Evaluation. src/eval/tables.py's load_or_run_table trusts ANY existing table
# CSV and never recomputes it -- that's what makes an interrupted evaluation
# resumable across sessions, but it also means retrained models are silently
# ignored if stale tables are lying around. So the orchestrator records which
# training state an evaluation was computed from, and clears the cached tables
# when that state has changed. See eval_unified_check / run_eval_stage.
EVAL_DIR_REL = "experiments/evaluation/eval_outputs/unified_eval"
EVAL_SUMMARY_REL = f"{EVAL_DIR_REL}/summary_matrices/metric_matrix_overview.csv"
EVAL_STATE_REL = f"{EVAL_DIR_REL}/orchestrator_eval_state.json"

# --- multi-seed runs -------------------------------------------------------
#
# --seed N redirects everything this run OWNS into experiments/seed_N/ and points
# it at that seed's split file, leaving the original tree untouched. The stage
# table itself is not rewritten; the literals stay as they are and get remapped at
# the moment they become absolute paths, which keeps --seed from being able to
# corrupt the table for an un-seeded run in the same process.
#
# The remap is deliberately three-way rather than "prefix everything", because not
# every path literal here is an output. dataprep_base checks
# training_data_with_features.zarr, dataprep_lagged_monthly owns
# feature_cache_lagged_monthly.zarr, and dataprep_merge_monthly checks a zarr store
# -- all at the project root, all SHARED across seeds. Redirecting those would send
# a seed run looking for datasets that were never copied.
_ACTIVE_SEED = None


def set_active_seed(seed):
    """Activate a seed for this process. Also exports FONDA_SEED, which is how the
    notebooks pick up the same seed, split file and output tree -- every runner here
    builds its subprocess environment from os.environ, so setting it once covers
    nbconvert stages, ratio-sweep workers and the evaluation stage alike."""
    global _ACTIVE_SEED
    _ACTIVE_SEED = None if seed is None else int(seed)
    if _ACTIVE_SEED is not None:
        os.environ[SEED_ENV_VAR] = str(_ACTIVE_SEED)


def seeded_rel(rel):
    """Remap one PROJECT_ROOT-relative path for the active seed.

    experiments/...  -> experiments/seed_<N>/...   (owned outputs)
    data_split.npz   -> data_split_seed_<N>.npz    (per-seed input)
    anything else    -> unchanged                  (shared inputs, notebooks, logs)
    """
    if _ACTIVE_SEED is None:
        return rel
    rel = str(rel)
    if rel == "data_split.npz":
        return f"data_split_seed_{_ACTIVE_SEED}.npz"
    if rel == "experiments" or rel.startswith("experiments/"):
        return f"experiments/seed_{_ACTIVE_SEED}{rel[len('experiments'):]}"
    return rel


def seeded_path(root, rel):
    return Path(root) / seeded_rel(rel)

# Training stages whose models the evaluation reads. Deliberately excludes the
# reservoir-sampling stages: src/eval/families.py defines 50 families and none
# of them are reservoir, so those stages cannot affect evaluation output.
EVAL_PREREQ_STAGE_IDS = (
    "sgd_baseline", "sgd_prevyears", "sgd_prevyears_incr",
    "sgd_prevyears_monthly", "sgd_prevyears_monthly_incr",
    "mlp_baseline",
    "er_plain", "er_target_positive_rate", "er_confidently_correct",
    "er_hard_example_mining", "er_uncertainty_prioritization",
    "er_misclassification_buffer", "er_combined",
)


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
    group: str             # dataprep | sgd | mlp_baseline | er_parallel | er_sequential | combined_strategies
    notebook: str          # PROJECT_ROOT-relative
    check: "object"        # Callable[[Path, "Stage"], StageStatus]
    runner: str            # "nbconvert" | "ratio_sweep"
    ratios: tuple = ()
    requires: tuple = ()
    inputs: tuple = ()     # PROJECT_ROOT-relative paths that must exist before this can run at all
    default_selected: bool = True
    notes: str = ""
    env_override: dict = field(default_factory=dict)  # extra env vars for this stage's subprocess,
                            # beyond SINGLE_THREAD_BLAS_ENV -- e.g. the combined-strategy stages use
                            # this to pass MLP_COMBINED_CONFIG_OVERRIDE instead of the notebook's
                            # hardcoded constants. Empty for every "nbconvert"-runner stage that
                            # doesn't need it; "ratio_sweep" stages set their own override internally
                            # (see run_sweep_for_notebook) and ignore this field.
    artifacts: tuple = ()  # PROJECT_ROOT-relative paths this stage OWNS (checkpoints/models/history
                            # CSVs). "{ratio_key}" is expanded over stage.ratios. Only used by
                            # --reset-stage; empty for stages that don't support it yet. Always exact
                            # paths -- never a glob -- so a reset can never catch a sibling directory
                            # from a different hyperparameter config (e.g. the *_opt_fix, *_PR_0.10
                            # dirs living alongside the ones this stage owns).


# ---------------------------------------------------------------------------
# Completion-check helpers -- pure, unit-testable, no subprocess/import cost
# ---------------------------------------------------------------------------

def artifacts_exist(root, rel_paths):
    rel_paths = tuple(rel_paths)
    present = [p for p in rel_paths if seeded_path(root, p).exists()]
    missing = [p for p in rel_paths if p not in present]
    if not missing:
        return StageStatus("DONE", detail=f"{len(present)}/{len(rel_paths)} artifact(s) present")
    # Report where it actually looked, not the un-remapped literal -- under --seed
    # those differ, and printing the literal would send you hunting in the wrong tree.
    shown = ", ".join(seeded_rel(p) for p in missing)
    if not present:
        return StageStatus("TODO", detail=f"missing: {shown}")
    return StageStatus("PARTIAL", detail=f"missing: {shown}", remaining=tuple(missing))


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
    store = seeded_path(root, rel_store)
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


def _corrupt_status(path, exc):
    """A checkpoint file exists but can't be parsed -- most likely a crash
    mid-write from before checkpointing.py's saves were made atomic (see
    save_completion_status/save_all_training_histories). BLOCKED, not a crash:
    a single unreadable file must never take down status-checking for every
    other stage (previously it would have, since compute_status is called in
    one unguarded dict comprehension over all STAGES)."""
    return StageStatus(
        "BLOCKED",
        detail=f"checkpoint file exists but is corrupted/unreadable: {path} "
               f"({exc.__class__.__name__}: {exc})",
        remedy=(
            f"This file can't be parsed as JSON -- likely a crash mid-write. Check git history "
            f"for a clean version:\n"
            f'  git log --oneline -- "{path}"\n'
            f"Restore a good commit if one exists:\n"
            f'  git show <commit>:"{path}" > "{path}"\n'
            f"Or delete it and let this ratio retrain from scratch:\n"
            f'  rm "{path}"'
        ),
    )


def _load_completion_status_safe(path, **kwargs):
    """load_completion_status, but a decode failure becomes a (None, StageStatus)
    pair instead of propagating -- see _corrupt_status for why."""
    try:
        return load_completion_status(path, **kwargs), None
    except (json.JSONDecodeError, OSError, UnicodeDecodeError) as exc:
        return None, _corrupt_status(path, exc)


def ratios_complete_shared(root, rel_json, ratios):
    status, corrupt = _load_completion_status_safe(seeded_path(root, rel_json))
    if corrupt is not None:
        return corrupt
    completed = set(status.get("completed_ratios", []))
    keys = [format_ratio_key(r) for r in ratios]
    missing = [r for r, k in zip(ratios, keys) if k not in completed]

    if not missing:
        return StageStatus("DONE", detail=f"{len(ratios)}/{len(ratios)} ratios complete ({rel_json})")
    n_done = len(ratios) - len(missing)
    state = "TODO" if n_done == 0 else "PARTIAL"
    return StageStatus(state, detail=f"{n_done}/{len(ratios)} ratios complete", remaining=tuple(missing))


def ratios_complete_per_ratio(root, rel_json_base, ratios):
    base = seeded_path(root, rel_json_base)
    missing, done = [], []
    for r in ratios:
        key = format_ratio_key(r)
        status, corrupt = _load_completion_status_safe(per_ratio_path(base, key))
        if corrupt is not None:
            return corrupt
        (done if key in status.get("completed_ratios", []) else missing).append(r)

    note = ""
    if missing and base.exists():
        # Pre-refactor notebooks wrote one shared completion file across all ratios;
        # the refactored notebook only ever reads its own per-ratio files, so a
        # legacy shared file claiming completion is silently ignored by the
        # notebook itself -- surface that here so a TODO doesn't look like a bug.
        # Purely advisory: if THIS file is unreadable, just skip the note rather
        # than blocking the stage over a file nothing actually depends on.
        legacy_status, legacy_corrupt = _load_completion_status_safe(base)
        legacy_completed = set(legacy_status.get("completed_ratios", [])) if legacy_corrupt is None else set()
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


COMBINED_STRATEGY_CHECKPOINT_DIR = (
    "experiments/mlp/combined/training_checkpoints_mlp_experience_replay_combined"
)


def _other_combined_configs_note(root, rel_json, known_files):
    """Warn if completion files for OTHER strategy-hyperparameter combos exist in the
    (shared) checkpoint directory that AREN'T one of the combos this orchestrator tracks
    as its own stage -- this notebook's completion filename is config-coupled (encodes 6
    tunable hyperparameters), so it's easy to hand-tune a value outside the tracked set and
    have the result silently invisible to --list forever."""
    combined_dir = seeded_path(root, COMBINED_STRATEGY_CHECKPOINT_DIR)
    if not combined_dir.exists():
        return ""
    current_name = Path(rel_json).name
    known_names = {Path(f).name for f in known_files}
    strays = sorted(
        p.name for p in combined_dir.glob("mlp_replay_completion_status_combined_*.json")
        if p.name != current_name and p.name not in known_names
    )
    if not strays:
        return ""
    return (f"{len(strays)} completion file(s) for UNTRACKED strategy hyperparameters exist "
            f"in this dir (not one of the combos this orchestrator tracks): {strays}")


def combined_strategy_check(
    rel_json, ratio, hard_example, confidently_correct, uncertainty_prioritization,
    positive_rate, misclassification_buffer, replay_weight_scale, known_files,
):
    """Unlike every other replay stage, this notebook writes a COMPOSITE key into
    completed_ratios (ratio + the 6-hyperparameter suffix + '_RR=<ratio>'), because
    CHECKPOINT_DIR is shared across every hyperparameter combo -- only the filenames inside
    it are combo-scoped, so a bare ratio key would be ambiguous between combos sharing that
    ratio. ratios_complete_shared/shared_check assume the bare key every OTHER notebook
    writes; reusing them here was the pre-existing bug (a completed run could never be
    detected as DONE). This computes the exact key the notebook actually writes instead."""
    expected_key = format_combined_run_key(
        ratio, hard_example, confidently_correct, uncertainty_prioritization,
        positive_rate, misclassification_buffer, replay_weight_scale,
    )

    def _check(root, stage):
        status, corrupt = _load_completion_status_safe(seeded_path(root, rel_json))
        if corrupt is not None:
            return corrupt
        note = _other_combined_configs_note(root, rel_json, known_files)
        if expected_key in set(status.get("completed_ratios", [])):
            return StageStatus("DONE", detail=f"1/1 ratios complete ({rel_json})", note=note)
        return StageStatus("TODO", detail="0/1 ratios complete", note=note)

    return _check


def training_signature(root):
    """Which training results an evaluation would be computed from: every
    prerequisite stage's completion state, including exactly which ratios are
    still outstanding. Comparing this against the signature stored beside the
    eval outputs is what distinguishes 'resume an interrupted evaluation'
    (same signature -> keep the cached tables) from 'the models changed under
    us' (different signature -> the cached tables are stale)."""
    signature = {}
    for stage_id in EVAL_PREREQ_STAGE_IDS:
        stage = STAGES_BY_ID[stage_id]
        status = compute_status(stage, root)
        signature[stage_id] = {
            "state": status.state,
            "remaining": [float(r) for r in status.remaining],
        }
    return signature


def read_eval_state(root):
    path = seeded_path(root, EVAL_STATE_REL)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError, UnicodeDecodeError):
        return None  # unreadable state == no state; worst case we recompute


def write_eval_state(root, signature, status):
    path = seeded_path(root, EVAL_STATE_REL)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / (path.name + ".tmp")
    tmp.write_text(json.dumps({"status": status, "signature": signature}, indent=2, sort_keys=True),
                    encoding="utf-8")
    os.replace(tmp, path)


def eval_unified_check(root, stage):
    current = training_signature(root)
    state = read_eval_state(root)
    stored = state.get("signature") if state else None
    matches = stored == current

    if state and state.get("status") == "complete" and matches:
        return StageStatus("DONE", detail="evaluation complete for the current training state")

    if state and state.get("status") == "in_progress" and matches:
        return StageStatus(
            "PARTIAL",
            detail="evaluation was interrupted; will resume from its cached tables",
        )

    note = ""
    if seeded_path(root, EVAL_SUMMARY_REL).exists():
        why = ("was not produced by this orchestrator (no state file)" if state is None
               else "was computed from a different training state")
        note = (
            f"cached evaluation tables exist but the evaluation {why} -- they will be cleared "
            f"and recomputed. src/eval/tables.py's load_or_run_table trusts any existing CSV, so "
            f"reusing them would silently report results from the wrong models."
        )
    return StageStatus("TODO", detail="evaluation not yet run for the current training state", note=note)


# ---------------------------------------------------------------------------
# Stage table
# ---------------------------------------------------------------------------

ER = "notebooks/training/mlp/experience_replay"
ERC = "experiments/mlp/experience_replay"
SGD_RATIOS = ()  # SGD notebooks have no ratio sweep

# --- combined-strategy configs (er_combined + er_combined_<id>) -------------
#
# The combined-strategy notebook's 6 tunable constants (HARD_EXAMPLE, CONFIDENTLY_CORRECT,
# UNCERTAINITY_PRIORITIZATION, POSITIVE_RATE, MISCLASSIFICATION_BUFFER, REPLAY_WEIGHT_SCALE)
# are currently hardcoded to one config -- that's the "er_combined" stage below. Before this
# pipeline-rerun effort, 6 different combos of those constants were explored by hand, each
# leaving its own completion-status file in COMBINED_STRATEGY_CHECKPOINT_DIR. One of those 6
# (byte-for-byte: HE=0, CC=0.2, UP=0.1, PR=(0.2,10)) is identical to the current hardcoded
# default -- MC/RWS are absent from its filename only because those two knobs didn't exist yet
# when it ran, and their absence is exactly their current default value -- so it's already
# covered by "er_combined" below and isn't repeated here. The other 5 are regenerated as their
# own stages so the now-fixed training code (streaming replay pool, per-year cadence) produces
# correct results for every historical combo, not just the current default.
#
# Each ratio is the ONLY one that satisfies the notebook's own validation guard
# (HARD_EXAMPLE + CONFIDENTLY_CORRECT + UNCERTAINITY_PRIORITIZATION + PR_buffer_fraction +
# MISCLASSIFICATION_BUFFER <= ratio, else ValueError) for that combo -- confirmed by hand for
# each one below. Not expanded to a 4-ratio sweep: the smallest of these sums (0.25) already
# exceeds RR=0.2, so no combo could ever validly run at that ratio.
COMBINED_STRATEGY_CHECKPOINT_DIR = (
    "experiments/mlp/combined/training_checkpoints_mlp_experience_replay_combined"
)

DEFAULT_COMBINED_STRATEGY_PARAMS = dict(
    ratio=0.5, hard_example=0.0, confidently_correct=0.2, uncertainty_prioritization=0.1,
    positive_rate=(0.2, 10), misclassification_buffer=0.0, replay_weight_scale=1.0,
)

COMBINED_STRATEGY_CONFIGS = (
    dict(combo_id="1", hard_example=0.1, confidently_correct=0.1, uncertainty_prioritization=0.1,
         positive_rate=(0.2, 10), misclassification_buffer=0.0, replay_weight_scale=1.0, ratio=0.5),
    dict(combo_id="2", hard_example=0.0, confidently_correct=0.15, uncertainty_prioritization=0.0,
         positive_rate=(0.0, 10), misclassification_buffer=0.1, replay_weight_scale=1.0, ratio=0.4),
    dict(combo_id="3", hard_example=0.0, confidently_correct=0.1, uncertainty_prioritization=0.1,
         positive_rate=(0.0, 10), misclassification_buffer=0.1, replay_weight_scale=1.0, ratio=0.5),
    dict(combo_id="4", hard_example=0.0, confidently_correct=0.1, uncertainty_prioritization=0.2,
         positive_rate=(0.1, 10), misclassification_buffer=0.0, replay_weight_scale=1.0, ratio=0.5),
    dict(combo_id="6", hard_example=0.0, confidently_correct=0.2, uncertainty_prioritization=0.2,
         positive_rate=(0.1, 10), misclassification_buffer=0.0, replay_weight_scale=1.0, ratio=0.5),
)


def _combined_strategy_rel_json(params):
    suffix = format_combined_strategy_suffix(
        params["hard_example"], params["confidently_correct"], params["uncertainty_prioritization"],
        params["positive_rate"], params["misclassification_buffer"], params["replay_weight_scale"],
    )
    return f"{COMBINED_STRATEGY_CHECKPOINT_DIR}/mlp_replay_completion_status{suffix}.json", suffix


KNOWN_COMBINED_STRATEGY_FILES = tuple(
    _combined_strategy_rel_json(p)[0]
    for p in (DEFAULT_COMBINED_STRATEGY_PARAMS, *COMBINED_STRATEGY_CONFIGS)
)

# Every year this notebook ever trains over -- used only to enumerate misclass-snapshot
# paths exactly (never a glob), matching the "artifacts must be exact paths" rule.
COMBINED_STRATEGY_TRAINING_YEARS = (2017, 2018, 2019, 2020, 2021, 2022)


def _combined_strategy_artifacts(cfg):
    """Every file this notebook writes for one (combo, ratio) run -- what a clean
    re-run needs gone, not just the completion-status json. Missing any of these
    would leave a stale mlp_replay_all_training_histories{suffix}.pkl behind, which
    the notebook's own resume logic would then merge with fresh results under the
    SAME run_key, duplicating year entries in the output history."""
    rel_json, suffix = _combined_strategy_rel_json(cfg)
    ratio_suffix = f"{suffix}_RR={format_float_token(cfg['ratio'])}"
    models_dir = f"experiments/mlp/combined/models_mlp{ratio_suffix}"
    artifacts = [
        rel_json,
        f"{COMBINED_STRATEGY_CHECKPOINT_DIR}/mlp_replay_all_training_histories{suffix}.pkl",
        f"{COMBINED_STRATEGY_CHECKPOINT_DIR}/mlp_replay_training_log{suffix}.txt",
        models_dir,
        # write_combined_history_csv's output -- lives directly under EXPERIMENTS_DIR
        # (experiments/mlp/combined/), not inside CHECKPOINT_DIR or models_dir.
        f"experiments/mlp/combined/mlp_classifier_history_prevyears_monthly_features{suffix}_all_ratios.csv",
    ]
    if cfg["misclassification_buffer"] > 0:
        # Only written when MISCLASSIFICATION_BUFFER > 0 -- see the notebook's
        # `if REPLAY_ENABLED and year_idx >= 1 and MISCLASSIFICATION_BUFFER > 0:` guard.
        run_key = format_combined_run_key(
            cfg["ratio"], cfg["hard_example"], cfg["confidently_correct"],
            cfg["uncertainty_prioritization"], cfg["positive_rate"],
            cfg["misclassification_buffer"], cfg["replay_weight_scale"],
        )
        artifacts += [
            f"{COMBINED_STRATEGY_CHECKPOINT_DIR}/mlp_replay_misclass_snapshot_{run_key}_year_{year}.pkl"
            for year in COMBINED_STRATEGY_TRAINING_YEARS
        ]
    return tuple(artifacts)


def _make_combined_strategy_stage(cfg):
    rel_json, _suffix = _combined_strategy_rel_json(cfg)
    env = {"MLP_COMBINED_CONFIG_OVERRIDE": json.dumps({
        "REPLAY_RATIO": cfg["ratio"],
        "HARD_EXAMPLE": cfg["hard_example"],
        "CONFIDENTLY_CORRECT": cfg["confidently_correct"],
        "UNCERTAINITY_PRIORITIZATION": cfg["uncertainty_prioritization"],
        "POSITIVE_RATE": list(cfg["positive_rate"]),
        "MISCLASSIFICATION_BUFFER": cfg["misclassification_buffer"],
        "REPLAY_WEIGHT_SCALE": cfg["replay_weight_scale"],
    })}
    return Stage(
        id=f"er_combined_{cfg['combo_id']}", group="combined_strategies",
        notebook=f"{ER}/MLP_experience_replay_combined.ipynb",
        requires=("dataprep_merge_monthly",), ratios=(cfg["ratio"],),
        check=combined_strategy_check(
            rel_json, cfg["ratio"], cfg["hard_example"], cfg["confidently_correct"],
            cfg["uncertainty_prioritization"], cfg["positive_rate"],
            cfg["misclassification_buffer"], cfg["replay_weight_scale"], KNOWN_COMBINED_STRATEGY_FILES,
        ),
        runner="nbconvert", env_override=env, default_selected=False,
        artifacts=_combined_strategy_artifacts(cfg),
        notes=(f"Regenerates historical combined-strategy config #{cfg['combo_id']} "
               f"(HE={cfg['hard_example']}, CC={cfg['confidently_correct']}, "
               f"UP={cfg['uncertainty_prioritization']}, PR={cfg['positive_rate']}, "
               f"MC={cfg['misclassification_buffer']}, RWS={cfg['replay_weight_scale']}) "
               f"at its historical ratio RR={cfg['ratio']}."),
    )


COMBINED_STRATEGY_STAGES = tuple(_make_combined_strategy_stage(cfg) for cfg in COMBINED_STRATEGY_CONFIGS)


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
    #     Each declares the zarr it actually opens, so if one is ever re-run
    #     (--force) somewhere that dataset isn't present it reports
    #     BLOCKED-missing-input up front instead of dying mid-notebook. This is
    #     live on Kaggle: only the _plus_monthly_indices store gets uploaded, so
    #     the first three below could not actually retrain there. It never comes
    #     up while they're DONE, since compute_status checks completion first.
    Stage(
        id="sgd_baseline", group="sgd",
        notebook="notebooks/training/sgd/SGD Classifier.ipynb",
        requires=("dataprep_base",),
        inputs=("training_data_with_features.zarr", "data_split.npz"),
        check=artifacts_check("experiments/sgd/models/model_year_2022.pkl"),
        runner="nbconvert",
        notes="No resume logic -- a rerun retrains all 6 years (2017-2022) from scratch.",
    ),
    Stage(
        id="sgd_prevyears", group="sgd",
        notebook="notebooks/training/sgd/SGD Classifier_prevyears.ipynb",
        requires=("dataprep_base",),
        inputs=("training_data_with_features.zarr", "data_split.npz"),
        check=artifacts_check("experiments/sgd/models_lagged_features/model_year_2022_lagged_features.pkl"),
        runner="nbconvert",
        notes="No resume logic.",
    ),
    Stage(
        id="sgd_prevyears_incr", group="sgd",
        notebook="notebooks/training/sgd/SGD Classifier_prevyears-incremental scaler.ipynb",
        requires=("dataprep_base",),
        inputs=("training_data_with_features.zarr", "data_split.npz"),
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
        inputs=("training_data_with_features_plus_monthly_indices.zarr", "data_split.npz"),
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
        inputs=("training_data_with_features_plus_monthly_indices.zarr", "data_split.npz"),
        check=artifacts_check(
            "experiments/sgd/models_prevyears_monthly_features_incremental_scaler/"
            "scaler_final_prevyears_monthly_features_incremental_scaler.pkl"
        ),
        runner="nbconvert",
        notes="No resume logic.",
    ),
    Stage(
        id="sgd_experience_replay", group="sgd",
        notebook="notebooks/training/sgd/Experience Replay-SGD Classifier_prevyears and monthly "
                  "features-incremental scaler.ipynb",
        requires=("dataprep_merge_monthly",), ratios=(0.2, 0.3, 0.4, 0.5),
        check=shared_check("experiments/sgd/training_checkpoints/training_completion_status.json"),
        runner="nbconvert",
        artifacts=(
            "experiments/sgd/training_checkpoints",
            "experiments/sgd/models_prevyears_monthly_features_incremental_scaler_experience_replay_{ratio_key}",
            "experiments/sgd/sgd_classifier_model_prevyears_monthly_features_incremental_scaler_experience_replay_{ratio_key}.pkl",
        ),
        notes="Has its own internal replay-ratio sweep with a shared completion file across ratios "
              "(same pattern as mlp_baseline) -- run as a single sequential process, never parallelized. "
              "Its 'Save Model and Results' section only runs once per notebook execution (for whichever "
              "ratio is last in REPLAY_RATIOS), so a from-scratch run only produces one ratio's final-model "
              "file directly; the other 3 were backfilled by copying each ratio's already-identical "
              "model_year_2022_....pkl checkpoint (verified byte-identical to the equivalent er_plain "
              "final-model file) since this notebook was already fully trained by hand before being wired "
              "in here.",
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
        artifacts=(
            f"{ERC}/training_checkpoints_mlp_experience_replay",
            f"{ERC}/models_mlp_prevyears_monthly_features_incremental_scaler_experience_replay_{{ratio_key}}",
            f"{ERC}/mlp_classifier_history_prevyears_monthly_features_incremental_scaler_experience_replay_all_ratios.csv",
        ),
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
        artifacts=(
            f"{ERC}/training_checkpoints_mlp_experience_replay_confidently_correct_memory",
            f"{ERC}/models_mlp_prevyears_monthly_features_incremental_scaler_experience_replay_confidently_correct_memory_{{ratio_key}}",
            f"{ERC}/mlp_classifier_history_prevyears_monthly_features_incremental_scaler_experience_replay_confidently_correct_memory_all_ratios.csv",
        ),
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
        artifacts=(
            f"{ERC}/training_checkpoints_mlp_experience_replay_hard_example_mining",
            f"{ERC}/models_mlp_prevyears_monthly_features_incremental_scaler_experience_replay_hard_example_mining_{{ratio_key}}",
            f"{ERC}/mlp_classifier_history_prevyears_monthly_features_incremental_scaler_experience_replay_hard_example_mining_all_ratios.csv",
        ),
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
        artifacts=(
            f"{ERC}/training_checkpoints_mlp_experience_replay_uncertainity_prioritization",
            f"{ERC}/models_mlp_prevyears_monthly_features_incremental_scaler_experience_replay_uncertainity_prioritization_{{ratio_key}}",
            f"{ERC}/mlp_classifier_history_prevyears_monthly_features_incremental_scaler_experience_replay_uncertainity_prioritization_all_ratios.csv",
        ),
    ),

    # --- experience replay: NOT refactored, shared completion file across
    #     ratios -- must run as a single sequential process each, never through
    #     the parallel launcher (would clobber each other's progress) ---------
    #
    #     The 6 *_reservoir_sampling stages below are default_selected=False:
    #     excluded from normal runs by user decision, and NOT evaluated either
    #     (src/eval/families.py defines 50 families, none of them reservoir), so
    #     skipping them costs the evaluation stage nothing. Run one explicitly
    #     with --only <id> if you ever want it back.
    Stage(
        id="er_plain_reservoir", group="er_sequential",
        notebook=f"{ER}/MLP-experience replay_reservoir_sampling.ipynb",
        requires=("dataprep_merge_monthly",), ratios=(0.2, 0.3, 0.4, 0.5),
        check=shared_check(f"{ERC}/training_checkpoints_mlp_experience_replay_reservoir_sampling/mlp_replay_completion_status.json"),
        runner="nbconvert",
        default_selected=False,
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
        default_selected=False,
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
        default_selected=False,
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
        default_selected=False,
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
        default_selected=False,
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
        artifacts=(
            f"{ERC}/training_checkpoints_mlp_missclassification_buffer",
            f"{ERC}/models_mlp_prevyears_monthly_features_incremental_scaler_missclassification_buffer_{{ratio_key}}",
            f"{ERC}/mlp_classifier_history_prevyears_monthly_features_incremental_scaler_missclassification_buffer_all_ratios.csv",
        ),
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
        default_selected=False,
        notes="Single fixed ratio (0.3).",
    ),
    Stage(
        id="er_combined", group="er_sequential",
        notebook=f"{ER}/MLP_experience_replay_combined.ipynb",
        requires=("dataprep_merge_monthly",), ratios=(0.5,),
        check=combined_strategy_check(
            _combined_strategy_rel_json(DEFAULT_COMBINED_STRATEGY_PARAMS)[0],
            DEFAULT_COMBINED_STRATEGY_PARAMS["ratio"],
            DEFAULT_COMBINED_STRATEGY_PARAMS["hard_example"],
            DEFAULT_COMBINED_STRATEGY_PARAMS["confidently_correct"],
            DEFAULT_COMBINED_STRATEGY_PARAMS["uncertainty_prioritization"],
            DEFAULT_COMBINED_STRATEGY_PARAMS["positive_rate"],
            DEFAULT_COMBINED_STRATEGY_PARAMS["misclassification_buffer"],
            DEFAULT_COMBINED_STRATEGY_PARAMS["replay_weight_scale"],
            KNOWN_COMBINED_STRATEGY_FILES,
        ),
        runner="nbconvert",
        artifacts=_combined_strategy_artifacts(DEFAULT_COMBINED_STRATEGY_PARAMS),
        notes="Sweeps 1 ratio while combining multiple replay strategies at fixed fractions; "
              "its completion filename encodes those fractions (HE/CC/UP/PR/MC/RWS), hardcoded "
              "here for the notebook's current default values. Writes to experiments/mlp/combined, "
              "not experiments/mlp/experience_replay.",
    ),
    *COMBINED_STRATEGY_STAGES,

    # --- evaluation: runs last, over everything trained above ---------------
    Stage(
        id="eval_unified", group="evaluation",
        notebook="notebooks/evaluation/Evaluations.ipynb",
        requires=EVAL_PREREQ_STAGE_IDS,
        inputs=("training_data_with_features_plus_monthly_indices.zarr", "data_split.npz"),
        check=eval_unified_check,
        runner="eval",
        notes="Evaluates all 50 families in src/eval/families.py. Declares every training stage "
              "it reads as `requires`, so a training failure cancels it rather than letting it "
              "evaluate half-trained models. Internally resumable per table (load_or_run_table), "
              "and run_eval_stage clears those cached tables when the training state they were "
              "computed from has changed.",
    ),
)

STAGES_BY_ID = {s.id: s for s in STAGES}


# ---------------------------------------------------------------------------
# Status computation, selection, dependency resolution
# ---------------------------------------------------------------------------

def compute_status(stage, root):
    # Check completion FIRST. A stage whose output already exists (e.g. because
    # a pre-built artifact was uploaded/symlinked directly, as dataprep_merge_monthly's
    # output is on Kaggle) is DONE regardless of whether its raw *inputs* are present --
    # those inputs are only needed to RUN the stage, and a DONE stage will never run.
    status = stage.check(root, stage)
    if status.state == "DONE":
        return status

    missing_inputs = [p for p in stage.inputs if not seeded_path(root, p).exists()]
    if missing_inputs:
        # seeded_rel, not the raw literal: under --seed N the split this stage needs
        # is data_split_seed_N.npz, and naming data_split.npz would point at a file
        # that exists and is the wrong one.
        joined = ", ".join(seeded_rel(p) for p in missing_inputs)
        return StageStatus(
            "BLOCKED",
            detail=f"missing input(s): {joined}",
            remedy=f"Ensure these exist under {root} before running this stage: {joined}",
        )
    return status


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


def resolve_stage_artifacts(root, stage):
    """Expand stage.artifacts (which may contain a literal "{ratio_key}" placeholder,
    one entry repeated across stage.ratios) into concrete, EXISTING paths under root.

    Pure and read-only -- only checks .exists(), never deletes. Paths that don't
    exist are silently dropped rather than erroring, since a partially-trained
    stage may be missing some of its ratio dirs. Exact paths only, by construction
    of how `artifacts` is declared on each Stage -- no globbing here, so a sibling
    directory from a different hyperparameter config (e.g. *_opt_fix, *_PR_0.10)
    can never be swept up by accident."""
    resolved = []
    for rel in stage.artifacts:
        if "{ratio_key}" in rel:
            for ratio in stage.ratios:
                candidate = seeded_path(root, rel.format(ratio_key=format_ratio_key(ratio)))
                if candidate.exists():
                    resolved.append(candidate)
        else:
            candidate = seeded_path(root, rel)
            if candidate.exists():
                resolved.append(candidate)
    return resolved


def reset_stages(root, stages, stage_ids, confirm):
    """Delete every existing artifact owned by the given stage ids. Returns
    {stage_id: [Path, ...]} of what was (or, if not confirm, would be) removed.

    Raises ValueError up front -- before deleting anything -- for an unknown
    stage id or one that declares no `artifacts` (most stages don't support
    --reset-stage yet; silently no-op'ing on those would look like success)."""
    by_id = {s.id: s for s in stages}
    unknown = [sid for sid in stage_ids if sid not in by_id]
    if unknown:
        close = difflib.get_close_matches(unknown[0], by_id.keys(), n=3) if unknown else []
        hint = f" Did you mean: {close}?" if close else ""
        raise ValueError(f"Unknown stage id(s) for --reset-stage: {unknown}.{hint}")

    no_artifacts = [sid for sid in stage_ids if not by_id[sid].artifacts]
    if no_artifacts:
        raise ValueError(
            f"Stage(s) {no_artifacts} declare no `artifacts` -- --reset-stage isn't wired up for "
            f"them yet. Delete their checkpoint/model directories by hand if you're sure."
        )

    plan = {sid: resolve_stage_artifacts(root, by_id[sid]) for sid in stage_ids}
    if confirm:
        import shutil
        for paths in plan.values():
            for path in paths:
                if path.is_dir():
                    shutil.rmtree(path)
                else:
                    path.unlink()
    return plan


def resolve_cancelled(stages, seed_ids, done_ids=frozenset()):
    """Every stage whose transitive `requires` closure touches seed_ids, excluding
    the seeds themselves. seed_ids is either "stages currently BLOCKED" (at listing
    time) or "stages BLOCKED or FAILED so far this run" (at execution time).

    done_ids stops propagation dead: a stage that's already DONE satisfies its
    dependents regardless of what's further upstream of it (e.g. a pre-built
    artifact was uploaded directly, so the stage that would normally regenerate
    it is DONE even though ITS OWN raw inputs, further upstream, are BLOCKED --
    that must not cancel everything downstream of the already-DONE stage)."""
    seed_ids = set(seed_ids) - set(done_ids)
    cancelled = set(seed_ids)
    changed = True
    while changed:
        changed = False
        for s in stages:
            if s.id in cancelled or s.id in done_ids:
                continue
            if any(dep in cancelled for dep in s.requires):
                cancelled.add(s.id)
                changed = True
    return cancelled - seed_ids


def auto_max_parallel(available_bytes):
    n = available_bytes // BYTES_PER_REPLAY_PROCESS
    return max(1, min(4, int(n)))


def resolve_max_parallel(arg_value):
    """Note the optimism baked into "auto": psutil reports memory available RIGHT
    NOW, at orchestrator startup, before any dataset is opened or any worker is
    launched -- so it is an upper bound on what the workers will actually find,
    never a promise. Pass --max-parallel explicitly if a run is being OOM-killed
    despite the sizing below."""
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

    env = {**os.environ, **SINGLE_THREAD_BLAS_ENV, **stage.env_override}
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


def run_eval_stage(stage, root, log_dir):
    """Wraps run_nbconvert_stage with the cached-table lifecycle the evaluation
    notebook can't manage itself: load_or_run_table never recomputes an existing
    CSV, so tables left over from a DIFFERENT training state must be cleared
    first or the run silently reports results for the wrong models. Tables from
    the SAME training state are kept -- that's what makes an evaluation
    interrupted by a session boundary resumable."""
    current = training_signature(root)
    state = read_eval_state(root)
    resuming = bool(state and state.get("signature") == current)

    eval_dir = seeded_path(root, EVAL_DIR_REL)
    if resuming:
        print("  resuming evaluation -- keeping cached tables from the same training state")
    elif eval_dir.exists():
        stale = sorted(eval_dir.rglob("*.csv"))
        for path in stale:
            path.unlink()
        print(f"  cleared {len(stale)} stale evaluation table(s) computed from a different "
              f"training state (recoverable from git history if needed)")

    write_eval_state(root, current, "in_progress")
    ok, log_path = run_nbconvert_stage(stage, root, log_dir)
    if ok:
        write_eval_state(root, current, "complete")
    return ok, log_path


def git_commit_and_push(root, message):
    """Best-effort persistence after a stage finishes (success OR failure --
    a failed stage can still have made real progress on disk, e.g. 2 of 4
    ratios in a sweep completing before the 3rd OOMs). This exists specifically
    for Kaggle, where /kaggle/working is wiped when the session ends and
    nothing survives unless it's pushed somewhere durable. Never raises --
    losing the ability to persist is far less bad than losing the actual
    training progress by aborting the run over a git/network hiccup."""

    def run(*args):
        return subprocess.run(["git", *args], cwd=root, capture_output=True, text=True)

    try:
        if not (root / "experiments").exists():
            print("  (git-push) experiments/ doesn't exist yet, nothing to commit")
            return

        add = run("add", "experiments/")
        if add.returncode != 0:
            print(f"  (git-push) 'git add' failed, skipping this checkpoint: {add.stderr.strip()}")
            return

        # Nothing staged -> `diff --cached --quiet` exits 0. Skip the commit
        # entirely rather than creating empty "nothing changed" commits.
        if run("diff", "--cached", "--quiet").returncode == 0:
            print("  (git-push) nothing new under experiments/ to commit")
            return

        commit = run("commit", "-m", message)
        if commit.returncode != 0:
            print(f"  (git-push) 'git commit' failed, skipping push: {commit.stderr.strip()}")
            return

        push = run("push")
        if push.returncode != 0:
            print(f"  (git-push) 'git push' failed -- committed locally but NOT pushed: {push.stderr.strip()}")
            print("  (git-push) this commit will still be picked up if the session survives long enough "
                  "to retry, but won't survive a session ending before a successful push.")
            return

        print(f"  (git-push) committed and pushed: {message}")
    except Exception as exc:  # noqa: BLE001 -- deliberately broad, see docstring
        print(f"  (git-push) unexpected error, continuing anyway: {exc}")


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
        done_ids = {s.id for s in stages if statuses[s.id].state == "DONE"}
        cancelled = resolve_cancelled(stages, blocked_ids, done_ids=done_ids)
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
                         help="dataprep | sgd | mlp_baseline | er_parallel | er_sequential | "
                              "combined_strategies | evaluation")
    parser.add_argument("--from", dest="from_id", metavar="ID")
    parser.add_argument("--skip", nargs="+", metavar="ID")
    parser.add_argument("--max-parallel", default="auto", help="Integer, or 'auto' (default; sizes from available RAM)")
    parser.add_argument("--time-budget", type=float, default=None, metavar="HOURS")
    parser.add_argument("--log-dir", default=None)
    parser.add_argument("--seed", type=int, default=None, metavar="N",
                        help="Run at seed N: read data_split_seed_N.npz, train with seed N, and "
                             "write everything under experiments/seed_N/. Omit for the original "
                             "experiments/ tree (seed 42), which a seeded run never touches. "
                             "Generate the split first: python scripts/make_seed_split.py --seed N")
    parser.add_argument("--force", action="store_true", help="Re-run DONE stages (refused for dataprep_merge_monthly)")
    parser.add_argument("--on-failure", choices=("continue", "stop"), default="continue")
    parser.add_argument("--no-auto-downgrade", action="store_true",
                         help="Don't drop --max-parallel to 1 after a sweep stage MemoryErrors")
    parser.add_argument("--git-push", action="store_true",
                         help="Commit and push experiments/ after each stage (success or failure). "
                              "Off by default -- meant for Kaggle, where /kaggle/working is wiped when "
                              "the session ends and this is the only way progress survives. Requires "
                              "push credentials already configured on the git remote (a token, etc.) --"
                              " a push failure is logged but never aborts the run.")
    parser.add_argument("--reset-stage", nargs="+", metavar="ID",
                         help="Delete a stage's checkpoints/models/history CSV so it retrains from "
                              "scratch (only wired up for stages that declare `artifacts` -- "
                              "currently the 5 non-reservoir experience-replay stages). With no "
                              "--yes, only PRINTS what would be deleted and exits -- nothing else "
                              "runs. Use after a code change that invalidates existing checkpoints "
                              "(e.g. changed sampling logic), since a stage whose artifacts still "
                              "look complete is reported DONE and never re-runs.")
    parser.add_argument("--yes", action="store_true",
                         help="Actually perform the deletion requested by --reset-stage. Without it, "
                              "--reset-stage is a dry-run preview only.")
    return parser


def main(argv=None):
    args = build_arg_parser().parse_args(argv)
    root = PROJECT_ROOT
    # Before anything reads a path or launches a subprocess: every rel->abs
    # resolution below goes through seeded_path, and every runner inherits
    # FONDA_SEED from os.environ.
    set_active_seed(args.seed)
    if args.seed is not None:
        print(f"seed {args.seed}: outputs under {seeded_rel('experiments')}/, "
              f"split {seeded_rel('data_split.npz')}\n")

    if args.reset_stage:
        try:
            plan = reset_stages(root, STAGES, args.reset_stage, confirm=args.yes)
        except ValueError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        verb = "Deleted" if args.yes else "Would delete (pass --yes to actually delete)"
        total = 0
        for stage_id, paths in plan.items():
            print(f"\n{stage_id}:")
            if not paths:
                print("  (nothing exists on disk for this stage -- already reset)")
                continue
            for path in paths:
                print(f"  {verb}: {path}")
                total += 1
        print(f"\n{total} path(s) {'deleted' if args.yes else 'would be deleted'}.")
        if not args.yes and total:
            print("Re-run with --yes to actually delete.")
        return 0

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
        done_ids = {s.id for s in STAGES if statuses[s.id].state == "DONE"}
        blocked_ids = {s.id for s in STAGES if statuses[s.id].state == "BLOCKED"}
        cancelled = resolve_cancelled(STAGES, blocked_ids, done_ids=done_ids)
        for s in selected:
            st = statuses[s.id]
            if st.state == "BLOCKED":
                action = "skip (BLOCKED -- see above)"
            elif s.id in cancelled:
                action = "skip (CANCELLED -- depends on a BLOCKED stage)"
            elif st.state == "DONE" and not args.force:
                action = "skip (DONE)"
            else:
                action = f"run ({s.runner})"
            print(f"  {s.id:<42} {action}")
        return 0

    log_dir = Path(args.log_dir) if args.log_dir else root / "pipeline_logs" / time.strftime("%Y%m%dT%H%M%S")
    log_dir.mkdir(parents=True, exist_ok=True)

    # A DONE stage's own dependencies no longer matter (it already succeeded whenever
    # they were last satisfied) -- used to stop cancellation propagating past it.
    done_ids = {s.id for s in STAGES if statuses[s.id].state == "DONE"}

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
        if st.state != "DONE" and any(
            dep not in done_ids and (dep in cancelled or statuses[dep].state == "BLOCKED")
            for dep in stage.requires
        ):
            cancelled |= {stage.id} | resolve_cancelled(STAGES, {stage.id}, done_ids=done_ids)
            print(f"\n[{i}/{len(selected)}] {stage.id}: CANCELLED (depends on a BLOCKED or failed stage)")
            continue
        if st.state == "BLOCKED":
            print(f"\n[{i}/{len(selected)}] {stage.id}: BLOCKED -- see status table above, skipping")
            cancelled |= resolve_cancelled(STAGES, {stage.id}, done_ids=done_ids)
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
        elif stage.runner == "eval":
            ok, log_ref = run_eval_stage(stage, root, log_dir)
        else:
            ok, log_ref = run_nbconvert_stage(stage, root, log_dir)

        duration = time.monotonic() - stage_start
        results.append((stage.id, ok, log_ref, duration))
        print(f"[{i}/{len(selected)}] {stage.id}  {'OK' if ok else 'FAILED'}  {duration/3600:.1f}h")

        if args.git_push:
            # Push regardless of ok/failed -- a failed sweep can still have real,
            # already-completed ratios on disk worth keeping (see docstring).
            git_commit_and_push(root, f"Pipeline: {stage.id} {'OK' if ok else 'FAILED'}")

        if not ok:
            cancelled |= resolve_cancelled(STAGES, {stage.id}, done_ids=done_ids)
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
