import json
import os
import subprocess

import pytest

from scripts.run_pipeline_stages import (
    COMBINED_STRATEGY_CONFIGS,
    COMBINED_STRATEGY_STAGES,
    DEFAULT_COMBINED_STRATEGY_PARAMS,
    EVAL_PREREQ_STAGE_IDS,
    EVAL_SUMMARY_REL,
    KNOWN_COMBINED_STRATEGY_FILES,
    PROJECT_ROOT,
    STAGES,
    STAGES_BY_ID,
    Stage,
    artifacts_exist,
    BYTES_PER_REPLAY_PROCESS,
    auto_max_parallel,
    combined_strategy_check,
    compute_status,
    eval_unified_check,
    git_commit_and_push,
    ratios_complete_per_ratio,
    ratios_complete_shared,
    reset_stages,
    resolve_cancelled,
    resolve_stage_artifacts,
    seeded_path,
    seeded_rel,
    select_stages,
    set_active_seed,
    shared_check,
    training_signature,
    write_eval_state,
    zarr_has_vars,
)
from src.mlp_replay.checkpointing import (
    format_combined_run_key,
    format_ratio_key,
    per_ratio_path,
    save_completion_status,
)
from src.seed_run import SEED_ENV_VAR


# ---------------------------------------------------------------------------
# artifacts_exist
# ---------------------------------------------------------------------------

def test_artifacts_exist_all_present(tmp_path):
    (tmp_path / "a.pkl").write_bytes(b"x")
    (tmp_path / "b.pkl").write_bytes(b"x")
    status = artifacts_exist(tmp_path, ["a.pkl", "b.pkl"])
    assert status.state == "DONE"


def test_artifacts_exist_none_present(tmp_path):
    status = artifacts_exist(tmp_path, ["a.pkl", "b.pkl"])
    assert status.state == "TODO"
    assert "a.pkl" in status.detail and "b.pkl" in status.detail


def test_artifacts_exist_some_present(tmp_path):
    (tmp_path / "a.pkl").write_bytes(b"x")
    status = artifacts_exist(tmp_path, ["a.pkl", "b.pkl"])
    assert status.state == "PARTIAL"
    assert status.remaining == ("b.pkl",)


# ---------------------------------------------------------------------------
# zarr_has_vars
# ---------------------------------------------------------------------------

def _write_fake_zarr_v3_store(path, var_names):
    path.mkdir(parents=True, exist_ok=True)
    metadata = {name: {"shape": [10], "data_type": "float32"} for name in var_names}
    payload = {
        "attributes": {},
        "zarr_format": 3,
        "node_type": "group",
        "consolidated_metadata": {"kind": "inline", "must_understand": False, "metadata": metadata},
    }
    (path / "zarr.json").write_text(json.dumps(payload), encoding="utf-8")


def test_zarr_has_vars_store_absent(tmp_path):
    status = zarr_has_vars(tmp_path, "store.zarr", ("a", "b", "c"))
    assert status.state == "TODO"


def test_zarr_has_vars_all_present(tmp_path):
    _write_fake_zarr_v3_store(tmp_path / "store.zarr", ["a", "b", "c", "extra"])
    status = zarr_has_vars(tmp_path, "store.zarr", ("a", "b", "c"))
    assert status.state == "DONE"


def test_zarr_has_vars_partial_is_blocked_not_todo(tmp_path):
    # 7 of 10 -- mirrors the real dataprep_merge_monthly hazard this check exists for.
    required = tuple(f"v{i}" for i in range(10))
    _write_fake_zarr_v3_store(tmp_path / "store.zarr", required[:7])
    status = zarr_has_vars(tmp_path, "store.zarr", required)
    assert status.state == "BLOCKED"
    for missing in required[7:]:
        assert missing in status.detail
    assert str(tmp_path / "store.zarr") in status.remedy
    assert "rm -rf" in status.remedy


def test_zarr_has_vars_zero_present_is_also_blocked(tmp_path):
    _write_fake_zarr_v3_store(tmp_path / "store.zarr", ["unrelated_var"])
    status = zarr_has_vars(tmp_path, "store.zarr", ("a", "b"))
    assert status.state == "BLOCKED"


# ---------------------------------------------------------------------------
# ratios_complete_shared
# ---------------------------------------------------------------------------

def test_ratios_complete_shared_all_done(tmp_path):
    rel = "status.json"
    save_completion_status(tmp_path / rel, {"completed_ratios": ["RR_0.2", "RR_0.3"], "completed_years": {}})
    status = ratios_complete_shared(tmp_path, rel, (0.2, 0.3))
    assert status.state == "DONE"


def test_ratios_complete_shared_partial(tmp_path):
    rel = "status.json"
    save_completion_status(tmp_path / rel, {"completed_ratios": ["RR_0.2"], "completed_years": {}})
    status = ratios_complete_shared(tmp_path, rel, (0.2, 0.3, 0.4))
    assert status.state == "PARTIAL"
    assert status.remaining == (0.3, 0.4)


def test_ratios_complete_shared_none_done(tmp_path):
    status = ratios_complete_shared(tmp_path, "missing_status.json", (0.2, 0.3))
    assert status.state == "TODO"
    assert status.remaining == (0.2, 0.3)


# ---------------------------------------------------------------------------
# ratios_complete_per_ratio
# ---------------------------------------------------------------------------

def test_ratios_complete_per_ratio_partial(tmp_path):
    base = tmp_path / "mlp_replay_completion_status.json"
    save_completion_status(per_ratio_path(base, format_ratio_key(0.2)),
                            {"completed_ratios": ["RR_0.2"], "completed_years": {}})
    status = ratios_complete_per_ratio(tmp_path, "mlp_replay_completion_status.json", (0.2, 0.3, 0.4, 0.5))
    assert status.state == "PARTIAL"
    assert status.remaining == (0.3, 0.4, 0.5)
    assert status.note == ""


def test_ratios_complete_per_ratio_ignores_legacy_shared_file(tmp_path):
    # Legacy pre-refactor file at the BASE path claims everything is done, but no
    # per-ratio files exist -- the refactored notebook never reads the base path,
    # so this must report TODO (not DONE), with a note explaining why.
    base = tmp_path / "mlp_replay_completion_status.json"
    save_completion_status(base, {"completed_ratios": ["RR_0.2", "RR_0.3"], "completed_years": {}})
    status = ratios_complete_per_ratio(tmp_path, "mlp_replay_completion_status.json", (0.2, 0.3))
    assert status.state == "TODO"
    assert status.remaining == (0.2, 0.3)
    assert "legacy" in status.note
    assert "RR_0.2" in status.note


# ---------------------------------------------------------------------------
# Corrupted checkpoint files -- a single unreadable file must report a clear,
# actionable status rather than crashing the whole status check (previously
# it would have: main() computes every stage's status in one unguarded dict
# comprehension, so an uncaught exception from any one stage killed --list
# entirely, telling you nothing about the other 21 stages).
# ---------------------------------------------------------------------------

def test_ratios_complete_shared_corrupted_file_is_blocked_not_raised(tmp_path):
    rel = "status.json"
    (tmp_path / rel).write_text("{not valid json", encoding="utf-8")

    status = ratios_complete_shared(tmp_path, rel, (0.2, 0.3))

    assert status.state == "BLOCKED"
    assert "corrupted" in status.detail
    assert str(tmp_path / rel) in status.detail
    assert "git log" in status.remedy


def test_ratios_complete_per_ratio_corrupted_file_is_blocked_not_raised(tmp_path):
    base = tmp_path / "mlp_replay_completion_status.json"
    good_path = per_ratio_path(base, format_ratio_key(0.2))
    bad_path = per_ratio_path(base, format_ratio_key(0.3))
    save_completion_status(good_path, {"completed_ratios": ["RR_0.2"], "completed_years": {}})
    bad_path.write_text("{not valid json", encoding="utf-8")

    status = ratios_complete_per_ratio(tmp_path, "mlp_replay_completion_status.json", (0.2, 0.3))

    assert status.state == "BLOCKED"
    assert str(bad_path) in status.detail


def test_ratios_complete_per_ratio_corrupted_legacy_file_is_advisory_only(tmp_path):
    # The legacy-file check is purely informational (a "note"); a corrupted
    # legacy file must not block a stage whose real (per-ratio) files are fine.
    base = tmp_path / "mlp_replay_completion_status.json"
    base.write_text("{not valid json", encoding="utf-8")
    save_completion_status(per_ratio_path(base, format_ratio_key(0.2)),
                            {"completed_ratios": ["RR_0.2"], "completed_years": {}})

    status = ratios_complete_per_ratio(tmp_path, "mlp_replay_completion_status.json", (0.2,))

    assert status.state == "DONE"
    assert status.note == ""


def test_one_corrupted_stage_does_not_block_computing_others(tmp_path):
    # Mirrors main()'s `{s.id: compute_status(s, root) for s in STAGES}` --
    # a corrupted file for one stage must not raise and take the whole
    # comprehension down with it.
    healthy_rel = "healthy_status.json"
    save_completion_status(tmp_path / healthy_rel, {"completed_ratios": ["RR_0.5"], "completed_years": {}})
    corrupt_rel = "corrupt_status.json"
    (tmp_path / corrupt_rel).write_text("{not valid json", encoding="utf-8")

    stages = [
        Stage(id="healthy", group="g", notebook="a.ipynb", runner="nbconvert",
              ratios=(0.5,), check=shared_check(healthy_rel)),
        Stage(id="corrupt", group="g", notebook="b.ipynb", runner="nbconvert",
              ratios=(0.5,), check=shared_check(corrupt_rel)),
    ]

    statuses = {s.id: compute_status(s, tmp_path) for s in stages}  # must not raise

    assert statuses["healthy"].state == "DONE"
    assert statuses["corrupt"].state == "BLOCKED"


# ---------------------------------------------------------------------------
# combined_strategy_check / COMBINED_STRATEGY_STAGES
#
# er_combined's notebook writes a COMPOSITE key into completed_ratios (ratio +
# the 6-hyperparameter suffix + '_RR=<ratio>'), unlike every other replay stage's
# bare ratio key -- shared_check/ratios_complete_shared assume the bare key and
# so could never detect a genuinely completed run. combined_strategy_check exists
# to compute the exact key the notebook actually writes instead.
# ---------------------------------------------------------------------------

def test_combined_strategy_check_done_only_with_the_exact_composite_key(tmp_path):
    rel_json = "combo.json"
    check = combined_strategy_check(
        rel_json, 0.5, 0.0, 0.2, 0.1, (0.2, 10), 0.0, 1.0, known_files=(),
    )

    # Never run: file doesn't exist at all.
    assert check(tmp_path, None).state == "TODO"

    # The bug this exists to fix: a file containing only the BARE ratio key
    # (what every other notebook writes, and what the broken check tested for)
    # must NOT be mistaken for done.
    save_completion_status(tmp_path / rel_json, {"completed_ratios": [format_ratio_key(0.5)]})
    assert check(tmp_path, None).state == "TODO"

    # The actual composite key the notebook writes -- this is what DONE requires.
    expected_key = format_combined_run_key(0.5, 0.0, 0.2, 0.1, (0.2, 10), 0.0, 1.0)
    save_completion_status(tmp_path / rel_json, {"completed_ratios": [expected_key]})
    status = check(tmp_path, None)
    assert status.state == "DONE"
    assert rel_json in status.detail


def test_combined_strategy_check_corrupted_file_is_blocked_not_raised(tmp_path):
    rel_json = "combo.json"
    (tmp_path / rel_json).write_text("{not json", encoding="utf-8")
    check = combined_strategy_check(rel_json, 0.5, 0.0, 0.2, 0.1, (0.2, 10), 0.0, 1.0, known_files=())
    assert check(tmp_path, None).state == "BLOCKED"


def test_combined_strategy_check_notes_untracked_sibling_files(tmp_path):
    combined_dir = tmp_path / "experiments/mlp/combined/training_checkpoints_mlp_experience_replay_combined"
    combined_dir.mkdir(parents=True)
    tracked = combined_dir / "mlp_replay_completion_status_combined_HE=0_CC=0.2_UP=0.1_PR=(0.2,10)_MC=0_RWS=1.json"
    stray = combined_dir / "mlp_replay_completion_status_combined_HE=0.9_CC=0_UP=0_PR=(0,10)_MC=0_RWS=1.json"
    tracked.write_text("{}", encoding="utf-8")
    stray.write_text("{}", encoding="utf-8")

    rel_json = str(tracked.relative_to(tmp_path)).replace("\\", "/")
    check = combined_strategy_check(
        rel_json, 0.5, 0.0, 0.2, 0.1, (0.2, 10), 0.0, 1.0, known_files=(rel_json,),
    )
    status = check(tmp_path, None)
    assert stray.name in status.note
    assert tracked.name not in status.note


def test_combined_strategy_stages_are_unique_and_distinct_from_the_default():
    ids = [s.id for s in COMBINED_STRATEGY_STAGES]
    assert len(ids) == len(COMBINED_STRATEGY_CONFIGS)
    assert len(ids) == len(set(ids))
    assert "er_combined" not in ids  # the default stage stays hand-written, not generated

    for stage in COMBINED_STRATEGY_STAGES:
        assert stage.group == "combined_strategies"
        assert stage.default_selected is False
        assert stage.runner == "nbconvert"
        assert len(stage.ratios) == 1


def test_combined_strategy_config_sums_satisfy_the_notebook_validation_guard():
    """MLP_experience_replay_combined.ipynb raises ValueError if HARD_EXAMPLE +
    CONFIDENTLY_CORRECT + UNCERTAINITY_PRIORITIZATION + PR_buffer_fraction +
    MISCLASSIFICATION_BUFFER > ratio. A future edit to COMBINED_STRATEGY_CONFIGS
    that violates this would crash the notebook hours into a Kaggle run instead
    of failing here in milliseconds."""
    for cfg in (DEFAULT_COMBINED_STRATEGY_PARAMS, *COMBINED_STRATEGY_CONFIGS):
        pr_buffer_fraction = cfg["positive_rate"][0]
        total = (
            cfg["hard_example"] + cfg["confidently_correct"] + cfg["uncertainty_prioritization"]
            + pr_buffer_fraction + cfg["misclassification_buffer"]
        )
        assert total <= cfg["ratio"] + 1e-12, cfg


def test_combined_strategy_env_override_round_trips_through_json():
    for stage in COMBINED_STRATEGY_STAGES:
        cfg = json.loads(stage.env_override["MLP_COMBINED_CONFIG_OVERRIDE"])
        assert cfg["POSITIVE_RATE"] == list(next(
            c["positive_rate"] for c in COMBINED_STRATEGY_CONFIGS
            if f"er_combined_{c['combo_id']}" == stage.id
        ))
        assert isinstance(cfg["REPLAY_RATIO"], float)


def test_known_combined_strategy_files_has_one_entry_per_tracked_combo():
    # Default config + the 5 regenerated ones -- these are what
    # _other_combined_configs_note excludes from its "untracked stray" warning.
    assert len(KNOWN_COMBINED_STRATEGY_FILES) == 1 + len(COMBINED_STRATEGY_CONFIGS)
    assert len(KNOWN_COMBINED_STRATEGY_FILES) == len(set(KNOWN_COMBINED_STRATEGY_FILES))


# ---------------------------------------------------------------------------
# select_stages
# ---------------------------------------------------------------------------

def _toy_stages():
    return [
        Stage(id="a", group="g1", notebook="a.ipynb", check=lambda r, s: None, runner="nbconvert"),
        Stage(id="b", group="g1", notebook="b.ipynb", check=lambda r, s: None, runner="nbconvert",
              requires=("a",)),
        Stage(id="c", group="g2", notebook="c.ipynb", check=lambda r, s: None, runner="nbconvert",
              default_selected=False),
    ]


def test_select_stages_default_selection_respects_default_selected():
    selected = select_stages(_toy_stages())
    assert [s.id for s in selected] == ["a", "b"]


def test_select_stages_only():
    selected = select_stages(_toy_stages(), only=["c"])
    assert [s.id for s in selected] == ["c"]


def test_select_stages_group():
    selected = select_stages(_toy_stages(), groups=["g1"])
    assert [s.id for s in selected] == ["a", "b"]


def test_select_stages_from_id():
    selected = select_stages(_toy_stages(), only=["a", "b", "c"], from_id="b")
    assert [s.id for s in selected] == ["b", "c"]


def test_select_stages_skip():
    selected = select_stages(_toy_stages(), only=["a", "b"], skip=["a"])
    assert [s.id for s in selected] == ["b"]


def test_select_stages_unknown_id_raises_with_suggestion():
    with pytest.raises(ValueError, match="Unknown stage id"):
        select_stages(_toy_stages(), only=["abc"])


def test_select_stages_unknown_group_raises():
    with pytest.raises(ValueError, match="Unknown group"):
        select_stages(_toy_stages(), groups=["nope"])


# ---------------------------------------------------------------------------
# resolve_cancelled
# ---------------------------------------------------------------------------

def test_resolve_cancelled_propagates_through_requires_chain():
    stages = _toy_stages()  # a <- b <- (nothing); c is independent
    cancelled = resolve_cancelled(stages, {"a"})
    assert cancelled == {"b"}  # b depends on a; c does not depend on anything


def test_resolve_cancelled_independent_stage_untouched():
    stages = _toy_stages()
    cancelled = resolve_cancelled(stages, {"c"})
    assert cancelled == set()  # nothing depends on c


# ---------------------------------------------------------------------------
# resolve_stage_artifacts / reset_stages
# ---------------------------------------------------------------------------

def _stage_with_artifacts(ratios=(0.2, 0.3)):
    return Stage(
        id="toy_er", group="er_parallel", notebook="toy.ipynb", runner="ratio_sweep",
        check=lambda r, s: None, ratios=ratios,
        artifacts=(
            "checkpoints/toy_er",
            "models/toy_er_{ratio_key}",
            "history/toy_er_all_ratios.csv",
        ),
    )


def test_resolve_stage_artifacts_expands_ratio_key_and_skips_missing(tmp_path):
    stage = _stage_with_artifacts()
    (tmp_path / "checkpoints" / "toy_er").mkdir(parents=True)
    (tmp_path / "models" / "toy_er_RR_0.2").mkdir(parents=True)
    # RR_0.3 models dir and the history CSV are deliberately absent.

    resolved = resolve_stage_artifacts(tmp_path, stage)

    assert resolved == [
        tmp_path / "checkpoints" / "toy_er",
        tmp_path / "models" / "toy_er_RR_0.2",
    ]


def test_resolve_stage_artifacts_nothing_on_disk_returns_empty(tmp_path):
    assert resolve_stage_artifacts(tmp_path, _stage_with_artifacts()) == []


def test_resolve_stage_artifacts_never_touches_sibling_directory(tmp_path):
    # A sibling from a different hyperparameter config (mirrors *_opt_fix,
    # *_PR_0.10 living next to the dirs a real ER stage owns) must never be
    # swept up -- resolve_stage_artifacts only expands the exact declared names.
    stage = _stage_with_artifacts(ratios=(0.2,))
    (tmp_path / "models" / "toy_er_RR_0.2").mkdir(parents=True)
    sibling = tmp_path / "models" / "toy_er_RR_0.2_opt_fix"
    sibling.mkdir(parents=True)

    resolved = resolve_stage_artifacts(tmp_path, stage)

    assert sibling not in resolved
    assert tmp_path / "models" / "toy_er_RR_0.2" in resolved


def test_reset_stages_preview_does_not_delete(tmp_path):
    stage = _stage_with_artifacts(ratios=(0.2,))
    target = tmp_path / "checkpoints" / "toy_er"
    target.mkdir(parents=True)

    plan = reset_stages(tmp_path, [stage], ["toy_er"], confirm=False)

    assert plan == {"toy_er": [target]}
    assert target.exists()  # preview only -- nothing removed


def test_reset_stages_confirm_deletes_dirs_and_files(tmp_path):
    stage = _stage_with_artifacts(ratios=(0.2,))
    ckpt_dir = tmp_path / "checkpoints" / "toy_er"
    ckpt_dir.mkdir(parents=True)
    (ckpt_dir / "status.json").write_text("{}", encoding="utf-8")
    models_dir = tmp_path / "models" / "toy_er_RR_0.2"
    models_dir.mkdir(parents=True)
    history_csv = tmp_path / "history" / "toy_er_all_ratios.csv"
    history_csv.parent.mkdir(parents=True)
    history_csv.write_text("year\n2020\n", encoding="utf-8")
    sibling = tmp_path / "models" / "toy_er_RR_0.2_opt_fix"
    sibling.mkdir(parents=True)

    plan = reset_stages(tmp_path, [stage], ["toy_er"], confirm=True)

    assert not ckpt_dir.exists()
    assert not models_dir.exists()
    assert not history_csv.exists()
    assert sibling.exists()  # sibling from a different config must survive
    assert len(plan["toy_er"]) == 3


def test_reset_stages_unknown_stage_id_raises_before_deleting(tmp_path):
    stage = _stage_with_artifacts(ratios=(0.2,))
    target = tmp_path / "checkpoints" / "toy_er"
    target.mkdir(parents=True)

    with pytest.raises(ValueError, match="Unknown stage id"):
        reset_stages(tmp_path, [stage], ["toy_er", "bogus"], confirm=True)

    assert target.exists()  # the valid stage's artifact must survive the raise


def test_reset_stages_stage_without_artifacts_raises(tmp_path):
    no_artifacts_stage = Stage(id="bare", group="g", notebook="a.ipynb",
                                runner="nbconvert", check=lambda r, s: None)

    with pytest.raises(ValueError, match="declare no `artifacts`"):
        reset_stages(tmp_path, [no_artifacts_stage], ["bare"], confirm=True)


def test_reset_stage_artifacts_declared_on_real_stages_are_exact_and_exist_in_repo():
    # Every real stage that declares `artifacts` must resolve to paths under
    # PROJECT_ROOT's experiments/ tree (a typo'd literal here would silently never
    # match anything on a real run). Combined-strategy stages write under
    # experiments/mlp/combined/ (see er_combined's own notes) -- both the original
    # "er_combined" (still in group er_sequential) and its generated
    # er_combined_<id> siblings (group combined_strategies). sgd_experience_replay
    # writes under experiments/sgd/. Every other ER stage writes under
    # experiments/mlp/experience_replay/.
    for stage in STAGES:
        if not stage.artifacts:
            continue
        is_combined_strategy = stage.id == "er_combined" or stage.id.startswith("er_combined_")
        if is_combined_strategy:
            expected_prefix = "experiments/mlp/combined/"
        elif stage.id == "sgd_experience_replay":
            expected_prefix = "experiments/sgd/"
        else:
            expected_prefix = "experiments/mlp/experience_replay/"
        for rel in stage.artifacts:
            assert rel.startswith(expected_prefix), stage.id
            assert "*" not in rel, f"{stage.id} artifact must be an exact path, not a glob: {rel}"


# ---------------------------------------------------------------------------
# auto_max_parallel
# ---------------------------------------------------------------------------

def test_auto_max_parallel_clamps_to_range():
    gib = 1024 ** 3
    per_process = BYTES_PER_REPLAY_PROCESS
    assert auto_max_parallel(0) == 1
    assert auto_max_parallel(per_process - 1) == 1  # never below 1, even if nothing fits
    assert auto_max_parallel(2 * per_process) == 2
    assert auto_max_parallel(1000 * gib) == 4  # clamped at 4 even with huge RAM


def test_auto_max_parallel_picks_two_on_a_kaggle_cpu_box():
    """Pins the number that actually matters. A Kaggle CPU session reports ~30GiB
    and has no swap, so this is the difference between the sweep finishing and the
    kernel being OOM-killed -- exactly what happened at --max-parallel 4."""
    assert auto_max_parallel(30 * 1024 ** 3) == 2


# ---------------------------------------------------------------------------
# Stage table integrity -- one cheap test that catches the class of typo that
# would otherwise cost a 9-hour Kaggle session.
# ---------------------------------------------------------------------------

def test_stage_table_is_well_formed():
    ids = [s.id for s in STAGES]
    assert len(ids) == len(set(ids)), "duplicate stage id"

    by_id = {s.id: s for s in STAGES}
    order = {s.id: i for i, s in enumerate(STAGES)}
    for s in STAGES:
        for dep in s.requires:
            assert dep in by_id, f"{s.id} requires unknown stage {dep}"
            assert order[dep] < order[s.id], f"{s.id} requires {dep}, which is declared later (not topological)"
        assert s.runner in {"nbconvert", "ratio_sweep", "eval"}, f"{s.id} has unknown runner {s.runner}"
        if s.runner == "ratio_sweep":
            assert s.group == "er_parallel", f"{s.id} uses ratio_sweep but is in group {s.group}"
            assert s.ratios, f"{s.id} uses ratio_sweep but declares no ratios"
        assert s.group != "er_sequential" or s.runner != "ratio_sweep", (
            f"{s.id} is er_sequential but would run through the parallel launcher"
        )
        # A typo'd notebook path only surfaces hours into a Kaggle session otherwise.
        assert (PROJECT_ROOT / s.notebook).exists(), f"{s.id} points at a missing notebook: {s.notebook}"

    # Every stage the evaluation declares as a prerequisite must actually exist,
    # and the evaluation must not depend on itself.
    for dep in EVAL_PREREQ_STAGE_IDS:
        assert dep in by_id, f"EVAL_PREREQ_STAGE_IDS names unknown stage {dep}"
    assert "eval_unified" not in EVAL_PREREQ_STAGE_IDS
    assert set(STAGES_BY_ID["eval_unified"].requires) == set(EVAL_PREREQ_STAGE_IDS)

    completion_paths = []
    for s in STAGES:
        # Recover each stage's completion-file path by calling check() against an
        # empty temp-like root is unnecessary here -- the curried checks close over
        # the literal path already, which we can recover via the closure cell. This
        # also covers the combined-strategy stages (combined_strategy_check(...) is
        # a closure too, unlike the old bespoke er_combined_check it replaced).
        closure = getattr(s.check, "__closure__", None)
        if not closure:
            continue
        for cell in closure:
            if isinstance(cell.cell_contents, str) and cell.cell_contents.endswith(".json"):
                completion_paths.append(cell.cell_contents)
    assert len(completion_paths) == len(set(completion_paths)), "two stages share a completion file path"


# ---------------------------------------------------------------------------
# git_commit_and_push -- real local git repos (bare "origin" + working clone),
# no network involved, so push success/failure is genuinely exercised.
# ---------------------------------------------------------------------------

def _run_git(*args, cwd):
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True)


def _init_repo_with_remote(tmp_path):
    bare = tmp_path / "origin.git"
    work = tmp_path / "work"
    _run_git("init", "--bare", str(bare), cwd=tmp_path)
    _run_git("clone", str(bare), str(work), cwd=tmp_path)
    _run_git("config", "user.email", "test@example.com", cwd=work)
    _run_git("config", "user.name", "Test", cwd=work)
    (work / "README.md").write_text("init\n", encoding="utf-8")
    _run_git("add", "README.md", cwd=work)
    _run_git("commit", "-m", "initial commit", cwd=work)
    _run_git("push", cwd=work)
    return work


def test_git_commit_and_push_no_experiments_dir(tmp_path, capsys):
    # experiments/ doesn't exist at all yet -- must not error on `git add`
    # (pathspec-did-not-match), just report nothing to commit.
    work = _init_repo_with_remote(tmp_path)
    log_before = _run_git("log", "--oneline", cwd=work).stdout

    git_commit_and_push(work, "should not create a commit")

    assert _run_git("log", "--oneline", cwd=work).stdout == log_before
    assert "nothing to commit" in capsys.readouterr().out


def test_git_commit_and_push_no_changes_within_experiments(tmp_path, capsys):
    work = _init_repo_with_remote(tmp_path)
    (work / "experiments").mkdir()
    (work / "experiments" / "model.pkl").write_bytes(b"already committed")
    _run_git("add", "experiments/", cwd=work)
    _run_git("commit", "-m", "pre-existing", cwd=work)
    log_before = _run_git("log", "--oneline", cwd=work).stdout

    git_commit_and_push(work, "should not create a commit")

    assert _run_git("log", "--oneline", cwd=work).stdout == log_before
    assert "nothing new" in capsys.readouterr().out


def test_git_commit_and_push_commits_and_pushes(tmp_path):
    work = _init_repo_with_remote(tmp_path)
    (work / "experiments").mkdir()
    (work / "experiments" / "model.pkl").write_bytes(b"fake model bytes")

    git_commit_and_push(work, "Pipeline: fake_stage OK")

    log = _run_git("log", "--oneline", "-1", cwd=work).stdout
    assert "Pipeline: fake_stage OK" in log

    # Confirm it actually reached the remote, not just the local clone -- clone
    # the bare "origin" fresh and check the file is there.
    fresh = tmp_path / "fresh_clone"
    _run_git("clone", str(tmp_path / "origin.git"), str(fresh), cwd=tmp_path)
    assert (fresh / "experiments" / "model.pkl").read_bytes() == b"fake model bytes"


def test_git_commit_and_push_survives_broken_remote(tmp_path, capsys):
    # No remote configured at all -- `git push` fails, but the function must not raise.
    work = tmp_path / "standalone"
    _run_git("init", str(work), cwd=tmp_path)
    _run_git("config", "user.email", "test@example.com", cwd=work)
    _run_git("config", "user.name", "Test", cwd=work)
    (work / "README.md").write_text("init\n", encoding="utf-8")
    _run_git("add", "README.md", cwd=work)
    _run_git("commit", "-m", "initial commit", cwd=work)

    (work / "experiments").mkdir()
    (work / "experiments" / "model.pkl").write_bytes(b"x")

    git_commit_and_push(work, "Pipeline: fake_stage OK")  # must not raise

    # The commit itself should still have succeeded locally even though push failed.
    log = _run_git("log", "--oneline", "-1", cwd=work).stdout
    assert "Pipeline: fake_stage OK" in log
    assert "push" in capsys.readouterr().out.lower()


# ---------------------------------------------------------------------------
# Reservoir-sampling stages are excluded from default runs
# ---------------------------------------------------------------------------

RESERVOIR_STAGE_IDS = {s.id for s in STAGES if s.id.endswith("_reservoir")}


def test_reservoir_stages_exist_but_are_not_default_selected():
    assert len(RESERVOIR_STAGE_IDS) == 6, "expected exactly 6 reservoir-sampling stages"
    selected = {s.id for s in select_stages(STAGES)}
    assert not (selected & RESERVOIR_STAGE_IDS)


def test_reservoir_stage_still_reachable_explicitly():
    selected = select_stages(STAGES, only=["er_plain_reservoir"])
    assert [s.id for s in selected] == ["er_plain_reservoir"]


def test_no_reservoir_stage_is_an_eval_prerequisite():
    # src/eval/families.py defines no reservoir families, so excluding these
    # stages must not affect what the evaluation can compute.
    assert not (set(EVAL_PREREQ_STAGE_IDS) & RESERVOIR_STAGE_IDS)


# ---------------------------------------------------------------------------
# eval_unified_check -- distinguishes "resume an interrupted evaluation" from
# "the models changed, cached tables are stale".
# ---------------------------------------------------------------------------

def _eval_stage():
    return STAGES_BY_ID["eval_unified"]


def test_eval_check_todo_when_never_run(tmp_path):
    status = eval_unified_check(tmp_path, _eval_stage())
    assert status.state == "TODO"
    assert status.note == ""  # no cached tables -> nothing to warn about


def test_eval_check_done_when_state_complete_and_signature_matches(tmp_path):
    write_eval_state(tmp_path, training_signature(tmp_path), "complete")
    status = eval_unified_check(tmp_path, _eval_stage())
    assert status.state == "DONE"


def test_eval_check_partial_when_interrupted_mid_run(tmp_path):
    # Same training state, but the evaluation didn't finish -> resume it,
    # keeping the cached tables it already computed.
    write_eval_state(tmp_path, training_signature(tmp_path), "in_progress")
    status = eval_unified_check(tmp_path, _eval_stage())
    assert status.state == "PARTIAL"
    assert "resume" in status.detail


def test_eval_check_stale_when_signature_differs(tmp_path):
    summary = tmp_path / EVAL_SUMMARY_REL
    summary.parent.mkdir(parents=True, exist_ok=True)
    summary.write_text("table_type,metric,row_count\n", encoding="utf-8")

    stale_signature = {"er_plain": {"state": "DONE", "remaining": []}}
    write_eval_state(tmp_path, stale_signature, "complete")

    status = eval_unified_check(tmp_path, _eval_stage())
    assert status.state == "TODO"
    assert "different training state" in status.note


def test_eval_check_warns_when_tables_predate_the_orchestrator(tmp_path):
    # Cached tables with no state file at all -- exactly the situation in this
    # repo, where evaluation outputs predate the orchestrator entirely.
    summary = tmp_path / EVAL_SUMMARY_REL
    summary.parent.mkdir(parents=True, exist_ok=True)
    summary.write_text("table_type,metric,row_count\n", encoding="utf-8")

    status = eval_unified_check(tmp_path, _eval_stage())
    assert status.state == "TODO"
    assert "no state file" in status.note


def test_training_signature_changes_when_a_prerequisite_completes(tmp_path):
    before = training_signature(tmp_path)

    # Make one prerequisite stage look DONE by creating its completion artifact.
    completion_path = (
        tmp_path / "experiments/mlp/experience_replay"
                   "/training_checkpoints_mlp_missclassification_buffer"
                   "/mlp_missclassification_buffer_completion_status.json"
    )
    completion_path.parent.mkdir(parents=True, exist_ok=True)
    save_completion_status(completion_path, {"completed_ratios": ["RR_0.3"], "completed_years": {}})
    after = training_signature(tmp_path)

    assert before != after, "signature must change when training progress changes"
    assert after["er_misclassification_buffer"]["state"] == "DONE"


# ---------------------------------------------------------------------------
# --seed / seeded_rel
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _no_seed_leaks_between_tests():
    """set_active_seed mutates module state and os.environ; every test in this file
    must start un-seeded or the un-seeded assertions elsewhere become meaningless."""
    yield
    set_active_seed(None)
    os.environ.pop(SEED_ENV_VAR, None)


class TestSeededRel:
    def test_unseeded_is_the_identity(self):
        """The property that protects the existing experiments/ tree."""
        for rel in ("experiments/sgd/models", "data_split.npz",
                    "training_data_with_features.zarr", "notebooks/x.ipynb"):
            assert seeded_rel(rel) == rel

    def test_experiments_paths_move_into_the_seed_tree(self):
        set_active_seed(1)
        assert seeded_rel("experiments/sgd/models") == "experiments/seed_1/sgd/models"
        assert seeded_rel("experiments") == "experiments/seed_1"

    def test_the_split_file_becomes_the_seeds_split(self):
        set_active_seed(2)
        assert seeded_rel("data_split.npz") == "data_split_seed_2.npz"

    def test_shared_inputs_are_never_redirected(self):
        """Datasets are shared across seeds and live at the project root. Redirecting
        them would send a seed run looking for a zarr that was never copied."""
        set_active_seed(1)
        for rel in ("training_data_with_features.zarr",
                    "training_data_with_features_plus_monthly_indices.zarr",
                    "full_dataset_resizedv2.zarr",
                    "feature_cache_lagged_monthly.zarr",
                    "notebooks/training/sgd/SGD Classifier.ipynb"):
            assert seeded_rel(rel) == rel

    def test_no_partial_prefix_match(self):
        """'experiments_old/...' must not be caught by an 'experiments' startswith."""
        set_active_seed(1)
        assert seeded_rel("experiments_old/x") == "experiments_old/x"
        assert seeded_rel("data_split.npz.bak") == "data_split.npz.bak"

    def test_seeded_path_joins_against_root(self, tmp_path):
        set_active_seed(3)
        assert seeded_path(tmp_path, "experiments/sgd") == tmp_path / "experiments/seed_3/sgd"

    def test_set_active_seed_exports_the_env_var_for_subprocesses(self):
        """Notebooks read FONDA_SEED; every runner builds its env from os.environ."""
        set_active_seed(7)
        assert os.environ[SEED_ENV_VAR] == "7"

    def test_distinct_seeds_cannot_collide(self):
        trees = set()
        for s in (1, 2, 3, 42):
            set_active_seed(s)
            trees.add(seeded_rel("experiments"))
        assert len(trees) == 4


class TestSeedRedirectsRealStages:
    def test_every_declared_artifact_lands_in_the_seed_tree(self):
        set_active_seed(1)
        for stage in STAGES:
            for rel in stage.artifacts:
                assert seeded_rel(rel).startswith("experiments/seed_1/"), stage.id

    def test_stage_inputs_split_correctly_between_shared_and_seeded(self):
        set_active_seed(1)
        for stage in STAGES:
            for rel in stage.inputs:
                if rel == "data_split.npz":
                    assert seeded_rel(rel) == "data_split_seed_1.npz", stage.id
                else:
                    # Every other input is a zarr store shared across all seeds.
                    assert seeded_rel(rel) == rel, f"{stage.id}: {rel}"

    def test_eval_constants_follow_the_seed(self):
        set_active_seed(1)
        assert seeded_rel(EVAL_SUMMARY_REL).startswith("experiments/seed_1/evaluation/")

    def test_reset_stage_resolves_the_seed_tree_and_never_the_original(self, tmp_path):
        """The safety property for --reset-stage, which DELETES what it resolves.

        Both trees are populated with the same artifact, so a resolver that ignored
        the seed would happily return the original tree's copy. resolve_stage_artifacts
        only returns paths that exist, which is why both have to be created for this
        test to mean anything.
        """
        stage = STAGES_BY_ID["er_plain"]
        for rel in stage.artifacts:
            for base in ("experiments", "experiments/seed_1"):
                concrete = rel.replace("experiments", base, 1).format(ratio_key="RR_0.2")
                path = tmp_path / concrete
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"x")

        set_active_seed(1)
        paths = resolve_stage_artifacts(tmp_path, stage)

        assert paths, "both trees were populated, so something must resolve"
        for p in paths:
            rel_parts = p.relative_to(tmp_path).parts
            assert rel_parts[:2] == ("experiments", "seed_1"), p

    def test_unseeded_reset_still_resolves_the_original_tree(self, tmp_path):
        """The mirror image: no seed active means the original tree, untouched."""
        stage = STAGES_BY_ID["er_plain"]
        for rel in stage.artifacts:
            path = tmp_path / rel.format(ratio_key="RR_0.2")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"x")

        paths = resolve_stage_artifacts(tmp_path, stage)

        assert paths
        for p in paths:
            assert "seed_" not in str(p.relative_to(tmp_path))
