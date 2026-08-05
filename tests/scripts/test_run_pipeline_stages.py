import json
import subprocess

import pytest

from scripts.run_pipeline_stages import (
    STAGES,
    Stage,
    artifacts_exist,
    auto_max_parallel,
    compute_status,
    git_commit_and_push,
    ratios_complete_per_ratio,
    ratios_complete_shared,
    resolve_cancelled,
    select_stages,
    shared_check,
    zarr_has_vars,
)
from src.mlp_replay.checkpointing import (
    format_ratio_key,
    per_ratio_path,
    save_completion_status,
)


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
# auto_max_parallel
# ---------------------------------------------------------------------------

def test_auto_max_parallel_clamps_to_range():
    gib = 1024 ** 3
    assert auto_max_parallel(0) == 1
    assert auto_max_parallel(1 * gib) == 1
    assert auto_max_parallel(11 * gib) == 2
    assert auto_max_parallel(1000 * gib) == 4  # clamped at 4 even with huge RAM


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
        if s.runner == "ratio_sweep":
            assert s.group == "er_parallel", f"{s.id} uses ratio_sweep but is in group {s.group}"
            assert s.ratios, f"{s.id} uses ratio_sweep but declares no ratios"
        assert s.group != "er_sequential" or s.runner != "ratio_sweep", (
            f"{s.id} is er_sequential but would run through the parallel launcher"
        )

    completion_paths = []
    for s in STAGES:
        # Recover each stage's completion-file path by calling check() against an
        # empty temp-like root is unnecessary here -- the curried checks close over
        # the literal path already, which we can recover via the closure cell for
        # the shared/per-ratio checks. Skip stages using bespoke check functions
        # (er_combined) since they're covered by their own dedicated test instead.
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
