import pytest

from src.seed_run import (
    BASELINE_SEED,
    SEED_ENV_VAR,
    experiments_root,
    model_seed,
    resolve_seed,
    seed_split_path,
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """Every test starts with no seed active, whatever the ambient environment is."""
    monkeypatch.delenv(SEED_ENV_VAR, raising=False)


# ---------------------------------------------------------------------------
# The safety property: unset env var == today's behaviour, unchanged
# ---------------------------------------------------------------------------

def test_unset_env_reproduces_the_original_layout(tmp_path):
    """This is the guarantee that protects the existing experiments/ tree: with no
    seed active, every path and the RNG itself must be what they were before
    seed support existed."""
    assert resolve_seed() is None
    assert model_seed() == 42
    assert experiments_root(tmp_path) == tmp_path / "experiments"
    assert seed_split_path(tmp_path) == tmp_path / "data_split.npz"


def test_no_seed_directory_component_leaks_in_when_unseeded(tmp_path):
    # Compare only the components below tmp_path -- pytest builds tmp_path from the
    # test's own name, which itself contains "seed_".
    assert experiments_root(tmp_path).relative_to(tmp_path).parts == ("experiments",)
    assert seed_split_path(tmp_path).relative_to(tmp_path).parts == ("data_split.npz",)


# ---------------------------------------------------------------------------
# resolve_seed
# ---------------------------------------------------------------------------

def test_env_var_is_read(monkeypatch):
    monkeypatch.setenv(SEED_ENV_VAR, "7")
    assert resolve_seed() == 7


def test_explicit_argument_beats_the_env_var(monkeypatch):
    """Same precedence as resolve_cache_root: arg > env > default."""
    monkeypatch.setenv(SEED_ENV_VAR, "7")
    assert resolve_seed(3) == 3


def test_empty_env_var_is_treated_as_unset(monkeypatch):
    # An exported-but-empty var is what a shell leaves behind after `FONDA_SEED=`,
    # and must not be read as seed 0.
    monkeypatch.setenv(SEED_ENV_VAR, "")
    assert resolve_seed() is None


def test_non_integer_env_var_raises_with_a_usable_message(monkeypatch):
    monkeypatch.setenv(SEED_ENV_VAR, "abc")
    with pytest.raises(ValueError, match=SEED_ENV_VAR):
        resolve_seed()


def test_seed_zero_is_a_real_seed_not_a_falsy_unset(monkeypatch):
    monkeypatch.setenv(SEED_ENV_VAR, "0")
    assert resolve_seed() == 0
    assert model_seed() == 0


# ---------------------------------------------------------------------------
# model_seed
# ---------------------------------------------------------------------------

def test_model_seed_falls_back_to_the_baseline(monkeypatch):
    assert model_seed() == BASELINE_SEED


def test_model_seed_follows_the_active_seed(monkeypatch):
    monkeypatch.setenv(SEED_ENV_VAR, "2")
    assert model_seed() == 2


# ---------------------------------------------------------------------------
# experiments_root / seed_split_path
# ---------------------------------------------------------------------------

def test_seeded_paths(tmp_path, monkeypatch):
    monkeypatch.setenv(SEED_ENV_VAR, "1")
    assert experiments_root(tmp_path) == tmp_path / "experiments" / "seed_1"
    assert seed_split_path(tmp_path) == tmp_path / "data_split_seed_1.npz"


def test_seed_tree_is_nested_under_experiments_so_gitignore_rules_apply(tmp_path, monkeypatch):
    """`!experiments/**/*.csv` re-includes eval tables against the blanket *.csv
    ignore. A sibling experiments_seed_1/ would fall outside it and every table
    would be silently untracked -- fatal for the Kaggle push/resume flow."""
    monkeypatch.setenv(SEED_ENV_VAR, "1")
    root = experiments_root(tmp_path)
    assert root.parent == tmp_path / "experiments"


def test_distinct_seeds_never_share_a_tree_or_a_split(tmp_path):
    roots = {str(experiments_root(tmp_path, seed=s)) for s in (1, 2, 3, 42)}
    splits = {str(seed_split_path(tmp_path, seed=s)) for s in (1, 2, 3, 42)}
    assert len(roots) == 4
    assert len(splits) == 4


def test_explicit_seed_42_is_a_fresh_tree_not_the_original(tmp_path):
    """An explicit seed is always its own tree, even when it equals BASELINE_SEED --
    otherwise `--seed 42` would overwrite the very results being compared against."""
    assert experiments_root(tmp_path, seed=42) == tmp_path / "experiments" / "seed_42"
    assert experiments_root(tmp_path, seed=42) != experiments_root(tmp_path)
    assert seed_split_path(tmp_path, seed=42) != seed_split_path(tmp_path)


def test_accepts_a_string_project_root(monkeypatch):
    monkeypatch.setenv(SEED_ENV_VAR, "5")
    assert experiments_root(".").name == "seed_5"
