import pandas as pd
import pytest

from scripts.aggregate_seed_results import (
    COMBINED_REL,
    METRICS,
    aggregate_per_cell,
    aggregate_per_family,
    discover_seeds,
)


def _frame(seed, rows):
    """rows: list of (family_id, table_type, model_year, eval_year, f1)."""
    df = pd.DataFrame(rows, columns=["family_id", "table_type", "model_year", "eval_year", "f1_score"])
    for m in METRICS:
        if m not in df.columns:
            df[m] = 0.0
    df["seed"] = seed
    return df


# ---------------------------------------------------------------------------
# discover_seeds
# ---------------------------------------------------------------------------

def test_discover_seeds_finds_seed_dirs_only(tmp_path):
    base = tmp_path / "experiments"
    for name in ("seed_1", "seed_10", "seed_2", "sgd", "mlp", "seed_", "seed_x", "seed_summary"):
        (base / name).mkdir(parents=True)
    assert discover_seeds(tmp_path) == [1, 2, 10]


def test_discover_seeds_on_a_tree_with_none(tmp_path):
    (tmp_path / "experiments" / "sgd").mkdir(parents=True)
    assert discover_seeds(tmp_path) == []


def test_discover_seeds_without_an_experiments_dir(tmp_path):
    assert discover_seeds(tmp_path) == []


# ---------------------------------------------------------------------------
# aggregate_per_cell
# ---------------------------------------------------------------------------

def test_per_cell_mean_and_std_across_seeds():
    frames = [
        _frame(42, [("baseline", "own_year", 2017, 2017, 0.10)]),
        _frame(1, [("baseline", "own_year", 2017, 2017, 0.20)]),
        _frame(2, [("baseline", "own_year", 2017, 2017, 0.30)]),
    ]
    out = aggregate_per_cell(frames)
    row = out[(out.metric == "f1_score")].iloc[0]
    assert row["mean"] == pytest.approx(0.20)
    assert row["std"] == pytest.approx(0.10)  # sample std of .1/.2/.3
    assert row["n_seeds"] == 3
    assert row["seeds"] == "1,2,42"


def test_per_cell_keeps_year_cells_separate():
    frames = [
        _frame(42, [("f", "own_year", 2017, 2017, 0.1), ("f", "own_year", 2018, 2018, 0.9)]),
        _frame(1, [("f", "own_year", 2017, 2017, 0.3), ("f", "own_year", 2018, 2018, 0.7)]),
    ]
    out = aggregate_per_cell(frames)
    f1 = out[out.metric == "f1_score"].set_index("model_year")
    assert f1.loc[2017, "mean"] == pytest.approx(0.2)
    assert f1.loc[2018, "mean"] == pytest.approx(0.8)


def test_per_cell_reports_a_short_n_rather_than_hiding_it():
    """A family that errored in one seed must not silently produce a mean that
    looks like it rests on every seed."""
    frames = [
        _frame(42, [("a", "own_year", 2017, 2017, 0.1), ("b", "own_year", 2017, 2017, 0.5)]),
        _frame(1, [("a", "own_year", 2017, 2017, 0.3)]),  # family b errored here
    ]
    out = aggregate_per_cell(frames)
    f1 = out[out.metric == "f1_score"].set_index("family_id")
    assert f1.loc["a", "n_seeds"] == 2
    assert f1.loc["b", "n_seeds"] == 1
    assert f1.loc["b", "seeds"] == "42"


# ---------------------------------------------------------------------------
# aggregate_per_family -- the ordering that matters
# ---------------------------------------------------------------------------

def test_per_family_averages_within_a_seed_before_across_seeds():
    """Seed 42 contributes 4 year-cells, seed 1 only 1. Collapsing within a seed
    first makes each seed one sample; pooling every cell instead would let seed 42
    dominate purely by row count.

    within-seed:  seed42 mean(0,0,0,1)=0.25, seed1 mean(1)=1.0  -> across = 0.625
    pooled:       mean(0,0,0,1,1) = 0.4                          <- wrong
    """
    frames = [
        _frame(42, [("f", "own_year", y, y, v) for y, v in zip(range(2017, 2021), [0, 0, 0, 1])]),
        _frame(1, [("f", "own_year", 2017, 2017, 1.0)]),
    ]
    out = aggregate_per_family(frames)
    row = out[out.metric == "f1_score"].iloc[0]
    assert row["mean"] == pytest.approx(0.625)
    assert row["mean"] != pytest.approx(0.4)
    assert row["n_seeds"] == 2


def test_per_family_std_is_across_seeds_not_across_years():
    """Within each seed the years vary a lot, but both seeds average to the same
    number -- so the cross-seed std must be 0, not the year-to-year spread."""
    frames = [
        _frame(42, [("f", "own_year", 2017, 2017, 0.0), ("f", "own_year", 2018, 2018, 1.0)]),
        _frame(1, [("f", "own_year", 2017, 2017, 0.4), ("f", "own_year", 2018, 2018, 0.6)]),
    ]
    out = aggregate_per_family(frames)
    row = out[out.metric == "f1_score"].iloc[0]
    assert row["mean"] == pytest.approx(0.5)
    assert row["std"] == pytest.approx(0.0)


def test_per_family_min_max_bracket_the_mean():
    frames = [
        _frame(42, [("f", "own_year", 2017, 2017, 0.1)]),
        _frame(1, [("f", "own_year", 2017, 2017, 0.5)]),
        _frame(2, [("f", "own_year", 2017, 2017, 0.9)]),
    ]
    row = aggregate_per_family(frames).query("metric == 'f1_score'").iloc[0]
    assert row["min"] == pytest.approx(0.1)
    assert row["max"] == pytest.approx(0.9)
    assert row["min"] <= row["mean"] <= row["max"]


def test_table_types_are_not_mixed():
    frames = [
        _frame(42, [("f", "own_year", 2017, 2017, 0.2), ("f", "next_year", 2017, 2018, 0.8)]),
        _frame(1, [("f", "own_year", 2017, 2017, 0.2), ("f", "next_year", 2017, 2018, 0.8)]),
    ]
    out = aggregate_per_family(frames).query("metric == 'f1_score'").set_index("table_type")
    assert out.loc["own_year", "mean"] == pytest.approx(0.2)
    assert out.loc["next_year", "mean"] == pytest.approx(0.8)


def test_combined_rel_points_at_the_real_table_name():
    assert COMBINED_REL.endswith("all_families_combined.csv")
