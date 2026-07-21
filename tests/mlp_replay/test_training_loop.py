import pickle

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler

from src.mlp_replay.checkpointing import create_empty_training_history
from src.mlp_replay.model import build_mlp_model
from src.mlp_replay.training_loop import (
    finalize_ratio_and_save,
    load_year_batch_or_none,
    resume_or_init_ratio_state,
    write_combined_history_csv,
)

MODEL_FILE_TEMPLATE = "model_year_{year}.pkl"
SCALER_FILE_TEMPLATE = "scaler_year_{year}.pkl"
YEAR_VALUE_TO_IDX = {2001: 1, 2002: 2, 2003: 3, 2004: 4}


@pytest.fixture
def models_dir(tmp_path):
    d = tmp_path / "models"
    d.mkdir()
    return d


def test_resume_or_init_fresh_start(models_dir, tmp_path):
    model, scaler, start_year_idx, completed_years, history = resume_or_init_ratio_state(
        ratio_key="RR_0.2",
        models_dir=models_dir,
        model_file_template=MODEL_FILE_TEMPLATE,
        scaler_file_template=SCALER_FILE_TEMPLATE,
        year_value_to_idx=YEAR_VALUE_TO_IDX,
        all_training_histories={},
        completion_status={"completed_ratios": [], "completed_years": {}},
        training_log_file=tmp_path / "log.txt",
    )
    assert start_year_idx == 1
    assert completed_years == set()
    assert model.hidden_layer_sizes == (64,)
    assert hasattr(scaler, "fit")
    assert history["year"] == []


def test_resume_or_init_resumes_from_valid_checkpoint(models_dir, tmp_path):
    fake_model = build_mlp_model()
    fake_model.partial_fit(np.zeros((10, 3)), np.array([0, 1] * 5), classes=np.array([0, 1]))
    fake_scaler = StandardScaler().fit(np.zeros((10, 3)))
    with open(models_dir / MODEL_FILE_TEMPLATE.format(year=2001), "wb") as f:
        pickle.dump(fake_model, f)
    with open(models_dir / SCALER_FILE_TEMPLATE.format(year=2001), "wb") as f:
        pickle.dump(fake_scaler, f)

    history_seed = create_empty_training_history()
    history_seed["year"] = [2001]
    for key in history_seed:
        if key != "year":
            history_seed[key] = [0.1]

    model, scaler, start_year_idx, completed_years, history = resume_or_init_ratio_state(
        ratio_key="RR_0.2",
        models_dir=models_dir,
        model_file_template=MODEL_FILE_TEMPLATE,
        scaler_file_template=SCALER_FILE_TEMPLATE,
        year_value_to_idx=YEAR_VALUE_TO_IDX,
        all_training_histories={"RR_0.2": history_seed},
        completion_status={"completed_ratios": [], "completed_years": {"RR_0.2": [2001]}},
        training_log_file=tmp_path / "log.txt",
    )
    assert start_year_idx == 2  # resume right after year 2001 (idx 1 -> 2)
    assert completed_years == {2001}
    assert history["year"] == [2001]


def test_resume_or_init_restarts_when_checkpoint_files_missing(models_dir, tmp_path):
    # completion_status claims 2002 is done, but no model/scaler files exist for it
    model, scaler, start_year_idx, completed_years, history = resume_or_init_ratio_state(
        ratio_key="RR_0.2",
        models_dir=models_dir,
        model_file_template=MODEL_FILE_TEMPLATE,
        scaler_file_template=SCALER_FILE_TEMPLATE,
        year_value_to_idx=YEAR_VALUE_TO_IDX,
        all_training_histories={"RR_0.2": {**create_empty_training_history(), "year": [2002]}},
        completion_status={"completed_ratios": [], "completed_years": {"RR_0.2": [2002]}},
        training_log_file=tmp_path / "log.txt",
    )
    assert start_year_idx == 1
    assert completed_years == set()
    assert history["year"] == []


def test_resume_or_init_passes_through_extra_history_keys(models_dir, tmp_path):
    _, _, _, _, history = resume_or_init_ratio_state(
        ratio_key="RR_0.2",
        models_dir=models_dir,
        model_file_template=MODEL_FILE_TEMPLATE,
        scaler_file_template=SCALER_FILE_TEMPLATE,
        year_value_to_idx=YEAR_VALUE_TO_IDX,
        all_training_histories={},
        completion_status={"completed_ratios": [], "completed_years": {}},
        training_log_file=tmp_path / "log.txt",
        extra_history_keys=["weight_policy", "hard_target_size"],
    )
    assert "weight_policy" in history
    assert "hard_target_size" in history


def test_load_year_batch_or_none_returns_data_when_present(tmp_path):
    train_cache = {1: (np.zeros((5, 2)), np.zeros(5))}
    val_cache = {1: (np.zeros((3, 2)), np.zeros(3))}
    result = load_year_batch_or_none(
        train_cache, val_cache, year_idx=1, year_val=2001, ratio_key="RR_0.2",
        completed_years=set(), completion_status={"completed_ratios": [], "completed_years": {"RR_0.2": []}},
        completion_status_file=tmp_path / "status.json", training_log_file=tmp_path / "log.txt",
    )
    assert result is not None
    X_train, y_train, X_val, y_val = result
    assert X_train.shape == (5, 2)
    assert X_val.shape == (3, 2)


def test_load_year_batch_or_none_records_skip_when_empty(tmp_path):
    empty_cache = {1: (np.empty((0, 2)), np.empty(0))}
    val_cache = {1: (np.zeros((3, 2)), np.zeros(3))}
    completed_years = set()
    completion_status = {"completed_ratios": [], "completed_years": {"RR_0.2": []}}
    status_file = tmp_path / "status.json"

    result = load_year_batch_or_none(
        empty_cache, val_cache, year_idx=1, year_val=2001, ratio_key="RR_0.2",
        completed_years=completed_years, completion_status=completion_status,
        completion_status_file=status_file, training_log_file=tmp_path / "log.txt",
    )
    assert result is None
    assert 2001 in completed_years
    assert status_file.exists()


def test_finalize_ratio_and_save(models_dir, tmp_path):
    scaler = StandardScaler().fit(np.random.default_rng(1).normal(size=(10, 3)))
    model = build_mlp_model()
    model.partial_fit(np.random.default_rng(1).normal(size=(10, 3)), np.array([0, 1] * 5), classes=np.array([0, 1]))

    history = create_empty_training_history()
    history["year"] = [2001, 2002]
    for key in history:
        if key != "year":
            history[key] = [0.5, 0.6]

    all_histories = {}
    completion_status = {"completed_ratios": [], "completed_years": {"RR_0.2": [2001, 2002]}}

    finalize_ratio_and_save(
        incremental_scaler=scaler,
        model=model,
        models_dir=models_dir,
        final_scaler_filename="final_scaler.pkl",
        final_model_filename="final_model.pkl",
        history_filename="history.csv",
        training_history=history,
        ratio_key="RR_0.2",
        all_target_year_values=[2001, 2002],
        completed_years={2001, 2002},
        completion_status=completion_status,
        completion_status_file=tmp_path / "status.json",
        training_log_file=tmp_path / "log.txt",
        all_training_histories=all_histories,
        all_histories_file=tmp_path / "hist.pkl",
    )

    assert (models_dir / "final_scaler.pkl").exists()
    assert (models_dir / "final_model.pkl").exists()
    assert (models_dir / "history.csv").exists()
    assert "RR_0.2" in completion_status["completed_ratios"]
    assert "RR_0.2" in all_histories
    assert (tmp_path / "hist.pkl").exists()


def test_finalize_ratio_and_save_not_marked_completed_when_years_missing(models_dir, tmp_path):
    scaler = StandardScaler().fit(np.random.default_rng(1).normal(size=(10, 3)))
    model = build_mlp_model()
    model.partial_fit(np.random.default_rng(1).normal(size=(10, 3)), np.array([0, 1] * 5), classes=np.array([0, 1]))
    history = create_empty_training_history()
    history["year"] = [2001]
    for key in history:
        if key != "year":
            history[key] = [0.5]

    completion_status = {"completed_ratios": [], "completed_years": {"RR_0.2": [2001]}}
    finalize_ratio_and_save(
        incremental_scaler=scaler, model=model, models_dir=models_dir,
        final_scaler_filename="final_scaler.pkl", final_model_filename="final_model.pkl",
        history_filename="history.csv", training_history=history, ratio_key="RR_0.2",
        all_target_year_values=[2001, 2002],  # 2002 missing -> not fully completed
        completed_years={2001},
        completion_status=completion_status,
        completion_status_file=tmp_path / "status.json",
        training_log_file=tmp_path / "log.txt",
        all_training_histories={},
        all_histories_file=tmp_path / "hist.pkl",
    )
    assert "RR_0.2" not in completion_status["completed_ratios"]


def test_write_combined_history_csv(tmp_path):
    histories = {
        "RR_0.2": {"year": [2001, 2002], "val_f1": [0.5, 0.6]},
        "RR_0.3": {"year": [2001, 2002], "val_f1": [0.55, 0.65]},
    }
    combined_path = tmp_path / "combined.csv"
    write_combined_history_csv(histories, combined_path)

    assert combined_path.exists()
    df = pd.read_csv(combined_path)
    assert len(df) == 4
    assert set(df["ratio_key"]) == {"RR_0.2", "RR_0.3"}


def test_write_combined_history_csv_no_data(tmp_path, capsys):
    combined_path = tmp_path / "combined.csv"
    write_combined_history_csv({}, combined_path)
    assert not combined_path.exists()
    assert "No history data" in capsys.readouterr().out
