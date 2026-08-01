import numpy as np

from src.mlp_replay.checkpointing import (
    append_training_log,
    create_empty_training_history,
    format_ratio_key,
    get_cached_raw_year,
    load_all_training_histories,
    load_completion_status,
    per_ratio_path,
    save_all_training_histories,
    save_completion_status,
)

BASE_KEYS = {
    "year", "train_accuracy", "train_precision", "train_recall", "train_f1",
    "val_accuracy", "val_precision", "val_recall", "val_f1", "val_roc_auc", "val_pr_auc",
    "replay_pool_size", "replay_target_size", "replay_used_size",
}


def test_create_empty_training_history_default_keys():
    history = create_empty_training_history()
    assert set(history.keys()) == BASE_KEYS
    assert all(v == [] for v in history.values())


def test_create_empty_training_history_extra_keys():
    history = create_empty_training_history(extra_keys=["weight_policy", "hard_target_size"])
    assert BASE_KEYS.issubset(history.keys())
    assert "weight_policy" in history
    assert "hard_target_size" in history
    assert history["weight_policy"] == []


def test_format_ratio_key():
    assert format_ratio_key(0.2) == "RR_0.2"
    assert format_ratio_key(0.35) == "RR_0.3" or format_ratio_key(0.35) == "RR_0.4"  # formatting rounds to 1 decimal


def test_get_cached_raw_year_present_and_missing():
    cache = {1: (np.zeros((3, 2)), np.zeros(3))}
    X, y = get_cached_raw_year(cache, 1)
    assert X.shape == (3, 2)

    X_missing, y_missing = get_cached_raw_year(cache, 99)
    assert X_missing.shape == (0, 0)
    assert y_missing.shape == (0,)
    assert X_missing.dtype == np.float32
    assert y_missing.dtype == np.int64


def test_save_and_load_all_training_histories_roundtrip(tmp_path):
    path = tmp_path / "histories.pkl"
    assert load_all_training_histories(path) == {}  # missing file -> empty dict

    histories = {"RR_0.2": create_empty_training_history()}
    histories["RR_0.2"]["year"] = [2001, 2002]
    save_all_training_histories(path, histories)

    loaded = load_all_training_histories(path)
    assert loaded == histories


def test_load_all_training_histories_rejects_non_dict(tmp_path):
    import pickle
    path = tmp_path / "bad.pkl"
    with open(path, "wb") as f:
        pickle.dump([1, 2, 3], f)
    assert load_all_training_histories(path) == {}


def test_load_completion_status_defaults(tmp_path):
    path = tmp_path / "status.json"
    status = load_completion_status(path)
    assert status == {"completed_ratios": [], "completed_years": {}}


def test_load_completion_status_extra_default_keys(tmp_path):
    path = tmp_path / "status.json"
    status = load_completion_status(path, extra_default_keys=["weight_policy_by_ratio"])
    assert status == {"completed_ratios": [], "completed_years": {}, "weight_policy_by_ratio": {}}


def test_save_completion_status_dedupes_and_sorts(tmp_path):
    path = tmp_path / "status.json"
    status = {
        "completed_ratios": ["RR_0.2", "RR_0.2", "RR_0.3"],
        "completed_years": {"RR_0.2": [3, 1, 2, 2]},
    }
    save_completion_status(path, status)

    reloaded = load_completion_status(path)
    assert reloaded["completed_ratios"] == ["RR_0.2", "RR_0.3"]
    assert reloaded["completed_years"] == {"RR_0.2": [1, 2, 3]}


def test_load_completion_status_backfills_missing_keys_from_disk(tmp_path):
    import json
    path = tmp_path / "status.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"completed_ratios": ["RR_0.2"]}, f)  # completed_years missing entirely

    status = load_completion_status(path, extra_default_keys=["weight_policy_by_ratio"])
    assert status["completed_ratios"] == ["RR_0.2"]
    assert status["completed_years"] == {}
    assert status["weight_policy_by_ratio"] == {}


def test_per_ratio_path_inserts_ratio_key_before_extension(tmp_path):
    base = tmp_path / "mlp_replay_all_training_histories.pkl"
    assert per_ratio_path(base, "RR_0.2") == tmp_path / "mlp_replay_all_training_histories_RR_0.2.pkl"


def test_per_ratio_path_preserves_existing_suffix_tokens(tmp_path):
    # Some notebooks already bake a token into the base filename (e.g. a strategy
    # name); per_ratio_path must not clobber it, just append the ratio before the extension.
    base = tmp_path / "mlp_replay_completion_status_confidently_correct_memory.json"
    result = per_ratio_path(base, "RR_0.4")
    assert result == tmp_path / "mlp_replay_completion_status_confidently_correct_memory_RR_0.4.json"


def test_per_ratio_path_different_ratios_do_not_collide(tmp_path):
    base = tmp_path / "log.txt"
    paths = {per_ratio_path(base, f"RR_{r:.1f}") for r in (0.2, 0.3, 0.4, 0.5)}
    assert len(paths) == 4


def test_append_training_log_writes_real_newlines(tmp_path):
    path = tmp_path / "log.txt"
    append_training_log(path, "first entry")
    append_training_log(path, "second entry")

    content = path.read_text(encoding="utf-8")
    assert content.count(chr(92)) == 0  # no literal backslashes
    assert content.count(chr(10)) == 2  # two real newlines
    assert "first entry" in content
    assert "second entry" in content
