import pickle

import pandas as pd
from sklearn.preprocessing import StandardScaler

from .checkpointing import (
    append_training_log,
    create_empty_training_history,
    get_cached_raw_year,
    save_all_training_histories,
    save_completion_status,
)
from .model import build_mlp_model


def resume_or_init_ratio_state(
    ratio_key,
    models_dir,
    model_file_template,
    scaler_file_template,
    year_value_to_idx,
    all_training_histories,
    completion_status,
    training_log_file,
    extra_history_keys=None,
):
    """Load an existing model/scaler checkpoint for this ratio if one exists, else start fresh.

    Returns (model, incremental_scaler, start_year_idx, completed_years, training_history).
    """
    training_history = all_training_histories.get(ratio_key, create_empty_training_history(extra_keys=extra_history_keys))
    for key in create_empty_training_history(extra_keys=extra_history_keys).keys():
        training_history.setdefault(key, [])

    completed_years = set(int(y) for y in completion_status.get('completed_years', {}).get(ratio_key, []))
    completed_years.update(int(y) for y in training_history.get('year', []))
    completion_status.setdefault('completed_years', {})[ratio_key] = sorted(list(completed_years))

    model = None
    incremental_scaler = None
    start_year_idx = 1

    if completed_years:
        resume_candidate_years = sorted(completed_years, reverse=True)
        resumed = False
        resume_year = None
        for resume_year in resume_candidate_years:
            year_model_path = models_dir / model_file_template.format(year=resume_year)
            year_scaler_path = models_dir / scaler_file_template.format(year=resume_year)
            if year_model_path.exists() and year_scaler_path.exists() and resume_year in year_value_to_idx:
                with open(year_model_path, 'rb') as f:
                    model = pickle.load(f)
                with open(year_scaler_path, 'rb') as f:
                    incremental_scaler = pickle.load(f)
                start_year_idx = year_value_to_idx[resume_year] + 1
                resumed = True
                print(f'[{ratio_key}] resuming from year {resume_year}; continuing at index {start_year_idx}.')
                append_training_log(training_log_file, f'[{ratio_key}] resumed from year {resume_year}.')
                break

        if resumed:
            # Clean up completed_years and training_history for years after the resumed year
            completed_years = {y for y in completed_years if y <= resume_year}
            if training_history.get('year'):
                keep_indices = [i for i, y in enumerate(training_history['year']) if y <= resume_year]
                for key in training_history.keys():
                    if isinstance(training_history[key], list):
                        training_history[key] = [training_history[key][i] for i in keep_indices]
        else:
            print(f'[{ratio_key}] checkpoint artifacts missing/inconsistent. Restarting this ratio from scratch.')
            append_training_log(training_log_file, f'[{ratio_key}] restart due to missing/inconsistent checkpoint artifacts.')
            completed_years = set()
            completion_status['completed_years'][ratio_key] = []
            training_history = create_empty_training_history(extra_keys=extra_history_keys)

    if model is None:
        model = build_mlp_model()
    if incremental_scaler is None:
        incremental_scaler = StandardScaler()

    return model, incremental_scaler, start_year_idx, completed_years, training_history


def load_year_batch_or_none(
    train_feature_cache,
    val_feature_cache,
    year_idx,
    year_val,
    ratio_key,
    completed_years,
    completion_status,
    completion_status_file,
    training_log_file,
):
    """Load the cached train/val arrays for one year, or record the year as skipped and return None."""
    X_train_raw, y_train_batch = get_cached_raw_year(train_feature_cache, year_idx)
    X_val_raw, y_val_batch = get_cached_raw_year(val_feature_cache, year_idx)

    if len(X_train_raw) == 0 or len(X_val_raw) == 0:
        print(f'[{ratio_key}] Year {year_val}: skipped (empty after filtering)')
        completed_years.add(year_val)
        completion_status['completed_years'][ratio_key] = sorted(list(completed_years))
        save_completion_status(completion_status_file, completion_status)
        append_training_log(training_log_file, f'[{ratio_key}] year {year_val} skipped (empty after filtering).')
        return None

    return X_train_raw, y_train_batch, X_val_raw, y_val_batch


def finalize_ratio_and_save(
    incremental_scaler,
    model,
    models_dir,
    final_scaler_filename,
    final_model_filename,
    history_filename,
    training_history,
    ratio_key,
    all_target_year_values,
    completed_years,
    completion_status,
    completion_status_file,
    training_log_file,
    all_training_histories,
    all_histories_file,
):
    if hasattr(incremental_scaler, 'mean_'):
        final_scaler_path = models_dir / final_scaler_filename
        with open(final_scaler_path, 'wb') as f:
            pickle.dump(incremental_scaler, f)
        print(f'[{ratio_key}] Final incremental scaler saved: {final_scaler_path.name}')

    final_model_path = models_dir / final_model_filename
    with open(final_model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f'[{ratio_key}] Final model saved: {final_model_path.name}')

    ratio_history_df = pd.DataFrame(training_history).sort_values('year').reset_index(drop=True)
    ratio_history_path = models_dir / history_filename
    ratio_history_df.to_csv(ratio_history_path, index=False)
    print(f'[{ratio_key}] History saved: {ratio_history_path}')

    if set(all_target_year_values).issubset(completed_years):
        if ratio_key not in completion_status.get('completed_ratios', []):
            completion_status.setdefault('completed_ratios', []).append(ratio_key)
        save_completion_status(completion_status_file, completion_status)
        append_training_log(training_log_file, f'[{ratio_key}] marked as fully completed.')

    all_training_histories[ratio_key] = training_history.copy()
    save_all_training_histories(all_histories_file, all_training_histories)


def write_combined_history_csv(all_training_histories, combined_history_path):
    combined_history_frames = []
    for ratio_key, history_dict in all_training_histories.items():
        if not isinstance(history_dict, dict):
            continue
        if len(history_dict.get('year', [])) == 0:
            continue
        df_ratio = pd.DataFrame(history_dict).sort_values('year').reset_index(drop=True)
        replay_ratio = float(ratio_key.split('_')[1])
        df_ratio['ratio_key'] = ratio_key
        df_ratio['replay_ratio'] = replay_ratio
        combined_history_frames.append(df_ratio)

    if combined_history_frames:
        combined_history_df = pd.concat(combined_history_frames, ignore_index=True)
        combined_history_df = combined_history_df.sort_values(['replay_ratio', 'year']).reset_index(drop=True)
        combined_history_df.to_csv(combined_history_path, index=False)
        print(f'Combined history saved: {combined_history_path}')
        print(f'Combined rows: {len(combined_history_df):,}')
    else:
        print('No history data available to write combined history.')
