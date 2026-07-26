"""Loading model/scaler artifacts and validating feature counts against them.

Moved verbatim out of Evaluations.ipynb cell 4. These are pure functions -- no free
globals, all inputs are arguments -- which is what makes them safe to lift without
any behaviour risk. `load_model` / `load_scaler` / `load_final_scaler` are three
variations on "format a {year} template, check it exists, unpickle," but each has a
distinct call shape (load_final_scaler takes a family config and a plain path, not
a template) and error semantics that call sites read by name, so they stay separate
rather than collapsing into one over-parameterized helper.
"""

import pickle
from pathlib import Path


def infer_years_from_models(models_dir, model_template):
    """Which years in [2010, 2035) have a model file on disk for this family."""
    years = []
    for y in range(2010, 2035):
        if (models_dir / model_template.format(year=y)).exists():
            years.append(y)
    return years


def load_model(models_dir, model_template, year):
    path = models_dir / model_template.format(year=year)
    if not path.exists():
        return None
    with open(path, 'rb') as f:
        return pickle.load(f)


def load_scaler(models_dir, scaler_template, year):
    """None when scaler_template is None -- the rbf family has no per-year scaler."""
    if scaler_template is None:
        return None
    path = models_dir / scaler_template.format(year=year)
    if not path.exists():
        return None
    with open(path, 'rb') as f:
        return pickle.load(f)


def load_final_scaler(family_cfg):
    final_scaler_path = family_cfg.get('final_scaler_path')
    if not final_scaler_path:
        return None
    if not Path(final_scaler_path).exists():
        return None
    with open(final_scaler_path, 'rb') as f:
        return pickle.load(f)


def resolve_scaler_year(family_cfg, model_year, eval_year):
    """Which year's scaler to use when scoring model_year on eval_year.

    For an incremental-scaler family evaluating a year before the model's own year,
    the model-year scaler is used (the eval-year scaler wouldn't have existed yet
    when the model was trained). Otherwise, use the eval year's own scaler.
    """
    if family_cfg['incremental_scaler'] and eval_year < model_year:
        return model_year, 'model_year_scaler_for_prior_year'
    return eval_year, 'evaluation_year_scaler'


def validate_feature_count(X, expected_n_features, owner_label, family_id, model_year, eval_year, split_name):
    if expected_n_features is None:
        return

    n_features = int(X.shape[1])
    expected_n_features = int(expected_n_features)
    if n_features != expected_n_features:
        raise ValueError(
            f'Feature mismatch for family={family_id}, model_year={model_year}, eval_year={eval_year}, split={split_name}: '
            f'X has {n_features} features, but {owner_label} expects {expected_n_features} features.'
        )


def validate_scaler_features(X, scaler_obj, family_id, model_year, eval_year, split_name):
    expected_n_features = getattr(scaler_obj, 'n_features_in_', None) if scaler_obj is not None else None
    validate_feature_count(
        X=X,
        expected_n_features=expected_n_features,
        owner_label='loaded scaler',
        family_id=family_id,
        model_year=model_year,
        eval_year=eval_year,
        split_name=split_name,
    )


def validate_model_features(X, model_obj, family_id, model_year, eval_year, split_name):
    expected_n_features = getattr(model_obj, 'n_features_in_', None) if model_obj is not None else None
    validate_feature_count(
        X=X,
        expected_n_features=expected_n_features,
        owner_label='loaded model',
        family_id=family_id,
        model_year=model_year,
        eval_year=eval_year,
        split_name=split_name,
    )
