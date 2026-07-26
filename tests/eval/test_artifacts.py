import pickle

import numpy as np
import pytest

from src.eval.artifacts import (
    infer_years_from_models,
    load_final_scaler,
    load_model,
    load_scaler,
    resolve_scaler_year,
    validate_feature_count,
    validate_model_features,
    validate_scaler_features,
)


class _FittedThing:
    """Stands in for a fitted sklearn model/scaler with n_features_in_."""

    def __init__(self, n_features_in_):
        self.n_features_in_ = n_features_in_


def _touch(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b'')


class TestInferYearsFromModels:
    def test_finds_only_existing_years(self, tmp_path):
        _touch(tmp_path / 'model_year_2018.pkl')
        _touch(tmp_path / 'model_year_2020.pkl')
        _touch(tmp_path / 'model_year_2021.pkl')
        assert infer_years_from_models(tmp_path, 'model_year_{year}.pkl') == [2018, 2020, 2021]

    def test_reports_a_gap(self, tmp_path):
        """A missing middle year must not be silently skipped over."""
        _touch(tmp_path / 'model_year_2017.pkl')
        _touch(tmp_path / 'model_year_2019.pkl')
        assert 2018 not in infer_years_from_models(tmp_path, 'model_year_{year}.pkl')
        assert infer_years_from_models(tmp_path, 'model_year_{year}.pkl') == [2017, 2019]

    def test_excludes_years_outside_the_probe_range(self, tmp_path):
        _touch(tmp_path / 'model_year_2009.pkl')
        _touch(tmp_path / 'model_year_2035.pkl')
        _touch(tmp_path / 'model_year_2020.pkl')
        assert infer_years_from_models(tmp_path, 'model_year_{year}.pkl') == [2020]

    def test_empty_dir_returns_empty_list(self, tmp_path):
        assert infer_years_from_models(tmp_path, 'model_year_{year}.pkl') == []

    def test_template_with_year_mid_string(self, tmp_path):
        _touch(tmp_path / 'model_2019_suffix.pkl')
        assert infer_years_from_models(tmp_path, 'model_{year}_suffix.pkl') == [2019]


class TestLoadModelAndScaler:
    def test_load_model_returns_none_when_absent(self, tmp_path):
        assert load_model(tmp_path, 'model_year_{year}.pkl', 2019) is None

    def test_load_model_unpickles_existing_file(self, tmp_path):
        path = tmp_path / 'model_year_2019.pkl'
        with open(path, 'wb') as f:
            pickle.dump({'a': 1}, f)
        assert load_model(tmp_path, 'model_year_{year}.pkl', 2019) == {'a': 1}

    def test_load_scaler_returns_none_when_template_is_none(self, tmp_path):
        """The rbf family has no per-year scaler template."""
        assert load_scaler(tmp_path, None, 2019) is None

    def test_load_scaler_returns_none_when_file_absent(self, tmp_path):
        assert load_scaler(tmp_path, 'scaler_year_{year}.pkl', 2019) is None

    def test_load_scaler_unpickles_existing_file(self, tmp_path):
        path = tmp_path / 'scaler_year_2019.pkl'
        with open(path, 'wb') as f:
            pickle.dump('a-scaler', f)
        assert load_scaler(tmp_path, 'scaler_year_{year}.pkl', 2019) == 'a-scaler'


class TestLoadFinalScaler:
    def test_returns_none_when_key_absent(self):
        assert load_final_scaler({}) is None

    def test_returns_none_when_key_is_falsy(self):
        assert load_final_scaler({'final_scaler_path': None}) is None
        assert load_final_scaler({'final_scaler_path': ''}) is None

    def test_returns_none_when_path_missing(self, tmp_path):
        missing = tmp_path / 'nope.pkl'
        assert load_final_scaler({'final_scaler_path': str(missing)}) is None

    def test_unpickles_existing_file(self, tmp_path):
        path = tmp_path / 'final_scaler.pkl'
        with open(path, 'wb') as f:
            pickle.dump('final', f)
        assert load_final_scaler({'final_scaler_path': str(path)}) == 'final'


class TestResolveScalerYear:
    def test_incremental_and_eval_before_model_uses_model_year(self):
        cfg = {'incremental_scaler': True}
        year, policy = resolve_scaler_year(cfg, model_year=2020, eval_year=2018)
        assert (year, policy) == (2020, 'model_year_scaler_for_prior_year')

    def test_incremental_and_eval_at_or_after_model_uses_eval_year(self):
        cfg = {'incremental_scaler': True}
        assert resolve_scaler_year(cfg, model_year=2020, eval_year=2020) == (2020, 'evaluation_year_scaler')
        assert resolve_scaler_year(cfg, model_year=2020, eval_year=2022) == (2022, 'evaluation_year_scaler')

    def test_non_incremental_always_uses_eval_year(self):
        cfg = {'incremental_scaler': False}
        assert resolve_scaler_year(cfg, model_year=2020, eval_year=2018) == (2018, 'evaluation_year_scaler')
        assert resolve_scaler_year(cfg, model_year=2020, eval_year=2022) == (2022, 'evaluation_year_scaler')


class TestValidateFeatureCount:
    def test_noop_when_expected_is_none(self):
        X = np.zeros((3, 5))
        validate_feature_count(X, None, 'thing', 'fam', 2019, 2019, 'test')  # must not raise

    def test_noop_when_counts_match(self):
        X = np.zeros((3, 5))
        validate_feature_count(X, 5, 'thing', 'fam', 2019, 2019, 'test')  # must not raise

    def test_raises_with_context_in_the_message(self):
        X = np.zeros((3, 5))
        with pytest.raises(ValueError) as exc:
            validate_feature_count(X, 7, 'loaded scaler', 'my_family', 2018, 2019, 'validation')
        msg = str(exc.value)
        for expected in ('my_family', '2018', '2019', 'validation', 'loaded scaler', '5', '7'):
            assert expected in msg


class TestValidateScalerAndModelFeatures:
    def test_scaler_none_is_a_noop(self):
        validate_scaler_features(np.zeros((2, 3)), None, 'fam', 2019, 2019, 'test')

    def test_scaler_without_n_features_in_is_a_noop(self):
        validate_scaler_features(np.zeros((2, 3)), object(), 'fam', 2019, 2019, 'test')

    def test_scaler_mismatch_raises(self):
        with pytest.raises(ValueError, match='loaded scaler'):
            validate_scaler_features(np.zeros((2, 3)), _FittedThing(9), 'fam', 2019, 2019, 'test')

    def test_model_mismatch_raises(self):
        with pytest.raises(ValueError, match='loaded model'):
            validate_model_features(np.zeros((2, 3)), _FittedThing(9), 'fam', 2019, 2019, 'test')

    def test_matching_count_does_not_raise(self):
        validate_scaler_features(np.zeros((2, 3)), _FittedThing(3), 'fam', 2019, 2019, 'test')
        validate_model_features(np.zeros((2, 3)), _FittedThing(3), 'fam', 2019, 2019, 'test')
