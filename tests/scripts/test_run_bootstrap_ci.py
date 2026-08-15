"""End-to-end tests for the bootstrap CI driver, on a synthetic registry.

The driver's real risk is not the statistics (covered in tests/eval/test_bootstrap_ci.py)
but the plumbing: joining a prediction back to the cube it came from, and refusing to
run when re-inference no longer reproduces the published table. Both are exercised
here against a miniature but genuine pipeline -- a real zarr store, real pickled
sklearn models, real scalers -- rather than mocks, because a mock cannot catch the
alignment bug this analysis is most exposed to.
"""
import pickle

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler

import scripts.run_bootstrap_ci as driver
from src.mlp_replay.data import prepare_raw_features_for_year

YEARS = [2016, 2017, 2018]
PREP_FLAGS = driver.PREP_KIND_FLAGS['prevyears']


def _build_dataset(n_cubes=8, pixels_per_cube=12, n_years=4, seed=0):
    """A synthetic store shaped like the real one, including cube membership.

    Pixels are grouped into cubes and given a per-cube disturbance rate, so cube
    identity carries signal -- without that, a broken pixel-to-cube join would still
    produce believable numbers and the alignment test below would not bite.
    """
    rng = np.random.default_rng(seed)
    n_pixels = n_cubes * pixels_per_cube
    bands = ['B02', 'B03', 'B04', 'B06']

    cube_idx = np.repeat(np.arange(n_cubes), pixels_per_cube)
    per_cube_rate = rng.uniform(0.15, 0.75, n_cubes)
    disturbances = (rng.random((n_pixels, n_years))
                    < per_cube_rate[cube_idx][:, None]).astype(np.int64)

    # Features carry the label signal so the fitted models are not degenerate.
    signal = disturbances[:, :, None] * 0.15
    return xr.Dataset(
        {
            's2_bands': (('pixel', 'year', 's2_band'),
                         (rng.normal(0.3, 0.05, (n_pixels, n_years, len(bands))) + signal).astype(np.float32)),
            'dem': (('pixel', 'year'), rng.normal(500, 50, (n_pixels, n_years)).astype(np.float32)),
            'ndvi': (('pixel', 'year'),
                     (rng.normal(0.5, 0.1, (n_pixels, n_years)) - disturbances * 0.2).astype(np.float32)),
            'ndwi': (('pixel', 'year'), rng.normal(0.1, 0.1, (n_pixels, n_years)).astype(np.float32)),
            'disturbances': (('pixel', 'year'), disturbances),
            'cube_idx': (('pixel',), cube_idx.astype(np.int64)),
        },
        coords={
            'pixel': np.arange(n_pixels),
            'year': np.arange(YEARS[0] - 1, YEARS[0] - 1 + n_years),
            's2_band': bands,
        },
    )


@pytest.fixture
def synthetic_project(tmp_path, monkeypatch):
    """A miniature registry of three families with fitted models, wired into the driver."""
    ds = _build_dataset()
    dataset_path = tmp_path / 'training_data_with_features.zarr'
    ds.to_zarr(dataset_path, mode='w')

    n_pixels = ds.sizes['pixel']
    rng = np.random.default_rng(3)
    perm = rng.permutation(n_pixels)
    test_pixel_indices = np.sort(perm[: n_pixels // 2])
    train_pixel_indices = np.sort(perm[n_pixels // 2:])

    split_path = tmp_path / 'data_split.npz'
    np.savez(split_path, train_pixel_indices=train_pixel_indices,
             val_pixel_indices=train_pixel_indices, test_pixel_indices=test_pixel_indices)

    year_to_idx = {int(y): i for i, y in enumerate(ds.year.values)}
    family_ids = [
        'mlp_prevyears_monthly_features_incremental_scaler',
        'mlp_prevyears_monthly_features_incremental_scaler_experience_replay_RR_0.2',
        'mlp_combined_HE=0_CC=0.2_UP=0.2_PR=(0.1,10)_RR=0.5',
    ]

    families = {}
    for offset, family_id in enumerate(family_ids):
        models_dir = tmp_path / f'models_{offset}'
        models_dir.mkdir()
        for year in YEARS:
            X, y = prepare_raw_features_for_year(
                ds, train_pixel_indices, year_to_idx[year], dtype=None, **PREP_FLAGS)
            scaler = StandardScaler().fit(X)
            model = SGDClassifier(loss='log_loss', random_state=offset, max_iter=50)
            model.fit(scaler.transform(X), y)
            with open(models_dir / f'model_year_{year}.pkl', 'wb') as f:
                pickle.dump(model, f)
            with open(models_dir / f'scaler_year_{year}.pkl', 'wb') as f:
                pickle.dump(scaler, f)

        families[family_id] = {
            'short_label': family_id,
            'label': f'Synthetic {family_id}',
            'models_dir': models_dir,
            'model_template': 'model_year_{year}.pkl',
            'scaler_template': 'scaler_year_{year}.pkl',
            'dataset_path': dataset_path,
            'prep_kind': 'prevyears',
            'incremental_scaler': True,
        }

    output_root = tmp_path / 'bootstrap_ci'
    monkeypatch.setattr(driver, 'OUTPUT_ROOT', output_root)
    monkeypatch.setattr(driver, 'PREDICTIONS_DIR', output_root / 'predictions')
    monkeypatch.setattr(driver, 'CONTEXT_DIR', output_root / 'predictions' / '_context')
    monkeypatch.setattr(driver, 'CI_RESULTS_DIR', output_root / 'ci_results')
    monkeypatch.setattr(driver, 'SPLIT_PATH', split_path)
    monkeypatch.setattr(driver, 'PUBLISHED_TABLE', tmp_path / 'all_families_combined.csv')
    monkeypatch.setattr(driver, 'build_families', lambda _root: families)

    return {
        'families': families,
        'family_ids': family_ids,
        'baseline_id': family_ids[0],
        'tmp_path': tmp_path,
        'output_root': output_root,
        'test_pixel_indices': test_pixel_indices,
    }


def _publish_table(synthetic_project, table_types=('own_year',), corrupt_f1=False):
    """Write a stand-in all_families_combined.csv from the saved predictions.

    The real table is produced by the evaluation notebook; here it is derived from
    the same probabilities the driver will read back, which is what lets the
    consistency check pass on an honest run and fail on a doctored one.
    """
    rows = []
    for family_id in synthetic_project['family_ids']:
        for model_year in YEARS:
            for eval_year in YEARS:
                if eval_year > model_year:
                    continue
                loaded = driver.load_prediction(family_id, model_year, eval_year)
                if loaded is None:
                    continue
                y_proba, y_true, cube_idx = loaded
                threshold = 0.5
                tp, fp, fn, _ = driver.cube_confusion_counts(
                    y_true, y_proba >= threshold, cube_idx)
                f1 = float(driver.f1_from_counts(tp.sum(), fp.sum(), fn.sum()))
                for table_type in table_types:
                    if table_type == 'own_year' and model_year != eval_year:
                        continue
                    rows.append({
                        'family_id': family_id,
                        'family_label': f'Synthetic {family_id}',
                        'table_type': table_type,
                        'model_year': model_year,
                        'eval_year': eval_year,
                        'threshold_used': threshold,
                        'f1_score': f1 + (0.25 if corrupt_f1 else 0.0),
                        'pr_auc': 0.5,
                    })
    df = pd.DataFrame(rows)
    df.to_csv(driver.PUBLISHED_TABLE, index=False)
    return df


class TestPredictStage:
    def test_writes_one_prediction_per_configuration(self, synthetic_project):
        written, skipped = driver.run_predict_stage(
            synthetic_project['families'], synthetic_project['family_ids'],
            synthetic_project['test_pixel_indices'], verify_manifest=False)

        assert skipped == 0
        # 3 families x the deduplicated (model_year, eval_year) pairs for 3 years.
        assert written == 3 * len(driver.triples_for_family(YEARS))
        assert driver.prediction_path(
            synthetic_project['baseline_id'], YEARS[0], YEARS[0]).exists()

    def test_rerun_skips_everything_already_written(self, synthetic_project):
        """The resume guard: an interrupted sweep must not redo finished work."""
        driver.run_predict_stage(
            synthetic_project['families'], synthetic_project['family_ids'],
            synthetic_project['test_pixel_indices'], verify_manifest=False)

        written, skipped = driver.run_predict_stage(
            synthetic_project['families'], synthetic_project['family_ids'],
            synthetic_project['test_pixel_indices'], verify_manifest=False)

        assert written == 0
        assert skipped == 3 * len(driver.triples_for_family(YEARS))

    def test_predictions_align_with_their_cube_labels(self, synthetic_project):
        """The join this whole analysis rests on.

        Feature preparation drops rows with invalid labels or NaN features, so the
        saved probabilities are a subset of the test pixels. If cube ids were taken
        without applying that same mask, every prediction would be attributed to the
        wrong cube. Checked by rebuilding the expected labels independently.
        """
        driver.run_predict_stage(
            synthetic_project['families'], synthetic_project['family_ids'],
            synthetic_project['test_pixel_indices'], verify_manifest=False)

        family_cfg = synthetic_project['families'][synthetic_project['baseline_id']]
        ds = xr.open_zarr(family_cfg['dataset_path'])
        year_to_idx = {int(y): i for i, y in enumerate(ds.year.values)}

        _, expected_y, row_mask = prepare_raw_features_for_year(
            ds, synthetic_project['test_pixel_indices'], year_to_idx[YEARS[1]],
            dtype=None, return_row_mask=True, **PREP_FLAGS)
        expected_cubes = ds['cube_idx'].values[synthetic_project['test_pixel_indices']][row_mask]

        _, y_true, cube_idx = driver.load_prediction(
            synthetic_project['baseline_id'], YEARS[1], YEARS[1])

        assert np.array_equal(y_true, expected_y)
        assert np.array_equal(cube_idx, expected_cubes)

    def test_labels_and_cubes_are_stored_once_per_year_not_per_config(self, synthetic_project):
        """Context files are shared; duplicating them 900x is what this avoids."""
        driver.run_predict_stage(
            synthetic_project['families'], synthetic_project['family_ids'],
            synthetic_project['test_pixel_indices'], verify_manifest=False)

        context_files = list(driver.CONTEXT_DIR.glob('*.npz'))
        assert len(context_files) == len(YEARS)


class TestBootstrapStage:
    def test_writes_ci_and_margin_tables_with_expected_columns(self, synthetic_project):
        driver.run_predict_stage(
            synthetic_project['families'], synthetic_project['family_ids'],
            synthetic_project['test_pixel_indices'], verify_manifest=False)
        _publish_table(synthetic_project)

        driver.run_bootstrap_stage(
            baseline_id=synthetic_project['baseline_id'],
            strategy_ids=synthetic_project['family_ids'][1:],
            table_types=['own_year'], n_bootstrap=50, seed=42, alpha=0.05,
            ci_method='percentile', fdr_correction=True, f1_tolerance=1e-4)

        ci = pd.read_csv(driver.CI_RESULTS_DIR / 'own_year.csv')
        margins = pd.read_csv(driver.CI_RESULTS_DIR / 'own_year_margins.csv')

        for column in ['family_id', 'model_year', 'eval_year', 'metric',
                       'point_estimate', 'ci_lower', 'ci_upper', 'n_cubes']:
            assert column in ci.columns
        for column in ['family_id', 'baseline_family_id', 'model_year', 'eval_year',
                       'metric', 'point_estimate', 'ci_lower', 'ci_upper',
                       'p_value_one_sided', 'q_value_bh']:
            assert column in margins.columns

        assert set(ci['metric']) == {'f1_score', 'pr_auc'}
        assert (ci['ci_lower'] <= ci['point_estimate']).all()
        assert (ci['point_estimate'] <= ci['ci_upper']).all()
        # Margins cover the two strategies only, never the baseline against itself.
        assert synthetic_project['baseline_id'] not in set(margins['family_id'])
        assert len(margins) == 2 * len(YEARS) * 2

    def test_json_twin_is_written_alongside_each_csv(self, synthetic_project):
        driver.run_predict_stage(
            synthetic_project['families'], synthetic_project['family_ids'],
            synthetic_project['test_pixel_indices'], verify_manifest=False)
        _publish_table(synthetic_project)

        driver.run_bootstrap_stage(
            baseline_id=synthetic_project['baseline_id'],
            strategy_ids=synthetic_project['family_ids'][1:],
            table_types=['own_year'], n_bootstrap=20, seed=42, alpha=0.05,
            ci_method='percentile', fdr_correction=False, f1_tolerance=1e-4)

        assert (driver.CI_RESULTS_DIR / 'own_year.json').exists()
        assert (driver.CI_RESULTS_DIR / 'own_year_margins.json').exists()

    def test_disagreement_with_published_table_aborts(self, synthetic_project):
        """A drifted re-inference must stop the run, not silently produce CIs.

        If the reloaded checkpoints no longer reproduce the reported F1, the interval
        would describe a different model than the thesis reports -- the one failure
        mode here that is invisible in the output.
        """
        driver.run_predict_stage(
            synthetic_project['families'], synthetic_project['family_ids'],
            synthetic_project['test_pixel_indices'], verify_manifest=False)
        _publish_table(synthetic_project, corrupt_f1=True)

        with pytest.raises(ValueError, match='Re-inference disagrees'):
            driver.run_bootstrap_stage(
                baseline_id=synthetic_project['baseline_id'],
                strategy_ids=synthetic_project['family_ids'][1:],
                table_types=['own_year'], n_bootstrap=10, seed=42, alpha=0.05,
                ci_method='percentile', fdr_correction=False, f1_tolerance=1e-4)

    def test_seed_makes_the_run_reproducible(self, synthetic_project):
        driver.run_predict_stage(
            synthetic_project['families'], synthetic_project['family_ids'],
            synthetic_project['test_pixel_indices'], verify_manifest=False)
        _publish_table(synthetic_project)

        def _run():
            driver.run_bootstrap_stage(
                baseline_id=synthetic_project['baseline_id'],
                strategy_ids=synthetic_project['family_ids'][1:],
                table_types=['own_year'], n_bootstrap=50, seed=7, alpha=0.05,
                ci_method='percentile', fdr_correction=False, f1_tolerance=1e-4)
            return pd.read_csv(driver.CI_RESULTS_DIR / 'own_year.csv')

        first, second = _run(), _run()
        pd.testing.assert_frame_equal(first, second)

    def test_missing_published_table_raises_clearly(self, synthetic_project):
        with pytest.raises(FileNotFoundError, match='run that pipeline first'):
            driver.run_bootstrap_stage(
                baseline_id=synthetic_project['baseline_id'],
                strategy_ids=synthetic_project['family_ids'][1:],
                table_types=['own_year'], n_bootstrap=10, seed=42, alpha=0.05,
                ci_method='percentile', fdr_correction=False, f1_tolerance=1e-4)


class TestMainCli:
    def test_all_stages_run_end_to_end(self, synthetic_project, monkeypatch):
        # The published table has to exist before the bootstrap stage, so the predict
        # stage runs first on its own and the table is derived from its output.
        driver.main([
            '--stage', 'predict', '--skip-manifest-check',
            '--baseline', synthetic_project['baseline_id'],
        ])
        _publish_table(synthetic_project)

        exit_code = driver.main([
            '--stage', 'bootstrap', '--n-bootstrap', '30', '--table-types', 'own_year',
            '--baseline', synthetic_project['baseline_id'], '--fdr-correction',
        ])

        assert exit_code == 0
        manifest_path = synthetic_project['output_root'] / 'run_manifest.json'
        assert manifest_path.exists()

        import json
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        assert manifest['n_bootstrap'] == 30
        assert manifest['resampling_unit'] == 'cube'
        assert manifest['baseline_family_id'] == synthetic_project['baseline_id']
        assert len(manifest['strategy_family_ids']) == 2
