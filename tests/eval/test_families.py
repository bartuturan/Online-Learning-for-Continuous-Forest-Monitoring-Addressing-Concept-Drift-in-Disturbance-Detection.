from pathlib import Path

import pytest
import xarray as xr

from src.eval.families import build_families, open_family_dataset

PROJECT_ROOT = Path(__file__).resolve().parents[2]

REQUIRED_KEYS = {'label', 'models_dir', 'model_template', 'scaler_template', 'dataset_path', 'prep_kind', 'incremental_scaler'}
KNOWN_DATASETS = {
    'training_data_with_features.zarr',
    'training_data_with_features_plus_monthly_indices.zarr',
    'training_data_with_neighbourhood_features.zarr',
}


@pytest.fixture(scope='module')
def families():
    return build_families(PROJECT_ROOT)


def test_family_count(families):
    assert len(families) == 47


class TestExperimentsRootRedirect:
    """build_families(project_root, experiments_root=...) is what lets an
    evaluation score a seed run's models. The invariant that matters: MODEL paths
    move, DATA paths do not -- the zarr stores are shared across seeds."""

    @pytest.fixture(scope='class')
    def seeded(self):
        return build_families(PROJECT_ROOT, experiments_root=PROJECT_ROOT / 'experiments' / 'seed_1')

    def test_defaults_to_the_original_tree(self, families):
        for cfg in families.values():
            assert (PROJECT_ROOT / 'experiments') in cfg['models_dir'].parents

    def test_every_models_dir_moves_under_the_new_root(self, seeded):
        seed_root = PROJECT_ROOT / 'experiments' / 'seed_1'
        for fid, cfg in seeded.items():
            assert seed_root in cfg['models_dir'].parents, fid

    #: The one family whose final_model_path points at the repo root instead of
    #: into experiments/, so it cannot follow an experiments_root redirect. It is
    #: already dead at seed 42 -- neither its models_dir nor its final_model_path
    #: exists, and evaluation reports "Model directory missing" for it on every
    #: run. A leftover from before the ratio sweep existed (no current notebook
    #: produces an un-suffixed *_experience_replay directory). Exempted here
    #: rather than deleted, because removing a family is a separate decision.
    KNOWN_UNREDIRECTABLE = 'mlp_prevyears_monthly_features_incremental_scaler_experience_replay'

    def test_final_model_and_scaler_paths_move_too(self, seeded):
        seed_root = PROJECT_ROOT / 'experiments' / 'seed_1'
        for fid, cfg in seeded.items():
            if fid == self.KNOWN_UNREDIRECTABLE:
                continue
            for key in ('final_model_path', 'final_scaler_path'):
                if cfg.get(key):
                    assert seed_root in Path(cfg[key]).parents, f'{fid}.{key}'

    def test_the_exempted_family_is_still_the_only_one_and_still_dead(self, families):
        """If this fails, either the dead family was fixed/removed (drop the
        exemption) or a NEW root-anchored path crept in (fix it, don't exempt it)."""
        offenders = {
            fid for fid, cfg in families.items()
            if cfg.get('final_model_path')
            and (PROJECT_ROOT / 'experiments') not in Path(cfg['final_model_path']).parents
        }
        assert offenders == {self.KNOWN_UNREDIRECTABLE}
        assert not families[self.KNOWN_UNREDIRECTABLE]['models_dir'].exists()

    def test_dataset_path_does_NOT_move(self, families, seeded):
        """Redirecting the data would send a seed run looking for a zarr store that
        was never copied -- and silently invalidate the disk feature cache."""
        for fid in families:
            assert seeded[fid]['dataset_path'] == families[fid]['dataset_path'], fid
            assert 'seed_1' not in str(seeded[fid]['dataset_path'])

    def test_redirect_changes_nothing_but_the_paths(self, families, seeded):
        assert set(seeded) == set(families)
        for fid, cfg in seeded.items():
            for key in ('short_label', 'label', 'model_template', 'scaler_template',
                        'prep_kind', 'incremental_scaler'):
                assert cfg[key] == families[fid][key], f'{fid}.{key}'


def test_every_family_has_the_required_keys(families):
    for fid, cfg in families.items():
        missing = REQUIRED_KEYS - set(cfg)
        assert not missing, f'{fid} missing {missing}'


def test_prep_kind_is_always_a_known_kind(families):
    from src.eval.features import prepare_features_common  # noqa: F401
    known_kinds = {'baseline_notebook', 'prevyears', 'monthly', 'neighbourhood'}
    for fid, cfg in families.items():
        assert cfg['prep_kind'] in known_kinds, f'{fid} has unknown prep_kind {cfg["prep_kind"]!r}'


def test_dataset_path_is_always_one_of_the_three_known_files(families):
    for fid, cfg in families.items():
        assert cfg['dataset_path'].name in KNOWN_DATASETS, fid


def test_short_label_present_nonempty_and_unique(families):
    labels = [cfg['short_label'] for cfg in families.values()]
    assert all(isinstance(l, str) and l for l in labels)
    assert len(labels) == len(set(labels)), 'short_label must be unique across families'


def test_no_reference_notebook_key_survives(families):
    """Orphaned by the exec-hack deletion (P0); must not have come back."""
    for fid, cfg in families.items():
        assert 'reference_notebook' not in cfg, fid


def test_incremental_scaler_is_true_for_most_families(families):
    values = [cfg['incremental_scaler'] for cfg in families.values()]
    assert sum(values) == 42
    assert sum(not v for v in values) == 5


class TestShortLabelReproducesOldBehaviour:
    """Locks the Phase C migration as output-neutral.

    Before this phase, 38 of 50 families had a hand-written rename in
    row_label_renames (9 of those 38 mapped a key to itself); the other 12 fell
    through to their raw family_id via pandas' Series.replace() no-op-on-missing-key
    behavior. short_label must reproduce exactly that split.
    """

    def test_a_previously_uncovered_family_keeps_its_raw_id(self, families):
        # 'baseline' was one of the 12 families with no entry in row_label_renames.
        assert families['baseline']['short_label'] == 'baseline'

    def test_a_previously_renamed_family_keeps_its_exact_rename(self, families):
        # Spot check against the literal fold-in done during migration: family_id
        # keeps the historical 'missclassification' typo (it names an actual
        # results directory on disk), but the display label was already fixed in
        # row_label_renames -- that fix must survive the fold-in unchanged.
        cfg = families['mlp_prevyears_monthly_features_incremental_scaler_missclassification_buffer_RR_0.3']
        assert cfg['short_label'] == 'mlp_misclassification_buffer_RR_0.3'


def test_open_family_dataset_memoizes_per_path(families, tmp_path):
    """50 families share only 3 dataset_path values; must not reopen per family."""
    ds = xr.Dataset({'x': (('a',), [1, 2, 3])})
    zarr_path = tmp_path / 'shared.zarr'
    ds.to_zarr(zarr_path, mode='w')

    cache = {}
    cfg_a = {'dataset_path': zarr_path}
    cfg_b = {'dataset_path': zarr_path}  # a different family, same path

    handle_a = open_family_dataset(cfg_a, cache)
    handle_b = open_family_dataset(cfg_b, cache)
    assert handle_a is handle_b
    assert len(cache) == 1


def test_open_family_dataset_raises_for_missing_path(tmp_path):
    missing = tmp_path / 'does_not_exist.zarr'
    with pytest.raises(FileNotFoundError):
        open_family_dataset({'dataset_path': missing}, {})
