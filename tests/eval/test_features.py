"""src/eval/features.py has had zero direct tests since it shipped in Phase 2.

Written alongside the fix that made FeatureCache always-on: it was previously gated
by a family-id substring heuristic that excluded 17 of 50 families from caching for
no measured reason. An instrumented real run showed the excluded shape ran
prepare_raw_features_for_year 106 times to cover 6 distinct eval years (17.7x
redundancy) versus 12 times (2.0x, correct: val + test per year) for an included
family of the same shape.
"""

import numpy as np
import pytest

from src.eval.features import FeatureCache, apply_scaler, prepare_features_common


class TestFeatureCacheEnabled:
    def test_default_is_enabled_for_any_family_id(self):
        cache = FeatureCache()
        for fid in ['prevyears_monthly_features', 'baseline', None, 'anything_at_all']:
            assert cache.enabled(fid) is True

    def test_previously_excluded_family_shape_is_now_included(self):
        """The exact family that measurement showed was excluded and shouldn't be."""
        cache = FeatureCache()
        assert cache.enabled('prevyears_monthly_features') is True

    def test_enabled_for_is_overridable(self):
        cache = FeatureCache(enabled_for=lambda fid: fid == 'only_this_one')
        assert cache.enabled('only_this_one') is True
        assert cache.enabled('anything_else') is False


class TestFeatureCacheKey:
    def test_impute_window_distinguishes_otherwise_identical_keys(self):
        """The hardening: same ds/pixel_indices/year/flags, different impute_window
        must not collide, even though no current prep_kind combination triggers this
        in practice -- the key must be correct by construction, not by coincidence.
        """
        ds, pix = object(), object()
        key_expanding = FeatureCache.make_key(ds, pix, 3, True, False, False, 'expanding')
        key_all_years = FeatureCache.make_key(ds, pix, 3, True, False, False, 'all_years')
        assert key_expanding != key_all_years

    def test_key_is_identical_across_calls_with_the_same_inputs(self):
        ds, pix = object(), object()
        k1 = FeatureCache.make_key(ds, pix, 3, True, False, False, 'all_years')
        k2 = FeatureCache.make_key(ds, pix, 3, True, False, False, 'all_years')
        assert k1 == k2

    def test_key_distinguishes_year(self):
        ds, pix = object(), object()
        k1 = FeatureCache.make_key(ds, pix, 3, True, False, False, 'all_years')
        k2 = FeatureCache.make_key(ds, pix, 4, True, False, False, 'all_years')
        assert k1 != k2

    def test_key_distinguishes_flags(self):
        ds, pix = object(), object()
        base = FeatureCache.make_key(ds, pix, 3, True, False, False, 'all_years')
        assert base != FeatureCache.make_key(ds, pix, 3, False, False, False, 'all_years')
        assert base != FeatureCache.make_key(ds, pix, 3, True, True, False, 'all_years')
        assert base != FeatureCache.make_key(ds, pix, 3, True, False, True, 'all_years')

    def test_key_distinguishes_object_identity_not_equality(self):
        """Two distinct pixel-index arrays with equal contents must not collide --
        matching the notebook's actual objects (val_pixel_indices/test_pixel_indices
        are distinct arrays that can be numerically equal on tiny synthetic data)."""
        ds = object()
        pix_a = np.array([1, 2, 3])
        pix_b = np.array([1, 2, 3])
        ka = FeatureCache.make_key(ds, pix_a, 3, True, False, False, 'all_years')
        kb = FeatureCache.make_key(ds, pix_b, 3, True, False, False, 'all_years')
        assert ka != kb


class TestFeatureCacheGetSetClear:
    def test_get_on_missing_key_counts_a_miss(self):
        cache = FeatureCache()
        assert cache.get(('nope',)) is None
        assert cache.stats == {'hits': 0, 'misses': 1}

    def test_set_then_get_counts_a_hit_and_returns_the_payload(self):
        cache = FeatureCache()
        X, y = np.zeros((2, 2)), np.zeros(2)
        cache.set(('k',), X, y)
        payload = cache.get(('k',))
        assert payload is not None
        assert payload[0] is X and payload[1] is y
        assert cache.stats == {'hits': 1, 'misses': 0}

    def test_lru_eviction_at_max_entries(self):
        cache = FeatureCache(max_entries=2)
        cache.set(('a',), 1, 1)
        cache.set(('b',), 2, 2)
        cache.set(('c',), 3, 3)  # evicts 'a', the least-recently-used
        assert cache.get(('a',)) is None
        assert cache.get(('b',)) is not None
        assert cache.get(('c',)) is not None

    def test_get_moves_entry_to_most_recently_used(self):
        cache = FeatureCache(max_entries=2)
        cache.set(('a',), 1, 1)
        cache.set(('b',), 2, 2)
        cache.get(('a',))  # touch 'a' so 'b' becomes the LRU entry
        cache.set(('c',), 3, 3)  # must evict 'b', not 'a'
        assert cache.get(('a',)) is not None
        assert cache.get(('b',)) is None

    def test_clear_resets_entries_and_stats(self):
        cache = FeatureCache()
        cache.set(('a',), 1, 1)
        cache.get(('a',))
        cache.get(('missing',))
        cache.clear()
        assert len(cache.entries) == 0
        assert cache.stats == {'hits': 0, 'misses': 0}


class TestApplyScaler:
    def test_none_mode_returns_unscaled(self):
        X = np.array([[1.0, 2.0]])
        X_out, scaler = apply_scaler(X, None, 'none')
        assert X_out is X
        assert scaler is None

    def test_fit_mode_fits_a_new_scaler_when_none_given(self):
        X = np.array([[1.0], [2.0], [3.0]])
        X_out, scaler = apply_scaler(X, None, 'fit')
        assert scaler is not None
        assert X_out.mean() == pytest.approx(0.0, abs=1e-9)

    def test_transform_mode_requires_a_fitted_scaler(self):
        with pytest.raises(ValueError, match='must be fitted'):
            apply_scaler(np.zeros((2, 2)), None, 'transform')


class TestPrepareFeaturesCommon:
    def test_year_zero_returns_empty(self, synthetic_dataset, synthetic_pixel_indices):
        train_idx, _ = synthetic_pixel_indices
        X, y, scaler = prepare_features_common(synthetic_dataset, train_idx, 0)
        assert X.shape == (0, 0) and y.shape == (0,)

    def test_caching_does_not_change_the_returned_values(self, synthetic_dataset, synthetic_pixel_indices):
        """The behaviour this whole fix is supposed to preserve: a cache hit must
        return numerically identical features to a cache miss, not just faster ones.
        """
        train_idx, _ = synthetic_pixel_indices
        cache = FeatureCache()

        X_cold, y_cold, _ = prepare_features_common(
            synthetic_dataset, train_idx, 2, scaler_mode='none',
            include_last_year=True, include_monthly=False, cache=cache, family_id='f')
        assert cache.stats['misses'] == 1

        X_warm, y_warm, _ = prepare_features_common(
            synthetic_dataset, train_idx, 2, scaler_mode='none',
            include_last_year=True, include_monthly=False, cache=cache, family_id='f')
        assert cache.stats['hits'] == 1

        assert np.array_equal(X_cold, X_warm)
        assert np.array_equal(y_cold, y_warm)

    def test_cache_hit_still_scales_correctly_per_call(self, synthetic_dataset, synthetic_pixel_indices):
        """Scaling happens after the cache lookup, with whatever scaler this call
        was given -- a cache hit must not reuse a *previous call's* scaled output.
        """
        train_idx, _ = synthetic_pixel_indices
        cache = FeatureCache()
        from sklearn.preprocessing import StandardScaler

        X_unscaled, _, _ = prepare_features_common(
            synthetic_dataset, train_idx, 2, scaler_mode='none', cache=cache, family_id='f')

        scaler = StandardScaler().fit(X_unscaled)
        X_scaled, _, _ = prepare_features_common(
            synthetic_dataset, train_idx, 2, scaler=scaler, scaler_mode='transform',
            cache=cache, family_id='f')

        assert cache.stats['hits'] == 1, 'the raw features should have come from cache'
        assert not np.array_equal(X_unscaled, X_scaled), 'but the output must reflect this call''s scaler'
        assert np.allclose(X_scaled, scaler.transform(X_unscaled))

    def test_disabled_cache_never_reuses_across_calls(self, synthetic_dataset, synthetic_pixel_indices):
        train_idx, _ = synthetic_pixel_indices
        cache = FeatureCache(enabled_for=lambda fid: False)
        prepare_features_common(synthetic_dataset, train_idx, 2, scaler_mode='none', cache=cache)
        prepare_features_common(synthetic_dataset, train_idx, 2, scaler_mode='none', cache=cache)
        assert cache.stats == {'hits': 0, 'misses': 0}, 'a disabled cache should not even be queried'

    def test_include_last_year_false_and_true_do_not_share_a_cache_entry(
        self, synthetic_dataset, synthetic_pixel_indices
    ):
        train_idx, _ = synthetic_pixel_indices
        cache = FeatureCache()
        X_with, _, _ = prepare_features_common(
            synthetic_dataset, train_idx, 2, scaler_mode='none', include_last_year=True, cache=cache)
        X_without, _, _ = prepare_features_common(
            synthetic_dataset, train_idx, 2, scaler_mode='none', include_last_year=False, cache=cache)
        assert cache.stats == {'hits': 0, 'misses': 2}
        assert X_with.shape[1] != X_without.shape[1]

    def test_impute_window_variants_do_not_share_a_cache_entry(self, synthetic_dataset, synthetic_pixel_indices):
        train_idx, _ = synthetic_pixel_indices
        cache = FeatureCache()
        prepare_features_common(synthetic_dataset, train_idx, 2, scaler_mode='none',
                                impute_window='expanding', cache=cache)
        prepare_features_common(synthetic_dataset, train_idx, 2, scaler_mode='none',
                                impute_window='all_years', cache=cache)
        assert cache.stats == {'hits': 0, 'misses': 2}

    def test_empty_result_bypasses_scaling(self, synthetic_dataset):
        """A pixel selection that yields zero valid rows must not reach apply_scaler."""
        X, y, scaler_out = prepare_features_common(
            synthetic_dataset, np.array([], dtype=int), 2, scaler_mode='fit')
        assert X.shape == (0, 0)
        assert scaler_out is None

    def test_invalid_scaler_mode_raises(self, synthetic_dataset, synthetic_pixel_indices):
        train_idx, _ = synthetic_pixel_indices
        with pytest.raises(ValueError, match='Invalid scaler_mode'):
            prepare_features_common(synthetic_dataset, train_idx, 2, scaler_mode='bogus')
