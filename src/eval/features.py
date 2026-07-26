"""Feature preparation for the unified evaluation pipeline.

This is the wrapper that Evaluations.ipynb used to carry inline as
`_prepare_features_common`: an LRU cache plus per-family flag dispatch around the
feature math. The math itself now comes from src/mlp_replay/data.py, so evaluation
and training can no longer drift apart -- the failure `to do.md` describes as a
"Ghost Hunt", where a column-order mismatch looks like a performance drop.

Two contracts from the notebook are preserved deliberately:

* `impute_window` defaults to 'all_years', the leaky full-window S2 imputation the
  notebook has always used. Fixing it is a separate, reviewable change.
* Empty results return a (0, 0) matrix rather than (0, n_features), because callers
  test `len(X) == 0` and the notebook's validators expect that shape.
"""

from collections import OrderedDict

import numpy as np
from sklearn.preprocessing import StandardScaler

from src.mlp_replay.data import prepare_raw_features_for_year

VALID_SCALER_MODES = {'auto', 'fit', 'partial_fit', 'transform', 'none'}
FEATURE_CACHE_MAX_ENTRIES = 128


class FeatureCache:
    """Bounded LRU cache of raw (unscaled) feature matrices.

    Keyed on object identity of `ds`/`pixel_indices` plus the flags, matching the
    notebook: within one evaluation run those objects are reused, and hashing the
    arrays themselves would cost more than recomputing.

    Caching is on by default for every family. It used to be gated by a family-id
    substring match ("only the incremental-scaler monthly families reuse identical
    inputs often enough to be worth caching"), which was never measured. Instrumented:
    a family excluded by that heuristic ran prepare_raw_features_for_year 106 times to
    cover 6 distinct eval years -- a 17.7x redundancy factor -- while an included
    family with the same shape ran it 12 times for the same 6 years (2.0x: val + test
    per year, which is correct). Every family that shares val/test features across
    multiple model_years within evaluate_family's own_year/prior_years/next_year/
    cumulative/final_model_each_year tables benefits identically; there was no
    family for which the exclusion was actually earning its keep. `enabled_for`
    stays overridable (e.g. to disable caching in a test) but no longer has a
    family-shaped default.
    """

    def __init__(self, max_entries=FEATURE_CACHE_MAX_ENTRIES, enabled_for=None):
        self.entries = OrderedDict()
        self.max_entries = max_entries
        self.stats = {'hits': 0, 'misses': 0}
        self.enabled_for = enabled_for or (lambda family_id: True)

    def enabled(self, family_id):
        return bool(self.enabled_for(family_id))

    @staticmethod
    def make_key(ds, pixel_indices, year_idx, include_last_year, include_monthly, include_neighbourhood,
                 impute_window):
        # impute_window matters here even though no two prep kinds currently collide
        # without it (each prep_kind's (include_last_year, include_monthly,
        # include_neighbourhood) tuple happens to be unique, and impute_window is
        # determined by prep_kind). That's a coincidence of the current family
        # configs, not a guarantee -- a future prep kind that reused an existing
        # flag tuple with a different impute_window would silently get another
        # kind's cached (and wrongly imputed) features. Including it makes the key
        # correct by construction instead of correct by luck.
        return (
            id(ds),
            id(pixel_indices),
            int(year_idx),
            bool(include_last_year),
            bool(include_monthly),
            bool(include_neighbourhood),
            str(impute_window),
        )

    def get(self, key):
        payload = self.entries.get(key)
        if payload is None:
            self.stats['misses'] += 1
            return None
        self.entries.move_to_end(key)
        self.stats['hits'] += 1
        return payload

    def set(self, key, X_raw, y_raw):
        self.entries[key] = (X_raw, y_raw)
        self.entries.move_to_end(key)
        if len(self.entries) > self.max_entries:
            self.entries.popitem(last=False)

    def clear(self):
        self.entries.clear()
        self.stats = {'hits': 0, 'misses': 0}


FEATURE_CACHE = FeatureCache()


def apply_scaler(X, scaler, scaler_mode):
    """Scale in place of the notebook's inline branch; returns (X, scaler)."""
    if scaler_mode == 'none':
        return X, scaler

    if scaler is None:
        scaler = StandardScaler()

    if scaler_mode == 'auto':
        if hasattr(scaler, 'n_features_in_'):
            X = scaler.transform(X)
        else:
            X = scaler.fit_transform(X)
    elif scaler_mode == 'fit':
        X = scaler.fit_transform(X)
    elif scaler_mode == 'partial_fit':
        scaler.partial_fit(X)
        X = scaler.transform(X)
    elif scaler_mode == 'transform':
        if not hasattr(scaler, 'n_features_in_'):
            raise ValueError("Scaler must be fitted before scaler_mode='transform'.")
        X = scaler.transform(X)

    return X, scaler


def prepare_features_common(
    ds,
    pixel_indices,
    year_idx,
    scaler=None,
    scaler_mode='auto',
    include_last_year=True,
    include_monthly=False,
    include_neighbourhood=False,
    family_id=None,
    impute_window='all_years',
    cache=None,
):
    """Build one evaluation year's features, cached per family.

    `include_monthly` defaults to False here (not None as in the module) because
    evaluation always states a family's feature set explicitly rather than inferring
    it from whatever dataset happens to be loaded -- a model must be scored on the
    columns it was trained with.
    """
    if year_idx == 0:
        return np.empty((0, 0)), np.empty((0,)), scaler

    if scaler_mode not in VALID_SCALER_MODES:
        raise ValueError(f'Invalid scaler_mode: {scaler_mode}')

    cache = FEATURE_CACHE if cache is None else cache
    use_cache = cache.enabled(family_id)
    cache_key = None
    payload = None

    if use_cache:
        cache_key = cache.make_key(ds, pixel_indices, year_idx, include_last_year,
                                   include_monthly, include_neighbourhood, impute_window)
        payload = cache.get(cache_key)

    if payload is not None:
        X, y = payload
    else:
        X, y = prepare_raw_features_for_year(
            ds,
            pixel_indices,
            year_idx,
            include_last_year=include_last_year,
            include_monthly=include_monthly,
            include_neighbourhood=include_neighbourhood,
            impute_window=impute_window,
            dtype=None,
        )
        if use_cache and cache_key is not None:
            cache.set(cache_key, X, y)

    if len(X) == 0:
        return np.empty((0, 0)), np.empty((0,)), scaler

    X, scaler = apply_scaler(X, scaler, scaler_mode)
    return X, y, scaler
