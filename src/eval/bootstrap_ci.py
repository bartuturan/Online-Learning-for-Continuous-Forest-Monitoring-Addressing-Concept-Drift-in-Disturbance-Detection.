"""Cube-level bootstrap confidence intervals for the evaluation metrics.

The published tables in experiments/evaluation/eval_outputs/unified_eval/ report one
F1 and one PR-AUC per (family, model_year, eval_year) with no uncertainty attached.
Margins between strategies there can be as small as ~0.006, and nothing in those
tables says whether a margin that size would survive a different sample of forest.
This module answers that.

Two design decisions carry the statistics, and both are about the sampling unit:

* **Resampling is over cubes, not pixels.** The test split's 1.28M pixels come from
  297 cubes sampled on a 7x7 grid, so pixels inside a cube are spatially
  autocorrelated -- they are not independent draws. A pixel bootstrap would treat
  each as its own trial and report an interval far narrower than the real sampling
  variability. Cubes are the unit the sampling design actually independently drew,
  so cubes are the unit resampled here.

* **Strategy-vs-baseline uses a paired bootstrap.** Two separate intervals, checked
  for overlap by eye, is a well-known way to get the wrong answer: overlap does not
  imply the difference is insignificant. `paired_bootstrap_margin` draws one cube
  resample per iteration and scores both arms on that same draw, so the correlation
  between the two models' errors on shared cubes is preserved and the interval lands
  directly on the quantity of interest -- the margin.

Both metrics are computed without ever materialising a resampled pixel array, which
is what makes ~2000 resamples over 1.28M rows tractable:

* F1 at a fixed threshold is a function of summed TP/FP/FN, so per-cube counts are
  reduced once and a resample only sums 297 numbers (`cube_confusion_counts`).
* PR-AUC is not decomposable that way -- it depends on the global score ranking --
  so `bootstrap_pr_auc_ci` sorts by score once, reduces to counts per (score-rank
  bin, cube), and turns each resample into a matmul against that fixed table. On the
  real test split this is the difference between roughly 160s and 1s per interval;
  at ~830 configurations the unbinned version does not finish. The point estimate is
  still computed on the exact curve (`_weighted_average_precision`), so only the
  interval width depends on the binning, and `DEFAULT_PR_AUC_BINS` is set high enough
  that small inputs are binned one-row-per-bin and come out exact.

Nothing here loads a model or touches zarr; inputs are plain arrays. That keeps the
statistics testable against hand-computable cases, which is where the correctness
risk actually lives.
"""

import re

import numpy as np

#: Matches the strategy families in scope: every MLP experience-replay variant
#: (including the misclassification-buffer one, whose id spells it
#: 'missclassification') and every MLP combined-strategy family. Deliberately a
#: pattern over the registry rather than a pasted list of ~35 ids -- the registry in
#: src/eval/families.py is the source of truth for what exists, and a hardcoded list
#: silently goes stale the next time a sweep adds a config.
DEFAULT_STRATEGY_PATTERN = r'^mlp_(combined_|prevyears_monthly_features_incremental_scaler_(experience_replay|missclassification_buffer))'

#: The arm every strategy is compared against: the MLP without any replay strategy.
#: Every family matched by DEFAULT_STRATEGY_PATTERN is this model plus a replay
#: mechanism, same prep_kind and same incremental-scaler policy, so it is the
#: like-for-like control.
DEFAULT_BASELINE_FAMILY = 'mlp_prevyears_monthly_features_incremental_scaler'


def select_families(families, baseline_id=DEFAULT_BASELINE_FAMILY, strategy_pattern=DEFAULT_STRATEGY_PATTERN):
    """Resolve the baseline id and the strategy ids to compare against it.

    Raises rather than silently returning an empty comparison set: a pattern that
    matches nothing means the run would do a great deal of inference and then produce
    no margins at all, which is worth finding out about before the inference, not
    after.
    """
    if baseline_id not in families:
        raise KeyError(
            f'Baseline family {baseline_id!r} is not in the registry. '
            f'Available ids matching "baseline": '
            f'{sorted(f for f in families if "baseline" in f.lower()) or "(none)"}'
        )

    pattern = re.compile(strategy_pattern)
    strategy_ids = sorted(f for f in families if f != baseline_id and pattern.search(f))

    if not strategy_ids:
        raise ValueError(
            f'Strategy pattern {strategy_pattern!r} matched no families in a registry of '
            f'{len(families)}. Nothing would be compared against the baseline.'
        )

    return baseline_id, strategy_ids


def cube_confusion_counts(y_true, y_pred, cube_idx, cube_ids=None):
    """Per-cube (TP, FP, FN) counts, plus the cube id order they are indexed by.

    F1 at a fixed threshold depends on the data only through these three sums, and
    sums decompose over any partition of the rows. So reducing to one row per cube
    here is exact, not an approximation, and it collapses the per-resample cost from
    1.28M pixels to 297 numbers.

    `cube_ids` pins the row order when it must agree across two arms of a paired
    comparison; when omitted the sorted distinct cubes present are used.
    """
    y_true = np.asarray(y_true).astype(bool)
    y_pred = np.asarray(y_pred).astype(bool)
    cube_idx = np.asarray(cube_idx)

    if not (len(y_true) == len(y_pred) == len(cube_idx)):
        raise ValueError(
            f'y_true/y_pred/cube_idx must be aligned, got lengths '
            f'{len(y_true)}, {len(y_pred)}, {len(cube_idx)}'
        )

    if cube_ids is None:
        cube_ids = np.unique(cube_idx)
    else:
        cube_ids = np.asarray(cube_ids)

    # Positions into cube_ids; np.searchsorted needs cube_ids sorted, which np.unique
    # guarantees and an explicit cube_ids is checked for below.
    if not np.all(np.diff(cube_ids) > 0):
        raise ValueError('cube_ids must be sorted and unique')
    # searchsorted returns len(cube_ids) for a value above every id, so clip before
    # indexing -- the comparison below still catches it as absent.
    positions = np.searchsorted(cube_ids, cube_idx)
    if np.any(cube_ids[np.clip(positions, 0, len(cube_ids) - 1)] != cube_idx):
        missing = np.setdiff1d(cube_idx, cube_ids)
        raise ValueError(f'cube_idx contains ids absent from cube_ids: {missing[:5]}')

    n = len(cube_ids)
    tp = np.bincount(positions[y_true & y_pred], minlength=n)
    fp = np.bincount(positions[~y_true & y_pred], minlength=n)
    fn = np.bincount(positions[y_true & ~y_pred], minlength=n)

    return tp.astype(np.int64), fp.astype(np.int64), fn.astype(np.int64), cube_ids


def f1_from_counts(tp, fp, fn):
    """F1 from summed counts, with the degenerate all-zero case defined as 0.

    Matches sklearn's `zero_division=0`, which is what src/eval/metrics.py scores
    with, so a bootstrap point estimate is comparable to the published number.
    """
    tp = np.asarray(tp, dtype=np.float64)
    fp = np.asarray(fp, dtype=np.float64)
    fn = np.asarray(fn, dtype=np.float64)
    denominator = 2 * tp + fp + fn
    return np.divide(2 * tp, denominator, out=np.zeros_like(denominator), where=denominator > 0)


def draw_cube_resamples(n_cubes, n_bootstrap, rng):
    """`n_bootstrap` x `n_cubes` matrix of per-cube draw multiplicities.

    Returning multiplicities rather than lists of drawn indices is what lets both
    metrics stay vectorised: a multiplicity vector is exactly the per-cube weight the
    resample implies, so F1 becomes a weighted sum of counts and PR-AUC a weighted
    sweep, with no row duplication anywhere.
    """
    draws = rng.integers(0, n_cubes, size=(n_bootstrap, n_cubes))
    # axis=1 bincount, done as a flat bincount with per-row offsets.
    offsets = (np.arange(n_bootstrap) * n_cubes)[:, None]
    flat = (draws + offsets).ravel()
    counts = np.bincount(flat, minlength=n_bootstrap * n_cubes)
    return counts.reshape(n_bootstrap, n_cubes)


def _percentile_ci(samples, alpha):
    lower = float(np.percentile(samples, 100 * alpha / 2))
    upper = float(np.percentile(samples, 100 * (1 - alpha / 2)))
    return lower, upper


def _bca_ci(samples, point_estimate, jackknife_values, alpha):
    """Bias-corrected and accelerated interval.

    Worth the extra machinery over a plain percentile interval here because F1 and
    PR-AUC on imbalanced data have skewed sampling distributions, where the
    percentile interval is visibly off-centre. Falls back to the percentile interval
    when the correction is undefined (no spread in the jackknife, or a point estimate
    outside the resample range), rather than emitting a nan interval.
    """
    from scipy.stats import norm

    samples = np.asarray(samples, dtype=np.float64)
    proportion_below = float(np.mean(samples < point_estimate))
    if proportion_below <= 0.0 or proportion_below >= 1.0:
        return _percentile_ci(samples, alpha)

    z0 = norm.ppf(proportion_below)

    jackknife_values = np.asarray(jackknife_values, dtype=np.float64)
    jackknife_mean = jackknife_values.mean()
    deviations = jackknife_mean - jackknife_values
    denominator = 6.0 * (np.sum(deviations ** 2) ** 1.5)
    if denominator == 0.0:
        return _percentile_ci(samples, alpha)
    acceleration = float(np.sum(deviations ** 3) / denominator)

    z_alpha_low = norm.ppf(alpha / 2)
    z_alpha_high = norm.ppf(1 - alpha / 2)

    def _adjusted(z):
        adjusted = z0 + (z0 + z) / (1 - acceleration * (z0 + z))
        return float(np.clip(norm.cdf(adjusted) * 100, 0, 100))

    lower = float(np.percentile(samples, _adjusted(z_alpha_low)))
    upper = float(np.percentile(samples, _adjusted(z_alpha_high)))
    return lower, upper


def summarise(samples, point_estimate, alpha=0.05, method='percentile', jackknife_values=None):
    """Package a resample distribution into the fields written to the result tables."""
    samples = np.asarray(samples, dtype=np.float64)
    if method == 'bca':
        if jackknife_values is None:
            raise ValueError("method='bca' requires jackknife_values")
        lower, upper = _bca_ci(samples, point_estimate, jackknife_values, alpha)
    elif method == 'percentile':
        lower, upper = _percentile_ci(samples, alpha)
    else:
        raise ValueError(f'Unknown CI method: {method!r}')

    return {
        'point_estimate': float(point_estimate),
        'ci_lower': lower,
        'ci_upper': upper,
        'bootstrap_mean': float(samples.mean()),
        'bootstrap_std': float(samples.std(ddof=1)) if len(samples) > 1 else 0.0,
        'n_bootstrap': int(len(samples)),
        'ci_method': method,
        'ci_alpha': float(alpha),
    }


def bootstrap_f1_ci(y_true, y_pred, cube_idx, n_bootstrap=2000, alpha=0.05,
                    method='percentile', rng=None, cube_ids=None, weights=None):
    """Cube-level bootstrap CI for F1 at an already-fixed threshold.

    `weights` accepts a precomputed multiplicity matrix so a paired comparison can
    hand both arms the identical draws; left None it draws its own.
    """
    rng = np.random.default_rng() if rng is None else rng
    tp, fp, fn, cube_ids = cube_confusion_counts(y_true, y_pred, cube_idx, cube_ids=cube_ids)

    point_estimate = f1_from_counts(tp.sum(), fp.sum(), fn.sum())

    if weights is None:
        weights = draw_cube_resamples(len(cube_ids), n_bootstrap, rng)

    samples = f1_from_counts(weights @ tp, weights @ fp, weights @ fn)

    jackknife_values = None
    if method == 'bca':
        # Leave-one-cube-out, computed by subtraction rather than by re-reducing.
        jackknife_values = f1_from_counts(tp.sum() - tp, fp.sum() - fp, fn.sum() - fn)

    return summarise(samples, point_estimate, alpha=alpha, method=method,
                     jackknife_values=jackknife_values)


def _weighted_average_precision(y_true_sorted, weights_sorted):
    """Average precision under per-row weights, along an already-sorted score order.

    This is the step-wise sum used by sklearn's `average_precision_score`:
    sum over thresholds of (recall_k - recall_{k-1}) * precision_k, evaluated at
    every distinct score. Weighting rows by their cube's draw multiplicity is exactly
    what resampling cubes does to the curve, so this reproduces a resampled PR-AUC
    without re-sorting or rebuilding the array.

    `weights_sorted` may be a (n_resamples, n_rows) matrix; the sweep is vectorised
    across rows of it.
    """
    y_true_sorted = np.asarray(y_true_sorted).astype(bool)
    weights_sorted = np.atleast_2d(np.asarray(weights_sorted, dtype=np.float64))

    weighted_positive = weights_sorted * y_true_sorted
    cumulative_tp = np.cumsum(weighted_positive, axis=1)
    cumulative_all = np.cumsum(weights_sorted, axis=1)

    total_positive = cumulative_tp[:, -1:]

    precision = np.divide(cumulative_tp, cumulative_all,
                          out=np.zeros_like(cumulative_tp), where=cumulative_all > 0)
    recall = np.divide(cumulative_tp, total_positive,
                       out=np.zeros_like(cumulative_tp), where=total_positive > 0)

    recall_increment = np.diff(recall, axis=1, prepend=0.0)
    average_precision = np.sum(recall_increment * precision, axis=1)

    # An arm with no positives left in the resample has an undefined curve; report 0
    # rather than nan so a downstream percentile is still computable.
    average_precision = np.where(total_positive.ravel() > 0, average_precision, 0.0)
    return average_precision


#: Score-rank bins used to evaluate resampled PR curves. The sweep only needs the
#: order of scores, and average precision varies smoothly along it, so grouping
#: adjacent ranks costs almost nothing in accuracy while making each resample a
#: matmul against a (n_bins, n_cubes) table instead of a pass over every row. At
#: 1.28M test rows that is the difference between ~160s and ~1s per interval --
#: without it a full sweep over the reported configurations does not finish.
#: Set at or above the row count, every row lands in its own bin and the result is
#: exact, which is what small inputs (and the tests) get automatically.
DEFAULT_PR_AUC_BINS = 65536


def _bin_and_cube_counts(y_true_sorted, positions_sorted, n_cubes, n_bins):
    """Positive and total counts per (score-rank bin, cube), in descending score order.

    Rows are binned by rank rather than by score value, so bins hold equal counts and
    no bin can be empty-but-for-ties. Grouping a run of adjacent ranks is the same
    operation average precision already applies to tied scores.
    """
    n_rows = len(y_true_sorted)
    n_bins = min(int(n_bins), n_rows)
    bin_of_row = (np.arange(n_rows) * n_bins) // n_rows

    combined = bin_of_row * n_cubes + positions_sorted
    size = n_bins * n_cubes
    all_counts = np.bincount(combined, minlength=size).reshape(n_bins, n_cubes)
    pos_counts = np.bincount(combined[y_true_sorted], minlength=size).reshape(n_bins, n_cubes)
    return pos_counts.astype(np.float64), all_counts.astype(np.float64)


def _average_precision_from_bin_counts(weights, pos_counts, all_counts):
    """Average precision per resample, from per-(bin, cube) counts.

    `weights` is (n_resamples, n_cubes) draw multiplicities. The two matmuls turn each
    resample into its own weighted PR curve over bins; the rest is the same step-wise
    sum as the exact sweep.
    """
    weights = np.atleast_2d(np.asarray(weights, dtype=np.float64))

    cumulative_tp = np.cumsum(weights @ pos_counts.T, axis=1)
    cumulative_all = np.cumsum(weights @ all_counts.T, axis=1)
    total_positive = cumulative_tp[:, -1:]

    precision = np.divide(cumulative_tp, cumulative_all,
                          out=np.zeros_like(cumulative_tp), where=cumulative_all > 0)
    recall = np.divide(cumulative_tp, total_positive,
                       out=np.zeros_like(cumulative_tp), where=total_positive > 0)

    average_precision = np.sum(np.diff(recall, axis=1, prepend=0.0) * precision, axis=1)
    return np.where(total_positive.ravel() > 0, average_precision, 0.0)


def pr_auc_bin_table(y_true, y_score, cube_idx, cube_ids=None, n_bins=DEFAULT_PR_AUC_BINS):
    """Everything about one arm that repeated resampling needs, computed once.

    Sorting 1.28M rows and reducing them to per-(bin, cube) counts is the expensive
    half of a PR-AUC interval; the resampling itself is a matmul. Exposing the table
    lets a caller build it once and reuse it -- which matters most for the baseline
    arm, since every strategy is compared against the same one at each (model_year,
    eval_year) and would otherwise rebuild it from scratch every time.
    """
    y_true = np.asarray(y_true).astype(bool)
    y_score = np.asarray(y_score, dtype=np.float64)
    cube_idx = np.asarray(cube_idx)

    cube_ids = np.unique(cube_idx) if cube_ids is None else np.asarray(cube_ids)
    positions = np.searchsorted(cube_ids, cube_idx)

    order = np.argsort(-y_score, kind='stable')
    y_true_sorted = y_true[order]

    pos_counts, all_counts = _bin_and_cube_counts(
        y_true_sorted, positions[order], len(cube_ids), n_bins)

    return {
        'pos_counts': pos_counts,
        'all_counts': all_counts,
        'cube_ids': cube_ids,
        # Exact, unbinned -- this is the number reported beside the interval.
        'point_estimate': float(_weighted_average_precision(
            y_true_sorted, np.ones(len(y_true_sorted)))[0]),
    }


def bootstrap_pr_auc_ci(y_true=None, y_score=None, cube_idx=None, n_bootstrap=2000,
                        alpha=0.05, method='percentile', rng=None, cube_ids=None,
                        weights=None, n_bins=DEFAULT_PR_AUC_BINS, table=None):
    """Cube-level bootstrap CI for PR-AUC (average precision).

    Unlike F1, PR-AUC cannot be reduced to per-cube counts -- it depends on how every
    row ranks against every other. So the rows are sorted once, reduced once to counts
    per (score-rank bin, cube), and every resample after that is a matmul against that
    fixed table. See DEFAULT_PR_AUC_BINS for why the binning is what makes a full
    sweep tractable.

    The point estimate is always computed on the exact, unbinned curve, so the value
    reported next to each interval is the same number the evaluation pipeline
    published -- only the interval's width comes from the binned approximation.

    Pass `table` from `pr_auc_bin_table` to reuse an already-reduced arm instead of
    rebuilding it.
    """
    rng = np.random.default_rng() if rng is None else rng

    if table is None:
        table = pr_auc_bin_table(y_true, y_score, cube_idx, cube_ids=cube_ids, n_bins=n_bins)
    cube_ids = table['cube_ids']
    pos_counts, all_counts = table['pos_counts'], table['all_counts']

    if weights is None:
        weights = draw_cube_resamples(len(cube_ids), n_bootstrap, rng)

    samples = _average_precision_from_bin_counts(weights, pos_counts, all_counts)

    jackknife_values = None
    if method == 'bca':
        leave_one_out = np.ones((len(cube_ids), len(cube_ids)), dtype=np.float64)
        np.fill_diagonal(leave_one_out, 0.0)
        jackknife_values = _average_precision_from_bin_counts(
            leave_one_out, pos_counts, all_counts)

    return summarise(samples, table['point_estimate'], alpha=alpha, method=method,
                     jackknife_values=jackknife_values)


def paired_bootstrap_margin(strategy, baseline, cube_idx, metric,
                            strategy_threshold=None, baseline_threshold=None,
                            n_bootstrap=2000, alpha=0.05, method='percentile',
                            rng=None, cube_ids=None, n_bins=DEFAULT_PR_AUC_BINS,
                            strategy_table=None, baseline_table=None):
    """CI on (strategy - baseline) from a single shared sequence of cube resamples.

    This is the estimate the whole module exists for. Scoring both arms on the *same*
    draw keeps the two error patterns correlated exactly as they are in the real test
    set -- cubes that are hard for one model are usually hard for the other -- so the
    variance of the difference is far smaller than the variance of either arm, and a
    margin can be resolved that two separate intervals would leave looking ambiguous.

    `strategy` and `baseline` are each (y_true, y_score) for the same test rows in the
    same order. `metric` is 'f1' or 'pr_auc'.

    Each arm carries its **own** threshold, because that is how the published tables
    score them: every model's threshold was tuned separately on the validation split.
    Forcing one shared threshold would compare a tuned model against a detuned one and
    attribute the handicap to the strategy.

    Also reports two bootstrap tail proportions. Neither is the output of a parametric
    test; read them as "in this fraction of resampled forests the margin reversed".

    `p_value_one_sided` is the share of resamples where the strategy did not beat the
    baseline. It is computed as (1 + losses) / (n_bootstrap + 1) rather than
    losses / n_bootstrap so that a strategy winning every single resample reports
    1/(B+1) instead of a bare 0. B resamples cannot resolve a tail finer than that, and
    a printed p = 0 claims they can -- which is not a number a thesis can defend. This
    matches `two_sided_p` in scripts/run_pairwise_ci.py, which already used the floored
    form.

    `p_value_two_sided` doubles the smaller of the two floored tails, capped at 1. It
    is the one the published tables correct with BH, because a one-sided p in the
    "strategy beats baseline" direction cannot flag a margin that is large and
    *negative*: such a row reports p close to 1 by construction, which then contradicts
    its own interval. Both tails carry the same floor, so the doubled value bottoms out
    at 2/(B+1) rather than at twice an unfloored zero.
    """
    rng = np.random.default_rng() if rng is None else rng

    strategy_y_true, strategy_score = strategy
    baseline_y_true, baseline_score = baseline

    strategy_y_true = np.asarray(strategy_y_true)
    baseline_y_true = np.asarray(baseline_y_true)
    if not np.array_equal(strategy_y_true, baseline_y_true):
        raise ValueError(
            'Paired bootstrap requires both arms to be scored on identical rows in '
            'identical order, but their y_true arrays differ. Pairing mismatched rows '
            'would silently compare different pixels.'
        )

    cube_idx = np.asarray(cube_idx)
    if cube_ids is None:
        cube_ids = np.unique(cube_idx)
    weights = draw_cube_resamples(len(cube_ids), n_bootstrap, rng)

    if metric == 'f1':
        if strategy_threshold is None or baseline_threshold is None:
            raise ValueError("metric='f1' requires strategy_threshold and baseline_threshold")
        strategy_pred = np.asarray(strategy_score) >= strategy_threshold
        baseline_pred = np.asarray(baseline_score) >= baseline_threshold

        strategy_result = bootstrap_f1_ci(
            strategy_y_true, strategy_pred, cube_idx,
            alpha=alpha, method=method, cube_ids=cube_ids, weights=weights)
        baseline_result = bootstrap_f1_ci(
            baseline_y_true, baseline_pred, cube_idx,
            alpha=alpha, method=method, cube_ids=cube_ids, weights=weights)

        tp_s, fp_s, fn_s, _ = cube_confusion_counts(
            strategy_y_true, strategy_pred, cube_idx, cube_ids=cube_ids)
        tp_b, fp_b, fn_b, _ = cube_confusion_counts(
            baseline_y_true, baseline_pred, cube_idx, cube_ids=cube_ids)
        margin_samples = (f1_from_counts(weights @ tp_s, weights @ fp_s, weights @ fn_s)
                          - f1_from_counts(weights @ tp_b, weights @ fp_b, weights @ fn_b))

        margin_jackknife = None
        if method == 'bca':
            # Leave-one-cube-out on the *margin*, not on either arm: the acceleration
            # term has to describe the skewness of the quantity being interval-ed.
            # Both arms drop the same cube, which is what keeps it paired.
            margin_jackknife = (
                f1_from_counts(tp_s.sum() - tp_s, fp_s.sum() - fp_s, fn_s.sum() - fn_s)
                - f1_from_counts(tp_b.sum() - tp_b, fp_b.sum() - fp_b, fn_b.sum() - fn_b))

    elif metric == 'pr_auc':
        # Each arm's table is built at most once here and reused for both its own
        # interval and the margin -- the naive version rebuilds each of them twice.
        if strategy_table is None:
            strategy_table = pr_auc_bin_table(
                strategy_y_true, strategy_score, cube_idx, cube_ids=cube_ids, n_bins=n_bins)
        if baseline_table is None:
            baseline_table = pr_auc_bin_table(
                baseline_y_true, baseline_score, cube_idx, cube_ids=cube_ids, n_bins=n_bins)

        strategy_samples = _average_precision_from_bin_counts(
            weights, strategy_table['pos_counts'], strategy_table['all_counts'])
        baseline_samples = _average_precision_from_bin_counts(
            weights, baseline_table['pos_counts'], baseline_table['all_counts'])

        margin_samples = strategy_samples - baseline_samples

        arm_jackknife = {}
        margin_jackknife = None
        if method == 'bca':
            leave_one_out = np.ones((len(cube_ids), len(cube_ids)), dtype=np.float64)
            np.fill_diagonal(leave_one_out, 0.0)
            arm_jackknife['strategy'] = _average_precision_from_bin_counts(
                leave_one_out, strategy_table['pos_counts'], strategy_table['all_counts'])
            arm_jackknife['baseline'] = _average_precision_from_bin_counts(
                leave_one_out, baseline_table['pos_counts'], baseline_table['all_counts'])
            margin_jackknife = arm_jackknife['strategy'] - arm_jackknife['baseline']

        strategy_result = summarise(strategy_samples, strategy_table['point_estimate'],
                                    alpha=alpha, method=method,
                                    jackknife_values=arm_jackknife.get('strategy'))
        baseline_result = summarise(baseline_samples, baseline_table['point_estimate'],
                                    alpha=alpha, method=method,
                                    jackknife_values=arm_jackknife.get('baseline'))
    else:
        raise ValueError(f'Unknown metric: {metric!r}')

    point_margin = strategy_result['point_estimate'] - baseline_result['point_estimate']
    summary = summarise(margin_samples, point_margin, alpha=alpha, method=method,
                        jackknife_values=margin_jackknife)
    margin_samples = np.asarray(margin_samples)
    summary.update({
        'strategy_point_estimate': strategy_result['point_estimate'],
        'baseline_point_estimate': baseline_result['point_estimate'],
        **margin_tail_p_values(margin_samples),
        'metric': metric,
    })
    return summary


def margin_tail_p_values(margin_samples):
    """Floored bootstrap tail proportions for one margin's resamples.

    Returns {'p_value_one_sided', 'p_value_two_sided'}. Both tails are floored at
    1/(B+1) -- see paired_bootstrap_margin for why -- and the two-sided value doubles
    the smaller of them, capped at 1. Ties (a resample where the margin is exactly 0)
    count against both directions, which is the conservative reading.
    """
    margin_samples = np.asarray(margin_samples)
    n_boot = len(margin_samples)
    p_lo = (1 + np.sum(margin_samples <= 0)) / (n_boot + 1)   # strategy did not win
    p_hi = (1 + np.sum(margin_samples >= 0)) / (n_boot + 1)   # strategy did not lose
    return {
        'p_value_one_sided': float(p_lo),
        'p_value_two_sided': float(min(1.0, 2 * min(p_lo, p_hi))),
    }


def benjamini_hochberg(p_values):
    """BH-adjusted q-values, in the input order.

    Comparing ~35 strategies across three table types and six years is hundreds of
    simultaneous margins; at alpha=0.05 roughly one in twenty null comparisons clears
    the bar by chance alone. This is offered so a thesis claim about "which strategies
    beat baseline" can be made against a controlled false-discovery rate instead of
    against raw per-comparison tails.
    """
    p_values = np.asarray(p_values, dtype=np.float64)
    n = len(p_values)
    if n == 0:
        return np.empty(0, dtype=np.float64)

    order = np.argsort(p_values)
    ranks = np.arange(1, n + 1)
    adjusted = p_values[order] * n / ranks
    # Enforce monotonicity from the largest p downward.
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0.0, 1.0)

    result = np.empty(n, dtype=np.float64)
    result[order] = adjusted
    return result


def holm_bonferroni(p_values):
    """Holm-adjusted p-values, in the input order.

    Step-down Bonferroni: sort ascending, multiply the i-th smallest by (n-i+1), then
    enforce monotonicity with a forward running max. Controls family-wise error --
    "with probability >= 1-alpha, every rejection in the family is a real effect" --
    which is the guarantee that matches a claim made about a *named* comparison, as
    opposed to BH's average-false-discovery-rate guarantee over the whole family.
    Requires no independence or PRDS assumption on the p-values, unlike BH, and is
    uniformly at least as powerful as a flat Bonferroni bar (alpha/n for every test):
    only the smallest p pays the full n-fold penalty, and each subsequent one is
    checked against a bar that has relaxed because fewer hypotheses remain live.
    """
    p_values = np.asarray(p_values, dtype=np.float64)
    n = len(p_values)
    if n == 0:
        return np.empty(0, dtype=np.float64)

    order = np.argsort(p_values)
    ranks = np.arange(1, n + 1)
    adjusted = p_values[order] * (n - ranks + 1)
    # Enforce monotonicity from the smallest p upward.
    adjusted = np.maximum.accumulate(adjusted)
    adjusted = np.clip(adjusted, 0.0, 1.0)

    result = np.empty(n, dtype=np.float64)
    result[order] = adjusted
    return result
