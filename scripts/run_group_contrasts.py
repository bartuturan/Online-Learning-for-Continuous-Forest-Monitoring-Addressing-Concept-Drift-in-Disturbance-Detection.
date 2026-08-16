"""Planned group contrasts: the primary inferential test for the replay comparison.

scripts/run_pairwise_ci.py asks "is strategy A different from strategy B?" for all 21
pairs among the seven reported strategies. That is the wrong question for what
Chapter 4 actually argues. The text repeatedly declines to rank individual strategies
-- "they are best read as a single indistinguishable group", "these five are better
treated as a band than as a ranking" -- and instead makes claims about *kinds* of
memory curation: that the presence of replay matters more than its criterion, that
error-driven selection is the one mechanism that reliably fails, that blending
criteria buys nothing over committing to one. Testing 21 pairs to support four
statements about groups spends the multiplicity budget on comparisons no claim
depends on, and leaves the claims themselves untested.

This script tests the claims directly. Arms are partitioned into groups by
*mechanism*, and a small set of planned contrasts between group means carries the
argument:

  G0  no replay            Baseline
  G1  uncurated replay     Uniform Random
  G2  curated non-error    PR Stratification, Confidently Correct, Uncertainty Prior.
  G3  error-driven         Hard Example Mining, Misclassification Buffer
  G4  multi-criterion      Combined

  C1  G1 - G0    Does replaying history help at all, without any curation?
  C2  G2 - G1    Does curating the buffer add anything over drawing at random?
  C3  G2 - G3    The noise-amplification hypothesis (Section 3.7, Section 5.2).
  C4  G4 - G2    Does blending criteria beat committing to one?
  C5  G3 - G0    Is error-driven replay better or worse than not replaying?
  C6  G3 - G1    Is error-driven replay worse than not curating at all?

C6 exists because C3 and C5 together cannot express it. C3 says error-driven
selection trails *other curation*; C5 says it clears (or fails to clear) *no replay*.
Neither states the sharper claim the retention pairs support -- that error-driven
selection is worse than the uncurated buffer it was meant to improve on -- and
differencing C3 and C5 recovers only a point estimate, with no interval and no place
in the family-wise guarantee. Adding it as a sixth planned contrast raises the shared
critical value slightly for C1-C5, which is the honest price of testing it.

The partition is fixed by mechanism *before* looking at any score, which is what
keeps it from being a post-hoc regrouping of the observed ordering. G1 and G4 are
singletons because the thesis assigns them singular roles -- the uncurated control
and the multi-criterion arm -- not because a group of one is statistically
convenient. A group's score on a resample is the unweighted mean of its members'
scores on that resample: "the average strategy of this kind", which is the quantity
the prose's group claims are about.

Two families of test, both single-step max-T (Westfall & Young, 1993) over the shared
2000 draws, exactly as scripts/run_group_ci.py applies it to pairs:

  group_contrasts   The five contrasts above, family-wise-error-controlled within
                    each (slice, metric). Five contrasts rather than 21 pairs is
                    most of the point: the critical value falls, so a margin the
                    pairwise table could not resolve becomes resolvable here.

  band_homogeneity  The claim "read these five as a band" is an absence-of-difference
                    claim, and absence of a significant pairwise result is never by
                    itself evidence for one. So this deliberately reports no
                    "homogeneous" verdict. It reports `resolution` -- the largest
                    simultaneous half-width across the 10 within-band pairs, i.e. the
                    smallest difference this design could have detected anywhere in
                    the band -- and `spread_over_resolution`, the observed spread
                    divided by it. Below 1 means the band's spread is inside what the
                    comparison could resolve, which is the strongest statement the
                    data supports and is the number Section 4.3 gestures at when it
                    says differences of a few thousandths are "smaller than a
                    comparison on this single six-year archive can meaningfully
                    resolve". A proper equivalence claim would need a pre-specified
                    margin and a TOST; this is the descriptive substitute.

Three caveats travel with every number here and belong beside them in the text:

* **Selection bias is uncorrected.** Every arm except the baseline is an argmax over
  4-8 swept configurations; the baseline is a single configuration. C1 and C5
  therefore compare selected arms against an unselected baseline. C3's asymmetry is
  gone: it compared an all-selected G2 against a G3 containing Misclassification
  Buffer, which had no selection advantage at all while it was pinned to one ratio,
  and that biased C3 toward the result the thesis argues for. That strategy now
  sweeps RR 0.2-0.5, so both sides of C3 are equally selected. Resampling test cubes cannot correct this; only nested or
  held-out selection could, and Section 3.8 records why no held-out data remained.
* **The six (slice, metric) families are not independent looks.** F1 and PR-AUC
  correlate at ~0.98 on the same arm, forecast_mean and retention_mean at ~0.87, so
  "significant in all six" is closer to two effective confirmations than six. FWER is
  controlled within each family, never across all 30 tests.
* **A group mean is an average over the strategies that happen to exist here**, not a
  draw from a population of curation methods. With three members in the largest
  group, C2 and C3 speak about these instantiations, not about curation in general.

Reads experiments/bootstrap_ci/pairwise/arm_samples.npz and mean_column_ci.csv,
written by scripts/run_pairwise_ci.py -- run that first. Touches no prediction cache
and no published evaluation table; this is pure reduction of an existing resample.

Usage:
    python -u scripts/run_group_contrasts.py
    python -u scripts/run_group_contrasts.py --alpha 0.10
"""
import argparse
import itertools
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.run_pairwise_ci import (  # noqa: E402
    METRICS,
    PAIRWISE_DIR,
    SAMPLES_PATH,
    SLICES,
)
from src.eval.tables import save_table  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MEAN_CI_PATH = PAIRWISE_DIR / 'mean_column_ci.csv'
GROUP_MEANS_PATH = PAIRWISE_DIR / 'group_means.csv'
CONTRASTS_PATH = PAIRWISE_DIR / 'group_contrasts.csv'
HOMOGENEITY_PATH = PAIRWISE_DIR / 'band_homogeneity.csv'

#: Group membership by row-label prefix. Prefixes rather than whole labels because
#: the forecast and retention panels report the same strategy at different
#: configurations ('Uniform Random (RR=0.4)' vs '(RR=0.5)'), and the group a strategy
#: belongs to is a property of its curation criterion, not of its budget.
GROUPS = {
    'G0_no_replay': ('Baseline',),
    'G1_uncurated': ('Uniform Random',),
    'G2_curated_non_error': ('PR Stratification', 'Confidently Correct',
                             'Uncertainty Prioritization'),
    'G3_error_driven': ('Hard Example Mining', 'Misclassification Buffer'),
    # Combined sits in its own group rather than in G3 even though its
    # *forecast*-optimal allocation carries HE=0.10, an error-driven share: the
    # question it answers is whether blending beats committing to one criterion. The
    # partition is therefore clean on the retention slice (whose Combined allocation
    # is CC/UP/PR only) but not perfectly on the forecast slice, which is worth
    # stating wherever C3 or C4 is quoted for forecasting.
    'G4_multi_criterion': ('Combined',),
}

#: (name, group_a, group_b, claim) -- the contrast is a - b, and `claim` is the
#: sentence in the thesis it is the test for.
CONTRASTS = [
    ('C1_replay_vs_none', 'G1_uncurated', 'G0_no_replay',
     'Replaying history helps without any curation applied'),
    ('C2_curation_vs_random', 'G2_curated_non_error', 'G1_uncurated',
     'Curating the buffer adds over drawing from it at random'),
    ('C3_non_error_vs_error', 'G2_curated_non_error', 'G3_error_driven',
     'Error-driven selection underperforms other curation (noise amplification)'),
    ('C4_blend_vs_single', 'G4_multi_criterion', 'G2_curated_non_error',
     'Blending criteria beats committing to one'),
    ('C5_error_driven_vs_none', 'G3_error_driven', 'G0_no_replay',
     'Error-driven replay is better than not replaying at all'),
    ('C6_error_driven_vs_random', 'G3_error_driven', 'G1_uncurated',
     'Error-driven replay is better than drawing from the buffer at random'),
]

#: The five arms Section 4.3/4.4 calls a band: everything that is replay but not
#: error-driven. Homogeneity within this set is what "read them as a band" asserts.
BAND_GROUPS = ('G1_uncurated', 'G2_curated_non_error', 'G4_multi_criterion')


def assign_groups(labels):
    """Map each arm label to its group, failing loudly on an unclassified arm.

    A label that matches no prefix, or more than one, means the registry and this
    partition have drifted apart. Silently dropping it would quietly change what a
    group mean is an average *of*, which is exactly the kind of error that produces a
    plausible-looking number nobody can reproduce.
    """
    membership = {}
    for label in labels:
        matched = [name for name, prefixes in GROUPS.items()
                   if any(label.startswith(prefix) for prefix in prefixes)]
        if len(matched) != 1:
            raise ValueError(
                f'Arm {label!r} matched {len(matched)} groups ({matched or "none"}); '
                f'every arm must land in exactly one. Update GROUPS.')
        membership.setdefault(matched[0], []).append(label)

    missing = [name for name in GROUPS if name not in membership]
    if missing:
        raise ValueError(f'No arms found for group(s) {missing} among {sorted(labels)}.')
    return membership


def group_samples(samples, point_estimates, membership):
    """Per-group resample vectors and point estimates: unweighted mean over members.

    Averaging elementwise over the resample index is well defined here only because
    every arm was scored on the identical draw sequence (see the docstring of
    scripts/run_pairwise_ci.py); it is the same reason the per-year mean is built the
    same way.
    """
    grouped_samples = {}
    grouped_points = {}
    for name, members in membership.items():
        grouped_samples[name] = np.mean([samples[label] for label in members], axis=0)
        grouped_points[name] = float(np.mean([point_estimates[label] for label in members]))
    return grouped_samples, grouped_points


def max_t_family(diff_samples, point_diffs, alpha):
    """Single-step max-T over one family of contrasts sharing the same resamples.

    Studentise each contrast by its own bootstrap SD, take the max |t| across the
    family on every resample, and use the (1-alpha) quantile of that max as one
    shared critical value. Because the max is taken across the family *within* each
    resample, the threshold reflects the contrasts' actual correlation rather than
    assuming any, and delivers a family-wise guarantee.

    Returns (critical_value, se, half_width, significant, p_value) with one entry per
    contrast. Identical to the construction in scripts/run_group_ci.py, factored to a
    plain-array helper so it can serve both the contrast family and the band family.
    """
    diff_samples = np.atleast_2d(diff_samples)
    point_diffs = np.asarray(point_diffs, dtype=np.float64)
    n_boot = diff_samples.shape[0]

    se = diff_samples.std(axis=0, ddof=1)
    # se == 0 only if two groups are bit-identical on every resample, in which case
    # point_diffs is 0 too and the statistic is defined as 0 rather than 0/0.
    se_safe = np.where(se > 0, se, 1.0)

    studentized = (diff_samples - point_diffs) / se_safe
    max_abs_t = np.max(np.abs(studentized), axis=1)
    critical_value = float(np.percentile(max_abs_t, 100 * (1 - alpha)))

    own_stat = np.where(se > 0, np.abs(point_diffs) / se_safe, 0.0)
    # Floored at 1/(B+1): 2000 resamples cannot resolve a tail finer than that, and
    # reporting a bare 0 would claim they could.
    p_value = np.array([(1 + np.sum(max_abs_t >= stat)) / (n_boot + 1)
                        for stat in own_stat])

    return (critical_value, se, critical_value * se,
            (se > 0) & (own_stat > critical_value), p_value)


def contrast_rows(grouped_samples, grouped_points, alpha):
    """One row per planned contrast, all sharing this family's critical value."""
    diffs = np.stack([grouped_samples[a] - grouped_samples[b]
                      for _, a, b, _ in CONTRASTS], axis=1)
    points = np.array([grouped_points[a] - grouped_points[b] for _, a, b, _ in CONTRASTS])

    critical_value, se, half_width, significant, p_value = max_t_family(diffs, points, alpha)

    return [
        {
            'contrast': name,
            'group_a': group_a,
            'group_b': group_b,
            'claim': claim,
            'a_point_estimate': grouped_points[group_a],
            'b_point_estimate': grouped_points[group_b],
            'point_estimate': float(points[j]),
            'se_diff': float(se[j]),
            'critical_value_maxt': critical_value,
            'sim_ci_lower': float(points[j] - half_width[j]),
            'sim_ci_upper': float(points[j] + half_width[j]),
            'sim_significant': bool(significant[j]),
            'p_value_maxt': float(p_value[j]),
            'n_contrasts_in_family': len(CONTRASTS),
        }
        for j, (name, group_a, group_b, claim) in enumerate(CONTRASTS)
    ]


def homogeneity_rows(samples, point_estimates, membership, alpha):
    """Within-band pairwise max-T, plus the resolution the band claim rests on.

    `resolution` is the largest simultaneous half-width over the within-band pairs:
    the smallest difference this design could have detected anywhere in the band. A
    band claim is only meaningful stated against it -- "no ordering is resolvable"
    means little without saying what would have been.
    """
    band = [label for group in BAND_GROUPS for label in membership[group]]
    pairs = list(itertools.combinations(sorted(band), 2))

    diffs = np.stack([samples[a] - samples[b] for a, b in pairs], axis=1)
    points = np.array([point_estimates[a] - point_estimates[b] for a, b in pairs])

    critical_value, se, half_width, significant, p_value = max_t_family(diffs, points, alpha)

    band_points = [point_estimates[label] for label in band]
    resolution = float(half_width.max())
    observed_spread = float(max(band_points) - min(band_points))
    summary = {
        'band_size': len(band),
        'n_pairs': len(pairs),
        'critical_value_maxt': critical_value,
        'n_significant_pairs': int(significant.sum()),
        # Deliberately not a 'homogeneous' verdict: no significant pair is absence of
        # evidence, not evidence of absence. These two describe what the design could
        # have seen, and leave the equivalence claim to the reader.
        'no_ordering_resolved': bool(significant.sum() == 0),
        'resolution': resolution,
        'observed_spread': observed_spread,
        'spread_over_resolution': observed_spread / resolution if resolution > 0 else float('nan'),
    }

    rows = [
        {
            **summary,
            'strategy_a': label_a,
            'strategy_b': label_b,
            'point_estimate': float(points[j]),
            'se_diff': float(se[j]),
            'sim_ci_lower': float(points[j] - half_width[j]),
            'sim_ci_upper': float(points[j] + half_width[j]),
            'sim_significant': bool(significant[j]),
            'p_value_maxt': float(p_value[j]),
        }
        for j, (label_a, label_b) in enumerate(pairs)
    ]
    return rows


def load_arm_samples():
    if not SAMPLES_PATH.exists():
        raise FileNotFoundError(
            f'Missing {SAMPLES_PATH}. Run scripts/run_pairwise_ci.py first -- this '
            f'script only reduces its output, it does not read predictions itself.')
    with np.load(SAMPLES_PATH) as payload:
        return {key: payload[key] for key in payload.files}


def run(alpha):
    samples_by_key = load_arm_samples()
    mean_df = pd.read_csv(MEAN_CI_PATH, float_precision='round_trip')
    PAIRWISE_DIR.mkdir(parents=True, exist_ok=True)

    mean_rows = []
    contrast_out = []
    homogeneity_out = []

    for slice_name, spec in SLICES.items():
        labels = [label for label, _ in spec['rows']]
        membership = assign_groups(labels)

        for metric in METRICS:
            samples = {label: samples_by_key[f'{slice_name}|{label}|{metric}']
                       for label in labels}
            point_estimates = {
                label: float(mean_df[(mean_df['slice'] == slice_name)
                                     & (mean_df.row_label == label)
                                     & (mean_df.metric == metric)].iloc[0].point_estimate)
                for label in labels
            }

            grouped_samples, grouped_points = group_samples(
                samples, point_estimates, membership)

            for name, members in membership.items():
                arm_samples = grouped_samples[name]
                lower, upper = np.percentile(arm_samples, [100 * alpha / 2,
                                                           100 * (1 - alpha / 2)])
                mean_rows.append({
                    'slice': slice_name, 'metric': metric, 'group': name,
                    'n_members': len(members), 'members': '; '.join(sorted(members)),
                    'point_estimate': grouped_points[name],
                    'ci_lower': float(lower), 'ci_upper': float(upper),
                    'bootstrap_mean': float(arm_samples.mean()),
                    'bootstrap_std': float(arm_samples.std(ddof=1)),
                    'n_bootstrap': int(len(arm_samples)), 'ci_alpha': float(alpha),
                })

            for row in contrast_rows(grouped_samples, grouped_points, alpha):
                contrast_out.append({'slice': slice_name, 'metric': metric, **row})

            for row in homogeneity_rows(samples, point_estimates, membership, alpha):
                homogeneity_out.append({'slice': slice_name, 'metric': metric, **row})

    means_df = pd.DataFrame(mean_rows)
    contrasts_df = pd.DataFrame(contrast_out)
    homogeneity_df = pd.DataFrame(homogeneity_out)

    save_table(means_df, GROUP_MEANS_PATH)
    save_table(contrasts_df, CONTRASTS_PATH)
    save_table(homogeneity_df, HOMOGENEITY_PATH)

    print(f'  wrote {len(means_df)} group-mean rows, {len(contrasts_df)} contrast rows, '
          f'{len(homogeneity_df)} band-pair rows')
    return means_df, contrasts_df, homogeneity_df


def _display_path(path):
    """Repo-relative when it is inside the repo, absolute otherwise."""
    try:
        return str(Path(path).relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def write_manifest(args):
    try:
        commit = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=PROJECT_ROOT,
                                capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        commit = None
    manifest = {
        'created_at': datetime.now(timezone.utc).isoformat(timespec='seconds').replace('+00:00', 'Z'),
        'git_commit': commit,
        'alpha': args.alpha,
        'method': 'single-step max-T (Westfall-Young), studentised, one shared '
                  'critical value per (slice, metric) family',
        'source': _display_path(SAMPLES_PATH),
        'groups': {name: list(prefixes) for name, prefixes in GROUPS.items()},
        'contrasts': [{'name': n, 'a': a, 'b': b, 'claim': c} for n, a, b, c in CONTRASTS],
        'band_groups': list(BAND_GROUPS),
        'group_score': 'unweighted mean of member arms within each resample',
    }
    with open(PAIRWISE_DIR / 'group_contrasts_manifest.json', 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2)
        f.write('\n')


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--alpha', type=float, default=0.05)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    run(args.alpha)
    write_manifest(args)
    print('\nDone.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
