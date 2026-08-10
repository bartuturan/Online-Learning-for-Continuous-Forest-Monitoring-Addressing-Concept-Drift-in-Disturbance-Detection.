"""Aggregate evaluation results across seed runs into mean +/- std.

This is the deliverable of a multi-seed run. One seed tells you what a model
scored; several tell you whether the gap between two strategies is real or is
just where the RNG landed.

Reads each seed's all_families_combined.csv (experiments/seed_<N>/evaluation/...)
plus the original experiments/ tree, which is seed 42.

Two outputs, because they answer different questions:

  per_cell.csv    mean/std across seeds for every
                  (family, table_type, model_year, eval_year, metric).
                  Fine-grained -- for spotting which year a strategy is unstable in.

  per_family.csv  the headline table. Averages a seed's year-cells FIRST, then
                  takes mean/std of those per-seed numbers. That ordering matters:
                  pooling every cell instead would weight a seed by how many rows
                  it happens to contribute, so a seed that failed on two years
                  would quietly count less than the others.

Every row carries n_seeds and the seed list that produced it. A family that
errored in one seed shows n_seeds=3 rather than a mean that silently rests on
fewer samples than the header claims.

IMPORTANT when the split varies per seed (the Monte-Carlo cross-validation
protocol): each seed scored a DIFFERENT test set, so the spread here reflects
data-partition variance as well as training stochasticity. That is a wider and
more honest interval than a fixed-split rerun, but it is not the same quantity --
say which one you ran when you report the numbers.

Usage:
    python scripts/aggregate_seed_results.py                    # auto-discover
    python scripts/aggregate_seed_results.py --seeds 1 2 3      # baseline + these
    python scripts/aggregate_seed_results.py --no-baseline      # seed dirs only
"""

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.seed_run import BASELINE_SEED, experiments_root  # noqa: E402

COMBINED_REL = "evaluation/eval_outputs/unified_eval/all_families_combined.csv"
OUT_DIRNAME = "seed_summary"

#: Aggregated columns. Everything else in all_families_combined.csv is either an
#: identifier or a per-run diagnostic (thresholds, sample counts) that has no
#: meaningful cross-seed mean.
METRICS = ("f1_score", "pr_auc", "roc_auc", "precision", "recall", "accuracy")
KEYS = ("family_id", "table_type", "model_year", "eval_year")


def discover_seeds(root):
    """Seed trees on disk, by directory name. The baseline lives in experiments/
    itself and has no seed_ directory, so it is added separately."""
    base = Path(root) / "experiments"
    if not base.exists():
        return []
    found = []
    for child in sorted(base.iterdir()):
        m = re.fullmatch(r"seed_(\d+)", child.name) if child.is_dir() else None
        if m:
            found.append(int(m.group(1)))
    return sorted(found)


def combined_path_for(root, seed):
    """None for the baseline, which lives in the un-suffixed experiments/ tree."""
    base = experiments_root(root, seed=seed) if seed is not None else experiments_root(root)
    return base / COMBINED_REL


def load_seed(root, seed, label):
    path = combined_path_for(root, seed)
    if not path.exists():
        print(f"  seed {label}: MISSING ({path.relative_to(root)})")
        return None
    df = pd.read_csv(path, float_precision="round_trip")
    missing = [m for m in METRICS if m not in df.columns]
    if missing:
        print(f"  seed {label}: skipped -- missing column(s) {missing}")
        return None
    df["seed"] = label
    print(f"  seed {label}: {len(df):,} rows, {df.family_id.nunique()} families")
    return df


def aggregate_per_cell(frames):
    long = pd.concat(frames, ignore_index=True).melt(
        id_vars=list(KEYS) + ["seed"], value_vars=list(METRICS),
        var_name="metric", value_name="value",
    ).dropna(subset=["value"])
    grouped = long.groupby(list(KEYS) + ["metric"], dropna=False)["value"]
    out = grouped.agg(mean="mean", std="std", n_seeds="count").reset_index()
    seeds = long.groupby(list(KEYS) + ["metric"])["seed"].apply(
        lambda s: ",".join(str(x) for x in sorted(s.unique()))
    ).reset_index(name="seeds")
    return out.merge(seeds, on=list(KEYS) + ["metric"])


def aggregate_per_family(frames):
    """Year-cells collapsed within a seed first, then across seeds -- see module
    docstring for why that ordering is not interchangeable."""
    long = pd.concat(frames, ignore_index=True).melt(
        id_vars=["family_id", "table_type", "seed"], value_vars=list(METRICS),
        var_name="metric", value_name="value",
    ).dropna(subset=["value"])

    per_seed = long.groupby(["family_id", "table_type", "metric", "seed"])["value"].mean().reset_index()
    grouped = per_seed.groupby(["family_id", "table_type", "metric"])["value"]
    out = grouped.agg(mean="mean", std="std", n_seeds="count",
                       min="min", max="max").reset_index()
    seeds = per_seed.groupby(["family_id", "table_type", "metric"])["seed"].apply(
        lambda s: ",".join(str(x) for x in sorted(s.unique()))
    ).reset_index(name="seeds")
    return out.merge(seeds, on=["family_id", "table_type", "metric"])


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--seeds", type=int, nargs="*", default=None,
                        help="Seed trees to include. Default: every experiments/seed_N/ found.")
    parser.add_argument("--no-baseline", action="store_true",
                        help=f"Exclude the original experiments/ tree (seed {BASELINE_SEED}).")
    parser.add_argument("--root", default=str(PROJECT_ROOT))
    parser.add_argument("--out", default=None, help=f"Default: experiments/{OUT_DIRNAME}/")
    args = parser.parse_args(argv)

    root = Path(args.root)
    seeds = args.seeds if args.seeds is not None else discover_seeds(root)

    print("Loading seed runs:")
    frames = []
    if not args.no_baseline:
        df = load_seed(root, None, BASELINE_SEED)
        if df is not None:
            frames.append(df)
    for s in seeds:
        df = load_seed(root, s, s)
        if df is not None:
            frames.append(df)

    if not frames:
        print("\nNo seed results found -- nothing to aggregate.")
        return 1
    if len(frames) == 1:
        # std is undefined for one sample; pandas would emit a column of NaN and
        # the output would look like a variance estimate when it is not one.
        print("\nOnly one run available. Mean would equal that run and std would be "
              "undefined -- run more seeds before aggregating.")
        return 1

    out_dir = Path(args.out) if args.out else root / "experiments" / OUT_DIRNAME
    out_dir.mkdir(parents=True, exist_ok=True)

    per_cell = aggregate_per_cell(frames)
    per_family = aggregate_per_family(frames)
    per_cell.to_csv(out_dir / "per_cell.csv", index=False)
    per_family.to_csv(out_dir / "per_family.csv", index=False)

    print(f"\nWrote {out_dir.relative_to(root)}/per_cell.csv    ({len(per_cell):,} rows)")
    print(f"Wrote {out_dir.relative_to(root)}/per_family.csv  ({len(per_family):,} rows)")

    expected = len(frames)
    short = per_family[per_family.n_seeds < expected]
    if len(short):
        print(f"\n{len(short)} of {len(per_family)} per-family rows rest on fewer than "
              f"{expected} seeds -- a family that errored in some run. Check 'seeds':")
        for _, r in short.head(10).iterrows():
            print(f"  {r.family_id[:52]:<54} {r.table_type:<22} {r.metric:<10} "
                  f"n={r.n_seeds} ({r.seeds})")

    headline = per_family[(per_family.table_type == "own_year") & (per_family.metric == "f1_score")]
    if len(headline):
        print(f"\nown_year f1_score, top 10 by mean (n={expected} seeds):")
        for _, r in headline.nlargest(10, "mean").iterrows():
            print(f"  {r.family_id[:56]:<58} {r['mean']:.4f} +/- {r['std']:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
