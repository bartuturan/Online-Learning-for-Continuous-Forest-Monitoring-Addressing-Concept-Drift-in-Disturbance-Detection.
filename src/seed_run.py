"""One knob -- FONDA_SEED -- that decides a run's seed, its split file, and where
it writes.

Every result under experiments/ was produced with three separate hardcoded 42s:
the model's random_state, REPLAY_RANDOM_STATE for the replay sampler, and the
cube-wise train/val/test split in notebooks/data_prep/Data-prep.ipynb. To run the
pipeline again at a different seed without destroying those results, all three
have to move together AND the outputs have to land somewhere else. Deriving the
output directory and the split path FROM the seed -- rather than exposing them as
independent knobs -- is what makes "wrong directory, right seed" unrepresentable:
there is no combination of env vars that silently overwrites another seed's tree.

Precedence follows resolve_cache_root in src/mlp_replay/disk_feature_cache.py:
explicit argument > environment variable > default.

The default (env var unset) reproduces the pre-seed behaviour exactly -- same
experiments/ tree, same data_split.npz, same seed 42 -- so the existing results,
the Kaggle resume flow, and every notebook run by hand are all unaffected. That
property is worth more than the tidiness of treating 42 as "just another seed",
and tests/test_seed_run.py pins it.
"""

import os
from pathlib import Path

SEED_ENV_VAR = "FONDA_SEED"

#: The seed the existing experiments/ tree was trained with. Used when no seed is
#: active, so an un-seeded run keeps producing bit-identical models.
BASELINE_SEED = 42


def resolve_seed(seed=None):
    """The active seed, or None when this is an ordinary (un-seeded) run.

    None is not the same as BASELINE_SEED: None means "the original experiments/
    tree", while an explicit 42 would mean "a fresh seed_42/ tree that happens to
    use the same RNG". Callers that only care about the RNG want model_seed().
    """
    if seed is not None:
        return int(seed)
    raw = os.environ.get(SEED_ENV_VAR)
    if raw is None or raw == "":
        return None
    try:
        return int(raw)
    except ValueError:
        raise ValueError(
            f"{SEED_ENV_VAR}={raw!r} is not an integer. Unset it for a normal run, "
            f"or set it to the seed you want (e.g. {SEED_ENV_VAR}=1)."
        ) from None


def model_seed(seed=None):
    """What to pass to random_state= / use as REPLAY_RANDOM_STATE."""
    active = resolve_seed(seed)
    return BASELINE_SEED if active is None else active


def experiments_root(project_root, seed=None):
    """Where this run's models, checkpoints and eval tables go.

    Seed trees live UNDER experiments/ rather than beside it so they inherit the
    .gitignore rules that already exist for it -- in particular the
    `!experiments/**/*.csv` re-include, without which every evaluation table a
    seed run produces would be silently ignored by the blanket `*.csv` rule and
    never pushed. On Kaggle that would mean losing them at session end.
    """
    root = Path(project_root) / "experiments"
    active = resolve_seed(seed)
    return root if active is None else root / f"seed_{active}"


def seed_split_path(project_root, seed=None):
    """This run's train/val/test split file.

    Named seed_split_path rather than split_path because every notebook already
    binds a local `split_path`; importing a function under that name would be
    shadowed by the first assignment and blow up on any later call.
    """
    active = resolve_seed(seed)
    name = "data_split.npz" if active is None else f"data_split_seed_{active}.npz"
    return Path(project_root) / name
