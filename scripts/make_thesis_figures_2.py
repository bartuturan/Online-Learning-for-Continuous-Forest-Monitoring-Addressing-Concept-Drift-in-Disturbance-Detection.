"""Generate the three thesis figures from the prepared zarr datasets.

Figure 1 -- disturbance rate per year (positive vs. negative label fraction),
the per-year analogue of the per-split rates reported in the methodology
chapter (train/val/test = 2.0% / 2.0% / 2.1%). Sourced from training_data.zarr,
the same sampled pixel set the split rates come from.

Figure 2 -- spatial coverage map of the 2,216 candidate cubes vs. the 1,975
kept after the "at least one disturbance pixel in any year" filter (see
training_data.manifest.json). Cube centroids come from the position array in
full_dataset_resizedv2.zarr (pos = [lat, lon]); the cube-name string encodes
the same centroid at 2-decimal precision as a sanity check.

Figure 3 -- a hand-picked Sentinel-2 true-colour example with its disturbance
reference mask. Candidate cube/year pairs were scored for (a) 100% cloud-free
top-1 (least-cloud) acquisition, (b) a forest-masked disturbance pixel count
in a visually legible range (150-3000 px out of 16384), and (c) membership in
the kept 1975-cube training set; the shortlist was rendered as thumbnails and
inspected by eye. mc_14.18_49.84 (Czech Republic), year 2022, was selected for
showing several distinct, well-shaped clearcut/bark-beetle patches against a
mostly-forested background with no cloud contamination.

Run from the repository root:

    python scripts/make_thesis_figures.py
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import zarr

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / "figures" / "thesis"

COLOR_POSITIVE = "#D6472B"  # disturbance -- matches src/vis/utils.py cmap_disturbances red
COLOR_NEGATIVE = "#B5B5B5"  # no disturbance -- matches the same colormap's gray
COLOR_KEPT = "#2E6F40"
COLOR_DROPPED = "#C9C9C9"
COLOR_TEXT = "#333333"


def _save(fig, name):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUT_DIR / f"{name}.{ext}", dpi=300, bbox_inches="tight")
    print(f"wrote {OUT_DIR / name}.[png|pdf]")


def fig1_disturbance_rate_per_year(dataset_path=PROJECT_ROOT / "training_data.zarr", year_start=2017):
    """year_start=2017 excludes 2016, which only exists in the data to supply lagged
    features for 2017 and is not one of the modeled years in Sec. 3.3."""
    z = zarr.open(str(dataset_path), mode="r")
    dist = z["disturbances"]  # (pixel, year) in {0, 1, 255}
    all_years = z["year"][:]
    year_mask = all_years >= year_start
    years = all_years[year_mask]
    year_indices = np.where(year_mask)[0]

    pos = np.empty(len(years), dtype=np.int64)
    neg = np.empty(len(years), dtype=np.int64)
    for i, t in enumerate(year_indices):
        col = dist[:, t]
        pos[i] = int(np.sum(col == 1))
        neg[i] = int(np.sum(col == 0))
    total = pos + neg
    pos_rate = 100.0 * pos / total
    neg_rate = 100.0 * neg / total
    overall_rate = 100.0 * pos.sum() / total.sum()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

    x = np.arange(len(years))
    ax1.bar(x, neg_rate, color=COLOR_NEGATIVE, label="Not disturbed", width=0.65)
    ax1.bar(x, pos_rate, bottom=neg_rate, color=COLOR_POSITIVE, label="Disturbed", width=0.65)
    ax1.set_xticks(x)
    ax1.set_xticklabels(years)
    ax1.set_ylabel("Share of labeled pixels (%)")
    ax1.set_ylim(0, 100)
    ax1.set_title("(a) Full label composition")
    ax1.legend(loc="lower right", frameon=False)
    for spine in ("top", "right"):
        ax1.spines[spine].set_visible(False)

    bars = ax2.bar(x, pos_rate, color=COLOR_POSITIVE, width=0.6)
    ax2.axhline(overall_rate, color=COLOR_TEXT, linestyle="--", linewidth=1, alpha=0.7, zorder=1)
    ax2.text(0.02, 0.94, f"mean = {overall_rate:.2f}%", transform=ax2.transAxes,
              ha="left", va="top", fontsize=9, color=COLOR_TEXT)
    for b, v in zip(bars, pos_rate):
        ax2.text(b.get_x() + b.get_width() / 2, v + 0.08, f"{v:.2f}%",
                  ha="center", va="bottom", fontsize=8.5, color=COLOR_TEXT,
                  bbox=dict(facecolor="white", edgecolor="none", pad=0.5))
    ax2.set_xlim(-0.75, len(years) - 0.25)
    ax2.set_xticks(x)
    ax2.set_xticklabels(years)
    ax2.set_ylabel("Disturbance rate (%)")
    ax2.set_ylim(0, pos_rate.max() * 1.35)
    ax2.set_title("(b) Disturbance (positive) rate")
    for spine in ("top", "right"):
        ax2.spines[spine].set_visible(False)

    fig.suptitle("Disturbance rate per year", y=1.03)
    fig.tight_layout()
    _save(fig, "fig1_disturbance_rate_per_year")
    plt.close(fig)

    print("\nyear  pos_rate  neg_rate  n_pos    n_total")
    for t, yr in enumerate(years):
        print(f"{yr}  {pos_rate[t]:7.3f}%  {neg_rate[t]:7.3f}%  {pos[t]:7d}  {total[t]:8d}")


def fig2_spatial_coverage_map(
    source_path=PROJECT_ROOT / "full_dataset_resizedv2.zarr",
    training_path=PROJECT_ROOT / "training_data.zarr",
):
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    z = zarr.open(str(source_path), mode="r")
    names = z["cube"][:]
    centers = z["position"][:, 64, 64, :]  # (n_cubes, 2) -> (lat, lon)
    lat, lon = centers[:, 0], centers[:, 1]

    zt = zarr.open(str(training_path), mode="r")
    kept_names = set(np.unique(zt["cube_name"][:]).tolist())
    kept_mask = np.array([n in kept_names for n in names])

    n_total = len(names)
    n_kept = int(kept_mask.sum())
    n_dropped = n_total - n_kept

    fig = plt.figure(figsize=(8, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent([-11, 40, 34, 71], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND, facecolor="#F2F2ED")
    ax.add_feature(cfeature.OCEAN, facecolor="#DCEBF5")
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5, edgecolor="#888888")
    ax.add_feature(cfeature.BORDERS, linewidth=0.4, edgecolor="#AAAAAA", linestyle=":")

    ax.scatter(lon[~kept_mask], lat[~kept_mask], s=10, color=COLOR_DROPPED,
               edgecolor="none", transform=ccrs.PlateCarree(),
               label=f"Dropped ({n_dropped})", zorder=3)
    ax.scatter(lon[kept_mask], lat[kept_mask], s=10, color=COLOR_KEPT,
               edgecolor="none", transform=ccrs.PlateCarree(),
               label=f"Kept ({n_kept})", zorder=4)

    ax.set_title(f"Spatial coverage of the {n_total} candidate cubes\n"
                 f"({n_kept} kept after the disturbance-presence filter)")
    ax.legend(loc="lower left", frameon=True, fontsize=9)
    ax.gridlines(draw_labels=True, linewidth=0.3, color="#CCCCCC", alpha=0.6)

    _save(fig, "fig2_spatial_coverage_map")
    plt.close(fig)


def _stretch(band, lo=2, hi=98):
    p_lo, p_hi = np.percentile(band, [lo, hi])
    out = np.clip((band.astype(np.float32) - p_lo) / max(p_hi - p_lo, 1e-6), 0, 1)
    return out


def fig3_s2_reference_example(
    source_path=PROJECT_ROOT / "full_dataset_resizedv2.zarr",
    cube_idx=585,
    year_idx=6,
):
    z = zarr.open(str(source_path), mode="r")
    s2 = z["S2"][cube_idx, year_idx, 0]  # (128, 128, 7) least-cloud composite
    s2_bands = list(z["s2_band"][:])
    dist = z["disturbances"][cube_idx, year_idx]  # (128, 128)
    fmask = z["forest_mask"][cube_idx, year_idx]  # (128, 128)
    name = "_".join(str(z["cube"][cube_idx]).split("_")[:3])  # drop version/date/index suffix
    year = int(z["year"][year_idx])

    r, g, b = (s2_bands.index(band) for band in ("B04", "B03", "B02"))
    rgb = np.stack([_stretch(s2[..., r]), _stretch(s2[..., g]), _stretch(s2[..., b])], axis=-1)

    overlay = np.zeros((*dist.shape, 3), dtype=np.float32)
    overlay[:] = [0.85, 0.85, 0.85]
    overlay[fmask == 1] = [0.62, 0.82, 0.62]
    overlay[dist == 1] = [0.84, 0.09, 0.09]

    fig, axes = plt.subplots(1, 2, figsize=(9, 4.6))
    axes[0].imshow(rgb)
    axes[0].set_title("(a) Sentinel-2 true colour (B04/B03/B02)")
    axes[0].axis("off")

    axes[1].imshow(overlay)
    axes[1].set_title("(b) Forest mask & disturbance reference")
    axes[1].axis("off")

    from matplotlib.patches import Patch
    legend_elems = [
        Patch(facecolor=[0.62, 0.82, 0.62], label="Forest"),
        Patch(facecolor=[0.85, 0.85, 0.85], label="Non-forest"),
        Patch(facecolor=[0.84, 0.09, 0.09], label="Disturbance"),
    ]
    axes[1].legend(handles=legend_elems, loc="lower right", fontsize=8, frameon=True)

    fig.suptitle(f"Example cube {name} -- year {year}", y=1.02)
    fig.tight_layout()
    _save(fig, "fig3_s2_reference_example")
    plt.close(fig)


if __name__ == "__main__":
    print("Figure 1: disturbance rate per year")
    fig1_disturbance_rate_per_year()
    print("\nFigure 2: spatial coverage map")
    fig2_spatial_coverage_map()
    print("\nFigure 3: Sentinel-2 + reference example")
    fig3_s2_reference_example()
